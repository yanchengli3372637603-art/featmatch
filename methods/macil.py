import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import os
import glob
import shutil
from contextlib import contextmanager, nullcontext

import logging
import random
import numpy as np
from tqdm import tqdm

from methods.base import BaseLearner
from utils.toolkit import tensor2numpy
from models.network import MANet
from models.attention import Attention_HLoRA, Attention_LoRA, Attention_GLoRA

from utils.schedulers import CosineSchedule
from torch.distributions.multivariate_normal import MultivariateNormal
from utils.toolkit import count_parameters
from models.losses import AngularPenaltySMLoss, MahalanobisLoss, compute_angle_weighted_patch_distillation_loss
import re
from PIL import Image, UnidentifiedImageError
import torchvision.transforms.functional as TF


def _maybe_purge_replay_root(self, replay_root: str):
    """Safely remove all files/subdirs under replay_root.
    Safety guards: never delete '/', HOME, or empty path. Only deletes inside the directory.
    In distributed setting, only runs on rank 0 when torch.distributed is initialized.
    """
    if replay_root is None:
        return
    root = os.path.expanduser(str(replay_root))
    root = os.path.abspath(root)

    # rank-0 only (best-effort)
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            if dist.get_rank() != 0:
                return
    except Exception:
        pass

    # safety guards
    if root in ("/", os.path.expanduser("~")) or len(root.strip()) < 3:
        logging.warning(f"[Replay] Refuse to purge unsafe path: {root}")
        return
    if not os.path.isdir(root):
        return

    for name in os.listdir(root):
        fp = os.path.join(root, name)
        try:
            if os.path.isdir(fp):
                shutil.rmtree(fp, ignore_errors=True)
            else:
                os.remove(fp)
        except Exception as e:
            logging.warning(f"[Replay] Failed to remove {fp}: {e}")
    logging.info(f"[Replay] Purged replay_root: {root}")


class BalancedReplayDataset(torch.utils.data.Dataset):
    """Balanced replay dataset that samples classes uniformly and images uniformly within a class.
    Returns tuples compatible with the rest of MACIL code: (idx, image_tensor, label).
    """

    def __init__(self, by_class_paths, trsf, length=1000000, max_resample=10):
        self.by_class = {int(k): list(v) for k, v in by_class_paths.items() if len(v) > 0}
        self.classes = sorted(self.by_class.keys())
        self.trsf = trsf
        self.length = int(length)
        self.max_resample = int(max_resample)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # resample on IO errors
        for _ in range(self.max_resample):
            c = random.choice(self.classes)
            fp = random.choice(self.by_class[c])
            try:
                with Image.open(fp) as im:
                    im = im.convert("RGB")
                    x = self.trsf(im) if self.trsf is not None else TF.to_tensor(im)
                return idx, x, c
            except (UnidentifiedImageError, OSError, ValueError):
                continue
        # if still failing, raise to surface dataset corruption
        raise UnidentifiedImageError(f"Failed to read replay image after {self.max_resample} retries (last={fp})")


class InfluenceReplayDataset(torch.utils.data.Dataset):
    """Replay dataset with *class sampling weights* that can be updated online.
    Designed for single-process DataLoader (num_workers=0); with multi-workers, weight updates
    won't propagate reliably because each worker holds its own dataset copy.

    Returns: (idx, image_tensor, label).
    """

    def __init__(self, by_class_paths, trsf, length=1000000, max_resample=10, init_weight=1.0):
        self.by_class = {int(k): list(v) for k, v in by_class_paths.items() if len(v) > 0}
        self.classes = sorted(self.by_class.keys())
        self.trsf = trsf
        self.length = int(length)
        self.max_resample = int(max_resample)
        self.class_weights = {int(c): float(init_weight) for c in self.classes}

    def __len__(self):
        return self.length

    def get_class_weights(self):
        return {int(k): float(v) for k, v in self.class_weights.items()}

    def update_class_weights(self, updates: dict, ema: float = 1.0, min_weight: float = 1e-3):
        """EMA update: w <- (1-ema)*w + ema*new_w for classes in `updates`.
        updates: {class_id: new_weight_value}
        """
        ema = float(ema)
        min_weight = float(min_weight)
        for k, new_w in updates.items():
            k = int(k)
            if k not in self.class_weights:
                continue
            old = float(self.class_weights[k])
            nw = float(new_w)
            self.class_weights[k] = max(min_weight, (1.0 - ema) * old + ema * nw)

    def __getitem__(self, idx):
        for _ in range(self.max_resample):
            weights = [self.class_weights[c] for c in self.classes]
            c = random.choices(self.classes, weights=weights, k=1)[0]
            fp = random.choice(self.by_class[c])
            try:
                with Image.open(fp) as im:
                    im = im.convert("RGB")
                    x = self.trsf(im) if self.trsf is not None else TF.to_tensor(im)
                return idx, x, c
            except (UnidentifiedImageError, OSError, ValueError):
                continue
        raise UnidentifiedImageError(f"Failed to read replay image after {self.max_resample} retries (last={fp})")


class Learner(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = MANet(args)
        for module in self._network.modules():
            if isinstance(module, Attention_HLoRA):
                module.init_param()
            if isinstance(module, Attention_LoRA):
                module.init_param()
            if isinstance(module, Attention_GLoRA):
                module.init_param()

        self.args = args
        self.optim = args["optim"]
        self.EPSILON = args["EPSILON"]
        self.init_epoch = args["init_epoch"]
        self.init_lr = args["init_lr"]
        self.init_lr_decay = args["init_lr_decay"]
        self.init_weight_decay = args["init_weight_decay"]
        self.epochs = args["epochs"]
        self.lrate = args["lrate"]
        self.lrate_decay = args["lrate_decay"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]
        self.num_workers = args["num_workers"]
        self.scale = args["scale"]
        self.margin = args["margin"]
        self.total_sessions = args["total_sessions"]
        self.dataset = args["dataset"]
        self.logit_norm = 0.1
        self.topk = 1  # origin is 5
        self.class_num = self._network.class_num
        self.task_sizes = []

        # class prototypes
        self._class_means = None
        self._class_covs = None
        self._old_class_covs = None
        self.acc_matrix = np.zeros((self.total_sessions, self.total_sessions))

        # ===== distilled replay (old classes) =====
        # Expected directory layout (your case):
        #   <replay_root>/<class_id>/0000.png
        # where <class_id> is the ORIGINAL ImageFolder label (0..199 before shuffle).
        # If your replay labels are already mapped to DataManager's shuffled label space,
        # set `replay_labels_are_new=true`.
        self.enable_replay = bool(args.get("enable_replay", False))
        self.replay_root = args.get("replay_root", None)  # e.g. "exported_images"
        self.replay_bs = int(args.get("replay_bs", self.batch_size))
        self.replay_lambda = float(args.get("replay_lambda", 1.0))
        self.replay_ipc = int(args.get("replay_ipc", 1))  # images per class
        self.replay_labels_are_new = bool(args.get("replay_labels_are_new", False))

        # ===== Teacher-KL on replay (logits-level distillation during training) =====
        # When enabled, for replay images we align the current model's class distribution
        # to the frozen old model using KL divergence. This is complementary to patch/feature
        # distillation used on new-class images.
        self.replay_teacher_kd = bool(args.get('replay_teacher_kd', False))
        self.replay_teacher_kd_lambda = float(args.get('replay_teacher_kd_lambda', 1.0))
        self.replay_teacher_kd_T = float(args.get('replay_teacher_kd_T', 2.0))

        # ===== inv-scale for cosine-style interface logits (CE/KD) =====
        # If interface() returns L2-normalized cosine logits in [-1, 1], CE/KD can be near-uniform.
        # We scale logits by replay_inv_scale / replay_inv_temp before CE/KD.
        self.replay_inv_scale = float(args.get('replay_inv_scale', getattr(self, 'scale', 20.0)))
        self.replay_inv_temp = float(args.get('replay_inv_temp', 1.0))

        # ===== In-training classifier alignment =====
        # Use global seen-class logits during task training so new data and replay data
        # directly compete across old/new heads, instead of relying only on post-hoc CA.
        self.train_global_ce = bool(args.get('train_global_ce', False))
        self.global_ce_lambda = float(args.get('global_ce_lambda', 1.0))
        self.local_ce_lambda = float(args.get('local_ce_lambda', 0.3))
        self.global_ce_use_guidance = bool(args.get('global_ce_use_guidance', False))
        self.global_ce_detach_features = bool(args.get('global_ce_detach_features', True))

        # ===== LoRA interference suppression (orthogonality regularization) =====
        # Penalize correlations between the current task's LoRA (A_t/B_t) and previous tasks' LoRA (A_{<t}/B_{<t}).
        # This is data-agnostic and can be optimized together with both new-task data and distilled replay images.
        # Set lora_ortho_lambda>0 to enable.
        self.lora_ortho_lambda = float(args.get('lora_ortho_lambda', 0.0))
        # 'B' (default) tends to work well; options: 'A', 'B', 'both'
        self.lora_ortho_mode = str(args.get('lora_ortho_mode', 'B')).lower()


        # ===== route-2: replay generation by logit inversion (inside CIL) =====
        # Generates replay images for the *current* task classes using the just-trained model.
        # Output layout:
        #   <replay_root>/<new_label>/<0000.png>
        # Here <new_label> is the CIL label space (0..199 when shuffle=True).
        # IMPORTANT: if you use this generator, set `replay_labels_are_new=true`.
        self.generate_replay = bool(args.get("generate_replay", False))
        self.replay_gen_per_class = int(args.get("replay_gen_per_class", 1))
        self.replay_gen_steps = int(args.get("replay_gen_steps", 200))
        self.replay_gen_lr = float(args.get("replay_gen_lr", 0.1))
        self.replay_gen_tv = float(args.get("replay_gen_tv", 1e-4))
        self.replay_gen_l2 = float(args.get("replay_gen_l2", 1e-4))
        self.replay_gen_size = int(args.get("replay_gen_size", 224))
        self.replay_gen_pad = int(args.get("replay_gen_pad", 8))  # random shift padding
        self.replay_gen_skip_existing = bool(args.get("replay_gen_skip_existing", True))
        self.replay_gen_use_fp16 = bool(args.get("replay_gen_use_fp16", False))
        # Normalization assumed by the backbone / dataloader.
        # Keep consistent with your dataset pipeline (ImageNet mean/std is typical).
        self.replay_gen_mean = args.get("replay_gen_mean", [0.5, 0.5, 0.5])
        self.replay_gen_std = args.get("replay_gen_std", [0.5, 0.5, 0.5])

        # Purge any leftover distilled replay images at the beginning of each run/seed.
        # This avoids cross-seed contamination when running multiple seeds sequentially.
        # Default: enabled when generate_replay is enabled.
        self.replay_purge_on_run_start = bool(args.get("replay_purge_on_run_start", self.generate_replay))
        if self.replay_purge_on_run_start and self.replay_root:
            self._maybe_purge_replay_root(self.replay_root)

        # ===== route-2b: refine existing replay images after each task =====
        # Motivation: after learning a new task, decision boundaries change. Refining previously generated
        # replay images against the *latest* (frozen) model can keep them aligned with the new boundaries,
        # while a "keep" penalty prevents drifting too far from earlier replay images.
        #
        # Default behavior: refine ONLY old-task classes (i.e., classes before the just-finished task).
        self.replay_refine = bool(args.get("replay_refine", False))
        self.replay_refine_steps = int(args.get("replay_refine_steps", 50))
        self.replay_refine_lr = float(args.get("replay_refine_lr", 0.05))
        self.replay_refine_tv = float(args.get("replay_refine_tv", self.replay_gen_tv))
        self.replay_refine_l2 = float(args.get("replay_refine_l2", self.replay_gen_l2))
        self.replay_refine_keep = float(args.get("replay_refine_keep", 1.0))
        self.replay_refine_pad = int(args.get("replay_refine_pad", self.replay_gen_pad))
        self.replay_refine_skip_missing = bool(args.get("replay_refine_skip_missing", True))
        # Optional speed knob: refine at most N classes per task (random subset), -1 means all.
        self.replay_refine_max_classes = int(args.get("replay_refine_max_classes", -1))
        self.replay_gen_patch_weight = float(args.get("replay_gen_patch_weight", args.get("distill_patch_weight", 0.0)))
        self.replay_refine_patch_weight = float(
            args.get("replay_refine_patch_weight", args.get("refresh_patch_weight", 0.0)))
        self.replay_patch_sample_k = int(
            args.get("replay_patch_sample_k", args.get("distill_patch_sample_k", args.get("refresh_patch_sample_k", 0))))
        self.replay_patch_real_bs_small = int(args.get("replay_patch_real_bs_small", 8))
        self.replay_patch_real_batches = int(args.get("replay_patch_real_batches", 1))

        # ===== Gradient matching (LinearGM) ===== (LinearGM) =====
        # Old-class update: match LinearGM gradients of (old_model, img0) (target) and (new_model, img) (current).
        # New-class generation: LinearGM targets can be estimated from a small real batch of the current task.
        self.replay_gradmatch = bool(args.get("replay_gradmatch", True))
        self.replay_gradmatch_lambda = float(args.get("replay_gradmatch_lambda", 1.0))
        self.replay_gradmatch_real_bs = int(args.get("replay_gradmatch_real_bs", 32))
        self.replay_gm_views = int(args.get('replay_gm_views', 1))
        # ---- SECOND: GM mode ----
        # headgrad : match dL/d(W,b) of random linear head (original LinearGM)
        # featgrad : match dL/dz where logits = Wz + b (more stable, lower-dim)
        self.replay_gm_mode = str(args.get('replay_gm_mode', 'featgrad')).lower()
        if self.replay_gm_mode not in ('headgrad', 'featgrad'):
            logging.warning(f"[ReplayGM] Unknown replay_gm_mode={self.replay_gm_mode}; fallback to featgrad")
            self.replay_gm_mode = 'featgrad'
        self.replay_proto_lambda = float(args.get('replay_proto_lambda', 0.0))

        # ---- First-priority GM improvements ----
        # (1) Stabilize real GM targets: average over multiple small batches + EMA
        self.replay_gradmatch_real_batches = int(args.get("replay_gradmatch_real_batches", 4))
        self.replay_gradmatch_real_bs_small = int(args.get("replay_gradmatch_real_bs_small", 8))
        self.replay_gradmatch_ema = float(args.get("replay_gradmatch_ema", 0.2))
        # (2) Multi-head random linear classifiers (reduce projection bias)
        self.replay_gm_num_heads = int(args.get("replay_gm_num_heads", 4))
        # Match the LinearGM protocol: re-sample randomly initialized linear heads at every
        # image optimization step instead of reusing one fixed probe for all steps.
        self.replay_gm_reinit_each_step = bool(args.get("replay_gm_reinit_each_step", True))
        # (3) Soft category-aware GM (mix one-hot with class-similarity distribution)
        self.replay_gm_soft = bool(args.get("replay_gm_soft", True))
        self.replay_gm_soft_alpha = float(args.get("replay_gm_soft_alpha", 0.2))
        self.replay_gm_soft_tau = float(args.get("replay_gm_soft_tau", 0.5))

        # ===== Set-level LinearGM replay distillation =====
        # Optimize a small group of class images together so their *batch* gradient matches
        # a balanced real/replay batch. This better captures inter-class boundaries than
        # independent per-class inversion.
        self.replay_set_gm = bool(args.get("replay_set_gm", False))
        self.replay_set_size = int(args.get("replay_set_size", 8))
        self.replay_set_real_per_class = int(args.get("replay_set_real_per_class", 2))
        self.replay_set_gm_reinit_each_step = bool(
            args.get("replay_set_gm_reinit_each_step", args.get("replay_gm_reinit_each_step", True))
        )
        self.replay_set_real_mode = str(args.get("replay_set_real_mode", "test"))

        # caches for GM improvements
        self._gm_real_ema = {}
        self._gm_soft_label_cache = {}
        # ===== Influence-guided replay sampling (gradient conflict) =====
        # When enabled, the replay loader samples old classes with weights updated online.
        # Weight update uses gradient conflict score computed on a small subset of parameters
        # (trainable classifier heads / last-task LoRA params) and favors replay classes whose
        # gradients conflict with current new-task gradients.
        self.replay_influence_sampling = bool(args.get("replay_influence_sampling", False))
        self.replay_infl_update_interval = int(args.get("replay_infl_update_interval", 20))
        self.replay_infl_ema = float(args.get("replay_infl_ema", 0.2))
        self.replay_infl_eps = float(args.get("replay_infl_eps", 0.05))
        self.replay_infl_max_classes = int(args.get("replay_infl_max_classes", 8))
        self.replay_infl_min_weight = float(args.get("replay_infl_min_weight", 1e-3))

        # ===== Replay gradient guidance =====
        # Instead of adding replay loss directly to the scalar objective, use replay gradients
        # as a guidance signal for the main task gradient (A-GEM-style projection).
        self.replay_grad_guidance = bool(args.get("replay_grad_guidance", self.enable_replay))
        self.replay_grad_guidance_eps = float(args.get("replay_grad_guidance_eps", 1e-12))
        self._replay_dataset = None  # set in incremental_train if replay is built
        self._infl_params_cache = None

        # Keep a reference to DataManager for post-task generation (so we can sample real images).
        self._data_manager = None

    @contextmanager
    def _force_autograd(self):
        """Force-enable gradients inside the context (compatible across torch versions).
        Note: this does NOT disable torch.inference_mode if the caller wrapped us inside it.
        Ensure trainer does not call after_task() under inference_mode.
        """
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        try:
            yield
        finally:
            torch.set_grad_enabled(prev)

    @contextmanager
    def _maybe_disable_inference(self):
        """Best-effort: disable torch.inference_mode if supported by this torch version.
        If not supported, this is a no-op. Prefer keeping after_task() outside inference_mode in trainer.
        """
        try:
            ctx = torch.inference_mode(False)
        except Exception:
            ctx = nullcontext()
        with ctx:
            yield

    def after_task(self):
        # `self._old_network` at this moment is the PRE-task frozen model (teacher),
        # while `self._network` is the just-trained model for the current task (student).
        pre_task_old = self._old_network  # may be None on task0
        old_end = int(self._known_classes)
        new_end = int(self._total_classes)
        start_new = old_end  # classes [start_new, new_end) are the just-learned ones

        # Last task does not need replay image generation/refinement (no future task to benefit).
        is_last_task = (int(getattr(self, "_cur_task", 0)) >= int(self.total_sessions) - 1)

        logging.info('Exemplar size: {}'.format(self.exemplar_size))
        self._old_class_covs = None

        if is_last_task:
            logging.info('[Replay] Last task: skip replay refinement and generation.')
        else:
            # ===== (1) Update OLD-class replay images =====
            if self.replay_refine and self.replay_root and (pre_task_old is not None) and old_end > 0:
                try:
                    with self._maybe_disable_inference(), self._force_autograd():
                        self._update_old_replay_images_after_task(
                            old_model=pre_task_old,
                            new_model=self._network,
                            old_end=old_end,
                            total_classes=new_end,
                        )
                except Exception as e:
                    logging.exception(f'[ReplayUpdate] update old replay failed: {e}')
            elif self.replay_refine and (pre_task_old is None or old_end == 0):
                logging.info('[ReplayUpdate] skip (no old classes / no pre-task teacher).')

            # ===== (2) Generate replay images for NEW classes =====
            if self.generate_replay:
                if not self.replay_root:
                    logging.warning('[ReplayGen] generate_replay is True but replay_root is empty; skip generation.')
                else:
                    try:
                        with self._maybe_disable_inference(), self._force_autograd():
                            self._generate_replay_images_for_current_task(
                                start_c=start_new, end_c=new_end, model=self._network
                            )
                    except Exception as e:
                        logging.exception(f'[ReplayGen] generation failed: {e}')

        # ===== (3) Freeze a snapshot for the NEXT task =====
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes

    def _scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale cosine-style logits before CE/KD.
        Effective scale = replay_inv_scale / replay_inv_temp.
        """
        inv_scale = float(getattr(self, 'replay_inv_scale', getattr(self, 'scale', 20.0)))
        inv_temp = float(getattr(self, 'replay_inv_temp', 1.0))
        return logits * (inv_scale / max(inv_temp, 1e-6))

    def _seen_class_logits_from_features(self, model, features: torch.Tensor) -> torch.Tensor:
        """Compute interface-style logits from cached class-token features."""
        logits = []
        feat = F.normalize(features, p=2, dim=1)
        for head in model.classifier_pool[:model.numtask]:
            logits.append(F.linear(feat, F.normalize(head.weight, p=2, dim=1)))
        return torch.cat(logits, dim=1)

    def _lora_ortho_loss(self, task_id: int) -> torch.Tensor:
        """Orthogonality penalty between current-task LoRA params and previous-task LoRA params.

        We detect per-task LoRA parameters by pattern-matching named parameters:
          - ParameterList style:  ... .lora_XXX.<task_id>
          - Bank tensor style:     ... lora_XXX with shape[0] == total_sessions

        Old-task tensors are detached so gradients flow only into the current task's LoRA.
        """
        lam = float(getattr(self, 'lora_ortho_lambda', 0.0))
        if lam <= 0.0 or task_id <= 0:
            return torch.zeros((), device=self._device)

        mode = str(getattr(self, 'lora_ortho_mode', 'b')).lower()
        if mode not in ('a', 'b', 'both'):
            mode = 'b'

        n_tasks = int(self.args.get('total_sessions', task_id + 1))

        groups = {}
        for name, p in self._network.named_parameters():
            if 'lora_' not in name.lower():
                continue
            m = re.search(r'\.(\d+)$', name)
            if m is not None:
                tid = int(m.group(1))
                if tid < 0 or tid >= n_tasks:
                    continue
                base = name[:name.rfind('.')]
                groups.setdefault(base, {})[tid] = p
            else:
                if p.dim() >= 1 and int(p.shape[0]) == n_tasks:
                    groups.setdefault(name, {})['bank'] = p

        total_loss = torch.zeros((), device=self._device)

        def _as2d(t: torch.Tensor) -> torch.Tensor:
            t = t.float()
            if t.dim() == 1:
                return t.view(1, -1)
            if t.dim() == 2:
                return t
            return t.reshape(t.shape[0], -1)

        for base, d in groups.items():
            base_l = base.lower()
            is_a = 'lora_a' in base_l
            is_b = 'lora_b' in base_l

            if mode == 'a' and not is_a:
                continue
            if mode == 'b' and not is_b:
                continue
            if mode == 'both' and not (is_a or is_b):
                continue

            if 'bank' in d:
                bank = d['bank']
                cur = bank[task_id]
                olds = [bank[k].detach() for k in range(task_id)]
            else:
                if task_id not in d:
                    continue
                cur = d[task_id]
                old_keys = [k for k in d.keys() if isinstance(k, int) and k < task_id]
                if not old_keys:
                    continue
                olds = [d[k].detach() for k in sorted(old_keys)]

            if len(olds) == 0:
                continue

            cur2 = _as2d(cur)
            old2 = [_as2d(o) for o in olds]

            if all(o.shape[0] == cur2.shape[0] for o in old2):
                old_cat = torch.cat(old2, dim=1)
                prod = cur2.t().matmul(old_cat)
            elif all(o.shape[1] == cur2.shape[1] for o in old2):
                old_cat = torch.cat(old2, dim=0)
                prod = cur2.matmul(old_cat.t())
            else:
                curv = cur2.reshape(1, -1)
                old_cat = torch.cat([o.reshape(1, -1) for o in old2], dim=0)
                prod = curv.matmul(old_cat.t())

            total_loss = total_loss + prod.pow(2).mean()

        return total_loss * lam



    def _sync_replay_gen_stats_from_trsf(self, trsf):
        """Sync replay_gen_mean/std from the *actual* training transform.

        Goal: make distilled replay generation/refinement see the exact same input distribution
        as the real training dataloader.
        - If config explicitly sets replay_gen_mean/std, we respect it (do not overwrite).
        """
        if not getattr(self, 'enable_replay', False):
            return

        # Respect explicit user settings
        if isinstance(getattr(self, 'args', None), dict) and ('replay_gen_mean' in self.args or 'replay_gen_std' in self.args):
            return

        if trsf is None:
            return

        # torchvision.transforms.Compose has .transforms
        if hasattr(trsf, 'transforms'):
            trsf_list = list(trsf.transforms)
        elif isinstance(trsf, (list, tuple)):
            trsf_list = list(trsf)
        else:
            trsf_list = []

        for t in trsf_list:
            if t.__class__.__name__.lower() != 'normalize':
                continue
            mean = getattr(t, 'mean', None)
            std = getattr(t, 'std', None)
            if mean is None or std is None:
                continue
            try:
                mean_list = [float(x) for x in mean]
                std_list = [float(x) for x in std]
            except Exception:
                try:
                    mean_list = [float(x) for x in mean.tolist()]
                    std_list = [float(x) for x in std.tolist()]
                except Exception:
                    continue

            self.replay_gen_mean = mean_list
            self.replay_gen_std = std_list
            logging.info(f"[Replay] Synced replay_gen_mean/std from train transform: mean={mean_list}, std={std_list}")
            return

        return

    def incremental_train(self, data_manager):

        # Keep reference for post-task replay generation (sample real images for feature prototypes).
        self._data_manager = data_manager

        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(data_manager.get_task_size(self._cur_task))
        self._network.update_fc(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train')
        # Keep distilled replay generation/refinement normalization consistent with the real training pipeline
        self._sync_replay_gen_stats_from_trsf(getattr(train_dataset, 'trsf', None))
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers, pin_memory=True)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                      num_workers=self.num_workers, pin_memory=True)

        # Semantic Shift old embedding
        if self._cur_task > 0 and self._old_network is not None:
            self._old_network.to(self._device)
            train_embeddings_old, _ = self.extract_features(self.train_loader, self._old_network, self._cur_task - 1)
            if self.args.get('cc', False) is True:
                self._old_class_covs = self._compute_class_invcov(data_manager)

        # ===== build replay loader (distilled old-class images) =====
        replay_loader = None
        if self.enable_replay and self._cur_task > 0 and self.replay_root:
            rep_paths, rep_labels = self._collect_replay_from_dir(self.replay_root)
            if rep_paths is not None and len(rep_paths) > 0:
                rep_labels = self._map_replay_labels(rep_labels, data_manager)

                # keep only already learned classes (0..known-1)
                keep = rep_labels < self._known_classes
                rep_paths, rep_labels = rep_paths[keep], rep_labels[keep]

                # optional cap per class (keep most recent ipc images). set replay_ipc<=0 to keep all.
                rep_paths, rep_labels = self._limit_replay_ipc(rep_paths, rep_labels, self.replay_ipc)

                if len(rep_paths) > 0:
                    # build per-class path lists for uniform class sampling
                    by_class = {}
                    for p, y in zip(rep_paths.tolist(), rep_labels.tolist()):
                        by_class.setdefault(int(y), []).append(str(p))

                    # use the SAME training transform as current task data for distribution match
                    trsf = getattr(train_dataset, 'trsf', None)

                    # long-length dataset so we never exhaust it within an epoch
                    approx_len = max(len(train_dataset), 10000)
                    # If influence sampling is enabled, use InfluenceReplayDataset so we can update class sampling weights online.
                    if self.replay_influence_sampling:
                        replay_dataset = InfluenceReplayDataset(by_class, trsf=trsf, length=approx_len)
                    else:
                        replay_dataset = BalancedReplayDataset(by_class, trsf=trsf, length=approx_len)

                    # keep a handle so we can update weights during training (only meaningful for InfluenceReplayDataset)
                    self._replay_dataset = replay_dataset

                    # IMPORTANT: influence sampling requires num_workers=0 so weight updates take effect immediately.
                    rep_workers = 0 if self.replay_influence_sampling else self.num_workers

                    replay_loader = DataLoader(
                        replay_dataset,
                        batch_size=self.replay_bs,
                        shuffle=False,  # dataset itself is random
                        num_workers=rep_workers,
                        pin_memory=True,
                        drop_last=True,
                    )

        self._train(self.train_loader, self.test_loader, replay_loader=replay_loader)

        # Semantic Shift
        if self._cur_task > 0:
            train_embeddings_new, _ = self.extract_features(self.train_loader, self._network)
            old_class_mean = self._class_means[:self._known_classes]
            gap = self.displacement(train_embeddings_old, train_embeddings_new, old_class_mean, 1.0)
            if self.args.get('msc', False) is True:
                old_class_mean += gap
                self._class_means[:self._known_classes] = old_class_mean

        # update mean and cov and classifier alignment
        self._compute_class_mean(data_manager, check_diff=False, oracle=False)
        if self._cur_task > 0 and self.args.get('ca', False) is True:
            self._stage2_compact_classifier(self.task_sizes[-1])

    def _train(self, train_loader, test_loader, replay_loader=None):
        self._network.to(self._device)

        network_params = []
        # When doing joint training/global CE, update all seen classifier heads so the
        # old-vs-new boundary is learned during task training rather than only in CA.
        train_all_seen_heads = (
                self._cur_task > 0
                and (
                        (self.enable_replay and replay_loader is not None)
                        or bool(getattr(self, "train_global_ce", False))
                )
        )

        for name, param in self._network.named_parameters():
            param.requires_grad_(False)

            # ---- classifier / logit parameters ----
            if train_all_seen_heads:
                m = re.search(r"(^|\.)classifier_pool\.(\d+)($|\.)", name)
                if m is not None:
                    head_id = int(m.group(2))
                    if head_id <= (self._network.numtask - 1):
                        param.requires_grad_(True)
                        network_params.append({'params': param})
            else:
                if re.search(rf"(^|\.)classifier_pool\.{self._network.numtask - 1}($|\.)", name) is not None:
                    param.requires_grad_(True)
                    network_params.append({'params': param})

            if self.args['lora_type'] == 'elora':
                if re.search(rf"(^|\.)lora_B_k\.{self._network.numtask - 1}($|\.)", name) is not None:
                    param.requires_grad_(True)
                    network_params.append({'params': param})
                if re.search(rf"(^|\.)lora_B_v\.{self._network.numtask - 1}($|\.)", name) is not None:
                    param.requires_grad_(True)
                    network_params.append({'params': param})
                if re.search(rf"(^|\.)lora_A_k\.{self._network.numtask - 1}($|\.)", name) is not None:
                    param.requires_grad_(True)
                    network_params.append({'params': param})
                if re.search(rf"(^|\.)lora_A_v\.{self._network.numtask - 1}($|\.)", name) is not None:
                    param.requires_grad_(True)
                    network_params.append({'params': param})
            if self.args['lora_type'] == 'hlora' or self.args['lora_type'] == 'glora':
                if re.search(rf"(^|\.)elora_B_k\.{self._network.numtask - 1}($|\.)", name) is not None:
                    param.requires_grad_(True)
                    network_params.append({'params': param})
                if re.search(rf"(^|\.)elora_B_v\.{self._network.numtask - 1}($|\.)", name) is not None:
                    param.requires_grad_(True)
                    network_params.append({'params': param})
                if re.search(rf"(^|\.)glora_B_k($|\.)", name) is not None:
                    param.requires_grad_(True)
                    network_params.append({'params': param})
                if re.search(rf"(^|\.)glora_B_v($|\.)", name) is not None:
                    param.requires_grad_(True)
                    network_params.append({'params': param})
                if re.search(rf"(^|\.)glora_A_k($|\.)", name) is not None:
                    param.requires_grad_(True)
                    network_params.append({'params': param})
                if re.search(rf"(^|\.)glora_A_v($|\.)", name) is not None:
                    param.requires_grad_(True)
                    network_params.append({'params': param})

        if self._cur_task == 0:
            if self.optim == 'sgd':
                optimizer = optim.SGD(params=network_params, momentum=0.9, lr=self.init_lr,
                                      weight_decay=self.init_weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.init_epoch)
            elif self.optim == 'adam':
                optimizer = optim.Adam(params=network_params, lr=self.init_lr, weight_decay=self.init_weight_decay,
                                       betas=(0.9, 0.999))
                scheduler = CosineSchedule(optimizer=optimizer, K=self.init_epoch)
            else:
                raise Exception
            self.run_epoch = self.init_epoch
            self.train_function(train_loader, test_loader, optimizer, scheduler, replay_loader=replay_loader)
        else:
            if self.optim == 'sgd':
                optimizer = optim.SGD(params=network_params, momentum=0.9, lr=self.lrate,
                                      weight_decay=self.weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.epochs)
            elif self.optim == 'adam':
                optimizer = optim.Adam(params=network_params, lr=self.lrate, weight_decay=self.weight_decay,
                                       betas=(0.9, 0.999))
                scheduler = CosineSchedule(optimizer=optimizer, K=self.epochs)
            else:
                raise Exception
            self.run_epoch = self.epochs
            self.train_function(train_loader, test_loader, optimizer, scheduler, replay_loader=replay_loader)


# =========================
# Influence-guided replay sampling (gradient conflict)
# =========================
def _get_influence_params(self):
    """Select a small subset of parameters to compute gradient conflict on.
    We use *trainable* parameters (requires_grad=True), which in this code are
    the classifier heads and last-task LoRA/adapter params.
    Cached per-epoch to avoid repeated scans.
    """
    if self._infl_params_cache is not None:
        return self._infl_params_cache
    params = []
    for _n, _p in self._network.named_parameters():
        if _p.requires_grad:
            params.append(_p)
    self._infl_params_cache = params
    return params


def _cosine_from_grads(grads_a, grads_b, eps=1e-8):
    """Cosine similarity between two gradient lists returned by torch.autograd.grad.
    Missing grads (None) are treated as zeros (by skipping their contributions).
    """
    dot = 0.0
    na = 0.0
    nb = 0.0
    for ga, gb in zip(grads_a, grads_b):
        if ga is not None:
            ga_det = ga.detach()
            na = na + (ga_det * ga_det).sum()
        if gb is not None:
            gb_det = gb.detach()
            nb = nb + (gb_det * gb_det).sum()
        if ga is not None and gb is not None:
            dot = dot + (ga_det * gb_det).sum()
    denom = (na.sqrt() * nb.sqrt() + eps)
    return dot / denom


def _compute_param_grads(self, loss, params):
    if (loss is None) or (not isinstance(loss, torch.Tensor)) or (not loss.requires_grad):
        return tuple(None for _ in params)
    return torch.autograd.grad(
        loss,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )


def _set_param_grads(self, params, grads):
    for p, g in zip(params, grads):
        if not p.requires_grad:
            continue
        if g is None:
            p.grad = None
        else:
            p.grad = g.detach()


def _replay_grad_guidance(self, main_grads, replay_grads, eps: float = 1e-12):
    """Project the main gradient away from replay conflicts.

    If the replay gradient points in a conflicting direction, remove its component from the
    main gradient (A-GEM-style). Parameters that are touched only by the replay loss keep
    their replay gradient so old heads can still adapt.
    """
    eps = float(eps)
    dot = torch.zeros((), device=self._device)
    main_norm = torch.zeros((), device=self._device)
    replay_norm = torch.zeros((), device=self._device)

    for gm, gr in zip(main_grads, replay_grads):
        if gm is not None:
            gm_det = gm.detach()
            main_norm = main_norm + (gm_det * gm_det).sum()
        if gr is not None:
            gr_det = gr.detach()
            replay_norm = replay_norm + (gr_det * gr_det).sum()
        if gm is not None and gr is not None:
            dot = dot + (gm.detach() * gr.detach()).sum()

    conflict = bool(dot.item() < 0.0)
    denom = replay_norm + eps
    coeff = dot / denom if conflict else torch.zeros_like(dot)

    guided = []
    for gm, gr in zip(main_grads, replay_grads):
        if gm is None and gr is None:
            guided.append(None)
        elif gm is None:
            guided.append(gr.detach())
        elif gr is None:
            guided.append(gm.detach())
        elif conflict:
            guided.append((gm - coeff * gr).detach())
        else:
            guided.append(gm.detach())

    cos = dot / (main_norm.sqrt() * replay_norm.sqrt() + eps)
    stats = {
        "conflict": conflict,
        "dot": float(dot.detach().item()),
        "cos": float(cos.detach().item()),
        "main_norm": float(main_norm.detach().sqrt().item()),
        "replay_norm": float(replay_norm.detach().sqrt().item()),
    }
    return guided, stats


def _update_replay_influence_weights(self, loss_cos, inputs_new, targets_new, rep_inputs, rep_targets):
    """Update class sampling weights in InfluenceReplayDataset based on gradient conflict.

    Score: s_c = max(0, -cos(g_c, g_new)), where:
      - g_new: gradients on current task batch (classification loss only),
      - g_c: gradients on replay samples of old class c (CE on interface logits).

    For efficiency, we update at most `replay_infl_max_classes` classes per call.
    """
    if (not self.replay_influence_sampling) or (self._replay_dataset is None) or (
            not isinstance(self._replay_dataset, InfluenceReplayDataset)):
        return
    if inputs_new is None or rep_inputs is None or rep_targets is None:
        return

    params = self._get_influence_params()
    if len(params) == 0:
        return

    # g_new: current-task classification loss only
    if bool(getattr(self, "train_global_ce", False)) and self._cur_task > 0:
        out_new = self._network(inputs_new)
        feat_new = out_new["features"].detach() if bool(getattr(self, "global_ce_detach_features", True)) else out_new["features"]
        logits_new_full = self._seen_class_logits_from_features(self._network, feat_new)
        end_c = min(int(self._total_classes), int(logits_new_full.shape[1]))
        loss_new = F.cross_entropy(self._scale_logits(logits_new_full[:, :end_c]), targets_new)
    else:
        out_new = self._network(inputs_new)
        logits_new = out_new['logits']
        targets_local = targets_new - int(self._known_classes)
        loss_new = loss_cos(logits_new, targets_local)
    grads_new = self._compute_param_grads(loss_new, params)

    uniq = torch.unique(rep_targets).detach().cpu().tolist()
    if len(uniq) == 0:
        return
    random.shuffle(uniq)
    uniq = uniq[:max(1, int(self.replay_infl_max_classes))]

    updates = {}
    total_classes = int(self._total_classes)

    for c in uniq:
        c = int(c)
        idx = (rep_targets == c).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        x_c = rep_inputs[idx]
        y_c = rep_targets[idx]
        if bool(getattr(self, "train_global_ce", False)) and self._cur_task > 0 and bool(
                getattr(self, "global_ce_detach_features", True)):
            feat_old = self._network.extract_vector(x_c).detach()
            logits_old_full = self._seen_class_logits_from_features(self._network, feat_old)
        else:
            logits_old_full = self._network.interface(x_c)
        if bool(getattr(self, "train_global_ce", False)) and self._cur_task > 0:
            end_c = min(int(self._total_classes), int(logits_old_full.shape[1]))
            logits_old_scaled = self._scale_logits(logits_old_full[:, :end_c])
        else:
            old_end = min(int(self._known_classes), int(logits_old_full.shape[1]))
            logits_old_scaled = self._scale_logits(logits_old_full[:, :old_end])
        loss_c = F.cross_entropy(logits_old_scaled, y_c)
        grads_c = self._compute_param_grads(loss_c, params)
        cos = float(self._cosine_from_grads(grads_c, grads_new).detach().item())
        score = max(0.0, -cos)  # conflict => higher score
        updates[c] = float(self.replay_infl_eps + score)

    if not updates:
        return

    self._replay_dataset.update_class_weights(
        updates,
        ema=float(self.replay_infl_ema),
        min_weight=float(self.replay_infl_min_weight),
    )


def train_function(self, train_loader, test_loader, optimizer, scheduler, replay_loader=None):
    logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))
    # Double check
    enabled = set()
    for name, param in self._network.named_parameters():
        if param.requires_grad:
            enabled.add(name)
    logging.info(f"Parameters to be updated: {enabled}")

    prog_bar = tqdm(range(self.run_epoch))

    loss_cos = AngularPenaltySMLoss(loss_type='cosface', s=self.scale, m=self.margin)
    if self._cur_task > 0 and self.args.get('cc', False) is True:
        loss_maha = MahalanobisLoss(self._old_class_covs)
    use_global_ce = bool(getattr(self, "train_global_ce", False)) and self._cur_task > 0

    for _, epoch in enumerate(prog_bar):
        self._network.train()
        self._infl_params_cache = None  # refresh param list each epoch
        losses = 0.
        correct, total = 0, 0
        rep_iter = iter(replay_loader) if replay_loader is not None else None
        log_interval = 50  # print losses every 50 iters
        loss_new_sum, loss_old_sum, loss_tot_sum = 0.0, 0.0, 0.0
        loss_new_cnt, loss_old_cnt = 0, 0
        replay_guide_cnt = 0
        replay_guide_conflict_cnt = 0
        params = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)

            # ----- mix old distilled replay into the same step -----
            if replay_loader is not None:
                try:
                    _, rep_inputs, rep_targets = next(rep_iter)
                except StopIteration:
                    rep_iter = iter(replay_loader)
                    _, rep_inputs, rep_targets = next(rep_iter)
                rep_inputs = rep_inputs.to(self._device)
                rep_targets = rep_targets.to(self._device)

                # Influence-guided replay: periodically update per-class sampling weights
                if self.replay_influence_sampling and (self._replay_dataset is not None) and (
                        self.replay_infl_update_interval > 0):
                    if (i % int(self.replay_infl_update_interval)) == 0:
                        try:
                            self._update_replay_influence_weights(
                                loss_cos=loss_cos,
                                inputs_new=inputs,  # current-task batch (train_loader contains only new classes)
                                targets_new=targets,
                                rep_inputs=rep_inputs,
                                rep_targets=rep_targets,
                            )
                        except Exception as _e:
                            logging.warning(f"[InfluenceReplay] weight update failed: {_e}")

                inputs = torch.cat([inputs, rep_inputs], dim=0)
                targets = torch.cat([targets, rep_targets], dim=0)

            is_new = targets >= self._known_classes
            is_old = ~is_new

            loss = 0.0

            loss_new_val = 0.0
            loss_old_val = 0.0
            loss_old_ce_val = 0.0
            loss_kd_val = 0.0
            loss_ortho_val = 0.0
            loss_new_global_val = 0.0
            loss_new_local_val = 0.0
            replay_cos_val = 0.0
            replay_conflict = False
            # ===== current task supervised training (new classes) =====
            loss_new = torch.zeros((), device=self._device)
            if is_new.any():
                inputs_new = inputs[is_new]
                targets_new_global = targets[is_new]
                targets_new = targets_new_global - self._known_classes  # -> [0..task_size-1]

                output = self._network(inputs_new)
                logits = output['logits']
                features = output['features']
                patch_tokens = output['patch_tokens']
                loss_new_local = loss_cos(logits, targets_new)
                loss_new_local_val = float(loss_new_local.detach().item())

                if use_global_ce:
                    global_features = features.detach() if bool(getattr(self, "global_ce_detach_features", True)) else features
                    logits_global_full = self._seen_class_logits_from_features(self._network, global_features)
                    end_c = min(int(self._total_classes), int(logits_global_full.shape[1]))
                    logits_global = self._scale_logits(logits_global_full[:, :end_c])
                    loss_new_global = F.cross_entropy(logits_global, targets_new_global)
                    loss_new_global_val = float(loss_new_global.detach().item())
                    loss_new = (
                            float(getattr(self, "global_ce_lambda", 1.0)) * loss_new_global
                            + float(getattr(self, "local_ce_lambda", 0.3)) * loss_new_local
                    )
                else:
                    loss_new = loss_new_local

                # keep your existing feature/patch distillation (for new-class batch)
                if self._cur_task > 0 and self.args.get('cc', False) is True:
                    with torch.no_grad():
                        old_output = self._old_network(inputs_new)
                        old_features = old_output['features']
                        old_patch_tokens = old_output['patch_tokens']
                    loss_new = loss_new + loss_maha(old_features, features, targets_new)
                    loss_new = loss_new + self.args['lamb_p'] * compute_angle_weighted_patch_distillation_loss(
                        patch_tokens, old_patch_tokens, features
                    )

                loss_new_val = float(loss_new.detach().item())

                loss_new_sum += loss_new_val

                loss_new_cnt += 1
                # train acc follows the active supervised objective.
                if use_global_ce:
                    _, preds = torch.max(logits_global, dim=1)
                    correct += preds.eq(targets_new_global.expand_as(preds)).cpu().sum()
                else:
                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets_new.expand_as(preds)).cpu().sum()
                total += len(targets_new)

            # ===== old-class rehearsal (distilled images) =====
            # Use `interface()` to get logits over all seen classes.
            if is_old.any():
                inputs_old = inputs[is_old]
                targets_old = targets[is_old]
                # (1) old-class supervised replay loss (simulate joint training)
                if use_global_ce and bool(getattr(self, "global_ce_detach_features", True)):
                    old_features = self._network.extract_vector(inputs_old).detach()
                    logits_old_full = self._seen_class_logits_from_features(self._network, old_features)
                else:
                    logits_old_full = self._network.interface(inputs_old)
                old_end = int(self._known_classes)
                old_end = min(old_end, int(logits_old_full.shape[1]))
                if use_global_ce:
                    total_end = min(int(self._total_classes), int(logits_old_full.shape[1]))
                    logits_old_ce = logits_old_full[:, :total_end]
                else:
                    logits_old_ce = logits_old_full[:, :old_end]
                logits_old_scaled = self._scale_logits(logits_old_ce)
                loss_old_ce = F.cross_entropy(logits_old_scaled, targets_old)
                logits_old = logits_old_full[:, :old_end]

                # (1b) Teacher-KL on replay (optional): logits-level distillation from frozen old model
                loss_old_kd = None
                loss_kd_val = 0.0
                if self.replay_teacher_kd and (self._cur_task > 0) and (self._old_network is not None):
                    with torch.no_grad():
                        t_logits_full = self._old_network.interface(inputs_old)
                    # Align teacher/student to old classes only; teacher has no new-class logits.
                    t_end = min(int(t_logits_full.shape[1]), int(old_end))
                    kd_end = min(int(logits_old.shape[1]), int(t_end))
                    if kd_end <= 0:
                        loss_old_kd = None
                        loss_kd_val = 0.0
                    else:
                        t_logits = t_logits_full[:, :kd_end]
                        s_logits_kd = logits_old[:, :kd_end]
                        T = float(self.replay_teacher_kd_T)
                        s_logits_kd_scaled = self._scale_logits(s_logits_kd)
                        t_logits_scaled = self._scale_logits(t_logits)
                        # KL( teacher || student ) with temperature scaling (standard distillation)
                        loss_old_kd = F.kl_div(
                            F.log_softmax(s_logits_kd_scaled / T, dim=1),
                            F.softmax(t_logits_scaled / T, dim=1),
                            reduction='batchmean',
                        ) * (T * T)
                        loss_kd_val = float(loss_old_kd.detach().item())

                # total replay loss (raw, before replay_lambda)
                loss_old = loss_old_ce
                if loss_old_kd is not None:
                    loss_old = loss_old + (float(self.replay_teacher_kd_lambda) * loss_old_kd)
                # log raw (unweighted) old loss for tuning
                loss_old_val = float(loss_old.detach().item())
                loss_old_ce_val = float(loss_old_ce.detach().item())
                loss_old_sum += loss_old_val
                loss_old_cnt += 1

            # ===== LoRA orthogonality regularization (interference suppression) =====
            loss_main = loss_new
            if float(getattr(self, 'lora_ortho_lambda', 0.0)) > 0.0 and self._cur_task > 0:
                loss_ortho = self._lora_ortho_loss(task_id=self._cur_task)
                loss_ortho_val = float(loss_ortho.detach().item())
                loss_main = loss_main + loss_ortho

            optimizer.zero_grad(set_to_none=True)
            use_replay_guidance = (
                    replay_loader is not None
                    and is_old.any()
                    and bool(getattr(self, 'replay_grad_guidance', True))
                    and ((not use_global_ce) or bool(getattr(self, 'global_ce_use_guidance', False)))
            )
            if use_replay_guidance:
                main_grads = self._compute_param_grads(loss_main, params)
                replay_grads = self._compute_param_grads(float(self.replay_lambda) * loss_old, params)
                guided_grads, guide_stats = self._replay_grad_guidance(
                    main_grads,
                    replay_grads,
                    eps=float(getattr(self, 'replay_grad_guidance_eps', 1e-12)),
                )
                self._set_param_grads(params, guided_grads)
                replay_cos_val = float(guide_stats["cos"])
                replay_conflict = bool(guide_stats["conflict"])
                replay_guide_cnt += 1
                if replay_conflict:
                    replay_guide_conflict_cnt += 1
                loss_tot_val = float(loss_main.detach().item())
            else:
                loss = loss_main
                if is_old.any() and replay_loader is not None:
                    loss = loss + (self.replay_lambda * loss_old)
                loss.backward()
                loss_tot_val = float(loss.detach().item())

            optimizer.step()
            losses += loss_tot_val
            loss_tot_sum += loss_tot_val

            if (i + 1) % log_interval == 0:
                old_ratio = float(is_old.float().mean().item())
                logging.info(
                    f"[Task {self._cur_task}][Epoch {epoch + 1}/{self.run_epoch}] "
                    f"Iter {i + 1}/{len(train_loader)} | "
                    f"loss_new={loss_new_val:.4f} "
                    f"(global={loss_new_global_val:.4f} local={loss_new_local_val:.4f}) "
                    f"loss_old={loss_old_val:.4f} loss_ortho={loss_ortho_val:.4f} "
                    f"(ce={loss_old_ce_val:.4f} kd={loss_kd_val:.4f}) "
                    f"(lambda={self.replay_lambda} global_ce={int(use_global_ce)}) loss_total={loss_tot_val:.4f} "
                    f"replay_mode={'guide' if use_replay_guidance else 'loss'} "
                    f"replay_cos={replay_cos_val:.4f} replay_conflict={int(replay_conflict)} | "
                    f"old_ratio={old_ratio:.3f}"
                )
        avg_new = loss_new_sum / max(loss_new_cnt, 1)

        avg_old = loss_old_sum / max(loss_old_cnt, 1)

        avg_tot = loss_tot_sum / max(len(train_loader), 1)

        logging.info(

            f"[Task {self._cur_task}][Epoch {epoch + 1}/{self.run_epoch}] "

            f"AVG loss_new={avg_new:.4f} AVG loss_old={avg_old:.4f} AVG loss_total={avg_tot:.4f} "
            f"(lambda={self.replay_lambda}) "
            f"replay_guidance={int(bool(getattr(self, 'replay_grad_guidance', True)))} "
            f"guide_steps={replay_guide_cnt} conflicts={replay_guide_conflict_cnt}"

        )

        scheduler.step()
        train_acc = 0.0 if total == 0 else np.around(tensor2numpy(correct) * 100 / total, decimals=2)
        info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
            self._cur_task,
            epoch + 1,
            self.run_epoch,
            losses / len(train_loader),
            train_acc
        )
        prog_bar.set_description(info)

    # task train finished
    test_acc = self._compute_accuracy(self._network, test_loader)
    final_info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
        self._cur_task,
        epoch + 1,
        self.run_epoch,
        losses / len(train_loader),
        train_acc,
        test_acc,
    )
    logging.info(final_info)


# =========================
# Distilled replay helpers
# =========================
def _collect_replay_from_dir(self, root_dir: str):
    """Collect replay image paths & labels from a directory.

    Supported layout:
      root_dir/<class_id>/*.png  (class_id are digits: 0..199)
    """
    root_dir = os.path.expanduser(str(root_dir))
    if not os.path.exists(root_dir):
        logging.warning(f"[Replay] replay_root not found: {root_dir}")
        return None, None

    class_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    class_dirs = sorted(class_dirs)
    if len(class_dirs) == 0:
        logging.warning(f"[Replay] no class subfolders under: {root_dir}")
        return None, None

    paths, labels = [], []
    for d in class_dirs:
        if not d.isdigit():
            continue
        lab = int(d)
        files = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            files.extend(sorted(glob.glob(os.path.join(root_dir, d, ext))))
        for fp in files:
            if os.path.isfile(fp):
                # skip broken/partial images (common if previous run was interrupted)
                try:
                    if os.path.getsize(fp) <= 0:
                        raise ValueError('empty file')
                    with Image.open(fp) as _im:
                        _im.verify()
                    paths.append(fp)
                    labels.append(lab)
                except Exception as e:
                    logging.warning(f"[Replay] skip unreadable image: {fp} ({e})")

    if len(paths) == 0:
        logging.warning(f"[Replay] found no images under: {root_dir}")
        return None, None

    return np.array(paths, dtype=object), np.array(labels, dtype=np.int64)


def _map_replay_labels(self, labels: np.ndarray, data_manager):
    """Map replay labels into DataManager's remapped label space (handles shuffle)."""
    labels = labels.astype(np.int64)
    if self.replay_labels_are_new:
        return labels
    # DataManager mapping: new_label = class_order.index(old_label)
    return np.array([data_manager._class_order.index(int(t)) for t in labels], dtype=np.int64)


def _limit_replay_ipc(self, paths: np.ndarray, labels: np.ndarray, ipc: int):
    """Keep at most `ipc` images per class (deterministic)."""
    if ipc is None or ipc <= 0:
        return paths, labels
    out_p, out_y = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        idx = idx[-ipc:]
        out_p.append(paths[idx])
        out_y.append(labels[idx])
    return (np.concatenate(out_p) if len(out_p) else paths), (np.concatenate(out_y) if len(out_y) else labels)


# =========================
# route-2: replay generation by logit inversion
# =========================

def _generate_replay_images_for_current_task(self, start_c: int = None, end_c: int = None, model=None):
    # CE inversion + LinearGM + prototype(Mahalanobis)
    if not self.replay_root:
        return

    if not self.replay_labels_are_new:
        logging.warning('[ReplayGen] You enabled generate_replay but replay_labels_are_new=false. '
                        'For route-2 outputs, please set replay_labels_are_new=true to avoid label remapping.')

    if end_c is None:
        end_c = int(self._total_classes)
    if start_c is None:
        start_c = int(self._known_classes)

    start_c = int(start_c)
    end_c = int(end_c)
    total_classes = int(end_c)

    if start_c >= end_c:
        logging.info('[ReplayGen] No new classes to generate (start_c>=end_c).')
        return

    class_ids = list(range(start_c, end_c))

    out_root = os.path.expanduser(str(self.replay_root))
    os.makedirs(out_root, exist_ok=True)

    if model is None:
        model = self._network
    model.to(self._device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    mean = torch.tensor(self.replay_gen_mean, device=self._device).view(1, 3, 1, 1)
    std = torch.tensor(self.replay_gen_std, device=self._device).view(1, 3, 1, 1)

    logging.info(
        f"[ReplayGen] Generating {self.replay_gen_per_class} img(s)/class for NEW classes {start_c}-{end_c - 1} "
        f"(steps={self.replay_gen_steps}, lr={self.replay_gen_lr}, size={self.replay_gen_size}, "
        f"gm_reinit_each_step={int(bool(getattr(self, 'replay_gm_reinit_each_step', True)))})."
    )

    if bool(getattr(self, "replay_set_gm", False)):
        self._generate_replay_images_for_current_task_setlevel(
            class_ids=class_ids,
            total_classes=total_classes,
            model=model,
            mean=mean,
            std=std,
            out_root=out_root,
        )
        return

    pbar = tqdm(class_ids, desc='ReplayGen', leave=False)
    for y in pbar:
        y = int(y)

        # prototype stats from _compute_class_mean
        proto_mean, proto_invcov = self._get_proto_stats(class_id=y)
        if float(getattr(self, 'replay_proto_lambda', 0.0)) > 0 and (proto_mean is None or proto_invcov is None):
            logging.warning(
                f"[ReplayUpdate] proto stats missing for class {y} (means/covs not ready). proto term disabled.")
        if float(getattr(self, 'replay_proto_lambda', 0.0)) > 0 and (proto_mean is None or proto_invcov is None):
            logging.warning(
                f"[ReplayGen] proto stats missing for class {y} (means/covs not ready). proto term disabled.")
        # LinearGM target from REAL data (new classes only) with multi-head & stabilization.
        # When replay_gm_reinit_each_step=True, this is sampled inside every image-optimization step
        # instead of once per class, matching Dataset Distillation for Pre-Trained SSL Vision Models.
        gm_fc, gm_target = None, None
        if (
                self.replay_gradmatch
                and (self._data_manager is not None)
                and (not bool(getattr(self, "replay_gm_reinit_each_step", True)))
        ):
            try:
                K = max(1, int(getattr(self, "replay_gm_num_heads", 1)))
                seed_base = 100000 * int(self._cur_task) + int(y)
                gm_fcs = []
                gm_targets = []
                for k in range(K):
                    fc = self._build_random_linear_fc(
                        num_feats=int(self.feature_dim),
                        num_classes=int(total_classes),
                        seed=int(seed_base + 97 * k),
                        device=self._device,
                    )
                    gm_fcs.append(fc)
                    gm_targets.append(
                        self._linear_gm_real_grad_from_real_data(
                            model=model,
                            data_manager=self._data_manager,
                            class_id=y,
                            bs=int(self.replay_gradmatch_real_bs),
                            gm_fc=fc,
                        )
                    )
                gm_fc, gm_target = gm_fcs, gm_targets
            except Exception as e:
                logging.warning(f"[ReplayGen] gradmatch target failed for class {y}: {e}")
                gm_fc, gm_target = None, None

        patch_target = None
        if float(getattr(self, "replay_gen_patch_weight", 0.0)) > 0.0 and (self._data_manager is not None):
            try:
                patch_target = self._real_patch_template_from_real_data(
                    model=model,
                    data_manager=self._data_manager,
                    class_id=y,
                )
            except Exception as e:
                logging.warning(f"[ReplayGen] patch target failed for class {y}: {e}")
                patch_target = None

        cls_dir = os.path.join(out_root, str(y))
        os.makedirs(cls_dir, exist_ok=True)

        existing = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            existing += glob.glob(os.path.join(cls_dir, ext))
        if self.replay_gen_skip_existing and (len(existing) >= int(self.replay_gen_per_class)):
            continue

        for _k in range(int(self.replay_gen_per_class)):
            nums = []
            for fp in existing:
                base = os.path.splitext(os.path.basename(fp))[0]
                if base.isdigit():
                    nums.append(int(base))
            next_idx = (max(nums) + 1) if len(nums) > 0 else 0
            out_path = os.path.join(cls_dir, f"{next_idx:04d}.png")

            img = self._optimize_one_replay_image(
                model=model,
                target_class=y,
                total_classes=total_classes,
                mean=mean,
                std=std,
                proto_mean=proto_mean,
                proto_invcov=proto_invcov,
                gm_fc=gm_fc,
                gm_target=gm_target,
                patch_target=patch_target,
            )
            self._save_tensor_as_png(img, out_path)
            existing.append(out_path)


def _update_old_replay_images_after_task(self, old_model, new_model, old_end: int, total_classes: int):
    # refine OLD replay with CE(new) + LinearGM(old->new) + prototype + keep
    if not self.replay_root:
        return
    old_end = int(old_end)
    total_classes = int(total_classes)
    if old_end <= 0:
        return

    out_root = os.path.expanduser(str(self.replay_root))
    if not os.path.isdir(out_root):
        logging.warning(f"[ReplayUpdate] replay_root not found: {out_root}")
        return

    class_ids = list(range(0, old_end))

    if getattr(self, "replay_refine_max_classes", None) is not None and int(self.replay_refine_max_classes) > 0:
        maxc = int(self.replay_refine_max_classes)
        if len(class_ids) > maxc:
            rng = np.random.default_rng(
                seed=int(self.args.get('seed', 0)) if isinstance(self.args.get('seed', 0), int) else 0
            )
            class_ids = list(rng.choice(class_ids, size=maxc, replace=False).tolist())

    old_model.to(self._device);
    old_model.eval()
    for p in old_model.parameters(): p.requires_grad_(False)
    new_model.to(self._device);
    new_model.eval()
    for p in new_model.parameters(): p.requires_grad_(False)

    mean = torch.tensor(self.replay_gen_mean, device=self._device).view(1, 3, 1, 1)
    std = torch.tensor(self.replay_gen_std, device=self._device).view(1, 3, 1, 1)

    logging.info(
        f"[ReplayUpdate] Updating replay images for {len(class_ids)} OLD class(es) 0..{old_end - 1} "
        f"in expanded space 0..{total_classes - 1} (steps={self.replay_refine_steps}, lr={self.replay_refine_lr})."
    )

    if bool(getattr(self, "replay_set_gm", False)):
        self._update_old_replay_images_after_task_setlevel(
            old_model=old_model,
            new_model=new_model,
            class_ids=class_ids,
            old_end=old_end,
            total_classes=total_classes,
            mean=mean,
            std=std,
            out_root=out_root,
        )
        return

    pbar = tqdm(class_ids, desc='ReplayUpdate', leave=False)
    for y in pbar:
        y = int(y)
        cls_dir = os.path.join(out_root, str(y))
        if not os.path.isdir(cls_dir):
            if self.replay_refine_skip_missing:
                continue
            os.makedirs(cls_dir, exist_ok=True)

        files = []
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            files.extend(sorted(glob.glob(os.path.join(cls_dir, ext))))
        if len(files) == 0:
            if self.replay_refine_skip_missing:
                continue
            continue

        proto_mean, proto_invcov = self._get_proto_stats(class_id=y)
        gm_fc = None
        if self.replay_gradmatch and (not bool(getattr(self, "replay_gm_reinit_each_step", True))):
            # multi-head GM for old-class refinement
            K = max(1, int(getattr(self, "replay_gm_num_heads", 1)))
            seed_base = 200000 * int(self._cur_task) + int(y)
            gm_fc = [
                self._build_random_linear_fc(
                    num_feats=int(self.feature_dim),
                    num_classes=int(old_end),
                    seed=int(seed_base + 97 * k),
                    device=self._device,
                )
                for k in range(K)
            ]

        for fp in files:
            try:
                img0 = self._load_image_as_tensor_01(fp, size=self.replay_gen_size)
            except Exception as e:
                logging.warning(f"[ReplayUpdate] failed to load {fp}: {e}")
                continue

            img_ref = self._refine_one_replay_image(
                new_model=new_model,
                old_model=old_model,
                img0=img0,
                target_class=y,
                total_classes=total_classes,
                old_end=old_end,
                mean=mean,
                std=std,
                proto_mean=proto_mean,
                proto_invcov=proto_invcov,
                gm_fc=gm_fc,
            )
            self._save_tensor_as_png(img_ref, fp)


def _chunk_classes(self, class_ids, chunk_size: int):
    chunk_size = max(1, int(chunk_size))
    class_ids = [int(c) for c in class_ids]
    for start in range(0, len(class_ids), chunk_size):
        yield class_ids[start:start + chunk_size]


def _next_replay_path(self, out_root: str, class_id: int):
    cls_dir = os.path.join(out_root, str(int(class_id)))
    os.makedirs(cls_dir, exist_ok=True)
    nums = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for fp in glob.glob(os.path.join(cls_dir, ext)):
            base = os.path.splitext(os.path.basename(fp))[0]
            if base.isdigit():
                nums.append(int(base))
    next_idx = (max(nums) + 1) if len(nums) > 0 else 0
    return os.path.join(cls_dir, f"{next_idx:04d}.png")


def _existing_replay_files(self, out_root: str, class_id: int):
    cls_dir = os.path.join(out_root, str(int(class_id)))
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        files.extend(sorted(glob.glob(os.path.join(cls_dir, ext))))
    return [fp for fp in files if os.path.isfile(fp)]


def _generate_replay_images_for_current_task_setlevel(self, class_ids, total_classes: int, model, mean, std, out_root: str):
    class_ids = [int(c) for c in class_ids]
    chunk_size = max(1, int(getattr(self, "replay_set_size", 8)))
    per_class = max(1, int(getattr(self, "replay_gen_per_class", 1)))
    logging.info(
        f"[ReplaySetGen] Set-level GM enabled: {len(class_ids)} class(es), "
        f"chunk_size={chunk_size}, ipc={per_class}, real_per_class={int(getattr(self, 'replay_set_real_per_class', 2))}."
    )

    for ipc_idx in range(per_class):
        pending = []
        for y in class_ids:
            existing = self._existing_replay_files(out_root, y)
            if self.replay_gen_skip_existing and len(existing) > ipc_idx:
                continue
            pending.append(y)
        if len(pending) == 0:
            continue

        pbar = tqdm(list(self._chunk_classes(pending, chunk_size)), desc=f"ReplaySetGen[{ipc_idx}]", leave=False)
        for group in pbar:
            imgs = self._optimize_replay_image_set(
                model=model,
                class_ids=group,
                total_classes=total_classes,
                mean=mean,
                std=std,
                old_model=None,
                img0_batch=None,
                old_end=None,
                mode="gen",
            )
            for img, y in zip(imgs, group):
                self._save_tensor_as_png(img, self._next_replay_path(out_root, y))


def _update_old_replay_images_after_task_setlevel(
        self,
        old_model,
        new_model,
        class_ids,
        old_end: int,
        total_classes: int,
        mean,
        std,
        out_root: str,
):
    class_ids = [int(c) for c in class_ids]
    files_by_class = {y: self._existing_replay_files(out_root, y) for y in class_ids}
    files_by_class = {y: fps for y, fps in files_by_class.items() if len(fps) > 0}
    if len(files_by_class) == 0:
        logging.info("[ReplaySetUpdate] skip: no replay files found for old classes.")
        return

    chunk_size = max(1, int(getattr(self, "replay_set_size", 8)))
    max_slots = max(len(v) for v in files_by_class.values())
    logging.info(
        f"[ReplaySetUpdate] Set-level GM enabled: {len(files_by_class)} old class(es), "
        f"chunk_size={chunk_size}, slots={max_slots}."
    )

    for slot in range(max_slots):
        slot_classes = [y for y, fps in files_by_class.items() if slot < len(fps)]
        pbar = tqdm(list(self._chunk_classes(slot_classes, chunk_size)), desc=f"ReplaySetUpdate[{slot}]", leave=False)
        for group in pbar:
            imgs0, save_paths = [], []
            active_classes = []
            for y in group:
                fp = files_by_class[y][slot]
                try:
                    imgs0.append(self._load_image_as_tensor_01(fp, size=self.replay_gen_size))
                    save_paths.append(fp)
                    active_classes.append(int(y))
                except Exception as e:
                    logging.warning(f"[ReplaySetUpdate] failed to load {fp}: {e}")
            if len(imgs0) == 0:
                continue
            img0_batch = torch.stack(imgs0, dim=0)
            imgs = self._optimize_replay_image_set(
                model=new_model,
                class_ids=active_classes,
                total_classes=total_classes,
                mean=mean,
                std=std,
                old_model=old_model,
                img0_batch=img0_batch,
                old_end=old_end,
                mode="refine",
            )
            for img, fp in zip(imgs, save_paths):
                self._save_tensor_as_png(img, fp)


def _optimize_replay_image_set(
        self,
        model,
        class_ids,
        total_classes: int,
        mean: torch.Tensor,
        std: torch.Tensor,
        old_model=None,
        img0_batch: torch.Tensor = None,
        old_end: int = None,
        mode: str = "gen",
) -> torch.Tensor:
    class_ids = [int(c) for c in class_ids]
    labels = torch.tensor(class_ids, device=self._device, dtype=torch.long)
    batch_size = len(class_ids)
    is_refine = (str(mode).lower() == "refine")
    steps = int(self.replay_refine_steps if is_refine else self.replay_gen_steps)
    lr = float(self.replay_refine_lr if is_refine else self.replay_gen_lr)
    pad = int(self.replay_refine_pad if is_refine else self.replay_gen_pad)
    tv_w = float(self.replay_refine_tv if is_refine else self.replay_gen_tv)
    l2_w = float(self.replay_refine_l2 if is_refine else self.replay_gen_l2)
    patch_w = float(getattr(self, "replay_refine_patch_weight" if is_refine else "replay_gen_patch_weight", 0.0))
    keep_w = float(getattr(self, "replay_refine_keep", 1.0)) if is_refine else 0.0
    gm_num_classes = int(old_end) if is_refine and old_end is not None else int(total_classes)

    if is_refine:
        if img0_batch is None:
            raise RuntimeError("Set-level refine requires img0_batch.")
        img0_batch = img0_batch.to(self._device)
        eps = 1e-4
        img0c = img0_batch.clamp(eps, 1 - eps)
        w = torch.log(img0c / (1 - img0c)).detach().clone().requires_grad_(True)
    else:
        w = torch.randn(
            batch_size, 3, int(self.replay_gen_size), int(self.replay_gen_size),
            device=self._device,
            requires_grad=True,
        )

    opt = optim.Adam([w], lr=lr)
    best_img = torch.sigmoid(w).detach().clone()
    best_loss = float("inf")

    from contextlib import nullcontext
    use_fp16 = bool(self.replay_gen_use_fp16 and self._device.type == 'cuda')
    amp_ctx = torch.cuda.amp.autocast(enabled=True) if use_fp16 else nullcontext()

    model.eval()
    if old_model is not None:
        old_model.eval()

    static_gm_fc, static_gm_target = None, None
    if self.replay_gradmatch and (not bool(getattr(self, "replay_set_gm_reinit_each_step", True))):
        K = max(1, int(getattr(self, "replay_gm_num_heads", 1)))
        seed_base = 300000000 * int(self._cur_task) + 100000 * int(sum((i + 1) * c for i, c in enumerate(class_ids)))
        static_gm_fc = [
            self._build_random_linear_fc(
                num_feats=int(self.feature_dim),
                num_classes=gm_num_classes,
                seed=int(seed_base + 97 * k),
                device=self._device,
            )
            for k in range(K)
        ]
        if is_refine:
            static_gm_target = self._linear_gm_real_grad_from_image_batch(
                model=old_model,
                img=img0_batch,
                labels=labels,
                gm_fc=static_gm_fc,
                mean=mean,
                std=std,
                pad=pad,
                views=max(1, int(getattr(self, "replay_gm_views", 1))),
            )
        else:
            static_gm_target = self._linear_gm_real_grad_from_real_class_set(
                model=model,
                data_manager=self._data_manager,
                class_ids=class_ids,
                gm_fc=static_gm_fc,
            )

    with self._maybe_disable_inference(), self._force_autograd():
        for t in range(max(1, steps)):
            img = torch.sigmoid(w)
            if is_refine and patch_w > 0.0 and old_model is not None:
                img_aug, img0_aug_for_patch = self._random_shift_and_flip_pair(img, img0_batch, pad=pad)
            else:
                img_aug = self._random_shift_and_flip(img, pad=pad)
                img0_aug_for_patch = None
            img_norm = (img_aug - mean) / std

            with amp_ctx:
                logits_full = model.interface(img_norm)
                end_c = min(int(total_classes), int(logits_full.shape[1]))
                logits = logits_full[:, :end_c]
                ce = F.cross_entropy(self._scale_logits(logits), labels)

                out = model(img_norm)
                feat = out["features"]
                patch_tokens = out.get("patch_tokens", None) if isinstance(out, dict) else None

                proto_terms = []
                if float(getattr(self, "replay_proto_lambda", 0.0)) > 0.0:
                    for row, y in enumerate(class_ids):
                        proto_mean, proto_invcov = self._get_proto_stats(class_id=y)
                        if proto_mean is not None and proto_invcov is not None:
                            proto_terms.append(self._proto_mahalanobis_loss(feat[row:row + 1], proto_mean, proto_invcov))
                proto = torch.stack(proto_terms).mean() if len(proto_terms) > 0 else torch.tensor(0.0, device=img.device)

                patch = torch.tensor(0.0, device=img.device)
                if is_refine and patch_w > 0.0 and patch_tokens is not None and old_model is not None:
                    with torch.no_grad():
                        img0_norm = (img0_aug_for_patch - mean) / std
                        if hasattr(old_model, "extract_tokens"):
                            _, old_patch = old_model.extract_tokens(img0_norm)
                        else:
                            old_out = old_model(img0_norm)
                            old_patch = old_out.get("patch_tokens", None) if isinstance(old_out, dict) else None
                    if old_patch is not None:
                        patch = self._angle_weighted_patch_loss(
                            patch_tokens,
                            old_patch.detach(),
                            feat,
                            patch_sample_k=int(getattr(self, "replay_patch_sample_k", 0)),
                        )

                gm = torch.tensor(0.0, device=img.device)
                if self.replay_gradmatch:
                    try:
                        if bool(getattr(self, "replay_set_gm_reinit_each_step", True)):
                            K = max(1, int(getattr(self, "replay_gm_num_heads", 1)))
                            seed_base = (
                                    300000000 * int(self._cur_task)
                                    + 100000 * int(sum((i + 1) * c for i, c in enumerate(class_ids)))
                                    + int(t)
                            )
                            gm_fc = [
                                self._build_random_linear_fc(
                                    num_feats=int(self.feature_dim),
                                    num_classes=gm_num_classes,
                                    seed=int(seed_base + 97 * k),
                                    device=self._device,
                                )
                                for k in range(K)
                            ]
                            if is_refine:
                                gm_target = self._linear_gm_real_grad_from_image_batch(
                                    model=old_model,
                                    img=img0_batch,
                                    labels=labels,
                                    gm_fc=gm_fc,
                                    mean=mean,
                                    std=std,
                                    pad=pad,
                                    views=max(1, int(getattr(self, "replay_gm_views", 1))),
                                )
                            else:
                                gm_target = self._linear_gm_real_grad_from_real_class_set(
                                    model=model,
                                    data_manager=self._data_manager,
                                    class_ids=class_ids,
                                    gm_fc=gm_fc,
                                )
                        else:
                            gm_fc = static_gm_fc
                            gm_target = static_gm_target
                        g_syn = self._linear_gm_syn_grad_from_image_batch(
                            model=model,
                            img=img,
                            labels=labels,
                            gm_fc=gm_fc,
                            mean=mean,
                            std=std,
                            pad=pad,
                            views=max(1, int(getattr(self, "replay_gm_views", 1))),
                        )
                        gm = torch.stack([self._grad_match_loss(gs, gt) for gs, gt in zip(g_syn, gm_target)]).mean()
                    except Exception as e:
                        logging.warning(f"[ReplaySetGM] target/synthetic GM failed for classes={class_ids}: {e}")

                tv = self._tv_loss(img_aug)
                l2 = (img_aug ** 2).mean()
                keep = ((img - img0_batch) ** 2).mean() if is_refine and img0_batch is not None else torch.tensor(0.0, device=img.device)
                loss = (
                        ce
                        + (tv_w * tv)
                        + (l2_w * l2)
                        + (float(getattr(self, "replay_proto_lambda", 0.0)) * proto)
                        + (float(self.replay_gradmatch_lambda) * gm)
                        + (patch_w * patch)
                        + (keep_w * keep)
                )

            opt.zero_grad(set_to_none=True)
            if not loss.requires_grad:
                raise RuntimeError("Set-level replay optimization graph is disabled.")
            loss.backward()
            opt.step()

            loss_val = float(loss.detach().item())
            if loss_val < best_loss:
                best_loss = loss_val
                best_img = img.detach().clone()

            if (t + 1) % 50 == 0 or (t + 1) == steps:
                logging.info(
                    f"[ReplaySet{'Update' if is_refine else 'Gen'}] classes={class_ids} "
                    f"step={t + 1}/{steps} loss={loss_val:.4f} ce={float(ce.detach().item()):.3f} "
                    f"proto={float(proto.detach().item()):.3f} gm={float(gm.detach().item()):.3f} "
                    f"patch={float(patch.detach().item()):.3f}"
                )

    return best_img.clamp(0, 1)


def _refine_one_replay_image(
        self,
        new_model,
        old_model,
        img0: torch.Tensor,
        target_class: int,
        total_classes: int,
        old_end: int,
        mean: torch.Tensor,
        std: torch.Tensor,
        proto_mean: torch.Tensor = None,
        proto_invcov: torch.Tensor = None,
        gm_fc=None,
) -> torch.Tensor:
    pad = int(self.replay_refine_pad)
    steps = int(self.replay_refine_steps)
    lr = float(self.replay_refine_lr)
    old_end = int(old_end)

    eps = 1e-4
    img0c = img0.clamp(eps, 1 - eps)
    w = torch.log(img0c / (1 - img0c)).unsqueeze(0)
    w = w.detach().clone().requires_grad_(True)
    opt = optim.Adam([w], lr=lr)
    target = torch.tensor([int(target_class)], device=self._device, dtype=torch.long)

    gm_target = None
    if self.replay_gradmatch and (gm_fc is not None) and (old_model is not None) and (old_end > 0):
        try:
            gm_target = self._linear_gm_real_grad_from_image(
                model=old_model,
                img=img0.unsqueeze(0),
                class_id=int(target_class),
                gm_fc=gm_fc,
                mean=mean,
                std=std,
                pad=pad,
                views=max(1, int(getattr(self, "replay_gm_views", 1))),
            )
        except Exception as e:
            logging.warning(f"[ReplayUpdate] per-image GM target failed for class {int(target_class)}: {e}")
            gm_target = None

    best_img = img0.unsqueeze(0).detach().clone()
    best_loss = float('inf')

    from contextlib import nullcontext
    use_fp16 = bool(self.replay_gen_use_fp16 and self._device.type == 'cuda')
    amp_ctx = torch.cuda.amp.autocast(enabled=True) if use_fp16 else nullcontext()

    # Force-enable autograd in case caller is inside a `torch.no_grad()` context.
    with self._maybe_disable_inference(), self._force_autograd():
        for t in range(steps):
            img = torch.sigmoid(w)
            if float(getattr(self, "replay_refine_patch_weight", 0.0)) > 0.0 and old_model is not None:
                img_aug, img0_aug_for_patch = self._random_shift_and_flip_pair(img, img0.unsqueeze(0), pad=pad)
            else:
                img_aug = self._random_shift_and_flip(img, pad=pad)
                img0_aug_for_patch = None
            img_norm = (img_aug - mean) / std

            with amp_ctx:
                logits_new_full = new_model.interface(img_norm)
                end_c = min(int(total_classes), int(logits_new_full.shape[1]))
                logits_new = logits_new_full[:, :end_c]
                logits_new_scaled = self._scale_logits(logits_new)
                ce = F.cross_entropy(logits_new_scaled, target)

                out = new_model(img_norm)
                feat = out['features']
                patch_tokens = out.get('patch_tokens', None) if isinstance(out, dict) else None

                proto = torch.tensor(0.0, device=img_aug.device)
                if (proto_mean is not None) and (proto_invcov is not None) and (
                        float(getattr(self, "replay_proto_lambda", 0.0)) > 0):
                    proto = self._proto_mahalanobis_loss(feat, proto_mean, proto_invcov)

                patch = torch.tensor(0.0, device=img_aug.device)
                if (
                        float(getattr(self, "replay_refine_patch_weight", 0.0)) > 0.0
                        and patch_tokens is not None
                        and old_model is not None
                ):
                    with torch.no_grad():
                        img0_norm = (img0_aug_for_patch - mean) / std
                        if hasattr(old_model, "extract_tokens"):
                            _, old_patch = old_model.extract_tokens(img0_norm)
                        else:
                            old_out = old_model(img0_norm)
                            old_patch = old_out.get('patch_tokens', None) if isinstance(old_out, dict) else None
                    if old_patch is not None:
                        patch = self._angle_weighted_patch_loss(
                            patch_tokens,
                            old_patch.detach(),
                            feat,
                            patch_sample_k=int(getattr(self, "replay_patch_sample_k", 0)),
                        )

                gm = torch.tensor(0.0, device=img_aug.device)
                gm_fc_step, gm_target_step = gm_fc, gm_target
                if (
                        self.replay_gradmatch
                        and bool(getattr(self, "replay_gm_reinit_each_step", True))
                        and old_model is not None
                        and old_end > 0
                ):
                    try:
                        K = max(1, int(getattr(self, "replay_gm_num_heads", 1)))
                        seed_base = 200000000 * int(self._cur_task) + 100000 * int(target_class) + int(t)
                        gm_fc_step = [
                            self._build_random_linear_fc(
                                num_feats=int(self.feature_dim),
                                num_classes=int(old_end),
                                seed=int(seed_base + 97 * k),
                                device=self._device,
                            )
                            for k in range(K)
                        ]
                        gm_target_step = self._linear_gm_real_grad_from_image(
                            model=old_model,
                            img=img0.unsqueeze(0),
                            class_id=int(target_class),
                            gm_fc=gm_fc_step,
                            mean=mean,
                            std=std,
                            pad=pad,
                            views=max(1, int(getattr(self, "replay_gm_views", 1))),
                        )
                    except Exception as e:
                        logging.warning(
                            f"[ReplayUpdate] step-wise GM target failed for class {int(target_class)} step={t + 1}: {e}"
                        )
                        gm_fc_step, gm_target_step = None, None

                if self.replay_gradmatch and (gm_fc_step is not None) and (gm_target_step is not None):
                    g_syn = self._linear_gm_syn_grad_from_image(
                        model=new_model,
                        img=img,
                        class_id=int(target_class),
                        gm_fc=gm_fc_step,
                        mean=mean,
                        std=std,
                        pad=pad,
                        views=max(1, int(getattr(self, "replay_gm_views", 1))),
                    )
                    if isinstance(g_syn, (list, tuple)):
                        _gm_losses = []
                        for _gs, _gt in zip(g_syn, gm_target_step):
                            _gm_losses.append(self._grad_match_loss(_gs, _gt))
                        gm = torch.stack(_gm_losses).mean() if len(_gm_losses) > 0 else torch.tensor(0.0,
                                                                                                     device=img_aug.device)
                    else:
                        gm = self._grad_match_loss(g_syn, gm_target_step)

                tv = self._tv_loss(img_aug)
                l2 = (img_aug ** 2).mean()
                keep = ((img - img0.unsqueeze(0)) ** 2).mean()

                loss = (
                        ce
                        + (self.replay_refine_tv * tv)
                        + (self.replay_refine_l2 * l2)
                        + (float(getattr(self, "replay_refine_keep", 1.0)) * keep)
                        + (float(getattr(self, "replay_proto_lambda", 0.0)) * proto)
                        + (float(self.replay_gradmatch_lambda) * gm)
                        + (float(getattr(self, "replay_refine_patch_weight", 0.0)) * patch)
                )

            opt.zero_grad(set_to_none=True)
            active_gm_fc = locals().get("gm_fc_step", gm_fc)
            if active_gm_fc is not None:
                if isinstance(active_gm_fc, (list, tuple)):
                    for _fc in active_gm_fc:
                        _fc.zero_grad(set_to_none=True)
                else:
                    active_gm_fc.zero_grad(set_to_none=True)

            if not loss.requires_grad:
                raise RuntimeError(
                    "Replay image optimization graph is disabled. Ensure after_task() is not called under torch.inference_mode()/no_grad.")

            loss.backward()
            opt.step()

            loss_val = float(loss.detach().item())
            if loss_val < best_loss:
                best_loss = loss_val
                best_img = img.detach().clone()

            if (t + 1) % 50 == 0 or (t + 1) == steps:
                with torch.no_grad():
                    p = torch.softmax(logits_new_scaled.float(), dim=1)[0, int(target_class)].item()
                logging.info(
                    f"[ReplayUpdate] class={int(target_class)} step={t + 1}/{steps} loss={loss_val:.4f} p={p:.3f} "
                    f"ce={float(ce.detach().item()):.3f} proto={float(proto.detach().item()):.3f} "
                    f"gm={float(gm.detach().item()):.3f} patch={float(patch.detach().item()):.3f}"
                )

    return best_img.squeeze(0).clamp(0, 1)


def _optimize_one_replay_image(
        self,
        model,
        target_class: int,
        total_classes: int,
        mean: torch.Tensor,
        std: torch.Tensor,
        proto_mean: torch.Tensor = None,
        proto_invcov: torch.Tensor = None,
        gm_fc=None,
        gm_target: torch.Tensor = None,
        patch_target: torch.Tensor = None,
) -> torch.Tensor:
    steps = int(self.replay_gen_steps)
    lr = float(self.replay_gen_lr)
    pad = int(self.replay_gen_pad)

    w = torch.randn(1, 3, int(self.replay_gen_size), int(self.replay_gen_size), device=self._device,
                    requires_grad=True)
    opt = optim.Adam([w], lr=lr)
    target = torch.tensor([int(target_class)], device=self._device, dtype=torch.long)

    best_img = None
    best_loss = float('inf')

    from contextlib import nullcontext
    use_fp16 = bool(self.replay_gen_use_fp16 and self._device.type == 'cuda')
    amp_ctx = torch.cuda.amp.autocast(enabled=True) if use_fp16 else nullcontext()

    # NOTE: Some training pipelines call `after_task()` inside a `torch.no_grad()` block
    # (e.g., right after evaluation). Replay image optimization *must* track gradients
    # w.r.t. the image parameters, so we force-enable autograd here.
    with self._maybe_disable_inference(), self._force_autograd():
        for t in range(steps):
            img = torch.sigmoid(w)
            img_aug = self._random_shift_and_flip(img, pad=pad)
            img_norm = (img_aug - mean) / std

            with amp_ctx:
                logits_full = model.interface(img_norm)
                end_c = min(int(total_classes), int(logits_full.shape[1]))
                logits = logits_full[:, :end_c]
                logits_scaled = self._scale_logits(logits)
                ce = F.cross_entropy(logits_scaled, target)

                out = model(img_norm)
                feat = out['features']
                patch_tokens = out.get('patch_tokens', None) if isinstance(out, dict) else None

                proto = torch.tensor(0.0, device=img_aug.device)
                if (proto_mean is not None) and (proto_invcov is not None) and (
                        float(getattr(self, "replay_proto_lambda", 0.0)) > 0):
                    proto = self._proto_mahalanobis_loss(feat, proto_mean, proto_invcov)

                patch = torch.tensor(0.0, device=img_aug.device)
                if (
                        float(getattr(self, "replay_gen_patch_weight", 0.0)) > 0.0
                        and patch_target is not None
                        and patch_tokens is not None
                ):
                    pt = patch_target.to(device=patch_tokens.device, dtype=patch_tokens.dtype)
                    if pt.dim() == 2:
                        pt = pt.unsqueeze(0)
                    if pt.size(0) == 1 and patch_tokens.size(0) != 1:
                        pt = pt.expand(patch_tokens.size(0), -1, -1)
                    patch = self._angle_weighted_patch_loss(
                        patch_tokens,
                        pt.detach(),
                        feat,
                        patch_sample_k=int(getattr(self, "replay_patch_sample_k", 0)),
                    )

                gm = torch.tensor(0.0, device=img_aug.device)
                gm_fc_step, gm_target_step = gm_fc, gm_target
                if (
                        self.replay_gradmatch
                        and bool(getattr(self, "replay_gm_reinit_each_step", True))
                        and (self._data_manager is not None)
                ):
                    try:
                        K = max(1, int(getattr(self, "replay_gm_num_heads", 1)))
                        seed_base = 100000000 * int(self._cur_task) + 100000 * int(target_class) + int(t)
                        gm_fc_step = []
                        gm_target_step = []
                        for k in range(K):
                            fc = self._build_random_linear_fc(
                                num_feats=int(self.feature_dim),
                                num_classes=int(total_classes),
                                seed=int(seed_base + 97 * k),
                                device=self._device,
                            )
                            gm_fc_step.append(fc)
                            gm_target_step.append(
                                self._linear_gm_real_grad_from_real_data(
                                    model=model,
                                    data_manager=self._data_manager,
                                    class_id=int(target_class),
                                    bs=int(self.replay_gradmatch_real_bs),
                                    gm_fc=fc,
                                )
                            )
                    except Exception as e:
                        logging.warning(
                            f"[ReplayGen] step-wise GM target failed for class {int(target_class)} step={t + 1}: {e}"
                        )
                        gm_fc_step, gm_target_step = None, None

                if self.replay_gradmatch and (gm_fc_step is not None) and (gm_target_step is not None):
                    g_syn = self._linear_gm_syn_grad_from_image(
                        model=model,
                        img=img,
                        class_id=int(target_class),
                        gm_fc=gm_fc_step,
                        mean=mean,
                        std=std,
                        pad=pad,
                        views=max(1, int(getattr(self, "replay_gm_views", 1))),
                    )
                    if isinstance(g_syn, (list, tuple)):
                        _gm_losses = []
                        for _gs, _gt in zip(g_syn, gm_target_step):
                            _gm_losses.append(self._grad_match_loss(_gs, _gt))
                        gm = torch.stack(_gm_losses).mean() if len(_gm_losses) > 0 else torch.tensor(0.0,
                                                                                                     device=img_aug.device)
                    else:
                        gm = self._grad_match_loss(g_syn, gm_target_step)

                tv = self._tv_loss(img_aug)
                l2 = (img_aug ** 2).mean()

                loss = (
                        ce
                        + (self.replay_gen_tv * tv)
                        + (self.replay_gen_l2 * l2)
                        + (float(getattr(self, "replay_proto_lambda", 0.0)) * proto)
                        + (float(self.replay_gradmatch_lambda) * gm)
                        + (float(getattr(self, "replay_gen_patch_weight", 0.0)) * patch)
                )

            opt.zero_grad(set_to_none=True)
            active_gm_fc = locals().get("gm_fc_step", gm_fc)
            if active_gm_fc is not None:
                if isinstance(active_gm_fc, (list, tuple)):
                    for _fc in active_gm_fc:
                        _fc.zero_grad(set_to_none=True)
                else:
                    active_gm_fc.zero_grad(set_to_none=True)

            if not loss.requires_grad:
                raise RuntimeError(
                    "Replay image optimization graph is disabled. Ensure after_task() is not called under torch.inference_mode()/no_grad.")

            loss.backward()
            opt.step()

            loss_val = float(loss.detach().item())
            if loss_val < best_loss:
                best_loss = loss_val
                best_img = img.detach().clone()

            if (t + 1) % 50 == 0 or (t + 1) == steps:
                with torch.no_grad():
                    p = torch.softmax(logits_scaled.float(), dim=1)[0, int(target_class)].item()
                logging.info(
                    f"[ReplayGen] class={int(target_class)} step={t + 1}/{steps} loss={loss_val:.4f} p={p:.3f} "
                    f"ce={float(ce.detach().item()):.3f} proto={float(proto.detach().item()):.3f} "
                    f"gm={float(gm.detach().item()):.3f} patch={float(patch.detach().item()):.3f}"
                )

    if best_img is None:
        best_img = torch.sigmoid(w).detach()
    return best_img.squeeze(0).clamp(0, 1)


def _angle_weighted_patch_loss(self, new_patch: torch.Tensor, target_patch: torch.Tensor, new_cls: torch.Tensor,
                               patch_sample_k: int = 0) -> torch.Tensor:
    if new_patch is None or target_patch is None or new_cls is None:
        return torch.tensor(0.0, device=self._device)
    n_tok = min(int(new_patch.size(1)), int(target_patch.size(1)))
    if n_tok <= 0:
        return torch.tensor(0.0, device=new_patch.device)
    new_patch = new_patch[:, :n_tok, :]
    target_patch = target_patch[:, :n_tok, :]
    new_patch_norm = F.normalize(new_patch, p=2, dim=-1)
    target_patch_norm = F.normalize(target_patch, p=2, dim=-1)
    alpha_cos = F.cosine_similarity(new_cls.unsqueeze(1), new_patch, dim=-1).clamp(min=-1.0, max=1.0)
    alpha_angle = 1 - (torch.acos(alpha_cos) / torch.pi)
    distances = torch.norm(new_patch_norm - target_patch_norm, p=2, dim=-1)
    weights = 1 - alpha_angle.detach()
    patch_sample_k = int(patch_sample_k or 0)
    if patch_sample_k > 0 and patch_sample_k < weights.size(1):
        top_idx = torch.topk(weights, k=patch_sample_k, dim=1, largest=True).indices
        distances = torch.gather(distances, 1, top_idx)
        weights = torch.gather(weights, 1, top_idx)
    return (weights * distances).mean()


def _real_patch_template_from_real_data(self, model, data_manager, class_id: int) -> torch.Tensor:
    class_id = int(class_id)
    bs_small = max(1, int(getattr(self, "replay_patch_real_bs_small", 8)))
    num_batches = max(1, int(getattr(self, "replay_patch_real_batches", 1)))
    _, _, ds = data_manager.get_dataset(
        np.arange(class_id, class_id + 1),
        source="train",
        mode="test",
        ret_data=True,
    )
    loader = DataLoader(ds, batch_size=bs_small, shuffle=True, num_workers=0, drop_last=False)
    it = iter(loader)
    patch_sum = None
    count = 0
    model_was_training = bool(model.training)
    model.eval()
    with torch.no_grad():
        for _ in range(num_batches):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                try:
                    batch = next(it)
                except StopIteration:
                    break
            if isinstance(batch, (tuple, list)):
                if len(batch) >= 3:
                    x = batch[-2]
                elif len(batch) == 2:
                    x = batch[0]
                else:
                    continue
            elif isinstance(batch, dict):
                x = batch.get("image", None)
                if x is None:
                    continue
            else:
                continue
            x = x.to(self._device, non_blocking=True)
            if hasattr(model, "extract_tokens"):
                _, patch = model.extract_tokens(x)
            else:
                out = model(x)
                patch = out.get("patch_tokens", None) if isinstance(out, dict) else None
            if patch is None:
                continue
            patch_sum = patch.detach().sum(dim=0) if patch_sum is None else patch_sum + patch.detach().sum(dim=0)
            count += int(patch.size(0))
    if model_was_training:
        model.train()
    if patch_sum is None or count <= 0:
        return None
    return (patch_sum / float(count)).detach()


def _random_shift_and_flip_pair(self, x_a: torch.Tensor, x_b: torch.Tensor, pad: int = 8):
    """Apply the same reflect-pad crop and horizontal flip to two image batches."""
    if pad is None or pad <= 0:
        out_a, out_b = x_a, x_b
    else:
        pa = TF.pad(x_a, [pad, pad, pad, pad], padding_mode='reflect')
        pb = TF.pad(x_b, [pad, pad, pad, pad], padding_mode='reflect')
        H = W = int(self.replay_gen_size)
        i = int(torch.randint(0, 2 * pad + 1, (1,), device=x_a.device).item())
        j = int(torch.randint(0, 2 * pad + 1, (1,), device=x_a.device).item())
        out_a = pa[:, :, i:i + H, j:j + W]
        out_b = pb[:, :, i:i + H, j:j + W]
    if torch.rand((), device=x_a.device).item() < 0.5:
        out_a = torch.flip(out_a, dims=[3])
        out_b = torch.flip(out_b, dims=[3])
    return out_a, out_b


# =========================
# Prototype constraint (mean + inv-cov) helpers
# =========================
def _get_proto_stats(self, class_id: int):
    class_id = int(class_id)
    if not hasattr(self, "_class_means") or (self._class_means is None):
        return None, None
    if not hasattr(self, "_class_covs") or (self._class_covs is None):
        return None, None
    if class_id < 0 or class_id >= int(self._class_means.shape[0]):
        return None, None

    mu = self._class_means[class_id].to(self._device).float().view(1, -1)

    if not hasattr(self, "_proto_invcov_cache") or (self._proto_invcov_cache is None):
        self._proto_invcov_cache = {}

    if class_id in self._proto_invcov_cache:
        inv = self._proto_invcov_cache[class_id]
        if inv.device != self._device:
            inv = inv.to(self._device)
        return mu, inv

    cov = self._class_covs[class_id].to(self._device).float()
    cov = self.shrink_cov(cov)
    cov = cov + torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype) * 1e-3
    inv = torch.linalg.pinv(cov).detach()
    self._proto_invcov_cache[class_id] = inv
    return mu, inv


def _proto_mahalanobis_loss(self, feat: torch.Tensor, mean: torch.Tensor, invcov: torch.Tensor) -> torch.Tensor:
    if feat is None or mean is None or invcov is None:
        return torch.tensor(0.0, device=self._device)
    f = feat.float()
    mu = mean.to(f.device).float()
    inv = invcov.to(f.device).float()
    if mu.dim() == 1:
        mu = mu.view(1, -1)
    diff = f - mu
    md = torch.einsum('bi,ij,bj->b', diff, inv, diff)
    return md.mean()


# =========================
# Linear Gradient Matching (LinearGM-style) helpers
# =========================
class _GMFC(torch.nn.Module):
    def __init__(self, num_feats: int, num_classes: int):
        super().__init__()
        self.linear = torch.nn.Linear(num_feats, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)


def _build_random_linear_fc(self, num_feats: int, num_classes: int, seed: int, device):
    # NOTE: some torch versions don't support `generator=` for torch.randn_like.
    # Use torch.randn(..., generator=g) with a safe fallback.
    try:
        g = torch.Generator(device=device)
    except TypeError:
        # Older torch: Generator() may not accept `device=`.
        g = torch.Generator()
    seed = int(seed) & 0x7fffffff
    g.manual_seed(seed)
    fc = self._GMFC(int(num_feats), int(num_classes)).to(device)
    # attach for stable EMA keying
    fc._gm_seed = int(seed)
    with torch.no_grad():
        try:
            w_init = torch.randn(fc.linear.weight.shape, device=device, generator=g) * 0.02
            b_init = torch.randn(fc.linear.bias.shape, device=device, generator=g) * 0.02
        except TypeError:
            # Very old torch: no `generator=` support. Fall back to global seeding.
            torch.manual_seed(seed)
            if str(device).startswith('cuda'):
                torch.cuda.manual_seed_all(seed)
            w_init = torch.randn(fc.linear.weight.shape, device=device) * 0.02
            b_init = torch.randn(fc.linear.bias.shape, device=device) * 0.02
        fc.linear.weight.copy_(w_init)
        fc.linear.bias.copy_(b_init)
    return fc


def _gm_soft_targets_for_class(self, class_id: int, num_classes: int, device):
    """Category-aware soft target distribution q over [0..num_classes-1] for a given class.
    q = (1-alpha)*onehot + alpha*softmax(sim(mu_c, mu_j)/tau)
    Cached per (num_classes, class_id).
    """
    class_id = int(class_id)
    num_classes = int(num_classes)
    alpha = float(getattr(self, "replay_gm_soft_alpha", 0.0))
    tau = float(getattr(self, "replay_gm_soft_tau", 1.0))
    if (not bool(getattr(self, "replay_gm_soft", False))) or alpha <= 0.0:
        q = torch.zeros(num_classes, device=device)
        if 0 <= class_id < num_classes:
            q[class_id] = 1.0
        return q
    if (not hasattr(self, "_class_means")) or (self._class_means is None):
        q = torch.zeros(num_classes, device=device)
        if 0 <= class_id < num_classes:
            q[class_id] = 1.0
        return q
    if not hasattr(self, "_gm_soft_label_cache") or (self._gm_soft_label_cache is None):
        self._gm_soft_label_cache = {}
    key = (int(num_classes), int(class_id))
    cached = self._gm_soft_label_cache.get(key, None)
    if cached is not None:
        if cached.device != device:
            cached = cached.to(device)
        return cached

    # Use class prototypes to derive inter-class similarities (stable, sample-independent).
    C = min(int(num_classes), int(self._class_means.shape[0]))
    mus = self._class_means[:C].to(device).float()  # [C, D]
    mus = F.normalize(mus, dim=1)
    if class_id < 0 or class_id >= C:
        q = torch.zeros(num_classes, device=device)
        if 0 <= class_id < num_classes:
            q[class_id] = 1.0
        self._gm_soft_label_cache[key] = q.detach()
        return q

    v = mus[class_id:class_id + 1]  # [1, D]
    sim = (v @ mus.t()).squeeze(0)  # [C]
    if tau is None or tau <= 0:
        tau = 1.0
    base = torch.softmax(sim / float(tau), dim=0)  # [C]
    q = torch.zeros(num_classes, device=device)
    q[:C] = base
    one = torch.zeros_like(q)
    one[class_id] = 1.0
    q = (1.0 - alpha) * one + alpha * q
    self._gm_soft_label_cache[key] = q.detach()
    return q


def _gm_loss_from_logits(self, logits: torch.Tensor, class_id: int, num_classes: int) -> torch.Tensor:
    """Soft-target cross entropy for GM.
    If replay_gm_soft enabled: use category-aware q; else standard CE with hard label.
    """
    class_id = int(class_id)
    num_classes = int(num_classes)
    if bool(getattr(self, "replay_gm_soft", False)) and float(getattr(self, "replay_gm_soft_alpha", 0.0)) > 0.0:
        q = self._gm_soft_targets_for_class(class_id, num_classes, device=logits.device)  # [C]
        q = q.view(1, -1).expand(logits.size(0), -1)
        logp = F.log_softmax(logits, dim=1)
        return -(q * logp).sum(dim=1).mean()
    else:
        y = torch.full((logits.size(0),), class_id, device=logits.device, dtype=torch.long)
        return F.cross_entropy(logits, y)


def _linear_gm_real_grad_from_real_data(self, model, data_manager, class_id: int, bs: int, gm_fc):
    """Stabilized real GM target:
    - average over multiple *small* batches (reduces batch noise)
    - optional EMA over repeated calls (per class & gm head seed)
    - optionally uses soft category-aware targets
    """
    class_id = int(class_id)
    bs = int(bs)

    num_batches = max(1, int(getattr(self, "replay_gradmatch_real_batches", 1)))
    bs_small = int(getattr(self, "replay_gradmatch_real_bs_small", bs))
    bs_small = max(1, min(bs_small, bs))

    _, _, ds = data_manager.get_dataset(
        np.arange(class_id, class_id + 1),
        source="train",
        mode="test",
        ret_data=True,
    )
    loader = DataLoader(ds, batch_size=bs_small, shuffle=True, num_workers=0, drop_last=True)
    it = iter(loader)

    grads = []
    for _b in range(num_batches):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        # batch can be (idx, x, y) or (x, y)
        if isinstance(batch, (tuple, list)):
            if len(batch) >= 3:
                x, y = batch[-2], batch[-1]
            elif len(batch) == 2:
                x, y = batch
            else:
                raise RuntimeError(f"Unexpected batch tuple length: {len(batch)}")
        elif isinstance(batch, dict):
            x = batch.get("image", None)
            y = batch.get("label", None)
            if x is None or y is None:
                raise RuntimeError("Unexpected batch dict keys for replay GM target.")
        else:
            raise RuntimeError(f"Unexpected batch type: {type(batch)}")

        x = x.to(self._device, non_blocking=True)
        y = y.to(self._device, non_blocking=True)
        c = int(y.view(-1)[0].item())
        num_classes = int(gm_fc.linear.out_features)

        with torch.no_grad():
            out = model(x)
            z = out["features"]

        # ---- GM target vector ----
        if str(getattr(self, "replay_gm_mode", "headgrad")).lower() == "featgrad":
            # g_z = (p - q) @ W where logits = W z + b (W: [C,D])
            W = gm_fc.linear.weight.detach()
            b = gm_fc.linear.bias.detach()
            logits = F.linear(z, W, b)  # [B, C]
            if bool(getattr(self, "replay_gm_soft", False)) and float(getattr(self, "replay_gm_soft_alpha", 0.0)) > 0.0:
                q = self._gm_soft_targets_for_class(c, num_classes, device=logits.device).view(1, -1).expand_as(logits)
            else:
                q = torch.zeros_like(logits)
                q[:, c] = 1.0
            p = torch.softmax(logits, dim=1)
            g_z = (p - q) @ W  # [B, D]
            grads.append(g_z.mean(dim=0).detach())
        else:
            logits = gm_fc(z)
            loss = self._gm_loss_from_logits(logits, class_id=c, num_classes=num_classes)
            grad_w, grad_b = torch.autograd.grad(
                loss,
                [gm_fc.linear.weight, gm_fc.linear.bias],
                retain_graph=False,
                create_graph=False,
            )
            grads.append(torch.cat([grad_w.detach().flatten(), grad_b.detach().flatten()], dim=0))

    g_avg = torch.stack(grads, dim=0).mean(dim=0)

    # EMA over repeated calls (keyed by class + gm head init seed + output classes)
    ema = float(getattr(self, "replay_gradmatch_ema", 0.0))
    if ema is not None and ema > 0.0:
        if not hasattr(self, "_gm_real_ema") or (self._gm_real_ema is None):
            self._gm_real_ema = {}
        seed = int(getattr(gm_fc, "_gm_seed", 0))
        key = (int(c), int(seed), int(num_classes), str(getattr(self, 'replay_gm_mode', 'headgrad')).lower())
        prev = self._gm_real_ema.get(key, None)
        if prev is None:
            cur = g_avg
        else:
            if prev.device != g_avg.device:
                prev = prev.to(g_avg.device)
            cur = (1.0 - ema) * prev + ema * g_avg
        self._gm_real_ema[key] = cur.detach()
        return cur.detach()

    return g_avg.detach()


def _linear_gm_real_grad_from_image(self, model, img: torch.Tensor, class_id: int, gm_fc, mean, std, pad: int,
                                    views: int):
    """Real GM target from a single reference image (teacher).
    Supports gm_fc as a single head or a list/tuple of heads.
    """
    class_id = int(class_id)
    views = int(views)
    pad = int(pad)

    feats = []
    with torch.no_grad():
        for _ in range(max(1, views)):
            v_img = self._random_shift_and_flip(img, pad=pad)
            v_norm = (v_img - mean) / std
            out = model(v_norm)
            feats.append(out["features"])
        z = torch.cat(feats, dim=0)

    def _one_fc(fc):
        num_classes = int(fc.linear.out_features)
        if str(getattr(self, "replay_gm_mode", "headgrad")).lower() == "featgrad":
            W = fc.linear.weight.detach()
            b = fc.linear.bias.detach()
            logits = F.linear(z, W, b)
            if bool(getattr(self, "replay_gm_soft", False)) and float(getattr(self, "replay_gm_soft_alpha", 0.0)) > 0.0:
                q = self._gm_soft_targets_for_class(class_id, num_classes, device=logits.device).view(1, -1).expand_as(
                    logits)
            else:
                q = torch.zeros_like(logits)
                q[:, class_id] = 1.0
            p = torch.softmax(logits, dim=1)
            g_z = (p - q) @ W
            return g_z.mean(dim=0).detach()
        else:
            logits = fc(z)
            loss = self._gm_loss_from_logits(logits, class_id=class_id, num_classes=num_classes)
            grad_w, grad_b = torch.autograd.grad(
                loss,
                [fc.linear.weight, fc.linear.bias],
                retain_graph=False,
                create_graph=False,
            )
            return torch.cat([grad_w.detach().flatten(), grad_b.detach().flatten()], dim=0)

    if isinstance(gm_fc, (list, tuple)):
        return [_one_fc(fc) for fc in gm_fc]
    else:
        return _one_fc(gm_fc)


def _gm_targets_for_labels(self, labels: torch.Tensor, num_classes: int, device):
    labels = labels.to(device=device, dtype=torch.long).view(-1)
    num_classes = int(num_classes)
    if bool(getattr(self, "replay_gm_soft", False)) and float(getattr(self, "replay_gm_soft_alpha", 0.0)) > 0.0:
        qs = [self._gm_soft_targets_for_class(int(y.item()), num_classes, device=device) for y in labels]
        return torch.stack(qs, dim=0)
    q = torch.zeros(labels.numel(), num_classes, device=device)
    valid = (labels >= 0) & (labels < num_classes)
    if valid.any():
        q[torch.arange(labels.numel(), device=device)[valid], labels[valid]] = 1.0
    return q


def _gm_loss_from_logits_batch(self, logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    labels = labels.to(device=logits.device, dtype=torch.long).view(-1)
    if bool(getattr(self, "replay_gm_soft", False)) and float(getattr(self, "replay_gm_soft_alpha", 0.0)) > 0.0:
        q = self._gm_targets_for_labels(labels, num_classes, device=logits.device)
        logp = F.log_softmax(logits, dim=1)
        return -(q * logp).sum(dim=1).mean()
    return F.cross_entropy(logits, labels)


def _linear_gm_grad_from_features_batch(self, z: torch.Tensor, labels: torch.Tensor, gm_fc, create_graph: bool):
    labels = labels.to(device=z.device, dtype=torch.long).view(-1)
    if labels.numel() != z.size(0):
        repeat = max(1, int(z.size(0) // max(labels.numel(), 1)))
        labels = labels.repeat_interleave(repeat)[:z.size(0)]

    def _one_fc(fc, retain: bool):
        num_classes = int(fc.linear.out_features)
        if str(getattr(self, "replay_gm_mode", "headgrad")).lower() == "featgrad":
            W = fc.linear.weight.detach()
            b = fc.linear.bias.detach()
            logits = F.linear(z, W, b)
            q = self._gm_targets_for_labels(labels, num_classes, device=logits.device)
            p = torch.softmax(logits, dim=1)
            g_z = (p - q) @ W
            return g_z.mean(dim=0) if create_graph else g_z.mean(dim=0).detach()
        logits = fc(z)
        loss = self._gm_loss_from_logits_batch(logits, labels=labels, num_classes=num_classes)
        grad_w, grad_b = torch.autograd.grad(
            loss,
            [fc.linear.weight, fc.linear.bias],
            retain_graph=retain,
            create_graph=create_graph,
        )
        out = torch.cat([grad_w.flatten(), grad_b.flatten()], dim=0)
        return out if create_graph else out.detach()

    if isinstance(gm_fc, (list, tuple)):
        outs = []
        K = len(gm_fc)
        for i, fc in enumerate(gm_fc):
            outs.append(_one_fc(fc, retain=(i < K - 1) or create_graph))
        return outs
    return _one_fc(gm_fc, retain=create_graph)


def _linear_gm_real_grad_from_image_batch(self, model, img: torch.Tensor, labels: torch.Tensor, gm_fc, mean, std,
                                         pad: int, views: int):
    views = max(1, int(views))
    feats = []
    with torch.no_grad():
        for _ in range(views):
            v_img = self._random_shift_and_flip(img, pad=pad)
            v_norm = (v_img - mean) / std
            out = model(v_norm)
            feats.append(out["features"])
        z = torch.cat(feats, dim=0)
    labels_v = labels.to(self._device).repeat(views)
    return self._linear_gm_grad_from_features_batch(z, labels_v, gm_fc, create_graph=False)


def _linear_gm_syn_grad_from_image_batch(self, model, img: torch.Tensor, labels: torch.Tensor, gm_fc, mean, std,
                                        pad: int, views: int):
    views = max(1, int(views))
    feats = []
    for _ in range(views):
        v_img = self._random_shift_and_flip(img, pad=pad)
        v_norm = (v_img - mean) / std
        out = model(v_norm)
        feats.append(out["features"])
    z = torch.cat(feats, dim=0)
    labels_v = labels.to(self._device).repeat(views)
    return self._linear_gm_grad_from_features_batch(z, labels_v, gm_fc, create_graph=True)


def _linear_gm_real_grad_from_real_class_set(self, model, data_manager, class_ids, gm_fc):
    if data_manager is None:
        raise RuntimeError("DataManager is not set")
    xs, ys = [], []
    bs_per_class = max(1, int(getattr(self, "replay_set_real_per_class", 2)))
    mode = str(getattr(self, "replay_set_real_mode", "test"))
    for c in [int(x) for x in class_ids]:
        _, _, ds = data_manager.get_dataset(
            np.arange(c, c + 1),
            source="train",
            mode=mode,
            ret_data=True,
        )
        loader = DataLoader(ds, batch_size=bs_per_class, shuffle=True, num_workers=0, drop_last=False)
        batch = next(iter(loader))
        if isinstance(batch, (tuple, list)):
            if len(batch) >= 3:
                x, y = batch[-2], batch[-1]
            elif len(batch) == 2:
                x, y = batch
            else:
                raise RuntimeError(f"Unexpected batch tuple length: {len(batch)}")
        elif isinstance(batch, dict):
            x = batch.get("image", None)
            y = batch.get("label", None)
            if x is None or y is None:
                raise RuntimeError("Unexpected batch dict keys for set-level GM target.")
        else:
            raise RuntimeError(f"Unexpected batch type: {type(batch)}")
        xs.append(x)
        ys.append(y)

    x = torch.cat(xs, dim=0).to(self._device, non_blocking=True)
    y = torch.cat(ys, dim=0).to(self._device, non_blocking=True)
    with torch.no_grad():
        out = model(x)
        z = out["features"]
    return self._linear_gm_grad_from_features_batch(z, y, gm_fc, create_graph=False)


def _linear_gm_syn_grad_from_image(self, model, img: torch.Tensor, class_id: int, gm_fc, mean, std, pad: int,
                                   views: int):
    """Synthetic GM gradient from the current (optimizable) image.
    Supports gm_fc as a single head or a list/tuple of heads.
    Returns:
      - tensor vector if single head
      - list[tensor vector] if multi-head
    """
    class_id = int(class_id)
    views = int(views)
    pad = int(pad)

    feats = []
    for _ in range(max(1, views)):
        v_img = self._random_shift_and_flip(img, pad=pad)
        v_norm = (v_img - mean) / std
        out = model(v_norm)
        feats.append(out["features"])
    z = torch.cat(feats, dim=0)  # keep graph for backprop to img

    def _one_fc(fc, retain: bool):
        num_classes = int(fc.linear.out_features)
        if str(getattr(self, "replay_gm_mode", "headgrad")).lower() == "featgrad":
            W = fc.linear.weight.detach()
            b = fc.linear.bias.detach()
            logits = F.linear(z, W, b)
            if bool(getattr(self, "replay_gm_soft", False)) and float(getattr(self, "replay_gm_soft_alpha", 0.0)) > 0.0:
                q = self._gm_soft_targets_for_class(class_id, num_classes, device=logits.device).view(1, -1).expand_as(
                    logits)
            else:
                q = torch.zeros_like(logits)
                q[:, class_id] = 1.0
            p = torch.softmax(logits, dim=1)
            g_z = (p - q) @ W
            return g_z.mean(dim=0)
        else:
            logits = fc(z)
            loss = self._gm_loss_from_logits(logits, class_id=class_id, num_classes=num_classes)
            grad_w, grad_b = torch.autograd.grad(
                loss,
                [fc.linear.weight, fc.linear.bias],
                retain_graph=retain,
                create_graph=True,
            )
            return torch.cat([grad_w.flatten(), grad_b.flatten()], dim=0)

    if isinstance(gm_fc, (list, tuple)):
        outs = []
        K = len(gm_fc)
        for i, fc in enumerate(gm_fc):
            outs.append(_one_fc(fc, retain=True))
        return outs
    else:
        return _one_fc(gm_fc, retain=True)


def _select_head_params(self, model):
    """Select (approx.) classifier-head parameters for gradient matching.

    We try common names first ('fc', 'classifier', 'head'). If none matched, fallback to the last
    couple of trainable parameters.
    """
    params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        ln = n.lower()
        if (('fc' in ln) or ('classifier' in ln) or ('head' in ln)) and (('weight' in ln) or ('bias' in ln)):
            params.append(p)
    if len(params) == 0:
        # Fallback: last few trainable params
        tail = []
        for _n, _p in reversed(list(model.named_parameters())):
            if _p.requires_grad:
                tail.append(_p)
            if len(tail) >= 2:
                break
        params = list(reversed(tail))
    return params


def _flatten_grads(self, grads):
    flat = []
    for g in grads:
        if g is None:
            continue
        flat.append(g.reshape(-1))
    if len(flat) == 0:
        return torch.zeros(0, device=self._device)
    return torch.cat(flat, dim=0)


def _grad_match_loss(self, g_syn: torch.Tensor, g_real: torch.Tensor) -> torch.Tensor:
    """Cosine distance between two gradient vectors (both treated as vectors)."""
    if (g_syn is None) or (g_real is None) or (g_syn.numel() == 0) or (g_real.numel() == 0):
        return torch.tensor(0.0, device=self._device)
    if g_real.device != g_syn.device:
        g_real = g_real.to(g_syn.device)
    g_syn = g_syn.float()
    g_real = g_real.float()
    g_syn_n = g_syn / (g_syn.norm() + 1e-12)
    g_real_n = g_real / (g_real.norm() + 1e-12)
    return 1.0 - torch.sum(g_syn_n * g_real_n)


def _compute_real_kd_head_grad(self, old_model, new_model, class_id: int, bs: int, old_end: int) -> torch.Tensor:
    """Compute KD-head gradient on a small REAL batch for class_id (used as grad-match target).

    This explicitly uses NEW-vs-OLD discrepancy:
        KD( softmax(z_old/T) || softmax(z_new/T) )
    and returns d(KD)/d(theta_head_new) as a flattened vector (detached).
    """
    dm = self._data_manager
    if dm is None:
        raise RuntimeError('DataManager is not set')
    old_end = int(old_end)
    # Get a small real dataset for this class (train split)
    _, _, ds = dm.get_dataset(np.arange(class_id, class_id + 1), source='train', mode='train', ret_data=True)
    loader = DataLoader(ds, batch_size=min(64, int(bs)), shuffle=True, num_workers=0)

    batch = next(iter(loader))
    if isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            x, _y = batch
        elif len(batch) >= 3:
            x, _y = batch[-2], batch[-1]
        else:
            raise RuntimeError(f'Unexpected batch tuple length: {len(batch)}')
    elif isinstance(batch, dict):
        x = batch.get('image', None)
        _y = batch.get('label', None)
        if x is None:
            raise RuntimeError('Unexpected batch dict keys; expected image')
    else:
        raise RuntimeError(f'Unexpected batch type: {type(batch)}')

    x = x.to(self._device, non_blocking=True)

    old_model.eval()
    new_model.eval()

    T = float(self.args.get("kd_T", 2.0))

    # KD loss in OLD logits space only (0..old_end-1)
    logits_old = old_model.interface(x)[:, :old_end]
    logits_new = new_model.interface(x)[:, :old_end]
    p_old = torch.softmax(logits_old / T, dim=1).detach()
    log_p_new = torch.log_softmax(logits_new / T, dim=1)
    kd = F.kl_div(log_p_new, p_old, reduction='batchmean') * (T * T)

    head_params = self._select_head_params(new_model)
    if len(head_params) == 0:
        return torch.zeros(0, device=self._device)

    grads = torch.autograd.grad(kd, head_params, retain_graph=False, create_graph=False, allow_unused=True)
    g = self._flatten_grads(grads).detach()
    return g


# =========================
# helpers for per-image target stats (strict CIL: no old real data)
# =========================
def _logit_ce_grad_from_logits(self, logits: torch.Tensor, target_class: int, T: float = 1.0) -> torch.Tensor:
    """Backprop signal for CE in logit space.

    For CE(log_softmax(z/T), y_onehot), dL/dz = (softmax(z/T) - y)/T.
    This is cheap and avoids 2nd-order derivatives when used for gradient matching on images.
    """
    T = float(T) if T is not None else 1.0
    if T <= 0:
        T = 1.0
    p = torch.softmax(logits / T, dim=1)
    y = torch.zeros_like(p)
    y[:, int(target_class)] = 1.0
    return (p - y) / T


@torch.no_grad()
def _compute_replay_logit_grad_target(
        self,
        model,
        img: torch.Tensor,
        target_class: int,
        old_end: int,
        mean: torch.Tensor,
        std: torch.Tensor,
        T: float,
        views: int = 4,
        pad: int = 8,
) -> torch.Tensor:
    """Compute target logit-gradient (averaged over views) from (model, img) in OLD logits space."""
    model.eval()
    old_end = int(old_end)
    if img.dim() == 3:
        img = img.unsqueeze(0)
    grads = []
    V = max(1, int(views))
    for _v in range(V):
        v_img = self._random_shift_and_flip(img, pad=pad)
        v_norm = (v_img - mean) / std
        logits = model.interface(v_norm)[:, :old_end]
        g = self._logit_ce_grad_from_logits(logits, int(target_class), T)
        grads.append(g)
    g_mu = torch.stack(grads, dim=0).mean(dim=0)  # [1, old_end]
    return g_mu.squeeze(0).detach()


def _random_shift_and_flip(self, x: torch.Tensor, pad: int = 8) -> torch.Tensor:
    """Randomly shift via reflect-pad + crop, and random horizontal flip."""
    if pad is None or pad <= 0:
        out = x
    else:
        out = TF.pad(x, [pad, pad, pad, pad], padding_mode='reflect')
        H = W = int(self.replay_gen_size)
        i = int(torch.randint(0, 2 * pad + 1, (1,), device=x.device).item())
        j = int(torch.randint(0, 2 * pad + 1, (1,), device=x.device).item())
        out = out[:, :, i:i + H, j:j + W]
    if torch.rand((), device=x.device).item() < 0.5:
        out = torch.flip(out, dims=[3])
    return out


def _tv_loss(self, x: torch.Tensor) -> torch.Tensor:
    """Total variation regularization (L1)."""
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


def _load_image_as_tensor_01(self, path: str, size: int = 224) -> torch.Tensor:
    """Load an image file and return a tensor [3,H,W] in [0,1] on self._device.

    We always resize to (size,size) to keep replay tensors consistent.
    """
    img = Image.open(path).convert('RGB')
    if size is not None:
        img = TF.resize(img, [int(size), int(size)])
    x = TF.to_tensor(img).to(self._device)  # [0,1]
    return x


def _save_tensor_as_png(self, img_01: torch.Tensor, path: str):
    """Save a [3,H,W] tensor in [0,1] to PNG (atomic write)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # IMPORTANT: PIL infers format from extension; a bare `.tmp` will fail.
    tmp_path = path + '.tmp.png'
    img = img_01.detach().clamp(0, 1).cpu()
    arr = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    try:
        Image.fromarray(arr).save(tmp_path, format='PNG')
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def accuracy(self, y_pred, y_true, accuracy_matrix=False):
    assert len(y_pred) == len(y_true), 'Data length error.'
    all_acc = {}
    all_acc['total'] = np.around((y_pred == y_true).sum() * 100 / len(y_true), decimals=2)

    i = 0
    # Grouped accuracy
    for class_id in range(0, np.max(y_true), self.class_num):
        idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + self.class_num))[0]
        label = '{}-{}'.format(str(class_id).rjust(2, '0'), str(class_id + self.class_num - 1).rjust(2, '0'))
        all_acc[label] = np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)
        if accuracy_matrix:
            self.acc_matrix[i, self._cur_task] = all_acc[label]
        i += 1

    # Old accuracy
    idxes = np.where(y_true < self._known_classes)[0]
    all_acc['old'] = 0 if len(idxes) == 0 else np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes),
                                                         decimals=2)

    # New accuracy
    idxes = np.where(y_true >= self._known_classes)[0]
    all_acc['new'] = np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)

    return all_acc


def _evaluate(self, y_pred, y_true, accuracy_matrix=False):
    ret = {}
    # print(len(y_pred), len(y_true))
    grouped = self.accuracy(y_pred, y_true, accuracy_matrix=accuracy_matrix)
    ret['grouped'] = grouped
    ret['top1'] = grouped['total']
    return ret


def _eval_cnn(self, loader):
    self._network.eval()
    y_pred, y_true = [], []
    y_pred_with_task = []
    y_pred_task, y_true_task = [], []

    for _, (_, inputs, targets) in enumerate(loader):
        inputs = inputs.to(self._device)
        targets = targets.to(self._device)

        with torch.no_grad():
            task_id = (targets // self.class_num).cpu()
            y_true_task.append(task_id)

            outputs = self._network.interface(inputs)

        predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1].view(-1)  # [bs, topk]
        y_pred_task.append((predicts // self.class_num).cpu())

        outputs_with_task = torch.zeros_like(outputs)[:, :self.class_num]
        for idx, i in enumerate(targets // self.class_num):
            en, be = self.class_num * i, self.class_num * (i + 1)
            outputs_with_task[idx] = outputs[idx, en:be]
        predicts_with_task = outputs_with_task.argmax(dim=1)
        predicts_with_task = predicts_with_task + (targets // self.class_num) * self.class_num

        y_pred.append(predicts.cpu().numpy())
        y_pred_with_task.append(predicts_with_task.cpu().numpy())
        y_true.append(targets.cpu().numpy())

    return np.concatenate(y_pred), np.concatenate(y_pred_with_task), np.concatenate(y_true), torch.cat(
        y_pred_task), torch.cat(y_true_task)  # [N, topk]


def _compute_accuracy(self, model, loader):
    model.eval()
    correct, total = 0, 0
    for i, (_, inputs, targets) in enumerate(loader):
        inputs = inputs.to(self._device)
        with torch.no_grad():
            outputs = model.interface(inputs)
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == targets).sum()
        total += len(targets)

    return np.around(tensor2numpy(correct) * 100 / total, decimals=2)


def _stage2_compact_classifier(self, task_size, ca_epochs=5):
    for p in self._network.classifier_pool[:self._cur_task + 1].parameters():
        p.requires_grad = True

    run_epochs = ca_epochs
    crct_num = self._total_classes
    param_list = [p for p in self._network.classifier_pool.parameters() if p.requires_grad]
    network_params = [{'params': param_list, 'lr': 0.01,
                       'weight_decay': 0.0005}]
    optimizer = optim.SGD(network_params, lr=0.01, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

    self._network.to(self._device)

    self._network.eval()
    for epoch in range(run_epochs):
        losses = 0.

        sampled_data = []
        sampled_label = []
        num_sampled_pcls = 256

        for c_id in range(crct_num):
            t_id = c_id // task_size
            decay = (t_id + 1) / (self._cur_task + 1) * 0.1
            cls_mean = self._class_means[c_id].to(self._device) * (0.9 + decay)
            cls_cov = self._class_covs[c_id].to(self._device)

            m = MultivariateNormal(cls_mean.float(), cls_cov.float())

            sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
            sampled_data.append(sampled_data_single)
            sampled_label.extend([c_id] * num_sampled_pcls)

        sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
        sampled_label = torch.tensor(sampled_label).long().to(self._device)

        inputs = sampled_data
        targets = sampled_label

        sf_indexes = torch.randperm(inputs.size(0))
        inputs = inputs[sf_indexes]
        targets = targets[sf_indexes]

        for _iter in range(crct_num):
            inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
            # -stage two only use classifiers
            outputs = self._network(inp, fc_only=True)
            logits = outputs

            if self.logit_norm is not None:
                per_task_norm = []
                prev_t_size = 0
                cur_t_size = 0
                for _ti in range(self._cur_task + 1):
                    cur_t_size += self.task_sizes[_ti]
                    temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                    per_task_norm.append(temp_norm)
                    prev_t_size += self.task_sizes[_ti]
                per_task_norm = torch.cat(per_task_norm, dim=-1)
                norms = per_task_norm.mean(dim=-1, keepdim=True)

                norms_all = torch.norm(logits[:, :crct_num], p=2, dim=-1, keepdim=True) + 1e-7
                decoupled_logits = torch.div(logits[:, :crct_num], norms) / self.logit_norm
                loss = F.cross_entropy(decoupled_logits, tgt)
            else:
                loss = F.cross_entropy(logits[:, :crct_num], tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

        scheduler.step()
        test_acc = self._compute_accuracy(self._network, self.test_loader)
        info = 'CA Task {} => Loss {:.3f}, Test_accy {:.3f}'.format(
            self._cur_task, losses / self._total_classes, test_acc)
        logging.info(info)


def _compute_class_mean(self, data_manager, check_diff=False, oracle=False):
    if hasattr(self, '_class_means') and self._class_means is not None and not check_diff:
        ori_classes = self._class_means.shape[0]
        assert ori_classes == self._known_classes
        new_class_means = torch.zeros((self._total_classes, self.feature_dim))
        new_class_means[:self._known_classes] = self._class_means
        self._class_means = new_class_means
        new_class_cov = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
        new_class_cov[:self._known_classes] = self._class_covs
        self._class_covs = new_class_cov
    elif not check_diff:
        self._class_means = torch.zeros((self._total_classes, self.feature_dim))
        self._class_covs = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))

    for class_idx in range(self._known_classes, self._total_classes):
        data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx + 1), source='train',
                                                              mode='test', ret_data=True)
        idx_loader = DataLoader(idx_dataset, batch_size=64, shuffle=False, num_workers=4)
        vectors, _ = self._extract_vectors(idx_loader)

        class_mean = torch.mean(torch.tensor(vectors), dim=0)
        class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T) + torch.eye(class_mean.shape[-1]) * 1e-3

        self._class_means[class_idx, :] = class_mean.detach()
        self._class_covs[class_idx, ...] = class_cov.detach()

    # Invalidate cached inverse covariances because class covariances may have changed.
    if hasattr(self, '_proto_invcov_cache'):
        self._proto_invcov_cache = {}


def displacement(self, Y1, Y2, embedding_old, sigma):
    DY = Y2 - Y1
    distance = np.sum((np.tile(Y1[None, :, :], [embedding_old.shape[0], 1, 1]) - np.tile(
        embedding_old[:, None, :], [1, Y1.shape[0], 1])) ** 2, axis=2)
    W = np.exp(-distance / (2 * sigma ** 2)) + 1e-5
    W_norm = W / np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])
    displacement = np.sum(np.tile(W_norm[:, :, None], [
        1, 1, DY.shape[1]]) * np.tile(DY[None, :, :], [W.shape[0], 1, 1]), axis=1)
    return displacement


def extract_features(self, trainloader, model, task_id=None):
    model = model.eval()
    embedding_list = []
    label_list = []
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            (_, data, label) = batch
            data = data.to(self._device)
            label = label.to(self._device)
            embedding = model.extract_vector(data, task_id)
            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())

    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)
    return embedding_list, label_list


def _extract_vectors_adv(self, loader, old=False):
    if old:
        network = self._old_network
    else:
        network = self._network
    network.eval()
    vectors, targets = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            (_, _inputs, _targets) = batch
            _inputs = _inputs.to(self._device)
            _vectors = network.extract_vector(_inputs)
            vectors.append(_vectors)
            targets.append(_targets)

    return torch.cat(vectors, dim=0), torch.cat(targets, dim=0)


def shrink_cov(self, cov):
    alpha1 = 10
    alpha2 = 10
    # Compute the mean of the diagonal elements
    diag_mean = torch.mean(torch.diagonal(cov))

    # Create a copy of the covariance matrix with zeroed out diagonals
    off_diag = cov.clone()
    off_diag.fill_diagonal_(0.0)

    # Compute the mean of the off-diagonal elements (non-zero entries)
    mask = off_diag != 0.0
    off_diag_mean = (off_diag * mask).sum() / mask.sum()

    # Identity matrix
    iden = torch.eye(cov.size(0), device=cov.device)

    # Shrink the covariance matrix
    cov_ = cov + (alpha1 * diag_mean * iden) + (alpha2 * off_diag_mean * (1 - iden))

    return cov_


def _compute_class_invcov(self, data_manager):
    _class_invcovs = torch.zeros((self.class_num, self.feature_dim, self.feature_dim), device=self._device)

    for class_idx in range(self._known_classes, self._total_classes):
        data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx + 1), source='train',
                                                              mode='test', ret_data=True)
        idx_loader = DataLoader(idx_dataset, batch_size=64, shuffle=False, num_workers=4)
        vectors, _ = self._extract_vectors_adv(idx_loader, True)

        class_cov = self.shrink_cov(torch.cov(torch.tensor(vectors, dtype=torch.float64).T)) + torch.eye(
            self.feature_dim).to(self._device) * 1e-3
        _class_invcovs[class_idx - self._known_classes, ...] = torch.linalg.pinv(class_cov).detach()

    return _class_invcovs


# =========================
# Patch: bind free functions as Learner methods
# (Fixes: AttributeError: Learner has no train_function)
# =========================

try:
    _L = Learner  # noqa: F821
except NameError:
    _L = None

if _L is not None:
    _L._cosine_from_grads = staticmethod(_cosine_from_grads)
    _L._build_random_linear_fc = _build_random_linear_fc
    _L._collect_replay_from_dir = _collect_replay_from_dir
    _L._compute_accuracy = _compute_accuracy
    _L._compute_class_invcov = _compute_class_invcov
    _L._compute_class_mean = _compute_class_mean
    _L._compute_param_grads = _compute_param_grads
    _L._compute_real_kd_head_grad = _compute_real_kd_head_grad
    _L._compute_replay_logit_grad_target = _compute_replay_logit_grad_target
    _L._eval_cnn = _eval_cnn
    _L._evaluate = _evaluate
    _L._extract_vectors_adv = _extract_vectors_adv
    _L._flatten_grads = _flatten_grads
    _L._angle_weighted_patch_loss = _angle_weighted_patch_loss
    _L._generate_replay_images_for_current_task = _generate_replay_images_for_current_task
    _L._get_influence_params = _get_influence_params
    _L._get_proto_stats = _get_proto_stats
    _L._grad_match_loss = _grad_match_loss
    _L._limit_replay_ipc = _limit_replay_ipc
    _L._linear_gm_real_grad_from_image = _linear_gm_real_grad_from_image
    _L._linear_gm_real_grad_from_image_batch = _linear_gm_real_grad_from_image_batch
    _L._linear_gm_real_grad_from_real_class_set = _linear_gm_real_grad_from_real_class_set
    _L._linear_gm_real_grad_from_real_data = _linear_gm_real_grad_from_real_data
    _L._linear_gm_grad_from_features_batch = _linear_gm_grad_from_features_batch
    _L._linear_gm_syn_grad_from_image = _linear_gm_syn_grad_from_image
    _L._linear_gm_syn_grad_from_image_batch = _linear_gm_syn_grad_from_image_batch
    _L._load_image_as_tensor_01 = _load_image_as_tensor_01
    _L._logit_ce_grad_from_logits = _logit_ce_grad_from_logits
    _L._map_replay_labels = _map_replay_labels
    _L._optimize_one_replay_image = _optimize_one_replay_image
    _L._optimize_replay_image_set = _optimize_replay_image_set
    _L._proto_mahalanobis_loss = _proto_mahalanobis_loss
    _L._real_patch_template_from_real_data = _real_patch_template_from_real_data
    _L._random_shift_and_flip = _random_shift_and_flip
    _L._random_shift_and_flip_pair = _random_shift_and_flip_pair
    _L._refine_one_replay_image = _refine_one_replay_image
    _L._save_tensor_as_png = _save_tensor_as_png
    _L._select_head_params = _select_head_params
    _L._stage2_compact_classifier = _stage2_compact_classifier
    _L._tv_loss = _tv_loss
    _L._set_param_grads = _set_param_grads
    _L._update_old_replay_images_after_task = _update_old_replay_images_after_task
    _L._update_old_replay_images_after_task_setlevel = _update_old_replay_images_after_task_setlevel
    _L._update_replay_influence_weights = _update_replay_influence_weights
    _L._replay_grad_guidance = _replay_grad_guidance
    _L._chunk_classes = _chunk_classes
    _L._existing_replay_files = _existing_replay_files
    _L._generate_replay_images_for_current_task_setlevel = _generate_replay_images_for_current_task_setlevel
    _L._gm_loss_from_logits_batch = _gm_loss_from_logits_batch
    _L._gm_targets_for_labels = _gm_targets_for_labels
    _L._next_replay_path = _next_replay_path
    _L.accuracy = accuracy
    _L.displacement = displacement
    _L.shrink_cov = shrink_cov
    _L.extract_features = extract_features
    _L.train_function = train_function
    _L._maybe_purge_replay_root = _maybe_purge_replay_root
    _L._GMFC = _GMFC

    # -------------------------
    # Extra safety: auto-bind any orphan top-level functions whose first arg is `self`
    # (Prevents repeated AttributeError when functions were accidentally de-indented)
    # -------------------------
    import types as _types
    import inspect as _inspect

    for _name, _obj in list(globals().items()):
        if isinstance(_obj, _types.FunctionType) and (not hasattr(_L, _name)):
            try:
                _params = list(_inspect.signature(_obj).parameters.values())
            except Exception:
                continue
            if _params and _params[0].name == "self":
                setattr(_L, _name, _obj)
