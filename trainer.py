import os
import os.path
import sys
import logging
import copy
import time
import contextlib
import torch
import numpy as np
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters


def train(args):
    seed_list = copy.deepcopy(args.get('seed', [0]))
    device = copy.deepcopy(args.get('device', '0'))
    device = device.split(',')

    for seed in seed_list:
        args['seed'] = seed
        args['device'] = device
        _train(args)


def _train(args):
    logdir = 'logs/{}/{}_tasks'.format(args.get('dataset', 'dataset'), args.get('total_sessions', 'X'))
    args['logdir'] = logdir
    os.makedirs(logdir, exist_ok=True)

    # ---- avoid duplicated handlers when looping multiple seeds ----
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    def _exp_tag(a: dict) -> str:
        # core tags
        lr_show = a.get('lrate', a.get('init_lr', '?'))
        tag = [
            f"seed{a.get('seed','?')}",
            f"init{a.get('init_cls','?')}",
            f"inc{a.get('increment','?')}",
            f"rank{a.get('rank','?')}",
            f"{a.get('lora_type','lora')}",
            f"{a.get('model_name','model')}",
            f"{a.get('optim','opt')}",
            f"lr{lr_show}",
        ]

        # replay tags
        if a.get("enable_replay", False):
            tag += [
                "replay",
                f"rbs{a.get('replay_bs','?')}",
                f"rlam{a.get('replay_lambda','?')}",
                f"ipc{a.get('replay_ipc','?')}",
                f"rlnew{int(bool(a.get('replay_labels_are_new', False)))}",
            ]
        else:
            tag += ["noreplay"]

        # refine tags
        if a.get("replay_refine_stats", False):
            method = str(a.get("replay_refine_method", "stats")).lower()
            tag += [
                f"ref{method}",
                f"rstep{a.get('replay_refine_steps','?')}",
                f"rlr{a.get('replay_refine_lr','?')}",
            ]
            if method == "gm":
                tag += [f"gmh{a.get('replay_refine_gm_heads','?')}"]
        else:
            tag += ["noref"]

        return "_".join(tag)

    logfilename = os.path.join(logdir, _exp_tag(args))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info("Log file: %s", logfilename + '.log')

    _set_random(args)
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args.get('dataset'),
        args.get('shuffle', True),
        args.get('seed', 0),
        args.get('init_cls', 0),
        args.get('increment', 0),
        args
    )

    model = factory.get_model(args.get('model_name'), args)

    cnn_curve, cnn_curve_with_task, nme_curve, cnn_curve_task = {'top1': []}, {'top1': []}, {'top1': []}, {'top1': []
                                                                                                                 }
    for task_id in range(data_manager.nb_tasks):
        logging.info('All params: %s', count_parameters(model._network))

        time_start = time.time()
        model.incremental_train(data_manager)
        time_end = time.time()
        logging.info('Train Time: %.3f', (time_end - time_start))

        time_start = time.time()
        cnn_accy, cnn_accy_with_task, nme_accy, cnn_accy_task = model.eval_task()
        time_end = time.time()
        logging.info('Eval Time: %.3f', (time_end - time_start))

        # Ensure replay generation runs with autograd enabled even if eval_task used no_grad/inference_mode

        try:

            inf_ctx = torch.inference_mode(False)

        except Exception:

            inf_ctx = contextlib.nullcontext()

        with inf_ctx, torch.enable_grad():

            torch.set_grad_enabled(True)

            model.after_task()

        torch.set_grad_enabled(True)

        logging.info('CNN: %s', cnn_accy.get('grouped', cnn_accy))
        cnn_curve['top1'].append(cnn_accy.get('top1', 0.0))
        cnn_curve_with_task['top1'].append(cnn_accy_with_task.get('top1', 0.0))
        cnn_curve_task['top1'].append(cnn_accy_task)

        logging.info('CNN top1 curve: %s', cnn_curve['top1'])
        logging.info('CNN top1 with task curve: %s', cnn_curve_with_task['top1'])
        logging.info('CNN top1 task curve: %s', cnn_curve_task['top1'])

        # If acc_matrix exists (some methods compute it), report forgetting/backward transfer
        if task_id > 0 and hasattr(model, "acc_matrix") and model.acc_matrix is not None:
            try:
                diagonal = np.diag(model.acc_matrix)
                forgetting = np.mean((np.max(model.acc_matrix, axis=1) -
                                      model.acc_matrix[:, task_id])[:task_id])
                backward = np.mean((model.acc_matrix[:, task_id] - diagonal)[:task_id])

                result_str = "Forgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
                logging.info(result_str)
            except Exception as e:
                logging.info("Skip forgetting/backward calc due to: %s", str(e))

    if hasattr(model, "acc_matrix") and model.acc_matrix is not None:
        logging.info('Accuracy Matrix: \n %s', model.acc_matrix.T.round(2))

    logging.info('Average Accuracy: %.4f', float(np.mean(cnn_curve['top1'])) if len(cnn_curve['top1']) else 0.0)
    logging.info('Last Accuracy: %.4f', float(cnn_curve['top1'][-1]) if len(cnn_curve['top1']) else 0.0)


def _set_device(args):
    device_type = args.get('device', ['0'])
    gpus = []

    for dev in device_type:
        if dev == -1 or dev == '-1':
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(dev))
        gpus.append(device)

    args['device'] = gpus


def _set_random(args):
    seed = int(args.get('seed', 0))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    def log_group(title, keys):
        logging.info("===== %s =====", title)
        for k in keys:
            if k in args:
                logging.info("%s: %s", k, args[k])

    log_group("DATA", [
        "dataset", "data_path", "shuffle", "init_cls", "increment", "total_sessions",
        "memory_size", "memory_per_class", "fixed_memory"
    ])

    log_group("MODEL", [
        "model_name", "embd_dim", "num_heads", "rank", "scale", "margin", "lora_type"
    ])

    log_group("OPTIM", [
        "batch_size", "init_epoch", "epochs",
        "optim", "init_lr", "init_lr_decay", "init_weight_decay",
        "lrate", "lrate_decay", "weight_decay",
        "num_workers", "EPSILON"
    ])

    log_group("REPLAY", [
        "enable_replay", "replay_root", "replay_bs", "replay_lambda", "replay_ipc",
        "replay_labels_are_new"
    ])

    log_group("REPLAY_REFINE", [
        "replay_refine_stats", "replay_refine_method",
        "replay_refine_scope", "replay_refine_steps", "replay_refine_lr",
        "replay_refine_keep", "replay_refine_tv", "replay_refine_l2",
        "replay_refine_max_classes", "replay_refine_normalize",
        "replay_refine_gm_heads"
    ])

    # print the rest (so nothing gets silently ignored)
    logging.info("===== OTHERS (not listed above) =====")
    listed = set()
    for grp in [
        ["dataset", "data_path", "shuffle", "init_cls", "increment", "total_sessions",
         "memory_size", "memory_per_class", "fixed_memory"],
        ["model_name", "embd_dim", "num_heads", "rank", "scale", "margin", "lora_type"],
        ["batch_size", "init_epoch", "epochs", "optim", "init_lr", "init_lr_decay", "init_weight_decay",
         "lrate", "lrate_decay", "weight_decay", "num_workers", "EPSILON"],
        ["enable_replay", "replay_root", "replay_bs", "replay_lambda", "replay_ipc", "replay_labels_are_new"],
        ["replay_refine_stats", "replay_refine_method", "replay_refine_scope", "replay_refine_steps",
         "replay_refine_lr", "replay_refine_keep", "replay_refine_tv", "replay_refine_l2",
         "replay_refine_max_classes", "replay_refine_normalize", "replay_refine_gm_heads"],
    ]:
        listed.update(grp)

    for k in sorted(args.keys()):
        if k not in listed:
            logging.info("%s: %s", k, args[k])
