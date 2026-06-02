# KML-ARIS Workflow

```mermaid
flowchart TD
    A["KML CEO: define round objective"] --> B["KML Producer: implement or run"]
    B --> C["Producer packet: patch, config, logs, metrics"]
    C --> D["KML Critic: integrity verification"]
    D --> E["Result-to-claim mapping"]
    E --> F["Claim audit against raw evidence"]
    F --> G{Verdict}
    G -->|pass| H["CEO: advance or freeze"]
    G -->|revise| I["CEO: create action items"]
    G -->|reject| J["CEO: rollback or rerun"]
    I --> B
    J --> B
    H --> K{IMA-10 target met?}
    K -->|no| A
    K -->|yes| L["Final report and GitHub update"]
```

## Round Contract

Every round must produce the following files or links:

```text
round_id:
objective:
commit:
config:
command:
gpu:
raw_log:
parsed_metrics:
critic_verdict:
ceo_decision:
```

## Evidence Command

```bash
python tools/parse_cil_results.py logs/ImageNet_A/10_tasks --out evidence/ima10_results.csv
```

## Stop Rule

The loop stops only when all conditions hold:

- Three seeds finished.
- `ALast > 64.14`.
- `AAvg > 71.45`.
- Critic verdict is `pass`.
- Claim ledger links every accepted claim to raw logs.

