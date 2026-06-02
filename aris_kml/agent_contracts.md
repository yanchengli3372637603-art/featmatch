# Three KML Agent Contracts

## KML CEO

Role: ARIS-style coordinator and evidence owner.

Responsibilities:

- Maintain `docs/kml_auto_research_plan.md`.
- Keep one claim ledger per experiment cycle.
- Assign tasks to Producer and review Critic verdicts.
- Decide whether a result is strong enough to become a paper claim.
- Stop the loop when IMA-10 exceeds MACIL paper baseline with audited evidence.

Inputs:

- User objective.
- Repository state.
- Producer artifacts.
- Critic verdicts.

Outputs:

- Round plan.
- Accepted/rejected action items.
- Updated claim ledger.
- Final research report.

Decision states:

- `advance`: evidence is sufficient for next phase.
- `revise`: implementation or experiment needs correction.
- `rerun`: result is inconclusive or unstable.
- `freeze`: artifact is accepted as a fixed baseline.

## KML Producer

Role: implementation and experiment executor.

Responsibilities:

- Implement only scoped changes requested by CEO.
- Produce configs, scripts, logs, tables, and figures.
- Run sanity checks before full 8xH200 deployment.
- Never claim improvement without pointing to raw logs.
- Preserve reproducibility: commit hash, seed, config, CUDA device, data path.

Required artifact packet:

```text
producer_packet/
  patch_summary.md
  configs.txt
  commands.sh
  logs/
  metrics.csv
  failure_notes.md
```

## KML Critic

Role: adversarial reviewer and evidence auditor.

Responsibilities:

- Review code correctness and experimental validity.
- Check result-to-claim mapping.
- Reject claims with missing seeds, missing baseline, changed data split, or unverifiable logs.
- Request ablations when a proposed innovation has no isolated evidence.
- Verify that IMA-10 comparison uses the same metric definitions as MACIL.

Verdict schema:

```yaml
verdict: pass | revise | reject
score: 0-10
blocking_issues:
  - issue:
    evidence:
required_actions:
  - action:
accepted_claims:
  - claim:
    evidence:
```

