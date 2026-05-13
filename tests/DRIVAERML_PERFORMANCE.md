# DrivAerML W&B Performance Goldens

These online tests are not just smoke coverage. They pin the W&B interaction
patterns that `wandb-driven-dev` should hardcode instead of asking an agent to
write one-off query scripts.

Project: `milieu/drivaerml`

Stable filter: `created_at < 2026-05-08`

## Query Shapes

| Question | Hardcoded path | Latency budget | Golden |
| --- | --- | ---: | --- |
| Best ABUPT 20k-step run by `val/surface_rel_l2` | `fast_wandb_query.py top` | 5s | `srehoxzc`, `0.07429194545928113` |
| Finished runs with `train/global_step == 100000` | `fast_wandb_query.py count` | 2s | `52` |
| Compare top two ABUPT 20k-step runs at step `15305` | `fast_wandb_query.py compare-step` | 30s | `srehoxzc` beats `bo2blqjb` |
| Explain plot shape at step `15305` | `curve_analysis.py compare` | 30s | `srehoxzc` wins by train and validation value |
| Compare 20 long 200k-step curves at step `150000` | `curve_analysis.py compare` | 30s | `bf00cxqu` wins train and validation value |

## Commands

```bash
uv run --with wandb --with requests \
  python skills/wandb-driven-dev/scripts/fast_wandb_query.py \
  count milieu/drivaerml \
  --filter 'created_at<2026-05-08' \
  --filter 'summary_metrics.train/global_step=100000'
```

```bash
uv run --with wandb --with requests \
  python skills/wandb-driven-dev/scripts/fast_wandb_query.py \
  top milieu/drivaerml \
  --metric val/surface_rel_l2 \
  --filter 'created_at<2026-05-08' \
  --filter 'config.model.model_class=abupt' \
  --filter 'config.max_steps=20000' \
  --config max_steps,model \
  --summary train/global_step \
  --limit 2
```

```bash
uv run --with wandb --with requests \
  python skills/wandb-driven-dev/scripts/fast_wandb_query.py \
  compare-step milieu/drivaerml \
  --runs srehoxzc,bo2blqjb \
  --step 15305 \
  --step-key train/global_step \
  --metrics train/loss,val/surface_rel_l2,val/volume_rel_l2,val/u_rel_l2,val/loss
```

```bash
uv run --with wandb --with requests --with pandas \
  python skills/wandb-driven-dev/scripts/curve_analysis.py \
  compare \
  --runs srehoxzc,bo2blqjb \
  --metrics train/loss,val/surface_rel_l2,val/volume_rel_l2,val/u_rel_l2,val/loss \
  --steps 15305
```

## Expected Step Comparison

At target step `15305`, DrivAerML uses `train/global_step` from local config and
the latest logged row at or before the target, because train and validation
metrics log at different cadences. Generic projects should start with W&B
`_step`; setup pins semantic training steps in local config when present.

| Run | Train step | `train/loss` | Val step | `val/surface_rel_l2` | `val/volume_rel_l2` | `val/u_rel_l2` | `val/loss` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `srehoxzc` | 15300 | 0.04521029070019722 | 15000 | 0.07945780126933602 | 0.15117401261296323 | 0.13385340545687657 | 0.04099014331586659 |
| `bo2blqjb` | 15300 | 0.9314912557601929 | 15000 | 0.3133448169707625 | 0.5791143328250141 | 0.21636501578114703 | 0.9659046530723572 |

## Guardrail

Do not replace these paths with SDK iteration over `api.runs(...)` or broad
history scans. Exact counts use server-side lazy counts, top-k uses selected
GraphQL fields, and at-step comparison stops scanning once the selected step key
passes the requested target.

For curve analysis, the default parallel worker cap is 12. On the pinned
DrivAerML 20-run benchmark, 12 workers was faster than 4 or 20 workers while
avoiding the shared-`wandb.Api` thread-safety failure observed with one API
client reused across workers.

Curve slopes are noise-aware: the default analyzer uses a trailing rolling
median over 5 logged points, fits a line over the slope window, and emits
`noise_to_signal` plus `trend_confidence`. Raw point values remain in `value`;
use `smoothed_value` for interpreting noisy trajectories.

The analyzer also separates training stages. Early-stage checks focus on launch
health: enough points, stable direction, and bad-direction spikes. Progress
checks focus on local slope, slope shift versus the previous window, and recent
spike events.

Measured on 2026-05-08 against 200k-step DrivAerML jobs at target step
`150000`, using metrics `train/loss,val/surface_rel_l2`:

| Curves | Workers | Latency |
| ---: | ---: | ---: |
| 2 | 2 | 3.344s |
| 3 | 3 | 3.298s |
| 20 | 12 | 8.334s |
