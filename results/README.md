# Experiment exports (for agents & reports)

Save outputs here after running `graph_research_presentation.ipynb` so agents and collaborators can use **files** instead of parsing notebook JSON.

## How to generate

Run the **“Export results”** code cell at the end of the presentation notebook (after `grid_run()`, multi-seed `rows`, and shortcut cells if you use them).

Or paste this in a notebook cell:

```python
from pathlib import Path
import json
import pandas as pd

OUT = Path("results")
OUT.mkdir(parents=True, exist_ok=True)

if "df" in globals() and df is not None and len(df):
    df.to_csv(OUT / "grid_run.csv", index=False)
if "rows" in globals() and rows:
    pd.DataFrame(rows).to_csv(OUT / "multiseed_k3_summary.csv", index=False)
if "shortcut_baselines" in globals() and shortcut_baselines:
    with open(OUT / "shortcut_baselines.json", "w") as f:
        json.dump({str(k): v for k, v in shortcut_baselines.items()}, f, indent=2)
print("Wrote under results/ — see column descriptions below.")
```

Commit the CSV/JSON you want versioned (optional: add large files to `.gitignore` if preferred).

## Files

| File | Produced by | Contents |
|------|-------------|----------|
| `grid_run.csv` | `df = grid_run(...)` | Hyperparameter grid: per config, per method — accuracy, allowed %, etc. |
| `multiseed_k3_summary.csv` | Multi-seed loop (`rows`) | Aggregated runs over `SEEDS` × `DISTRACTORS` for k=3 |
| `shortcut_baselines.json` | `measure_shortcut_baselines(...)` | Heuristic baseline accuracies per distractor level |

### `grid_run.csv` (typical columns)

Columns depend on `grid_run()` implementation; commonly include:

- `k`, `distractors`, `method` — setup
- `final_full` — validation (or full) accuracy at end of training
- `allowed_pct` — fraction of attention pairs allowed (graph/window vs dense)
- Other hyperparameters if appended in `rows.append({...})`

Inspect with `pd.read_csv("results/grid_run.csv").columns`.

### `multiseed_k3_summary.csv`

Typically includes per-row aggregates such as `distractors`, `dense_acc_mean`, `window_acc_mean`, `graph_acc_mean`, and corresponding `*_allowed_mean` fields (see notebook cell that builds `rows`).

### `shortcut_baselines.json`

Dict keyed by distractor count (as string), values like `{"random", "dst_only", "lowest_deg"}`.

## Using with Claude Code / Cursor / other agents

Point the model at **`CLAUDE.md`** (and **`AGENTS.md`** if needed) + this folder:

> “Update `plans/08-final-report-narrative.md` using only the numbers in `results/grid_run.csv` and `results/multiseed_k3_summary.csv`.”
