import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

p300 = ROOT / "models/regression_grid_shard_300.json"
p500 = ROOT / "models/regression_grid_shard_500.json"

d300 = json.loads(p300.read_text())
d500 = json.loads(p500.read_text())

all_rows = {r["params"]: r for r in d300["rows"]}
for r in d500["rows"]:
    all_rows[r["params"]] = r

sorted_rows = sorted(all_rows.values(), key=lambda r: r["rmse_cv"])

combined = {
    "feature_cols": d500["feature_cols"],
    "split": d500["split"],
    "rows": sorted_rows,
}

out_p = ROOT / "models/regression_grid_combined_checkpoint.json"
out_p.write_text(json.dumps(combined, indent=2))

print(f"Total combinations combined: {len(sorted_rows)}")
for i, r in enumerate(sorted_rows, 1):
    p = json.loads(r["params"])
    trees = p["n_estimators"]
    depth = p["max_depth"]
    leaf = p["min_samples_leaf"]
    cv_rmse = r["rmse_cv"]
    valid_rmse = r["rmse_valid"]
    cv_r2 = r["r2_cv"]
    print(f"#{i} | Trees={trees}, Depth={depth}, Leaf={leaf} | CV RMSE: {cv_rmse:.6f} | CV R2: {cv_r2:.6f} | Valid RMSE: {valid_rmse:.6f}")
