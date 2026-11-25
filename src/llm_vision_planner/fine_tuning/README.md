# Final Transformer assets

Use [final_transformer_vertical.md](final_transformer_vertical.md) for the selected V1 checkpoint, local asset restoration and gateway launch.

Retained data: the matching teacher-only 80/10/10 split and raw teacher JSONL (gzip archives), plus the raw person-outline calibration captures. Synthetic inputs are in `../test/fixtures`. No earlier planner's controller scores are accepted on `main`.

`transformer_runtime.py` is the single inference implementation used by the live ROS node, batch postprocessor, and inspection gateway. The V1 training implementation remains because this runtime imports the saved architecture and scene encoder. HRRT, refinement, QP and recording/postprocessing helpers remain dependencies of the final workflow. Older Transformer variants, Llama/LoRA planner backends, and ablation tooling remain on research branches, not here.

The selected model dashboard is in `outputs/final-transformer-vertical-teacher_only_80_10_10/dashboard/index.html`. The corresponding reports and checkpoint metadata are in the same bundle. Local weights are intentionally ignored by Git.

## Transformer calibration commands

Run from `src/llm_vision_planner`. Restore the real weights described in
[the asset guide](final_transformer_vertical.md) first: the local zero-byte
weight files are placeholders and cannot run inference. OpenAI selects the
HRRT expert from L2; waypoint prediction uses only the local Transformer.

```bash
LOCAL_FINAL="$PWD/fine_tuning/outputs/final-transformer-vertical-teacher_only_80_10_10/final_transformer_vertical_teacher_only_80_10_10_best"
python3 fine_tuning/scripts/postprocess_residual_calibration.py \
  --raw-csv fine_tuning/datasets/calibration_residual_raw_person_outline.csv \
  --output-csv fine_tuning/datasets/calibration_residual_scored.csv \
  --all-captures --l1-text "Go to the goal" --l2-text "Take the shortest safe route" \
  --goal-x 3.5 --goal-y 0.0 --expert-selector openai \
  --planner-provider transformer --transformer-checkpoint "$LOCAL_FINAL/checkpoint" \
  --transformer-device auto --delta 0.20

python3 fine_tuning/scripts/calibration_verifier_gateway.py \
  --dataset fine_tuning/datasets/calibration_residual_raw_person_outline.csv \
  --planner-provider transformer --transformer-checkpoint "$LOCAL_FINAL/checkpoint" \
  --transformer-device auto --intent-provider openai --delta 0.20 \
  --scored-csv fine_tuning/datasets/calibration_residual_scored.csv \
  --host 127.0.0.1 --port 8091
```

These commands require the ROS/dependency environment and `OPENAI_API_KEY`
described in the asset guide. The postprocessor writes the output CSV; it does
not require that output to exist beforehand. L2 supplies the approved route
preference to both the expert selector and Transformer, matching the gateway.
Mock postprocessing is only for fixtures and requires `--placeholder`; its
scores must not be used as real calibration. Neither command supports vLLM,
Llama, or OpenAI as a waypoint planner. Residuals are computed numerically
from the expert and Transformer trajectories.

## Planner-only Case-A fixed tube

Generate the test-derived demonstration certificate (one seeded row per
layout; `shortest`/`balanced`; `beta=2`; no environment uncertainty):

```bash
cd /home/prachit/Desktop/starling_final_testing/src/llm_vision_planner
python3 fine_tuning/scripts/calibrate_planner_only.py \
  --device auto --delta 0.10 --beta 2.0 --seed 17 --overwrite
python3 - <<'PY'
import json
from pathlib import Path
p = Path("fine_tuning/datasets/planner_only_case_a/split_certificate.json")
c = json.loads(p.read_text())
print("n=", c["calibration_sample_count"], "rank=", c["rank"])
print("q=", c["calibrated_quantile"], "radius_m=", c["calibrated_radius_m"])
PY
```

The script records verifier-rejected plans in `abstentions.jsonl`; the fixed
tube covers only plans released by that same verifier. This certificate is not
an environment-uncertainty or hardware-tracking certificate.

The user-defined `--delta` sets `p=1-delta` for the empirical quantile over all
valid trajectory scores, irrespective of sample count. The gateway reads the
scored CSV; it never substitutes the currently displayed trajectory's score for
the calibration quantile. A header-only placeholder CSV is not calibration and
prints `no valid quantile found`. See the hardware guide for trajectory-level
ACP, its coverage scope, and independent publication logs.
