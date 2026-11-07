# Final vertical V1 Transformer — evaluation guide

Selected bundle: `final-transformer-vertical-teacher_only_80_10_10`, teacher-only 80/10/10 split, checkpoint epoch 67. Training metadata records best Euclidean loss 0.2772914745977947. This is the model selected for current testing, not a claim that every competing model has been conclusively beaten or that its tube certificate passed.

## Local assets

Run from the package directory, `src/llm_vision_planner`:

```bash
LOCAL_FINAL="$PWD/fine_tuning/outputs/final-transformer-vertical-teacher_only_80_10_10/final_transformer_vertical_teacher_only_80_10_10_best"
test -s "$LOCAL_FINAL/checkpoint/model.pt"
test -s "$LOCAL_FINAL/text_encoder/model.safetensors"
sha256sum -c LOCAL_ASSETS.sha256
```

Both weight files remain available in the prepared local checkout and original research checkout; they are NOT published to GitHub. On a new machine, copy these exact files from your trusted local backup into these paths and verify their hashes. Do not substitute a different checkpoint. Tokenizer, configuration, encoder configuration, metrics and training reports are tracked. Original NCSA paths inside saved configuration are preserved as provenance; the gateway receives an explicit local checkpoint path.

Restore the exact training data only if needed (already unpacked in the prepared local checkout):

```bash
for archive in fine_tuning/datasets/calibrated_vertical_dss/hrrt_teacher_raw.jsonl.gz fine_tuning/datasets/calibrated_vertical_dss/teacher_only_80_10_10/*.csv.gz; do
  test -f "${archive%.gz}" || gzip -dk "$archive"
done
```

## Offline calibration gateway

Use the ROS Humble Python environment with the workspace dependencies installed
and source `/opt/ros/humble/setup.bash` first. The Transformer additionally needs
PyTorch, transformers, NumPy, SciPy and PyYAML. The checked local environment
already provides these; the gateway imports shared ROS modules even offline.

```bash
source /opt/ros/humble/setup.bash
read -rsp "OpenAI API key: " OPENAI_API_KEY
export OPENAI_API_KEY
python3 fine_tuning/scripts/calibration_verifier_gateway.py \
  --dataset fine_tuning/datasets/calibration_residual_raw_person_outline.csv \
  --planner-provider transformer \
  --transformer-checkpoint "$LOCAL_FINAL/checkpoint" \
  --transformer-device auto --intent-provider openai \
  --host 127.0.0.1 --port 8091
```

Open http://127.0.0.1:8091/. Environment uncertainty mode gives the Transformer and goal grounding perceived E-hat; HRRT uses ground-truth E. Planner-only mode gives all three ground-truth E. This is an offline inspection tool, not the live ROS mission controller.

The gateway does not score the entire dataset at startup, does not automatically make human preference selections, and its current per-scene tube output must not be advertised as a 95% or 80% calibration-distribution guarantee. Retry feedback is not encoded into the Transformer input. See the implementation report and saved numerical audit before interpreting its blue tube.

## Evidence

- Selected-model training dashboard: `outputs/final-transformer-vertical-teacher_only_80_10_10/dashboard/index.html`.
- Saved pre/postflight reports: the bundle's `reports/` directory (offline checks, despite their names).
- Numerical replay and HTMLs: `../output/gateway_radius_audit_20261104/`.

Training studies and alternative model variants remain on the unchanged research branches.
