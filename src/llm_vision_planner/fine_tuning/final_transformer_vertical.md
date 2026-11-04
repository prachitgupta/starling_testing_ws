# Final vertical Transformer — copy/paste guide

This trains the historical V1 baseline: 8 learned waypoint queries, hidden size
256, 3 fusion layers, 3 decoder layers, planner LR `1e-4`, text LR `2e-5`,
batch 32, accumulation 2, seed 42, and Euclidean loss only. The two dataset
paths and all outputs remain separate.

## 1. Laptop/GPU server: generate the shared raw vertical data

Run from the package root. `--resume` safely fills missing scene/preference
records; the final check must print `26400` before either CSV path is created.

```bash
set -euo pipefail
cd /home/prachit/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner
export VERTICAL_ROOT="$PWD/fine_tuning/datasets/calibrated_vertical_dss"
export OPENAI_API_KEY=YOUR_KEY
export TEACHER_MODEL=gpt-6-luna
mkdir -p "$VERTICAL_ROOT"

python3 fine_tuning/scripts/generate_hrrt_finetuning_dataset.py \
  --output "$VERTICAL_ROOT/hrrt_teacher_raw.jsonl" \
  --scenes 4000 --teacher openai --teacher-model "$TEACHER_MODEL" \
  --reasoning-effort medium --distillation-mode dss \
  --scene-layout calibrated_vertical --preferences-per-scene 6 \
  --multi-goal-layout-ratio 0.10 --goal-sample-resolution-m 0.10 \
  --minimum-forward-progress-m 2.0 --hrrt-iterations 500 \
  --max-total-nodes 5000 --max-nodes-per-key 180 --max-candidates 8 \
  --min-routes 2 --scene-attempts 16 --clearance-m 0.40 --resume

test "$(wc -l < "$VERTICAL_ROOT/hrrt_teacher_raw.jsonl")" -eq 26400
```

## 2A. Create CSVs without human audit

```bash
python3 fine_tuning/scripts/finalize_hrrt_sft_dataset.py \
  --raw "$VERTICAL_ROOT/hrrt_teacher_raw.jsonl" \
  --output-dir "$VERTICAL_ROOT/teacher_only_80_10_10" \
  --clearance-m 0.40 --distillation-mode dss --teacher-only \
  --validation-scene-ratio 0.10 --test-scene-ratio 0.10 --split-seed 42 \
  --reasoning-aux-ratio 0 --counterfactual-aux-ratio 0
```

## 2B. Create CSVs with human audit

```bash
python3 fine_tuning/scripts/build_hrrt_human_audit.py \
  --input "$VERTICAL_ROOT/hrrt_teacher_raw.jsonl" \
  --output "$VERTICAL_ROOT/hrrt_human_audit.jsonl" \
  --train-audit-ratio 0.10 --validation-scene-ratio 0.10 \
  --test-scene-ratio 0.10 --seed 42

python3 fine_tuning/scripts/review_hrrt_labels.py \
  --raw "$VERTICAL_ROOT/hrrt_teacher_raw.jsonl" \
  --audit "$VERTICAL_ROOT/hrrt_human_audit.jsonl" \
  --reviews "$VERTICAL_ROOT/hrrt_human_reviews.jsonl" \
  --host 127.0.0.1 --port 8090
```

After every assigned row is reviewed, stop the site with `Ctrl-C`, then run:

```bash
python3 fine_tuning/scripts/finalize_hrrt_sft_dataset.py \
  --raw "$VERTICAL_ROOT/hrrt_teacher_raw.jsonl" \
  --audit "$VERTICAL_ROOT/hrrt_human_audit.jsonl" \
  --reviews "$VERTICAL_ROOT/hrrt_human_reviews.jsonl" \
  --output-dir "$VERTICAL_ROOT/human_audited_80_10_10" \
  --clearance-m 0.40 --distillation-mode dss \
  --reasoning-aux-ratio 0 --counterfactual-aux-ratio 0
```

Check both variants:

```bash
for TAG in teacher_only_80_10_10 human_audited_80_10_10; do
  for SPLIT in train validation test; do
    test -s "$VERTICAL_ROOT/$TAG/hrrt_sft_${SPLIT}.csv"
  done
done
```

## 3. Laptop: transfer both CSV datasets to NCSA

```bash
export NCSA=pgupta12@dtai-login.delta.ncsa.illinois.edu
export REMOTE_PACKAGE=/projects/bhkj/pgupta12/starling_multiple_trajectory_idea/src/llm_vision_planner
ssh "$NCSA" "mkdir -p '$REMOTE_PACKAGE/fine_tuning/datasets/calibrated_vertical_dss/'{teacher_only_80_10_10,human_audited_80_10_10}"
for TAG in teacher_only_80_10_10 human_audited_80_10_10; do
  rsync -avP "$VERTICAL_ROOT/$TAG/" \
    "$NCSA:$REMOTE_PACKAGE/fine_tuning/datasets/calibrated_vertical_dss/$TAG/"
done
```

## 4. NCSA: pull code and submit both V1 jobs

From the laptop:

```bash
ssh "$NCSA"
```

Then paste this at the NCSA prompt:

```bash
set -euo pipefail
export PROJECT_ROOT=/projects/bhkj/$USER/starling_multiple_trajectory_idea
export PACKAGE_ROOT="$PROJECT_ROOT/src/llm_vision_planner"
cd "$PROJECT_ROOT"
git switch starling_multiple_trajectory_idea
git pull --ff-only origin starling_multiple_trajectory_idea
mkdir -p logs

for TAG in teacher_only_80_10_10 human_audited_80_10_10; do
  DATA="$PACKAGE_ROOT/fine_tuning/datasets/calibrated_vertical_dss/$TAG"
  RUN_NAME="final-transformer-vertical-$TAG"
  JOB_ID=$(sbatch --parsable --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",RUN_NAME="$RUN_NAME",CONFIG="$PACKAGE_ROOT/fine_tuning/configs/hrrt_waypoint_transformer.yaml",TRAINER_SCRIPT="$PACKAGE_ROOT/fine_tuning/scripts/train_hrrt_waypoint_transformer.py",TRAIN_CSV="$DATA/hrrt_sft_train.csv",VALIDATION_CSV="$DATA/hrrt_sft_validation.csv",TEST_CSV="$DATA/hrrt_sft_test.csv",MAX_EPOCHS=100,SAFETY_RADIUS_THRESHOLD_M=0.40,CHECKPOINT_SELECTION=euclidean "$PACKAGE_ROOT/fine_tuning/verification/scripts/run_transformer_verification.sbatch")
  printf '%s %s\n' "$TAG" "$JOB_ID" | tee -a "$PACKAGE_ROOT/fine_tuning/outputs/final-transformer-vertical-job-ids.txt"
done
```

Monitor on NCSA:

```bash
squeue -u "$USER" -o "%.18i %.2t %.10M %R"
tail -n 60 "$PROJECT_ROOT/logs/hrrt-transformer-verify-<JOB_ID>.out"
```

In the `ST` column, `PD` means pending and `R` means running; the last column
shows the pending reason or active compute node.

## 5. Laptop: refresh the dashboard at any completed epoch

Set `TAG` to either dataset. Before training starts, the two JSON files will not
exist. Once one epoch completes, repeat this block whenever you want an update.

```bash
export TAG=teacher_only_80_10_10
export NCSA=pgupta12@dtai-login.delta.ncsa.illinois.edu
export REMOTE_PACKAGE=/projects/bhkj/pgupta12/starling_multiple_trajectory_idea/src/llm_vision_planner
export REMOTE_RUN="$REMOTE_PACKAGE/fine_tuning/outputs/final-transformer-vertical-$TAG"
export LOCAL_RUN=/home/prachit/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/outputs/final-transformer-vertical-$TAG
mkdir -p "$LOCAL_RUN/model" "$LOCAL_RUN/dashboard"
rsync -avP --include='training_*.json' --exclude='*' "$NCSA:$REMOTE_RUN/model/" "$LOCAL_RUN/model/"
python3 /home/prachit/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/verification/scripts/plot_verification_results.py \
  --run-root "$LOCAL_RUN" --model-output "$LOCAL_RUN/model" \
  --output-dir "$LOCAL_RUN/dashboard" --training-only
python3 -m http.server 8770 --bind 127.0.0.1 --directory "$LOCAL_RUN/dashboard"
```

Open <http://127.0.0.1:8770/>. Stop the server with `Ctrl-C`, repeat the sync,
and restart it to import the newest epoch. After the job completes, replace the
training-only page with the full same-scene/preference verification dashboard:

```bash
rsync -avP "$NCSA:$REMOTE_RUN/plots/" "$LOCAL_RUN/dashboard/"
rsync -avP "$NCSA:$REMOTE_RUN/reports/" "$LOCAL_RUN/reports/"
```

## 6. Laptop: transfer the best checkpoint with a stable name

```bash
export LOCAL_FINAL="$LOCAL_RUN/final_transformer_vertical_${TAG}_best"
BEST_REMOTE=$(ssh "$NCSA" "python3 -c 'import json; print(json.load(open(\"$REMOTE_RUN/model/training_summary.json\"))[\"best_euclidean_checkpoint\"])'")
mkdir -p "$LOCAL_FINAL"
rsync -avP "$NCSA:$BEST_REMOTE/" "$LOCAL_FINAL/checkpoint/"
rsync -avP "$NCSA:$REMOTE_RUN/model/text_encoder/" "$LOCAL_FINAL/text_encoder/"
test -s "$LOCAL_FINAL/checkpoint/model.pt"
```

## 7. Laptop: run the calibration gateway on the latest outline CSV

The gateway retains its original mode selector:

- **Environment uncertainty:** HRRT uses ground-truth `E`; goal grounding and
  the Transformer use perceived `E-hat`.
- **Planner only:** goal grounding, HRRT, and Transformer all use ground-truth
  `E`; perceived geometry is hidden.

```bash
cd /home/prachit/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner
export OPENAI_API_KEY=YOUR_KEY
python3 fine_tuning/scripts/calibration_verifier_gateway.py \
  --dataset fine_tuning/datasets/calibration_residual_raw_person_outline.csv \
  --planner-provider transformer \
  --transformer-checkpoint "$LOCAL_FINAL/checkpoint" \
  --transformer-device auto --intent-provider openai \
  --host 127.0.0.1 --port 8091
```

Open <http://127.0.0.1:8091/>.
