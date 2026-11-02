# Baseline eight-waypoint loss study — Hiroyasu GPU server

Use this guide for the original Transformer 1 architecture. The existing
[NCSA guide](README.md) remains available for the six-interior-waypoint study.
This runner uses the existing model, scene parser, verifier, refinement/QP,
checkpoint format, and dashboard. It needs no ROS launch, teacher API, or Slurm.

All new results go to `fine_tuning/outputs/baseline-eight-loss-001/`, with their
own `dashboard/index.html`, `logs/`, `runs/`, `reports/`, and `state.json`.
The NCSA `loss-ablation-001/` directory is not used. Initialization refuses an
existing nonempty directory; do not set the new study path to your NCSA copy.

## What will run

```text
Audit the three existing CSV splits and epoch-40 reference checkpoint
  → evaluate the reference on validation
  → tiny-set preflight with all-eight Euclidean loss
  → train fresh eight-output V1: E0, E1, E2, E3, seed 42
  → optionally E4 if both squared-error and corridor losses improve
  → repeat E0 and the leading auxiliary loss with seeds 43 and 44
  → freeze selection from validation → evaluate the selected model on test once
  → final dashboard and checkpoint
```

Each run learns all eight waypoints, including start and goal. There is no
endpoint concatenation/clamping, progress embedding, or scene-only Transformer.
All eight queries self-attend using the original decoder. Each run starts with
base MiniLM and newly initialized planner weights, using the same seed across
loss variants. The saved epoch-40 checkpoint is evaluated as a reference; its
trained weights do not initialize the fresh runs. This measures training with
different losses, rather than additional fine-tuning of epoch 40.

The configuration is copied from the reference `config.json`, including:

| Setting | Saved baseline value |
| --- | --- |
| Scene features / learned queries | 10 / 8 |
| Hidden size / attention heads | 256 / 8 |
| Fusion / decoder layers | 3 / 3 |
| Feed-forward size / dropout | 1024 / 0.1 |
| Planner / text learning rates | `1e-4` / `2e-5` |
| Batch size / gradient accumulation | 32 / 2 (nominal effective batch 64) |
| Validation batch / loader workers | 64 / 4 |
| AdamW weight decay / gradient clipping | 0.01 / 1.0 |
| Epochs / cosine warmup fraction | 100 / 0.05 |
| First seed | 42 |

The original accumulation divisor is preserved, including the final partial
group. The data, model and validation settings also come from that configuration;
paths are remapped to the server. There is no silent use of Transformer 2's
learning rate or six-point training loss.

For (d_j=\|\hat w_j-w_j\|_2\), define
\(L_0=\frac18\sum_{j=1}^{8}d_j\) and
\(S=\frac18\sum_{j=1}^{8}d_j^2/(1\,\mathrm m)\).

| Variant | Training objective |
| --- | --- |
| E0 | `L0`: mean Euclidean error across all 8 points |
| E1 | `L0 + 0.25 S` |
| E2 | `L0 + C` |
| E3 | `L0 + 5 C` |
| E4, conditional | `L0 + 0.25 S + selected corridor weight × C` |

`C` reuses `corridor_side_loss` from Transformer 2: mean teacher-side hinge with
margin 0.20 m and soft waypoint selection temperature 0.35 m. It supplies a loss
only; the model stays V1. Clearance-bin loss is disabled, but bin agreement is
reported. Squared error covers all eight points in this study.

Training runs the common 100-epoch budget with validation every epoch. Checkpoint
and loss selection use **validation mean error over the six interior points**,
as in the existing dashboard, so easy endpoints cannot hide interior mistakes.
All-eight and endpoint errors are also shown. The leading auxiliary loss needs
mean paired improvement ≥0.02 m and improvement on a majority of the three
seeds. Otherwise E0 wins. The representative checkpoint is the median-validation
seed of the selected loss. Test metrics do not influence these choices.

The original training script evaluated refinement/verifier/QP every epoch and
had a safety-certificate stopping condition; this study evaluates that pipeline
on each run's selected checkpoint and uses the fixed epoch budget. It copies
training hyperparameters, not the original certificate stopping policy. Current
production code and GPU/software differences also mean this is not a promise
of bit-identical reproduction of the old run. The old checkpoint records dataset
paths but no original CSV hashes; all inputs for the new study are hashed at init.

## 1. Laptop: prepare the server checkout

These commands use the server username/path from the existing Hiroyasu setup.
If the hostname requires your institute network/VPN, connect before starting.
Run this block on the laptop:

```bash
set -euo pipefail
export STUDY_REMOTE=prachit2@hiroyasu-serv-01
export LOCAL_PACKAGE=/home/prachit/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner
export REMOTE_PACKAGE=/home/prachit2/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner
cd /home/prachit/Desktop/starling_multiple_trajectory_idea
git switch starling_multiple_trajectory_idea
git pull --ff-only origin starling_multiple_trajectory_idea

ssh "$STUDY_REMOTE" 'bash -s' <<'REMOTE'
set -euo pipefail
PROJECT_ROOT=/home/prachit2/Desktop/starling_multiple_trajectory_idea
if [ ! -d "$PROJECT_ROOT/.git" ]; then
  mkdir -p /home/prachit2/Desktop
  git clone --branch starling_multiple_trajectory_idea https://github.com/prachitgupta/starling_testing_ws.git "$PROJECT_ROOT"
fi
cd "$PROJECT_ROOT"
git switch starling_multiple_trajectory_idea
git pull --ff-only origin starling_multiple_trajectory_idea
mkdir -p src/llm_vision_planner/fine_tuning/datasets/dss_teacher_only_80_10_10
mkdir -p src/llm_vision_planner/fine_tuning/outputs/overfitted/hrrt_waypoint_transformer
REMOTE
```

Resolve Git conflicts normally; do not reset away server work. Once initialized,
keep this checkout and Python environment fixed until the study finishes.

## 2. Laptop: upload data and the reference checkpoint

Run in the same laptop terminal. Git does not contain these artifacts:

```bash
export LOCAL_BASELINE="$LOCAL_PACKAGE/fine_tuning/outputs/overfitted/hrrt_waypoint_transformer"
for SPLIT in train validation test; do
  test -s "$LOCAL_PACKAGE/fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_${SPLIT}.csv"
  rsync -avP "$LOCAL_PACKAGE/fine_tuning/datasets/dss_teacher_only_80_10_10/hrrt_sft_${SPLIT}.csv" \
    "$STUDY_REMOTE:$REMOTE_PACKAGE/fine_tuning/datasets/dss_teacher_only_80_10_10/"
done
test -s "$LOCAL_BASELINE/checkpoint-epoch-040/model.pt"
test -s "$LOCAL_BASELINE/checkpoint-epoch-040/config.json"
test -d "$LOCAL_BASELINE/checkpoint-epoch-040/tokenizer"
test -d "$LOCAL_BASELINE/text_encoder"
rsync -avP "$LOCAL_BASELINE/checkpoint-epoch-040" "$LOCAL_BASELINE/text_encoder" \
  "$STUDY_REMOTE:$REMOTE_PACKAGE/fine_tuning/outputs/overfitted/hrrt_waypoint_transformer/"
ssh "$STUDY_REMOTE"
```

## 3. GPU server: prepare Python and initialize once

Paste at the **Hiroyasu server prompt**, after SSH:

```bash
set -euo pipefail
export PROJECT_ROOT=/home/prachit2/Desktop/starling_multiple_trajectory_idea
export PACKAGE_ROOT="$PROJECT_ROOT/src/llm_vision_planner"
export TRAIN_ENV_DIR="$PROJECT_ROOT/.venv-baseline-loss-study"
export STUDY_ROOT="$PACKAGE_ROOT/fine_tuning/outputs/baseline-eight-loss-001"
cd "$PROJECT_ROOT"
nvidia-smi
# Set this to the GPU assigned to you on the shared server.
export CUDA_VISIBLE_DEVICES=0
if [ ! -x "$TRAIN_ENV_DIR/bin/python3" ]; then
  /usr/bin/python3 -m venv "$TRAIN_ENV_DIR"
fi
"$TRAIN_ENV_DIR/bin/python3" -m pip install torch transformers numpy scipy pyyaml sentencepiece matplotlib
"$TRAIN_ENV_DIR/bin/python3" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: check GPU visibility, driver and PyTorch installation"; print(torch.__version__, torch.cuda.get_device_name(0))'

"$TRAIN_ENV_DIR/bin/python3" "$PACKAGE_ROOT/fine_tuning/verification/scripts/run_baseline_loss_ablation.py" init \
  --study-root "$STUDY_ROOT" \
  --dataset-root "$PACKAGE_ROOT/fine_tuning/datasets/dss_teacher_only_80_10_10" \
  --baseline-checkpoint "$PACKAGE_ROOT/fine_tuning/outputs/overfitted/hrrt_waypoint_transformer/checkpoint-epoch-040" \
  --device cuda --minimum-improvement-m 0.02
"$TRAIN_ENV_DIR/bin/python3" -m json.tool "$STUDY_ROOT/manifest.json"
```

Using the explicit interpreter avoids a stale activated `.venv-hrrt-data`
Python path. Initialization audits scene separation, duplicate IDs, coordinates,
input shapes and artifacts. It creates the manifest and dashboard but does not
train. Inspect `config.training` in the manifest to confirm the table above.
Missing real weights or data is an error; an empty placeholder is not usable.

## 4. GPU server: start all runs with one command

In that server terminal:

```bash
mkdir -p "$STUDY_ROOT/logs"
nohup "$TRAIN_ENV_DIR/bin/python3" -u \
  "$PACKAGE_ROOT/fine_tuning/verification/scripts/run_baseline_loss_ablation.py" run \
  --study-root "$STUDY_ROOT" >> "$STUDY_ROOT/logs/study.log" 2>&1 < /dev/null &
STUDY_PID=$!
printf '%s\n' "$STUDY_PID" > "$STUDY_ROOT/worker.pid"
printf 'Worker PID: %s\n' "$STUDY_PID"
tail -n 40 "$STUDY_ROOT/logs/study.log"
```

You can close SSH. One Python process runs experiments sequentially; it does not
submit jobs. There are normally eight fresh 100-epoch runs, or nine when E4 is
enabled, plus reference/preflight/final evaluation. A lock prevents two workers
writing to the same study. Check GPU availability before starting a second study.

## 5. GPU server: monitor, pause and resume

For a new SSH session, restore these variables first:

```bash
export PROJECT_ROOT=/home/prachit2/Desktop/starling_multiple_trajectory_idea
export PACKAGE_ROOT="$PROJECT_ROOT/src/llm_vision_planner"
export TRAIN_ENV_DIR="$PROJECT_ROOT/.venv-baseline-loss-study"
export STUDY_ROOT="$PACKAGE_ROOT/fine_tuning/outputs/baseline-eight-loss-001"
export CUDA_VISIBLE_DEVICES=0
"$TRAIN_ENV_DIR/bin/python3" -m json.tool "$STUDY_ROOT/state.json"
ps -p "$(tail -n 1 "$STUDY_ROOT/worker.pid")" -o pid,etime,args || true
tail -n 60 "$STUDY_ROOT/logs/study.log"
nvidia-smi
```

`state.json` shows the active or next run. `runs/<run>/history.json` and the log
contain per-epoch metrics; the dashboard is saved at initialization, each epoch,
and each completed evaluation. Refresh/download it to see the newest snapshot;
it is not a streaming connection. Same-scene preference examples appear once
that checkpoint's validation evaluation finishes. During an epoch only the last
completed epoch is visible.

To pause, first check that the PID above belongs to this study, then:

```bash
kill -USR1 "$(tail -n 1 "$STUDY_ROOT/worker.pid")"
```

The worker finishes the current epoch or evaluation and saves before stopping.
Wait for `paused` and for the process to exit. To resume, run the **step 4 block**
again; do not rerun `init`. It restores optimizer, cosine schedule, scaler, RNG,
history, best weights and model state. A crash midway through an epoch repeats
that epoch. After a failure, inspect the log; if code/data/settings must change,
initialize a different study directory. Do not edit frozen manifests or hashes.

## 6. Laptop: download the dashboard and open it

Run on the laptop, outside SSH. These are the existing transfer commands with
the Hiroyasu server and separate output paths:

```bash
export STUDY_REMOTE=prachit2@hiroyasu-serv-01
export REMOTE_STUDY=/home/prachit2/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/outputs/baseline-eight-loss-001
export LOCAL_STUDY=/home/prachit/Desktop/starling_multiple_trajectory_idea/src/llm_vision_planner/fine_tuning/outputs/baseline-eight-loss-001
mkdir -p "$LOCAL_STUDY/dashboard"
rsync -avP "$STUDY_REMOTE:$REMOTE_STUDY/dashboard/" "$LOCAL_STUDY/dashboard/"
rsync -avP "$STUDY_REMOTE:$REMOTE_STUDY/state.json" "$LOCAL_STUDY/"
xdg-open "$LOCAL_STUDY/dashboard/index.html"
```

The eight-waypoint loss table is built in. **Do not run the old
`add_local_dashboard_model_table.py` on this dashboard**; its descriptions assume
the six-output NCSA study. The leaderboard retains interior error/P80,
all-eight/endpoint error, corridor/bin agreement, pipeline pass rate, radius,
parameter counts, best epoch and runtime. It also retains the Model A/B scene
selector with all recorded preferences, raw teacher/predicted waypoints,
component diagnostics, training curves, seed comparison and final test.
The shared diagnostic gradient probe and `squared_tail_loss_m` evaluation metric
use the six interior points for comparison with the earlier study. Training E0
and E1 still use all eight points as defined above; their actual terms are logged
in `train_terms`. Preflight uses all-eight Euclidean loss.

For HTTP viewing, run this in a laptop terminal and keep it open:

```bash
/usr/bin/python3 -m http.server 8767 --bind 127.0.0.1 --directory "$LOCAL_STUDY/dashboard"
```

Open **http://127.0.0.1:8767/**. Repeat the two `rsync` lines and refresh the page
for updates. Port 8767 keeps this study separate from the existing local viewer.

## 7. Laptop: collect final reports and selected checkpoint

Once the downloaded state says `complete`, in the laptop terminal with step 6's
variables:

```bash
rsync -avP "$STUDY_REMOTE:$REMOTE_STUDY/manifest.json" \
  "$STUDY_REMOTE:$REMOTE_STUDY/audit.json" \
  "$STUDY_REMOTE:$REMOTE_STUDY/selection.json" \
  "$STUDY_REMOTE:$REMOTE_STUDY/final_test.json" "$LOCAL_STUDY/"
rsync -avP "$STUDY_REMOTE:$REMOTE_STUDY/reports/" "$LOCAL_STUDY/reports/"
WINNER=$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_run"])' "$LOCAL_STUDY/selection.json")
mkdir -p "$LOCAL_STUDY/runs/$WINNER"
rsync -avP "$STUDY_REMOTE:$REMOTE_STUDY/runs/$WINNER/best" \
  "$STUDY_REMOTE:$REMOTE_STUDY/runs/$WINNER/text_encoder" \
  "$STUDY_REMOTE:$REMOTE_STUDY/runs/$WINNER/tokenizer" \
  "$STUDY_REMOTE:$REMOTE_STUDY/runs/$WINNER/history.json" \
  "$LOCAL_STUDY/runs/$WINNER/"
printf 'Selected checkpoint: %s/runs/%s/best\n' "$LOCAL_STUDY" "$WINNER"
```

`best/model.pt` and `best/config.json` retain the V1 checkpoint format; the
sibling base encoder and tokenizer allow offline reconstruction. Unavailable
safety radius is not zero, and verification pass rates are offline results.
The existing test split has been inspected before; it is not a new prospective
holdout. No new training scores exist until you actually run this workflow.

## Development checks

From the package directory in an environment with the training dependencies:

```bash
OMP_NUM_THREADS=1 python3 -m unittest discover -s fine_tuning/verification/scripts -p 'test_baseline_loss_ablation.py' -v
SMOKE_ROOT=$(mktemp -d /tmp/baseline-eight-smoke.XXXXXX)
python3 fine_tuning/verification/scripts/run_baseline_loss_ablation.py init --study-root "$SMOKE_ROOT" --smoke --device cpu
OMP_NUM_THREADS=2 python3 fine_tuning/verification/scripts/run_baseline_loss_ablation.py run --study-root "$SMOKE_ROOT"
xdg-open "$SMOKE_ROOT/dashboard/index.html"
```

Smoke mode uses real model/data code with eight rows and one epoch, marks its
reports, and omits full pipeline evaluation. It writes several GB of temporary
checkpoint artifacts; its scores cannot support scientific comparisons.
