# Record Calibration Data 2.0 — person-only frozen split CP

Use this workflow for the initial residual experiments. ACP remains available separately. The vehicle only takes off and holds its initial position during collection; it does not fly the planned task. A trained checkpoint, a completed live survey and new measured data are required. This guide does not supply empirical results or certify flight safety.

The experiment fixes one person's measured dimensions and heading, camera/hover pose, camera calibration, perception settings, start and goal. The **HRRT hardware transformer** is trained on exactly the Case-A `shortest` and `balanced` preferences used by the main Transformer dataset generator. The person-location domain is a finite set of tested points, separate from the vehicle's navigation bounds. Synthetic Dtrain/Dval and measured Dcal/Dtest share **only the location sampling law**; synthetic geometry does not reproduce sensor errors. Perfect tracking is a modeling assumption, not evidence of hardware accuracy or exchangeability. Fixed settings and randomized interleaved trials reduce known distribution differences; drift, temporal dependence and unmodeled effects can still invalidate exchangeability.

## 1. Setup and preflight (no vehicle motion)

These absolute paths work from either the package or workspace directory. In each new terminal, set:

```bash
export PERSON_WS=/home/prachit/Desktop/starling_final_testing
export PERSON_PKG="$PERSON_WS/src/llm_vision_planner"
source /opt/ros/humble/setup.bash
cd "$PERSON_WS"
colcon build --packages-select llm_vision_planner --symlink-install
source "$PERSON_WS/install/setup.bash"
ros2 pkg prefix llm_vision_planner
ros2 run llm_vision_planner person_experiment.py --help
export PERSON_PARAMS="$PERSON_PKG/config/llm_vision_planner.yaml"
export STARLING_IP=192.168.0.28
ping -c 3 "$STARLING_IP"
```

The printed installation must be the final-testing workspace, not an older `starling_testing_ws` overlay. If wrong, open a fresh shell, source only ROS and this workspace, and check again. Never guess a missing YAML path. Saved November 7 logs referenced an older installation and a missing YAML; that is evidence of a possible setup issue, not a diagnosis of every later run.

Finish camera calibration, VOXL camera/TFLite/ToF bridges, PX4 bridge and the Vicon bridge using the hardware setup sections of [the original guide](Record%20Calibration%20Data.md). Review the source YAML before freezing it: camera intrinsics/extrinsics and frame conventions must match this vehicle, `camera_calibration_valid` must be truthful, and depth must be measured. The current outline configuration can fall back to measured shape fitting; it is not a guarantee that EfficientSAM weights are present. Keep the estimator configuration and installed assets unchanged throughout a version.

On the ground station, use the established DDS settings (the vehicle must use the same domain):

```bash
source "$(ros2 pkg prefix llm_vision_planner)/lib/llm_vision_planner/ros_wifi_dds.sh" enable auto 42
ros2 daemon start
export VICON_COMPUTER_IP=192.168.0.130
ros2 run vicon_bridge vicon_bridge --ros-args \
  -p host_name:="${VICON_COMPUTER_IP}:801" \
  -p stream_mode:="ServerPush" \
  -p update_rate_hz:=125.0 \
  -p expected_rate_hz:=100.0 \
  -p publish_specific_segment:=false \
  -p world_frame_id:="vicon_world" \
  -p tf_namespace:="vicon" \
  -r /vicon/Starling2/Starling2/pose:=/mavros/vision_pose/pose
```

That single bridge publishes all Vicon subjects, including the vehicle and person; do not start per-object bridges beside it. Leave it running. In another identically sourced/configured terminal, verify:

```bash
ros2 topic list
ros2 topic echo /fmu/out/vehicle_odometry --once
ros2 topic echo /vicon/Starling2/Starling2 --once
ros2 topic echo /vicon/person/person --once
ros2 topic hz /tflite
# Stop the rate check with Ctrl+C, then check:
ros2 topic hz /tof_pc
```

Stop each rate check before proceeding. Topics above are the current YAML defaults; if your bridge names differ, edit the YAML before creating a survey. Set the recorder's Vicon time offset in that YAML if clocks demonstrably require it; do not loosen tolerances to hide a clock mismatch. Guided mode reads frozen YAML settings, not legacy launch overrides. Verify Vicon person dimensions are body-local X width/Y depth in metres and its heading is expressed in aligned NED, not untransformed Vicon coordinates.

Select a new version name and supply measured values interactively (no placeholder geometry):

```bash
read -rp "New experiment version (letters, digits, _ or -): " PERSON_VERSION
[[ "$PERSON_VERSION" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid version name"; exit 1; }
export PERSON_EXP="$PERSON_PKG/fine_tuning/datasets/person_split_cp/$PERSON_VERSION"
read -rp "Survey bounds XMIN XMAX YMIN YMAX, aligned NED metres: " -a PERSON_BOUNDS
read -rp "Fixed hover X Y Z_NED (negative Z is up): " -a PERSON_HOVER
read -rp "Measured person body-local X width, metres: " PERSON_WIDTH
read -rp "Measured person body-local Y depth, metres: " PERSON_DEPTH
ros2 run llm_vision_planner person_experiment.py init-survey \
  --source-root "$PERSON_PKG" --output-dir "$PERSON_EXP" --params-file "$PERSON_PARAMS" \
  --bounds "${PERSON_BOUNDS[@]}" --hover "${PERSON_HOVER[@]}" \
  --person-width "$PERSON_WIDTH" --person-depth "$PERSON_DEPTH" \
  --person-topic /vicon/person/person \
  --spacing 0.50 --visits 3 --checks 10 --interval 1.0 --seed 17 \
  --vehicle-position-tolerance 0.15 \
  --person-position-tolerance 0.25 \
  --stability-position-tolerance 0.03 \
  --yaw-tolerance-deg 10.0
```

The person ground truth is translation-only: Vicon supplies the centre X/Y, while the measured width/depth form a fixed **0 degrees aligned NED** footprint. Person marker rotation is ignored and is not an acceptance gate. The camera heading is fixed automatically to Starling's measured PX4/NED takeoff heading, **84.426 degrees**, so it matches the vehicle's stable camera orientation. The proof-of-concept command uses a 0.50 m grid, 0.15 m vehicle-hover tolerance, 0.25 m person-target tolerance, 0.03 m person stability tolerance and 10 degrees of vehicle yaw tolerance. The initializer rejects missing configuration, invalid dimensions and existing version directories. `survey.json` freezes the grid, tolerance values, camera/perception configuration, output path and sampling seed. The independent Vicon/PX4 frame-alignment gate remains 0.15 m translation error and 5 degrees rotation error; that rotation is required only to transform Vicon positions into NED. Use physically reachable/safe arena bounds; a large grid takes substantial time. Read the printed scheduled count before flight.

## 2. Hover and survey the restricted workspace

Only an authorized hardware operator should run this step, with clear space, Vicon tracking and working land/kill controls. No other takeoff/Offboard publisher may run. Place the vehicle at the frozen hover X/Y and heading before starting: the existing executor holds its initial X/Y/heading and climbs to the frozen Z. It does not reposition to a requested X/Y.

```bash
ros2 launch llm_vision_planner vision_error_calibration.launch.py \
  params_file:="$PERSON_PARAMS" collection_mode:=survey \
  experiment_manifest:="$PERSON_EXP/survey.json"
```

The launch prints its resolved installation, YAML, absolute destination and scheduled count before starting nodes. Frame alignment still precedes arming/takeoff. Typical output:

```text
Waiting for synchronized vehicle pose.
Hover ready. Move to target 12/…: x=…, y=… (NED).
Hold still. Scheduled check 12/….
Survey check 12/… passed.
```

A minimal live survey window shows the complete grid in grey, the current requested point as an orange star and the Vicon person pose as a blue X. Once all scheduled checks for a point pass, it remains latched green. A point is removed immediately after its first recorded rejection (although its remaining frozen checks are still performed). The title repeats the current positioning/hold/result guidance. The same view is saved at `$PERSON_EXP/survey.live.png`; on a headless terminal that PNG is refreshed instead of opening a window. Plotting is operator guidance only and does not alter, retry or filter survey evidence. Set `survey_plot_enabled: False` in the YAML only when deliberately running without this display, before freezing a new survey configuration.

Follow the locations/headings exactly; each visit has ten checks at least one second apart. Leave the previous location by more than 0.15 m between visits, including successive visits to the same point. Do not move the hovering aircraft by hand. Missing infrastructure pauses visibly. Scheduled observations require fresh, distinct detector frames, measured depth, exactly one non-held person, sufficient confidence, valid matched geometry and synchronized stationary Vicon. A failed check is retained and excludes that location; it is not replaced by a successful retry. All other scheduled checks still run. Additional visible obstacles are rejected, not silently discarded.

At completion, land **before** stopping the launch. In another sourced terminal:

```bash
ros2 topic pub --once /llm_vision/executor_command std_msgs/msg/String \
  "{data: '{\"command\":\"LAND\",\"reason\":\"person survey or collection complete\"}'}"
ros2 topic echo /llm_vision/mission_state
```

Wait for mission `COMPLETE`, physically verify landed/disarmed, then stop the launch. The survey's own `COMPLETE` means observations finished, not that the aircraft landed. Do not use Ctrl+C as a landing command. If normal landing is unavailable, use the established hardware emergency procedure.

## 3. Freeze the domain and sampling plan

```bash
ros2 run llm_vision_planner person_experiment.py freeze-domain --survey "$PERSON_EXP/survey.json"
```

Inspect `survey.map.svg`, `survey.map.csv`, `survey.report.json` and `survey.events.jsonl`. `domain.json` retains **every tested location with zero observed failures** including disconnected sets. It does not fill gaps or claim future 100% detection. Survey observations never enter any dataset split.

Before any measured dataset capture, explicitly choose split sizes, delta, seed, fixed task and expert settings. Dtrain and Dval counts are **person-layout counts**; export creates two rows per layout, one `shortest` and one `balanced`, from the same HRRT candidate family. Dcal and Dtest counts remain measured-trial counts, with one of those two preferences frozen per trial to avoid treating dependent preference variants as independent conformal samples. Freeze these choices before inspecting calibration scores. At delta=0.10 at least 9 Dcal scenes are necessary for a finite quantile; 9 is a mathematical minimum, not a recommended experimental sample size. For delta=0.05 the minimum is 19. Choose larger counts based on the intended study.

```bash
read -rp "Dtrain Dval Dcal Dtest counts: " PERSON_NTRAIN PERSON_NVAL PERSON_NCAL PERSON_NTEST
read -rp "Preselected miscoverage delta, strictly between 0 and 1: " PERSON_DELTA
read -rp "Sampling seed: " PERSON_SEED
read -rp "Fixed task goal X Y, NED metres: " -a PERSON_GOAL
read -rp "Vehicle navigation bounds XMIN XMAX YMIN YMAX: " -a PERSON_NAV
read -rp "Fixed L1 task text: " PERSON_L1
ros2 run llm_vision_planner person_experiment.py prepare \
  --domain "$PERSON_EXP/domain.json" \
  --n-train "$PERSON_NTRAIN" --n-val "$PERSON_NVAL" --n-cal "$PERSON_NCAL" --n-test "$PERSON_NTEST" \
  --delta "$PERSON_DELTA" --seed "$PERSON_SEED" \
  --goal "${PERSON_GOAL[@]}" --workspace "${PERSON_NAV[@]}" \
  --l1 "$PERSON_L1" --clearance 0.40 \
  --expert-selector heuristic --hrrt-iterations 500
```

The exporter imports the canonical prompt variants directly from `generate_hrrt_finetuning_dataset.py`; no separate L2 text is entered for this proof of concept. The example explicitly selects the same deterministic teacher logic used by that generator: minimum path length for `shortest`, and normalized path-length/clearance compromise for `balanced`. For an OpenAI-selected HRRT expert, preselect `--expert-selector openai --expert-model` with your intended model and supply `OPENAI_API_KEY`; API availability and model drift are additional dependencies. Do not change selector after seeing residuals. An infeasible fixed task or planner failure is a failed experiment, not permission to drop difficult scenes.

`experiment.json` assigns unique scene/trial IDs before observing data, uniformly samples locations with replacement independently for each split, and randomly interleaves Dcal/Dtest assignments. Independently drawn trials may revisit the same location. The fixed start and fixed goal are identical across layouts; only person position and the two language preferences are ablated. Both training rows derived from one layout stay in the same split. Navigation bounds are not replaced by the person domain.

## 4. Prepare synthetic training/validation and freeze the model

```bash
ros2 run llm_vision_planner person_experiment.py export-synthetic --experiment "$PERSON_EXP/experiment.json"
python3 "$PERSON_PKG/fine_tuning/scripts/train_hrrt_waypoint_transformer.py" \
  --config "$PERSON_PKG/fine_tuning/configs/hrrt_waypoint_transformer.yaml" \
  --train "$PERSON_EXP/Dtrain.csv" --validation "$PERSON_EXP/Dval.csv" \
  --output-dir "$PERSON_EXP/hrrt_hardware_transformer"
```

This trains the **HRRT hardware transformer** using the same DSS row schema, HRRT generation, preference prompts, expert-selection rules and Transformer trainer as the main dataset. Each person layout generates one shared route family and exactly two rows: `shortest` and `balanced`. Inspect training history; choose the checkpoint using Dtrain/Dval only, before Dcal/Dtest. The trainer writes `checkpoint-epoch-NNN` directories, adjacent `text_encoder`, and `training_summary.json` under `$PERSON_EXP/hrrt_hardware_transformer`. Its Dval stopping metric is a selection diagnostic, **not** the held-out Dcal certificate. Do not pass `--smoke-test` for an experimental checkpoint. No existing model is silently substituted.

```bash
read -erp "Absolute selected checkpoint directory produced above: " PERSON_CHECKPOINT
ros2 run llm_vision_planner person_experiment.py freeze-model \
  --experiment "$PERSON_EXP/experiment.json" --checkpoint "$PERSON_CHECKPOINT"
```

The freeze loads real model weights, checks assets and verifies the checkpoint's saved Dtrain/Dval hashes. `model.freeze.json` fingerprints the model and scoring code. Missing/empty placeholder assets are rejected; paths entered above are operator choices, not bundled evidence. Keep source code, YAML, checkpoint, training data and sensor settings fixed. Changes require a new version and, when perception/domain changes, a new survey.

## 5. Guided person recording (Dcal and Dtest)

Reestablish the same hover pose, person geometry/heading, camera and lighting protocol. Use the hardware precautions from step 2. The launch loads paths and geometry from the frozen manifest, not the current working directory:

```bash
ros2 launch llm_vision_planner vision_error_calibration.launch.py \
  params_file:="$PERSON_PARAMS" collection_mode:=scheduled \
  experiment_manifest:="$PERSON_EXP/experiment.json"
```

Normal progress reads `Hold still.` then `Snapshot recorded: 12/100. Move to the next location.` Success is announced only after a complete CSV capture is flushed; durable sidecar events retain the trial/split identity and raw-row fingerprint. The first scheduled observation is used, not the best detection. A missing detection, held/stale frame, hover drift or other failed armed trial is retained and stops the experiment. There is no retry-until-success or surviving-row certificate. Resume only between completed captures; interrupted armed captures or mismatched CSV/event histories are explicitly rejected for review. Repeated sampled positions are not suppressed by the legacy movement duplicate filter, but you must leave and return between trials.

After completion, land and stop as in step 2. Files are `measured.raw.csv` (unchanged legacy raw schema) and `collection.events.jsonl` (metadata/evidence). Startup/shutdown print absolute paths and counts. The survey's `survey.raw.csv` is only a recorder-schema header; actual survey observations are in `survey.events.jsonl`.

## 6. Residual scoring, split certificate and held-out evaluation

```bash
ros2 run llm_vision_planner person_experiment.py score \
  --experiment "$PERSON_EXP/experiment.json" --device auto
```

The command first verifies complete scheduled-scene accounting and fingerprints, then scores **all** scenes through ground-truth HRRT, perceived-geometry Transformer prediction, production refinement and paired ideal-double-integrator QPs with the existing 0.5 m/s and 0.5 m/s² limits. It computes

\[
R(t)=B(\Delta u(t)+K\Delta x(t)),\qquad s_i=\sup_t\sqrt{R(t)^T P R(t)}.
\]

All interval endpoints, both sides of control jumps and real interior extrema are checked under piecewise-constant controls. Shared A, B, P, K and the certified contraction rate are used in scoring/tube construction. No sampled-time-only maximum or velocity damping is used.

Outputs, all under `$PERSON_EXP`:

| File | Role |
| --- | --- |
| `Dcal.scored.csv` | One residual score per scheduled calibration scene |
| `Dtest.scored.csv` | Separate untouched test-scene scores |
| `split_certificate.json` | Dcal-only rank, delta, n, quantile/radius and model/data fingerprints |
| `test.report.json` | Held-out residual coverage and exceedance IDs; never updates the certificate |
| `scoring.failures.jsonl` | Explicit failed-scene evidence if scoring stops |

Split CP uses `k=ceil((1-delta)*(n+1))`, with the k-th score if `k<=n`. Otherwise the mathematical quantile is unbounded: JSON records `quantile_unbounded=true`, null values and `unbounded_insufficient_samples`. It is **never clamped** to the observed maximum. Preflight prevents such a count for this workflow. Missing rows, invalid scores, failures or absent assets cannot yield a certificate from the remainder. Do not change delta after seeing results; do not add/drop calibration samples opportunistically. Existing outputs are not overwritten.

The frozen loader accepts only the matching `Dcal.scored.csv` and adjacent certificate with the preselected delta. It rejects Dtest and modified files. Perception uncertainty remains represented by the paired residual; perfect tracking does not remove that uncertainty or establish flight safety.

## 7. Runtime modes and recovery

Do not launch a new hardware mission merely to view results. For subsequent separately approved mission testing, set `calibration_csv` to the absolute `$PERSON_EXP/Dcal.scored.csv` and `delta` to the frozen value in both gateway and contraction verifier (the existing `full_plot.launch.py` forwards both). The ordinary calibrated mode uses frozen split CP. To intentionally select the retained ACP mode in an already running interactive system:

```bash
ros2 topic pub --once /llm_vision/interaction_mode std_msgs/msg/String \
  "{data: '{\"mode\":\"ADAPTIVE_CONFORMAL_PREDICTION\"}'}"
```

Return to `MISSION_EXECUTION` for frozen calibrated operation. ACP keeps its separate persisted update history and empirical rule; choose a separate `acp_state_file` for each experiment. Stored ACP history never activates adaptation in split mode. Do not feed Dtest back through the adaptive evaluation path. These mode commands are not part of dataset collection.

If recording stalls, follow the printed prerequisite: frame alignment → fresh PX4/Vicon → hover state → perception topics → target/heading/stability → scheduled observation. Detailed ROS process logs remain under `~/.ros/log`; the event sidecars preserve observation geometry and reasons. Do not repeatedly restart to select favorable detections. A failed experimental trial requires review; a changed domain/protocol requires a new version. Neither log files nor these tests replace live hover/reliability measurements.

For `Waiting for mission-state updates` after the executor reports `HOLDING_FOR_PLAN`, first rebuild and resource the updated package (the prior implementation had incompatible QoS). Confirm the fixed publisher/subscriber pairing and the live value:

```bash
ros2 topic info -v /llm_vision/mission_state
ros2 topic echo --qos-durability transient_local /llm_vision/mission_state --once
```

The publisher and recorder subscription should both report reliable, transient-local QoS. If `Waiting for fresh PX4 NED pose` or executor `odometry timeout` appears, recording is correctly blocked because `/fmu/out/vehicle_odometry` stopped; this is separate from mission-state delivery. Check the vehicle link and process on `192.168.0.28`, matching DDS domain/interface on both machines, and verify a continuous rate before relaunching:

```bash
ping -c 3 192.168.0.28
ros2 topic hz /fmu/out/vehicle_odometry
ros2 topic info -v /fmu/out/vehicle_odometry
```

Do not increase the one-second odometry timeout to mask a dropped PX4 stream. Land/disarm and restore continuous telemetry first.

Offline verification (no vehicle connection):

```bash
python3 -m pytest -q "$PERSON_PKG/test/test_person_split_cp.py" "$PERSON_PKG/test/test_person_collection_replay.py"
cd "$PERSON_WS"
ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=231 colcon test --packages-select llm_vision_planner --event-handlers console_direct+
colcon test-result --verbose
```

Replay uses isolated test observations; it is not empirical evidence. Live guided manifests reject replay/simulated-clock collection. Legacy free-form recording and bag replay remain available with `collection_mode:=legacy`; they do not produce a version-2 frozen certificate.
