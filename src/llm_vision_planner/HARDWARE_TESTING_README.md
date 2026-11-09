# Hardware testing — interactive full mission

This guide only launches and operates the mission. First complete [hardware setup and preflight](HARDWARE_SETUP.md). Camera calibration, bag recording, raw capture and scoring are in [Record Calibration Data](Record%20Calibration%20Data.md).

## Preconditions

- Confirm frames, Vicon alignment, PX4 estimation, RC abort and a clear test area with the responsible operator.
- This command can start real flight control. Do not run it during documentation review.
- The default vision CSV is a dummy placeholder. Supply a validated, non-placeholder vision calibration CSV for the exact perception configuration. The historical controller certificate does not certify the new outline estimator or final Transformer.
- The live mission and offline calibration both use the local V1 Transformer. Restore its real weights before launching; OpenAI is used for intent parsing and expert-route selection.

### Launch the real mission gateway

This command starts real flight control. Run it only when the vehicle is ready
to fly.

```bash
cd ~/Desktop/starling_final_testing
source install/setup.bash
source "$(ros2 pkg prefix llm_vision_planner)/lib/llm_vision_planner/ros_wifi_dds.sh" \
  enable auto 42
ros2 daemon start
read -rsp "OpenAI API key: " OPENAI_API_KEY
export OPENAI_API_KEY
TRANSFORMER_CHECKPOINT="$PWD/src/llm_vision_planner/fine_tuning/outputs/final-transformer-vertical-teacher_only_80_10_10/final_transformer_vertical_teacher_only_80_10_10_best/checkpoint"
RESIDUAL_CSV="$PWD/src/llm_vision_planner/fine_tuning/datasets/calibration_residual_scored.csv"
DELTA=0.20

# Required for Adaptive conformal mode in a real environment.
read -rp "Chair X width in metres: " CHAIR_WIDTH_M
read -rp "Chair Y depth in metres: " CHAIR_DEPTH_M
read -rp "Person X width in metres: " PERSON_WIDTH_M
read -rp "Person Y depth in metres: " PERSON_DEPTH_M
read -rp "Stop-sign X width in metres: " STOPSIGN_WIDTH_M
read -rp "Stop-sign Y depth in metres: " STOPSIGN_DEPTH_M
export CHAIR_WIDTH_M CHAIR_DEPTH_M PERSON_WIDTH_M PERSON_DEPTH_M \
  STOPSIGN_WIDTH_M STOPSIGN_DEPTH_M
VICON_OBJECTS_JSON="$(python3 - <<'PY'
import json
import os

print(json.dumps([
    {"object_id": "chair-1", "label": "chair", "topic": "/vicon/chair1/chair1",
     "dimensions_m": [float(os.environ["CHAIR_WIDTH_M"]), float(os.environ["CHAIR_DEPTH_M"])]},
    {"object_id": "person-1", "label": "person", "topic": "/vicon/person/person",
     "dimensions_m": [float(os.environ["PERSON_WIDTH_M"]), float(os.environ["PERSON_DEPTH_M"])]},
    {"object_id": "stop-sign-1", "label": "stop sign", "topic": "/vicon/stopsign/stopsign",
     "dimensions_m": [float(os.environ["STOPSIGN_WIDTH_M"]), float(os.environ["STOPSIGN_DEPTH_M"])]},
]))
PY
)"

read -rp "Validated vision-calibration CSV (absolute path): " VISION_CERT
test -s "$VISION_CERT" || { echo "Missing calibration CSV"; exit 1; }

ros2 launch llm_vision_planner full_plot.launch.py \
  vision_error_calibration_csv:="$VISION_CERT" \
  params_file:="$PWD/src/llm_vision_planner/config/llm_vision_planner.yaml" \
  environment:=real \
  interaction_mode:=interactive \
  intent_provider:=openai \
  transformer_checkpoint:="$TRANSFORMER_CHECKPOINT" \
  transformer_device:=auto \
  calibration_csv:="$RESIDUAL_CSV" delta:="$DELTA" acp_gamma:=0.01 \
  acp_state_file:="$HOME/.ros/llm_vision/acp_state.json" \
  log_mission:=true log_directory:="$HOME/.ros/llm_vision/mission_logs" \
  vicon_objects_json:="$VICON_OBJECTS_JSON" \
  visualizer:=contraction \
  web_ui_host:=0.0.0.0 \
  web_ui_port:=8081
```

Use the RC kill switch or change PX4/QGroundControl mode to abort.

In real Adaptive conformal mode, HRRT uses the Vicon boxes from this catalog
and the Transformer uses the perceived obstacle snapshot. The web and contraction views
show Vicon ground truth in red and perceived geometry in orange. Enter measured
dimensions above and omit objects that are not present.
The vehicle alignment topic is `/vicon/Starling2/Starling2`. `environment:=sim`
does not use these Vicon inputs and retains the existing simulation behavior.

From another device on the same network, open
`http://<ground-station-lan-ip>:8081` once `/llm_vision/mission_state` reports
`HOLDING_FOR_PLAN`. The `0.0.0.0` value is the server bind address, not the
address to type into the browser. Select either mode from the left panel before
submitting the first mission prompt. Review each proposed mission and plan before approving execution.
Add `land_after_complete:=false` to the launch command to hold the final goal
instead of landing automatically.


## Operating modes in the same interface

- **Adaptive conformal:** select before submitting a prompt. Review the grounded mission, HRRT reference and planner proposal before approval. Each complete verified expert/Transformer QP pair produces one maximum residual score and one persisted ACI update. The frozen quantile is evaluated before the update; the new value applies to the next trajectory.
- **Mission execution:** select before submitting a prompt. This is the normal full mission approval/execution flow and skips the adaptive mode's HRRT L2 stage. It does not replace calibration or flight checks.

Keep operator supervision throughout. Use the RC kill switch or PX4 mode change to abort. This integration has not performed real flight or bag/Vicon hardware validation. Do not expose the unauthenticated control interface publicly.

## Residual calibration and publication logs

`delta` is the user-selected target miscoverage, strictly between 0 and 1.
All valid scored trajectories contribute to the empirical order statistic
`sorted_scores[ceil((1-delta)*n)-1]`, for every positive sample count. `delta:=0.20`
selects the 80th percentile. There is no minimum-sample cutoff. Missing, empty,
placeholder-only, or incompatible scored data prints `no valid quantile found`
and supplies no tube radius. The vision containment CSV remains a separate input.

The final interface shows calibrated quantile/radius and adaptive quantile/radius
separately. Mission execution uses the latest adaptive state if available, and
otherwise the calibrated baseline. It does not generate or log HRRT expert
trajectories. ACP evaluates one complete time-aligned trajectory pair, not each
waypoint; duplicate plan IDs cannot update state twice. State survives restart.
Use a new `acp_state_file` when changing calibration data, delta, or learning rate.

ACI uses `alpha_next = alpha + gamma*(delta-error)`, where
`error = 1[trajectory_score > frozen_quantile]`. Alpha is not clipped; full/empty
prediction sets at the boundaries cannot release a finite mission tube. These
boundary events still update ACP when a valid trajectory score is available.
The identity `mean(error)-delta = (alpha_initial-alpha_final)/(gamma*T)` gives
long-run average residual-score coverage for the evaluated sequence with bounded
alpha. It does not assert conditional coverage, finite-sample split-conformal
coverage, or physical flight safety. Reference: [Gibbs and Candes, Section 2](https://jmlr.org/papers/volume25/22-1218/22-1218.pdf).

For the ideal residual metric, `r = q*sqrt(lambda_max(C inv(P) C.T))/lambda`,
where `C` selects XY and `lambda` is the contraction rate. The current metric is
isotropic, so its actual XY projection is circular. Plot axes share a scale;
physical drone radius and chord discretization allowance are separately included
in collision checking. The optimized reference shared with the executor uses
ideal dynamics. The existing PX4 velocity-command interface is unchanged; this
reference-model bound is not a validated physical tracking certificate.

`mission_data_logger.py` is a separate process launched by `full_plot` by default.
Each run writes `metadata.json` and append-only `events.jsonl` under
`log_directory`. Records retain original sensor timestamps/frames, Unix/ROS/
monotonic receipt times, ordered sequence numbers, raw PX4 odometry, raw Vicon
person and configured obstacle transforms, semantic obstacle corners, waypoint
stages, optimized reference states/controls, executor-accepted references, and
ACP events. HRRT expert state/control samples are logged only for ACP trajectories.
States are `[x,y,vx,vy]` in NED metres and metres/second; ideal controls are
`[ax,ay]` in metres/second squared. Raw Vicon frames remain identified by their
source header; clock/frame alignment must be performed explicitly when making
publication plots. Logging does not command the vehicle.
