# Record Calibration Data

Run commands from the workspace root unless a command explicitly changes directory. These procedures are separated from mission execution. Raw captures are not calibrated scores and must retain misses. Never use dummy fixtures as a real-flight certificate.

The retained capture dataset is `fine_tuning/datasets/calibration_residual_raw_person_outline.csv` (59 captures, including 8 misses). Before recording, choose a **new output filename**; preserve this reference dataset. Outline perception estimates depth locally; optional EfficientSAM weights remain local, and the configured fallback is shape fitting. Validate the estimator actually used in capture metadata.

### Calibrate the hires camera

Use the grey stream paired with the TFLite color stream:
`hires_small_grey`. Do not use a tracking-camera pipe.

The successful board had 5x6 internal corners and 30 mm squares. Keep it flat,
well lit, and sharp.

Run on VOXL:

```bash
voxl-inspect-services
voxl-list-pipes | grep -E '^hires_small_(color|grey)$'

voxl-set-cpu-mode perf
systemctl stop voxl-tflite-server voxl-qvio-server voxl-tag-detector voxl-dfs-server voxl-streamer 2>/dev/null || true
systemctl restart voxl-camera-server voxl-portal
```

Reduce exposure under bright lighting:

```bash
voxl-send-command hires_small_grey set_exp_gain 3.0 400
```

Open the calibration overlay in VOXL Portal, then run:

```bash
voxl-calibrate-camera hires_small_grey -s 5x6 -l 0.030
```

The accepted calibration had a 0.703523 px reprojection error and was saved to:

```text
/data/modalai/opencv_hires_small_grey_intrinsics.yml
```

Check and back it up:

```bash
sed -n '1,100p' /data/modalai/opencv_hires_small_grey_intrinsics.yml
cp -p /data/modalai/opencv_hires_small_grey_intrinsics.yml \
  /data/modalai/opencv_hires_small_grey_intrinsics.yml.accepted
```

Restore automatic exposure and perception services:

```bash
voxl-send-command hires_small_grey start_ae
systemctl restart voxl-camera-server
systemctl start voxl-qvio-server voxl-tflite-server
voxl-inspect-services | grep -E 'camera|qvio|tflite|mpa-to-ros2'
```

### Build

Run on the ground station:

```bash
cd ~/Desktop/starling_final_testing
source /opt/ros/humble/setup.bash
colcon build --packages-select llm_vision_planner
source install/setup.bash
```

### Run perception only

Terminal 1:

```bash
source ~/Desktop/starling_final_testing/install/setup.bash
source "$(ros2 pkg prefix llm_vision_planner)/lib/llm_vision_planner/ros_wifi_dds.sh" \
  enable auto 42
ros2 daemon start
ros2 run llm_vision_planner perception_detection.py --ros-args \
  --params-file ~/Desktop/starling_final_testing/src/llm_vision_planner/config/llm_vision_planner.yaml
```

Terminal 2:

```bash
source ~/Desktop/starling_final_testing/install/setup.bash
source "$(ros2 pkg prefix llm_vision_planner)/lib/llm_vision_planner/ros_wifi_dds.sh" \
  enable auto 42
ros2 daemon start
ros2 topic echo --full-length /llm_vision/semantic_obstacles
```

### Record all perception topics from a remote machine

Use a separate Ubuntu 22.04/ROS 2 Humble computer on the same flight network so
point-cloud recording does not consume resources on the flight computer or the
ground station. The recorder must use the same DDS option and ROS domain as the
vehicle and ground station. The commands below assume Option A and domain 42.

Run once on the remote recorder:

```bash
sudo apt update
sudo apt install -y git rsync python3-colcon-common-extensions python3-rosdep python3-vcstool \
  ros-humble-rosbag2 ros-humble-rosbag2-storage-mcap

mkdir -p ~/Desktop
git clone --depth 1 --single-branch --branch main \
  https://github.com/prachitgupta/starling_testing_ws.git ~/Desktop/starling_final_testing
cd ~/Desktop/starling_final_testing
source /opt/ros/humble/setup.bash
bash scripts/setup_workspace.sh
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  sudo rosdep init
fi
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select px4_msgs voxl_msgs llm_vision_planner
```

If that clone already exists, pull and rebuild it instead of cloning again.
Before each recording, start the vehicle bridges and the perception node, then
run this on the remote recorder:

```bash
cd ~/Desktop/starling_final_testing
source /opt/ros/humble/setup.bash
source install/setup.bash
source "$(ros2 pkg prefix llm_vision_planner)/lib/llm_vision_planner/ros_wifi_dds.sh" \
  enable auto 42
ros2 daemon stop
ros2 daemon start

printenv ROS_DOMAIN_ID RMW_IMPLEMENTATION ROS_LOCALHOST_ONLY
ros2 topic list | sort | grep -E \
  '^(/hires_small_color|/tof_pc|/tflite|/fmu/out/vehicle_odometry|/llm_vision/(semantic_obstacles|obstacles)|/tf|/tf_static)$'
ros2 topic hz /tof_pc
ros2 topic hz /llm_vision/semantic_obstacles
```

Both `ros2 topic hz` commands must receive messages. Stop each check with
`Ctrl+C`. If `/tof_pc` is missing, do not start the experiment: the bag would not
contain the raw depth data required for later point-cloud work.

Record the three reproduction topics:

```bash
mkdir -p ~/rosbags
STAMP="$(date +%Y%m%d-%H%M%S)"
BAG_DIR="$HOME/rosbags/perception-$STAMP"

ros2 bag record -o "$BAG_DIR" \
  /hires_small_color \
  /tof_pc \
  /tflite
```

Keep recording for the scene or flight. Press `Ctrl+C` once and wait for bag
metadata to finish. Required: `/hires_small_color`, `/tof_pc`, `/tflite`.

Verify the recording before moving or deleting it:

```bash
ros2 bag info "$BAG_DIR"
du -sh "$BAG_DIR"
```

Confirm non-zero counts, then copy the complete bag directory:

```bash
rsync -av --progress "$BAG_DIR/" \
  USER@ANALYSIS_COMPUTER:~/rosbags/"$(basename "$BAG_DIR")"/
```

Replace `USER@ANALYSIS_COMPUTER` with the analysis computer's SSH login and
address. For Option B DDS, omit `ros_wifi_dds.sh enable`.

Replay from this worktree:

```bash
cd ~/Desktop/starling_final_testing
source /opt/ros/humble/setup.bash
source install/setup.bash
BAG_DIR="$(ls -td "$HOME"/rosbags/perception-* | head -1)"
ros2 bag play "$BAG_DIR" --clock
```

## 13. Generate the residual calibration dataset

This workflow is deliberately split into two stages. The flight records only
comma-delimited perceived/ground-truth obstacle geometry. Each accepted capture
is an atomic environment: every configured Vicon object is written under one
`capture_id`, including Vicon yaw and four NED corners. Detected objects also
include the perceived front center, view/lateral axes, visible width, the
depth behind the front surface (measured by perception, Section 12; the column
keeps its old name `pred_chatgpt_depth_m`), and four reconstructed NED
footprint corners. Offline processing
then runs HRRT-star on ground truth, the Llama planner on perception, both
ideal-double-integrator QPs, and writes one residual conformity score per
capture. Never use the raw CSV as the conformal calibration input.

### 13.1 Check the supplied dummy files

```bash
cd ~/Desktop/starling_final_testing
source /opt/ros/humble/setup.bash
source install/setup.bash

RAW_DUMMY="$PWD/src/llm_vision_planner/test/fixtures/calibration_residual_raw_dummy.csv"
SCORED_DUMMY="$PWD/src/llm_vision_planner/test/fixtures/calibration_residual_scored_dummy.csv"

head -n 3 "$RAW_DUMMY"
head -n 3 "$SCORED_DUMMY"

python3 src/llm_vision_planner/fine_tuning/scripts/postprocess_residual_calibration.py \
  --raw-csv "$RAW_DUMMY" \
  --output-csv /tmp/calibration_residual_scored_smoke.csv \
  --delimiter ',' \
  --l1-text "Fly to the point beyond the chair." \
  --l2-text "Use a short route while staying away from the person." \
  --goal-x 3.0 \
  --goal-y 0.0 \
  --fixed-z -0.5 \
  --planner-provider mock \
  --expert-selector heuristic \
  --all-captures \
  --placeholder

head -n 3 /tmp/calibration_residual_scored_smoke.csv
```

The dummy rows have `placeholder=true`. They verify the schema only and must
not be mixed with real calibration data.

### 13.2 Start the all-object Vicon bridge

Stop any other Vicon bridge first. Run this on the ground station:

```bash
export VICON_COMPUTER_IP=10.117.229.124

ros2 run vicon_bridge vicon_bridge --ros-args \
  -p host_name:="${VICON_COMPUTER_IP}:801" \
  -p stream_mode:="ServerPush" \
  -p update_rate_hz:=125.0 \
  -p expected_rate_hz:=50.0 \
  -p publish_specific_segment:=false \
  -p world_frame_id:="vicon_world" \
  -p tf_namespace:="vicon" \
  -r /vicon/Starling2/Starling2/pose:=/mavros/vision_pose/pose
```

In another terminal, verify every required stream:

```bash
source ~/Desktop/starling_final_testing/install/setup.bash
ros2 topic echo /vicon/Starling2/Starling2 --once
ros2 topic echo /vicon/chair1/chair1 --once
ros2 topic echo /vicon/person/person --once
ros2 topic echo /vicon/stopsign/stopsign --once
ros2 topic echo /fmu/out/vehicle_odometry --once
```

### 13.3 Record raw hardware captures

This launch owns Offboard control: do not run `full_plot.launch.py` or another
Offboard publisher at the same time. Keep QGroundControl land/kill controls
available. The vehicle takes off, holds its initial X/Y position, and records
only after frame alignment reports `FRAME_READY`.

Depths come from perception (Section 12), so no OpenAI key is needed. To
record GPT nano's depths as before, export `OPENAI_API_KEY` and add
`nominal_depth_source:=gpt intent_provider:=openai` to the launch command.

In the launch terminal:

```bash
cd ~/Desktop/starling_final_testing
source /opt/ros/humble/setup.bash
source install/setup.bash
source "$(ros2 pkg prefix llm_vision_planner)/lib/llm_vision_planner/ros_wifi_dds.sh" \
  enable auto 42
ros2 daemon start

read -rp "Chair X width in metres: " CHAIR_WIDTH_M
read -rp "Chair Y depth in metres: " CHAIR_DEPTH_M
read -rp "Person target X width in metres: " PERSON_WIDTH_M
read -rp "Person target Y depth in metres: " PERSON_DEPTH_M
read -rp "Stop-sign target X width in metres: " STOPSIGN_WIDTH_M
read -rp "Stop-sign target Y depth in metres: " STOPSIGN_DEPTH_M

export CHAIR_WIDTH_M CHAIR_DEPTH_M PERSON_WIDTH_M PERSON_DEPTH_M \
  STOPSIGN_WIDTH_M STOPSIGN_DEPTH_M
VICON_OBJECTS_JSON="$(python3 - <<'PY'
import json
import os

print(json.dumps([
    {
        "object_id": "chair-1",
        "label": "chair",
        "topic": "/vicon/chair1/chair1",
        "dimensions_m": [float(os.environ["CHAIR_WIDTH_M"]), float(os.environ["CHAIR_DEPTH_M"])],
    },
    {
        "object_id": "person-1",
        "label": "person",
        "topic": "/vicon/person/person",
        "dimensions_m": [float(os.environ["PERSON_WIDTH_M"]), float(os.environ["PERSON_DEPTH_M"])],
    },
    {
        "object_id": "stop-sign-1",
        "label": "stop sign",
        "topic": "/vicon/stopsign/stopsign",
        "dimensions_m": [float(os.environ["STOPSIGN_WIDTH_M"]), float(os.environ["STOPSIGN_DEPTH_M"])],
    },
]))
PY
)"

RAW_CSV="$PWD/src/llm_vision_planner/fine_tuning/datasets/calibration_residual_raw.csv"
TRIAL_ID="residual-$(date +%Y%m%d-%H%M%S)"
POSITION_CHANGE_M=0.15
YAW_CHANGE_RAD=0.261799  # 15 degrees

ros2 launch llm_vision_planner vision_error_calibration.launch.py \
  params_file:="$PWD/src/llm_vision_planner/config/llm_vision_planner.yaml" \
  trial_id:="$TRIAL_ID" \
  output_csv:="$RAW_CSV" \
  vicon_objects_json:="$VICON_OBJECTS_JSON" \
  capture_position_change_threshold_m:="$POSITION_CHANGE_M" \
  capture_yaw_change_threshold_rad:="$YAW_CHANGE_RAD"
```

#### Person-only collection, with no chair or stop sign

Use this complete launch block instead of the all-object block above when the
arena contains only the tracked person. The recorder will wait only for
`/vicon/person/person`; do not include absent objects in `VICON_OBJECTS_JSON`.

```bash
cd ~/Desktop/starling_final_testing
source /opt/ros/humble/setup.bash
source install/setup.bash
source "$(ros2 pkg prefix llm_vision_planner)/lib/llm_vision_planner/ros_wifi_dds.sh" \
  enable auto 42
ros2 daemon start

read -rp "Person target X width in metres: " PERSON_WIDTH_M
read -rp "Person target Y depth in metres: " PERSON_DEPTH_M
export PERSON_WIDTH_M PERSON_DEPTH_M

VICON_OBJECTS_JSON="$(python3 - <<'PY'
import json
import os

print(json.dumps([
    {
        "object_id": "person-1",
        "label": "person",
        "topic": "/vicon/person/person",
        "dimensions_m": [
            float(os.environ["PERSON_WIDTH_M"]),
            float(os.environ["PERSON_DEPTH_M"]),
        ],
    },
]))
PY
)"

RAW_CSV="$PWD/src/llm_vision_planner/fine_tuning/datasets/calibration_residual_raw_person_outline.csv"
TRIAL_ID="person-residual-$(date +%Y%m%d-%H%M%S)"
POSITION_CHANGE_M=0.15
YAW_CHANGE_RAD=0.261799

ros2 topic echo /vicon/person/person --once

ros2 launch llm_vision_planner vision_error_calibration.launch.py \
  params_file:="$PWD/src/llm_vision_planner/config/llm_vision_planner.yaml" \
  trial_id:="$TRIAL_ID" \
  output_csv:="$RAW_CSV" \
  vicon_objects_json:="$VICON_OBJECTS_JSON" \
  capture_position_change_threshold_m:="$POSITION_CHANGE_M" \
  capture_yaw_change_threshold_rad:="$YAW_CHANGE_RAD"
```

Before starting that launch, use the remote recorder prepared in Section 12 and
run this in a separate terminal on that machine. Rosbag2 continues discovering
topics after startup, so it may be started before the calibration nodes:

```bash
cd ~/Desktop/starling_final_testing
source /opt/ros/humble/setup.bash
source install/setup.bash
source "$(ros2 pkg prefix llm_vision_planner)/lib/llm_vision_planner/ros_wifi_dds.sh" \
  enable auto 42
ros2 daemon stop
ros2 daemon start

mkdir -p ~/rosbags
STAMP="$(date +%Y%m%d-%H%M%S)"
BAG_DIR="$HOME/rosbags/person-calibration-$STAMP"

ros2 bag record -o "$BAG_DIR" \
  /hires_small_color \
  /tof_pc \
  /tflite \
  /tflite_data \
  /fmu/out/vehicle_odometry \
  /vicon/Starling2/Starling2 \
  /vicon/person/person \
  /llm_vision/semantic_obstacles
```

The detections, odometry and Vicon topics let Section 13.3.1 record the
dataset again from this bag later, for example with a different depth method.
Add `/vicon/chair1/chair1` and `/vicon/stopsign/stopsign` when those objects
are in the arena.

Keep recording during collection. After landing, press `Ctrl+C` once and verify:

```bash
ros2 bag info "$BAG_DIR"
du -sh "$BAG_DIR"
```

The bag must contain non-zero counts for `/hires_small_color`, `/tof_pc`, and
`/tflite`.

Replay from this worktree:

```bash
cd ~/Desktop/starling_final_testing
source /opt/ros/humble/setup.bash
source install/setup.bash
BAG_DIR="$(ls -td "$HOME"/rosbags/person-calibration-* | head -1)"
ros2 bag play "$BAG_DIR" --clock
```

In a monitor terminal:

```bash
source ~/Desktop/starling_final_testing/install/setup.bash
ros2 topic echo /llm_vision/mission_state --once
ros2 topic echo /llm_vision/vision_calibration_status --once
tail -f \
  ~/Desktop/starling_final_testing/src/llm_vision_planner/fine_tuning/datasets/calibration_residual_raw.csv
```

After each `RECORDED` status, move at least one tracked obstacle by
`POSITION_CHANGE_M` or rotate it by `YAW_CHANGE_RAD`, then leave every object
stationary. The recorder waits for all configured Vicon objects and writes the
complete environment together. `SKIPPED_DUPLICATE_ENVIRONMENT` means neither
threshold was reached; `SKIPPED_MOVING_OBJECT` means the scene was not stable.
Do not move the aircraft by hand while it holds position.

Land before stopping the launch:

```bash
ros2 topic pub --once /llm_vision/executor_command std_msgs/msg/String \
  "{data: '{\"command\":\"LAND\",\"reason\":\"residual calibration complete\"}'}"
ros2 topic echo /llm_vision/mission_state
```

Wait for `COMPLETE` and verify that the vehicle is landed and disarmed. Then
press `Ctrl+C` in the launch terminal.

### 13.3.1 Record the dataset from a rosbag instead of a flight

The same launch records the raw CSV from a bag, with nothing sent to a
vehicle: `replay:=true` leaves out the takeoff node and the mission-state gate
and runs every node on the bag's clock (`use_sim_time`), so the perception
timestamps match the recorded Vicon stamps. The bag needs `/hires_small_color`,
`/tof_pc`, `/tflite`, `/tflite_data`, `/fmu/out/vehicle_odometry`,
`/vicon/Starling2/Starling2` and the Vicon topic of every object in
`VICON_OBJECTS_JSON`; the 2026-10-11 person bag has all of them. Do not replay
`/llm_vision/semantic_obstacles` from the bag: the perception node publishes
it again with the new depths.

Check once that the bag's clock and the Vicon stamps agree (a bag recorded on
another computer can be offset by that computer's clock):

```bash
cd ~/Desktop/starling_final_testing
source /opt/ros/humble/setup.bash
source install/setup.bash
BAG_DIR=~/rosbags/person-calibration-20261011-114711   # the bag folder

python3 - "$BAG_DIR" <<'PY'
import itertools
import sys

import numpy as np

sys.path.insert(0, "src/llm_vision_planner/scripts")
import bag_reader as br

bag = br.open_bag(sys.argv[1])
messages = itertools.islice(bag.messages(["/vicon/person/person"]), 3000)
gaps = [log_ns - br.stamp_ns(msg.header) for _, log_ns, msg in messages]
print(f"bag clock minus Vicon stamps: {np.median(gaps) / 1e9:+.3f} s")
PY
```

Within +-0.05 s is fine. Otherwise add `vicon_time_offset_s:=<printed value>`
to the launch command below; the recorder adds it to every Vicon stamp.

Terminal 1, the calibration nodes (person-only example; use the dimensions
entered for that session; 0.38 x 0.29 m matches the 2026-10-11 ground-truth
boxes):

```bash
cd ~/Desktop/starling_final_testing
source /opt/ros/humble/setup.bash
source install/setup.bash

VICON_OBJECTS_JSON='[{"object_id": "person-1", "label": "person", "topic": "/vicon/person/person", "dimensions_m": [0.38, 0.29]}]'
RAW_CSV="$PWD/src/llm_vision_planner/fine_tuning/datasets/calibration_residual_raw_person_outline.csv"
TRIAL_ID="person-residual-replay-$(date +%Y%m%d-%H%M%S)"

ros2 launch llm_vision_planner vision_error_calibration.launch.py \
  params_file:="$PWD/src/llm_vision_planner/config/llm_vision_planner.yaml" \
  replay:=true \
  trial_id:="$TRIAL_ID" \
  output_csv:="$RAW_CSV" \
  vicon_objects_json:="$VICON_OBJECTS_JSON" \
  capture_position_change_threshold_m:=0.15 \
  capture_yaw_change_threshold_rad:=0.261799
```

Terminal 2, once terminal 1 prints `interactive gateway ready`:

```bash
cd ~/Desktop/starling_final_testing
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 bag play ~/rosbags/person-calibration-20261011-114711 --clock --topics \
  /hires_small_color /tof_pc /tflite /tflite_data /fmu/out/vehicle_odometry \
  /vicon/Starling2/Starling2 /vicon/person/person
```

Terminal 3, to watch it:

```bash
source ~/Desktop/starling_final_testing/install/setup.bash
ros2 topic echo /llm_vision/vision_calibration_status
```

`FRAME_READY` appears within a second of playback, then `RECORDED` for each
new arrangement of the objects. Play at normal speed: the outline depth takes
about 1 s per object on a laptop, so faster playback outruns it.
When the bag ends, press `Ctrl+C` in terminal 1 and validate the CSV with
Section 13.4 (set `RAW_CSV` to the file above). Rows with
`missed_detection=true` are captures where no detection reached
`min_confidence` (0.70); the 2026-10-11 bag has many person boxes at
0.6-0.7.

Compare the replayed CSV with the one recorded live (GPT nano depths): how far
each Vicon footprint reaches outside the predicted one, and the depths:

```bash
python3 src/llm_vision_planner/fine_tuning/scripts/compare_raw_calibration.py \
  src/llm_vision_planner/fine_tuning/datasets/calibration_residual_raw_person_outline.csv "$RAW_CSV" --label person
```

### 13.4 Validate the raw comma-delimited CSV

```bash
cd ~/Desktop/starling_final_testing
RAW_CSV="$PWD/src/llm_vision_planner/fine_tuning/datasets/calibration_residual_raw.csv"

python3 - "$RAW_CSV" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream, delimiter=","))
required = {
    "session_id", "capture_id", "object_id", "label",
    "pred_min_x", "pred_min_y", "pred_max_x", "pred_max_y",
    "gt_min_x", "gt_min_y", "gt_max_x", "gt_max_y", "placeholder",
    "gt_yaw_rad",
    *(f"gt_corner_{i}_{axis}" for i in range(4) for axis in ("x", "y")),
    "pred_front_center_x", "pred_front_center_y",
    "pred_view_axis_x", "pred_view_axis_y",
    "pred_lateral_axis_x", "pred_lateral_axis_y",
    "pred_visible_width_m", "pred_chatgpt_depth_m",
    *(f"pred_corner_{i}_{axis}" for i in range(4) for axis in ("x", "y")),
}
if not rows:
    raise SystemExit("No raw calibration rows were recorded.")
missing = required.difference(rows[0])
if missing:
    raise SystemExit(f"Missing raw CSV columns: {sorted(missing)}")
if any(row["placeholder"].lower() != "false" for row in rows):
    raise SystemExit("Real raw data contains a placeholder row.")
captures = {}
for row in rows:
    captures.setdefault(row["capture_id"], []).append(row)
if any(len({row["object_id"] for row in group}) != len(group) for group in captures.values()):
    raise SystemExit("A capture contains duplicate object rows.")
for row in rows:
    if row["missed_detection"].lower() == "false":
        perceived = [name for name in required if name.startswith("pred_")]
        if any(not row[name] for name in perceived):
            raise SystemExit(f"Detected object is missing perceived geometry: {row['object_id']}")
print(f"raw rows: {len(rows)}")
print(f"independent captures: {len(captures)}")
PY
```

### 13.5 Compute the scored calibration CSV offline

Restore the selected local V1 Transformer weights using the final Transformer guide.
Each command scores only the captures selected for that `L1`/`L2`/goal batch.
The local Transformer plans on perceived geometry; ChatGPT selects the expert HRRT
route on ground-truth geometry. If a human already selected an exact displayed
route, replace `--expert-selector openai` with
`--expert-route-id ROUTE_ID_FROM_GATEWAY`.

```bash
cd ~/Desktop/starling_final_testing
source /opt/ros/humble/setup.bash
source install/setup.bash

RAW_CSV="$PWD/src/llm_vision_planner/fine_tuning/datasets/calibration_residual_raw.csv"
CALIBRATION_CSV="$PWD/src/llm_vision_planner/fine_tuning/datasets/calibration_residual_scored.csv"
TRANSFORMER_CHECKPOINT="$PWD/src/llm_vision_planner/fine_tuning/outputs/final-transformer-vertical-teacher_only_80_10_10/final_transformer_vertical_teacher_only_80_10_10_best/checkpoint"
DELTA=0.20
read -rsp "OpenAI API key: " OPENAI_API_KEY
export OPENAI_API_KEY
export EXPERT_MODEL=gpt-5.4

python3 src/llm_vision_planner/fine_tuning/scripts/postprocess_residual_calibration.py \
  --raw-csv "$RAW_CSV" \
  --list-captures

python3 src/llm_vision_planner/fine_tuning/scripts/postprocess_residual_calibration.py \
  --raw-csv "$RAW_CSV" \
  --output-csv "$CALIBRATION_CSV" \
  --delimiter ',' \
  --l1-text "Fly to the point beyond the chair." \
  --l2-text "Use a short route while staying as far from the person as possible." \
  --goal-x 3.0 \
  --goal-y 0.0 \
  --fixed-z -0.5 \
  --workspace-x-min -4.0 \
  --workspace-x-max 4.0 \
  --workspace-y-min -3.0 \
  --workspace-y-max 3.0 \
  --clearance-m 0.4 \
  --dt 0.1 \
  --hrrt-iterations 500 \
  --max-candidates 8 \
  --max-waypoints 8 \
  --sample-count 25 \
  --sample-seed 17 \
  --expert-selector openai \
  --expert-model "$EXPERT_MODEL" \
  --planner-provider transformer \
  --transformer-checkpoint "$TRANSFORMER_CHECKPOINT" \
  --transformer-device auto --delta "$DELTA" \
  --append
```

Run the command again with a different `L1`/`L2`/goal and either a different
`--sample-seed` or repeated `--capture-id CAPTURE_ID` flags. `--append` builds
one calibration CSV and rejects duplicate `(capture_id, L1, L2)` labels. Use
`--all-captures` only when applying one variant to every capture intentionally.

For an exact batch, replace `--sample-count 25 --sample-seed 17` with:

```bash
--capture-id residual-YYYYMMDD-HHMMSS-capture-000003 \
--capture-id residual-YYYYMMDD-HHMMSS-capture-000011
```

### 13.6 Validate the scored CSV

```bash
cd ~/Desktop/starling_final_testing
CALIBRATION_CSV="$PWD/src/llm_vision_planner/fine_tuning/datasets/calibration_residual_scored.csv"

python3 - "$CALIBRATION_CSV" <<'PY'
import csv
import math
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream, delimiter=","))
required = {
    "capture_id", "l1_text", "l2_text", "expert_selector", "expert_model", "expert_route_id",
    "ground_truth_environment_json", "perceived_environment_json",
    "conformity_score", "dynamics_model", "planner_provider", "placeholder",
}
if not rows:
    raise SystemExit("No scored calibration rows were generated.")
missing = required.difference(rows[0])
if missing:
    raise SystemExit(f"Missing scored CSV columns: {sorted(missing)}")
scores = [float(row["conformity_score"]) for row in rows]
if not all(math.isfinite(score) and score >= 0.0 for score in scores):
    raise SystemExit("A conformity score is invalid.")
if any(row["placeholder"].lower() != "false" for row in rows):
    raise SystemExit("Real scored data contains a placeholder row.")
print(f"scored captures: {len(rows)}")
print(f"score range: [{min(scores):.6f}, {max(scores):.6f}]")
PY

head -n 2 "$CALIBRATION_CSV"
```

Use `calibration_residual_scored.csv`, not the raw file or either dummy file,
when the adaptive conformal initial-radius loader is enabled.


## Final Transformer evaluation

Use [the final V1 gateway guide](fine_tuning/final_transformer_vertical.md). The postprocessor uses the same Transformer loader as both gateways. Mock fixture scores are excluded from real calibration. The quantile uses all valid scored rows at probability `1-delta`, with no sample-count threshold. The interactive gateway does not batch-score this entire dataset on startup or obtain human preference selections automatically. Freeze the exact deployed pipeline and separately validate its scored calibration before flight.
