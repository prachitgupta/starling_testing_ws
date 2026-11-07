# Hardware testing — interactive full mission

This guide only launches and operates the mission. First complete [hardware setup and preflight](HARDWARE_SETUP.md). Camera calibration, bag recording, raw capture and scoring are in [Record Calibration Data](Record%20Calibration%20Data.md).

## Preconditions

- Confirm frames, Vicon alignment, PX4 estimation, RC abort and a clear test area with the responsible operator.
- This command can start real flight control. Do not run it during documentation review.
- The default vision CSV is a dummy placeholder. Supply a validated, non-placeholder vision calibration CSV for the exact perception configuration. The historical controller certificate does not certify the new outline estimator or final Transformer.
- The live mission calls the existing Llama service. The final V1 Transformer is used by the separate offline calibration gateway, not this launch.

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
LLAMA_MODEL_NAME=hrrt_planner_dss

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
  llm_provider:=llama \
  llama_model_name:="$LLAMA_MODEL_NAME" \
  vicon_objects_json:="$VICON_OBJECTS_JSON" \
  visualizer:=contraction \
  web_ui_host:=0.0.0.0 \
  web_ui_port:=8081
```

Use the RC kill switch or change PX4/QGroundControl mode to abort.

In real Adaptive conformal mode, HRRT uses the Vicon boxes from this catalog
and Llama uses the perceived obstacle snapshot. The web and contraction views
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

- **Adaptive conformal:** select before submitting a prompt. Review the grounded mission, HRRT reference and planner proposal before approval. The UI label does not establish that online adaptive conformal updates or a calibrated Transformer guarantee exist.
- **Mission execution:** select before submitting a prompt. This is the normal full mission approval/execution flow and skips the adaptive mode's HRRT L2 stage. It does not replace calibration or flight checks.

Keep operator supervision throughout. Use the RC kill switch or PX4 mode change to abort. This integration has not performed real flight or bag/Vicon hardware validation. Do not expose the unauthenticated control interface publicly.
