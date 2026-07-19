# Reproduce the retained QP calibration

Generate fresh verified RRT/Llama prediction pairs (contacts the configured vLLM server):

```bash
cd ~/Desktop/starling_testing_ws/src/llm_vision_planner
python3 fine_tuning/scripts/conformal_rrt_dataset.py \
  --samples 2001 \
  --output fine_tuning/datasets/conformal_rrt_calibration_dataset_2001.csv
```

Generate QP trajectories and conformal columns from the stored prediction pairs:

```bash
cd ~/Desktop/starling_testing_ws/src/llm_vision_planner
python3 fine_tuning/scripts/generate_qp_calibration_dataset.py
python3 fine_tuning/scripts/generate_position_score_calibration.py
```

For a separate 5,000-row run in tmux, use:

```bash
tmux new-session -s qp5000
cd ~/Desktop/starling_testing_ws/src/llm_vision_planner
source /opt/ros/humble/setup.bash
/usr/bin/python3 fine_tuning/scripts/conformal_rrt_dataset.py --samples 5000 --output fine_tuning/datasets/conformal_rrt_prediction_pairs_5000.csv
/usr/bin/python3 fine_tuning/scripts/generate_qp_calibration_dataset.py --input fine_tuning/datasets/conformal_rrt_prediction_pairs_5000.csv --output fine_tuning/datasets/calibration_min_control_qp_shared_clock_5000.csv --samples 5000
/usr/bin/python3 fine_tuning/scripts/generate_position_score_calibration.py --input fine_tuning/datasets/calibration_min_control_qp_shared_clock_5000.csv --output fine_tuning/datasets/calibration_min_control_qp_position_score_5000.csv --samples 5000
```

```bash
python3 fine_tuning/scripts/dconformal_contraction_verify.py \
  --calibration-csv fine_tuning/datasets/calibration_min_control_qp_position_score_2000.csv \
  --calibration-samples 2000 --sample-id 998 \
  --output-png /tmp/qp_position_verify.png \
  --report-json /tmp/qp_position_verify.json
```
