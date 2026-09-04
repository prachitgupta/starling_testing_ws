# Reproduce the retained QP calibration

This file retains the original RRT commands. For separate, complete RRT and
Semantic Theta* calibration/scoring/plot workflows, use
[`README.md`](README.md#calibration-prerequisites). Semantic Theta requires
`--expert semantic_theta` in all three downstream tools and its own CSV/plot paths;
the QP and shared-clock calculations are shared with RRT.

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
/usr/bin/python3 fine_tuning/scripts/generate_qp_calibration_dataset.py --input fine_tuning/datasets/conformal_rrt_prediction_pairs_5000.csv --output fine_tuning/datasets/calibration_min_control_qp_shared_clock_with_limits_5000.csv --samples 5000 --max-velocity-mps 0.5 --max-acceleration-mps2 0.5
/usr/bin/python3 fine_tuning/scripts/generate_position_score_calibration.py --input fine_tuning/datasets/calibration_min_control_qp_shared_clock_with_limits_5000.csv --output fine_tuning/datasets/calibration_min_control_qp_position_score_with_limits_5000.csv --samples 5000
```

```bash
python3 fine_tuning/scripts/dconformal_contraction_verify.py \
  --calibration-csv fine_tuning/datasets/calibration_min_control_qp_position_score_with_limits_2000.csv \
  --calibration-samples 2000 --sample-id 998 \
  --trajectory-dt 0.1 --max-velocity-mps 0.5 --max-acceleration-mps2 0.5 \
  --output-png fine_tuning/plots/contraction/qp_with_limits.png \
  --report-json fine_tuning/results/contraction/qp_with_limits.json
```
