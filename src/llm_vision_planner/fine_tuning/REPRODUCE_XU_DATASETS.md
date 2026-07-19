# Reproduce the retained QP calibration

```bash
cd ~/Desktop/starling_testing_ws/src/llm_vision_planner
python3 fine_tuning/scripts/generate_qp_calibration_dataset.py
python3 fine_tuning/scripts/generate_single_score_calibration.py
```

```bash
python3 fine_tuning/scripts/dconformal_contraction_verify.py \
  --calibration-csv fine_tuning/datasets/calibration_min_control_qp_single_score_2000.csv \
  --calibration-samples 2000 --sample-id 998 \
  --output-png /tmp/qp_single_score_verify.png \
  --report-json /tmp/qp_single_score_verify.json
```
