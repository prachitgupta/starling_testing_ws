# Reproduce waypoint-to-state/control datasets

```bash
cd ~/Desktop/starling_testing_ws/src/llm_vision_planner
python3 fine_tuning/scripts/generate_xu_comparison_datasets.py
```

Basic offline verification:

```bash
for csv in fine_tuning/datasets/calibration_min_snap_*.csv; do
  python3 fine_tuning/scripts/dconformal_contraction_verify.py \
    --calibration-csv "$csv" --calibration-samples 200 --sample-id 0 \
    --output-png /tmp/"$(basename "$csv" .csv)".png \
    --report-json /tmp/"$(basename "$csv" .csv)".json
done
```
