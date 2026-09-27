# HRRT-star multiple-trajectory workspace

This directory is intentionally limited to the multiple-trajectory expert and
its future data-generation entry points.

## Implemented

- `scripts/hrrt_star.py`: a deterministic 2-D, fixed-altitude HRRT-star
  implementation with winding-number H-signatures, RRT-star rewiring,
  inter-signature rewiring, label-conditioned clearance bins, finite composite
  route keys, JSON output, validation, and demo plotting.
- `scripts/min_control_qp.py`: the existing minimum-control QP implementation,
  copied unchanged for converting verified HRRT waypoints into state and control
  samples.

## Reserved for later steps

- `scripts/generate_hrrt_acp_dataset.py`: adaptive-conformal data capture.
- `scripts/generate_hrrt_finetuning_dataset.py`: synthetic HRRT teacher data.
- `outputs/llama31_8b_hrrt_lora/`: future preference-conditioned adapter.

The two dataset scripts remain empty until their input/output contracts and the
consolidated calibration-score equation are approved.

## Demo

```bash
python3 fine_tuning/scripts/hrrt_star.py \
  --demo \
  --seed 17 \
  --iterations 300 \
  --plot-output fine_tuning/plots/hrrt_star_demo.png \
  --json-output fine_tuning/datasets/hrrt_star_demo.json
```

The planner uses ground-truth obstacle boxes and a geometric hard clearance.
It does not apply a perception-uncertainty guard band or construct a safety
tube. Each returned route is the shortest retained representative of a finite
composite key `(H-signature, clearance-bin vector)`.
