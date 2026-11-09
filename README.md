# Starling — final testing implementation

This branch is the curated testing snapshot: the multiple-trajectory implementation plus the colleague's outline-depth integration. Research branches remain independent; this is not a flight-certification release.

## Start here

- [Hardware testing: launch and operate the full interactive mission](src/llm_vision_planner/HARDWARE_TESTING_README.md)
- [Hardware setup and preflight](src/llm_vision_planner/HARDWARE_SETUP.md)
- [Record Calibration Data](src/llm_vision_planner/Record%20Calibration%20Data.md)
- [Final V1 Transformer and calibration gateway](src/llm_vision_planner/fine_tuning/final_transformer_vertical.md)
- [Implementation report and evidence limits](src/llm_vision_planner/reports/FINAL_IMPLEMENTATION_REPORT.md)

## Runtime layout

| Component | Responsibility |
| --- | --- |
| `scripts/perception_detection.py` | Detection, ToF alignment and locally estimated obstacle depth |
| `scripts/obstacle_depth.py`, `outline_depth.py`, `outline_segmentation.py` | Geometric/outline depth, segmentation and fallbacks |
| `scripts/interactive_mission_gateway.py` | Mission state, grounding, approvals and planner coordination |
| `scripts/llm_planner.py` | Local structured Transformer waypoint inference |
| `scripts/refinment.py`, `scripts/verifier.py` | Route refinement and geometric verification |
| `scripts/verify_contraction.py` | Dataset residual quantiles, trajectory QPs and projected tube checks |
| `fine_tuning/scripts/mission_data_logger.py` | Independent sensor, trajectory and ACP JSONL logs |
| `launch/full_plot.launch.py`, `web/interactive.html` | Integrated ROS launch and interactive mission UI |
| `fine_tuning/scripts/calibration_verifier_gateway.py` | Offline HRRT/final Transformer comparison gateway |
| `launch/vision_error_calibration.launch.py` | Integrated capture/replay pipeline |

Paths in this table are relative to `src/llm_vision_planner`. Both offline and live planning use the Transformer. See the hardware guide for trajectory-level ACP, user-configurable delta, and independent publication logging.

## Assets retained

Only the selected final vertical V1 checkpoint metadata, tokenizer, training history, dashboard and pre/postflight reports are retained from Transformer experiments. The matching teacher-only 80/10/10 dataset and raw teacher data are distributed as lossless gzip archives. Checkpoint weights and unpacked training files remain local; see the restore guide and checksum manifest.

The only hardware raw capture dataset retained is `calibration_residual_raw_person_outline.csv`. Required historical controller scores live in `config/certificates`; synthetic fixtures live in `test/fixtures`. Neither is mislabeled as newly collected outline calibration.

Final interactive HTMLs include the mission UI, calibration gateway's generated UI, dataset viewers, the selected model dashboard and the numerical tube-audit pages. Older model weights, Transformer 2.0, LoRA output bundles, ablation scripts and comparison results are excluded from this curated snapshot; they remain recoverable on the research branches.

## Provenance

- Previous main: `5c5afd4e419c843f4b3a5119dea0041a4ac82c56`.
- Multiple-trajectory source: `531b28d29f2a7ec99e6c921f9511092978cbd03b`.
- Depth integration source: `a73618ae110952ddf97b16bbfa2a89ad8244d821` (`depth-outline-method`).

This is a selective snapshot commit on main, not a history merge importing every experimental commit. Original research branches and the original local workspace are preserved.
