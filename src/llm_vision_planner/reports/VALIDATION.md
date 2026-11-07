# Curated integration validation — 2026-11-06

Validated in the isolated `starling_final_testing` checkout, without changing the original research checkout.

| Check | Result |
| --- | --- |
| `colcon build --packages-select llm_vision_planner --symlink-install` with ROS Humble sourced | Passed |
| Installed `full_plot.launch.py --show-args` | Passed; documented arguments available |
| `colcon test --packages-select llm_vision_planner` | 7 tests, 0 errors/failures/skips |
| Gateway environment-contract pytest tests | 2 passed |
| Unified offboard ROS synthetic harness | Passed |
| Python compilation of scripts, launch and fine_tuning/scripts | Passed |
| Actual selected V1 checkpoint + HRRT numerical audit replay | Passed; saved JSON/HTML/SVG refreshed |

The seven registered tests cover web UI, mission gateway, ROS gateway, residual calibration, obstacle depth, outline depth and synthetic calibration replay. The offboard harness uses synthetic messages; it is not a real flight. Outline tests use synthetic geometry/masks rather than proving EfficientSAM accuracy on real imagery.

The audit replay reproduced an unchanged retry prediction (maximum coordinate difference 0), environment-uncertainty score 6.64685694185239 and hypothetical metric-derived XY radius 2.719945689969702. The displayed gateway score is not thereby a distribution-calibrated radius. The planner-only example produced score 6.50418205863837.

Two obsolete test files, `test_conformal_obstacle_safety.py` and `test_expert_calibration_pipeline.py`, referenced earlier calibration/semantic-Theta modules absent from the source branch. They are excluded from this final snapshot, not reported as passing; historical versions remain on research branches. Initial standalone test attempts without sourced ROS/Python paths failed imports; the successful checks above used the proper ROS environment.

Not performed: clean-machine dependency installation, real flight, new hardware/Vicon bag validation, newly calibrated safety coverage, or external planner-service accuracy evaluation. The package build is not a full clean-machine ROS dependency build.
