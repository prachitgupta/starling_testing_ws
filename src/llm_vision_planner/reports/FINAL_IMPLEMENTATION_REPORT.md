# Final implementation report

## Scope and provenance

This snapshot combines the multiple-trajectory implementation at 531b28d with the depth-outline integration at a73618a, on top of main at 5c5afd4. Source research branches are preserved. Selected local final-model artifacts are added without publishing weight binaries.

## Perception integration

The colleague's implementation replaces reliance on a language model for obstacle depth with local geometric estimation. Perception calls the outline/ToF modules, publishes compatible obstacle information, and the mission gateway consumes perception-derived nominal depth. The launch/config wiring and capture/replay tests are retained. EfficientSAM ONNX weights are optional local assets; without them the configured shape-fitting fallback is not equivalent evidence of neural segmentation accuracy. The model placeholder/download instructions are retained under models.

## Two distinct execution paths

1. Live hardware: perception → mission gateway → existing Llama/vLLM/ChatGPT planner → refinement/verification → controller/mission execution.
2. Offline inspection: recorded capture → HRRT reference and final V1 Transformer → refinement/QP comparison → calibration gateway visualization.

The final V1 Transformer has NOT been integrated as a live ROS planner backend by this cleanup. Adaptive conformal and Mission execution are web-interface modes of the live mission. Environment uncertainty and Planner only are the offline gateway's separate E-hat/E experiment controls.

## Model and dataset selection

The final vertical teacher-only V1 checkpoint was selected at epoch 67; stored best Euclidean loss is 0.2772914745977947. Stored verification results include a pass fraction around 0.86458 and an infinite safety radius, so small prediction loss is not a certificate of a small or valid safety tube. No claim of superiority over all alternatives is established by this integration.

The corresponding training/validation/test CSVs and raw teacher JSONL are kept in lossless compressed form. The retained hardware calibration raw dataset is the person-outline CSV: 59 captures with 8 misses. Misses must remain part of failure accounting, not be silently removed to improve reported coverage.

## Known safety-tube limitations — not changed here

- The Transformer backend does not encode the failed-trial feedback into its next input. Recalling it on the same scene is not feedback-conditioned replanning.
- Environment uncertainty uses E-hat for Transformer/goal grounding and E for HRRT; Planner only uses E throughout.
- The current interactive gateway assigns a current-scene score to its tube-quantile field. It does not automatically build a calibration distribution or obtain human preference labels for all captures when launched.
- A controller residual score, a state-metric tube radius, and its XY projection are different quantities. Low waypoint Euclidean error does not imply a small control-residual tube. The saved audit exposes a score around 6.64686 versus a separately calculated XY projection around 2.71995 for its specific example; read its JSON equations/parameters, not a universal conversion.
- A conformal claim requires one well-defined score per independent calibration example, a frozen pipeline/domain, retained failures, and a stated finite-sample order statistic. The UI alone establishes none of these.
- Archived equation PDFs are reference material, not proof that every displayed gateway quantity implements those equations.

These are documented limitations, not silently repaired in a branch-curation change.

## Reports and interactive artifacts

The selected-model dashboard/training history/preflight/postflight outputs are under fine_tuning/outputs/final-transformer-vertical-teacher_only_80_10_10. The November 4 numerical gateway audit, reproduction script, HTML/SVG views and JSON results are under output/gateway_radius_audit_20261104. Original saved absolute paths in report JSON are provenance, not portable launch paths. Web/interactive.html and the dataset viewers remain available.

## Validation boundary

Synthetic checks and software integration checks are recorded in VALIDATION.md. No real flight, new hardware bag/Vicon trial, new calibration certificate, or empirical neural-segmentation accuracy claim is made by this snapshot. A responsible operator must validate the complete frozen hardware pipeline before deployment.
