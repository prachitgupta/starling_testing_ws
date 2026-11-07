# Historical runtime certificate

`calibration_min_control_qp_position_score_with_limits_2000.csv` is retained because the existing mission verifier depends on it. It contains historical controller calibration scores, not the new raw person-outline captures.

This file does not establish coverage for changed outline-depth perception or the selected V1 Transformer. Real deployment requires compatible independently validated calibration for the frozen end-to-end pipeline. Synthetic placeholder vision calibration is under `test/fixtures` and is not a real certificate.
