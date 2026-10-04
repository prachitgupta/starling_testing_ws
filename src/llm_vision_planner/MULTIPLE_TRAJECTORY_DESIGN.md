# Starling Multiple-Trajectory Worktree

This branch isolates the HRRT-star multiple-trajectory study from `main`,
`reporoduce_hardeware`, and `finding_expert`.

## Fixed design decisions

- Replace the single RRT/Semantic Theta expert with HRRT-star after the
  standalone HRRT-star implementation is verified.
- HRRT-star returns sparse fixed-altitude waypoint sequences.
- Candidate identity is the composite key `(H-signature, clearance-bin vector)`.
- Preserve at most one lowest-cost trajectory per composite key and apply a
  configurable global route cap.
- HRRT-star expert environments use ground-truth obstacle positions.
- Do not add perception-uncertainty obstacle inflation or an explicit safety
  tube to this worktree. Environmental and predictive uncertainty belong in
  the final consolidated calibration score.
- Sparse waypoint visualization uses collision-preserving piecewise-linear
  interpolation. Dynamic refinement occurs only after a candidate is selected.

## Interactive mission gateway redesign

The existing gateway, web UI, hardware capture, perception, and beacon/Vicon
files from committed `main` remain available in this worktree as source
material. The gateway will later be redesigned for two explicit modes.

### Adaptive conformal prediction data-collection mode

1. Natural-language input `L1` grounds the proposed goal.
2. The camera pipeline supplies the perceived labeled environment `E_hat`.
3. The beacon pipeline supplies the ground-truth environment `E`.
4. HRRT-star generates a finite, colored candidate family from `E`.
5. The UI displays the candidate trajectories over the live 2-D environment.
6. Natural-language input `L2` expresses the operator's route preference and
   selects one HRRT-star candidate.
7. The black-box LLM prediction is generated from its permitted perceived
   inputs; the expert target is the selected HRRT-star waypoint sequence.
8. The captured pair is labelled with one consolidated nonconformity score.

The precise consolidated-score equation and the exact black-box input contract
must be confirmed before this mode is implemented.

### Mission-execution mode

1. The operator supplies `L1` to ground and approve the goal.
2. The mission follows the execution pipeline without a second human route
   selection section.
3. The route-selection policy used in this mode must be fixed when the trained
   model interface is implemented.

## Hardware calibration-data scaffold

Reuse the takeoff/hover, perception, beacon/Vicon, and synchronized capture
components already present in this worktree. Each accepted sample is intended
to retain at least:

- `L1` and grounded goal;
- perceived labeled environment `E_hat`;
- synchronized beacon ground-truth environment `E`;
- HRRT-star candidate route cards and plotted route IDs;
- `L2` and the selected expert route ID;
- selected expert sparse waypoints;
- black-box LLM sparse waypoints;
- consolidated score and all terms needed to reproduce it.

## Fine-tuning-data scaffold

- Generate about 2,000 synthetic ground-truth environments.
- Generate a configurable five to six `L2` preference variants per environment;
  retain support for larger paraphrase sets during experiments.
- Generate the finite HRRT-star candidate family for every environment.
- Use the high-capacity teacher/selection procedure to map environment plus
  `L2` to one candidate.
- Fine-tuning input: serialized environment context plus natural-language `L2`.
- Fine-tuning target: deployment-visible reasoning followed by the selected
  HRRT-star sparse waypoint sequence. The reasoning uses the fixed
  `Preference/Geometry/Decision/Safety` structure and never exposes route cards.
- Support `dss` and a minimal SFT-compatible `dss_scott` dataset mode. The latter
  tests counterfactual rationale conditioning without a custom loss. DSS-SCOTT counterfactual routes
  remain in explicitly prefixed training-only tasks; validation, test, and
  deployment keep the same one-call reasoning-plus-waypoints contract.
- Split evaluation by both unseen environments and unseen language paraphrases.

## Scaffolded implementation files

- `fine_tuning/scripts/hrrt_star.py`
- `fine_tuning/scripts/generate_hrrt_acp_dataset.py`
- `fine_tuning/scripts/generate_hrrt_finetuning_dataset.py`
- `fine_tuning/scripts/finalize_hrrt_sft_dataset.py`
- `fine_tuning/scripts/train_hrrt_peft.py`
- `fine_tuning/scripts/evaluate_hrrt_adapter.py`

The complete commands and mode-specific data directories are documented in
`fine_tuning/README.md`.
