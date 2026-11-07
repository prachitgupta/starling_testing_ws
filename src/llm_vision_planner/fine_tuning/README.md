# Final Transformer assets

Use [final_transformer_vertical.md](final_transformer_vertical.md) for the selected V1 checkpoint, local asset restoration and gateway launch.

Retained data: the matching teacher-only 80/10/10 split and raw teacher JSONL (gzip archives), plus the raw person-outline calibration captures. Synthetic inputs are in `../test/fixtures`; historical controller scores are in `../config/certificates`.

The V1 training implementation remains because the production checkpoint loader imports its architecture. HRRT, refinement, QP and recording/postprocessing helpers remain dependencies of the final workflow. Older Transformer variants, LoRA bundles and ablation/verification-study tooling are maintained on the research branches, not here.

The selected model dashboard is in `outputs/final-transformer-vertical-teacher_only_80_10_10/dashboard/index.html`. The corresponding reports and checkpoint metadata are in the same bundle. Local weights are intentionally ignored by Git.
