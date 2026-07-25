# Invalidated experiment results

Results kept for audit trail. They are NOT used by any downstream stage.

## search_lightgbm_INVALID_min_child_weight_zero.json
First LightGBM Stage-1 search. **Scientifically invalid.** The search space
included `min_child_weight = 0.0`, which makes LightGBM permit degenerate
zero-count splits and abort with
`Check failed: (best_split_info.left_count) > (0)`.
11 of 14 sampled configurations crashed, so only 3 were actually scored and
the good region of the space (low learning_rate + relaxed min_child_samples)
was never sampled. Its reported `best_f1_macro_mean = 0.2927` is therefore an
artifact of a broken search, not a property of LightGBM.
Superseded by a re-run with the corrected space.
