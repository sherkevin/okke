# CHORD Theory Locks

These invariants are mandatory and are enforced by `tests/test_chord_theory_locks.py`.

## Fused score

CHORD must fuse scores exactly as:

`S_CHORD(c) = S_opera(c) + lambda_cur * V_anchor(c) + lambda_fut * F_future(c)`

If `lambda_cur = 0` and `lambda_fut = 0`, the result must reduce exactly to OPERA.

## Current signal

The first implementation is enhance-only:

- matched-anchor visual tokens are strengthened
- anchored but unmatched visual tokens stay unchanged
- unanchored visual tokens stay unchanged

The visual-token weight is:

`w_j = 1 + alpha_anchor * max_i(M_ij * r_i * s_i)`

Detector failure or query-matching failure must force `w_j = 1` for every visual token.

## Future signal

Future scoring must use sums, not means:

- `V_tau(c) = sum_j w_j * attn_tau(c -> v_j)`
- `T_tau(c) = sum_q attn_tau(c -> t_q)`
- `F_future(c) = sum V_tau(c) - lambda_txt * sum T_tau(c)`
- `R_future(c) = sum V_tau(c) / (sum V_tau(c) + sum T_tau(c) + eps)`

`T_tau(c)` must include all non-image tokens in the branch prefix, including generated continuation tokens.

If rollout fails numerically or structurally, CHORD must continue and set `F_future = 0`.

## Rollback precedence

Rollback remains higher priority than CHORD reranking:

- rollback is decided by preserved OPERA state
- if rollback triggers, precomputed CHORD scores on invalidated states are discarded
- CHORD scoring may only resume after rollback state restoration
