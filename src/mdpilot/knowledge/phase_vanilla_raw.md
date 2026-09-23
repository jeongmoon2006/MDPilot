=== PHASE `vanilla` — unbiased MD ===

Action space is `extend` or `stop`, and — only when `switch_to_metad` appears
in the `decision` enum of your tool — `switch_to_metad`:

- `extend` — run more vanilla MD.
- `stop` — the observable is adequately sampled for the task; the campaign
ends.
- `switch_to_metad` — vanilla MD is inadequate for the transition the task
requires within its budget. Propose a collective variable for metadynamics.

`switch_to_metad` is offered only when `task_expectation` is non-null. When it
is absent, do not argue for it.

Report fields, all in `precision`: `mean`, `sem_blocked`, `sem_naive`,
`statistical_inefficiency_block`, `statistical_inefficiency_autocorr`,
`tau_int_frames`, `ess` (effective number of independent samples),
`bimodality_coefficient` (Sarle's coefficient of the observable's marginal),
`n_basins`, `minor_basin_occupancy`. The unbiased phase computes no
free-energy answer, so `correctness.falsifiers` is empty here.

Decide from these numbers whether the observable is adequately sampled, and
whether unbiased MD is reaching the states the task names within the budget
stated in `task_expectation`. State in `reason` the criteria you applied.
