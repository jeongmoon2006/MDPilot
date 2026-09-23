=== PHASE `metad` — well-tempered metadynamics ===

Action space is `extend` or `stop`, and — only when they appear in the
`decision` enum of your tool — `switch_cv` and `add_cv`. When they are absent
the campaign has spent its CV-revision allowance; do not argue for them.
Proposing another *pivot* is never available.

The equilibrium convergence fields are deliberately absent from this report. A
biased trajectory is not an equilibrium ensemble, so they do not describe it.
Do not ask for those numbers or reason as if you had them.

`validity` — `biased_cvs` (the coordinate(s) currently biased; after `add_cv`
several, and `cv_ranges` gives the range each covered), `cv_label` (the
primary coordinate the surface fields describe), `cv_start`, `cv_min`,
`cv_max` (the range the walker visited over the *whole* biased phase —
cumulative), `n_fes_estimates`.

`precision` — numbers from the deposited bias integrated into a free-energy
surface: `fes_drift_kj_per_mol` (how much the surface changed between the
half-way cumulative estimate and the latest one), `recrossings` (transitions
counted with hysteresis between `recrossing_low` and `recrossing_high` on
the task's states, on `recrossing_observable`; `null` means it could not be
counted), `n_basins_fes`, `barrier_kj_per_mol`, `fes_depth_kj_per_mol`.

`correctness.answer` — `delta_g_low_minus_high_kj_per_mol` on the campaign
observable, integrated over the task's two states, reweighted from COLVAR.
Positive means the high state is the more stable one.

`correctness.falsifiers` — each recomputed the answer under a transformation
it should be invariant under; read `statistic`, `magnitude` and `inputs`:

- `seed_invariance` — a second walker started in the other state; `inputs`
says how many walkers there were.
- `estimator_invariance` — the hills marginal vs the reweighted surface.
- `time_window_invariance` — ΔG from the last half of the biased phase vs all
of it.
- `occupancy_invariance` — how long, in biased nanoseconds, the walker has
been away from the state it started in (`start_state`), and how long it has
sat entirely inside one state; `inputs` also carries this round's own
observable range and the per-round counts.
- `state_definition_invariance` — ΔG with each threshold moved by half a
band.
- `bias_accounting_invariance` — ΔG with and without the frames a wall was
acting on.

Decide from these numbers whether the surface has stopped moving, whether the
walker is still making the transitions the task requires, and whether the
answer survived the transformations. State in `reason` what you judged each
number against. `stop` when you judge the campaign done; `extend` otherwise,
unless you judge the biased coordinate itself to be the problem — see the
`switch_cv` / `add_cv` sections when they are offered.
