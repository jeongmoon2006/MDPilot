=== PHASE `metad` — well-tempered metadynamics ===

Action space is `extend` or `stop`, and — only when they appear in the
`decision` enum of your tool — `switch_cv` and `add_cv`. When they are absent
the campaign has spent its CV-revision allowance and the choice is not yours
to make; do not argue for it. Proposing another *pivot* is never available:
the campaign has already pivoted.

The equilibrium convergence fields are deliberately absent from this report. A
biased trajectory is not an equilibrium ensemble — the bias drives the
observable — so a long autocorrelation would mean the bias is still filling
and a bimodal marginal would mean the bias worked, not that the system is
sampling freely. Do not ask for those numbers or reason as if you had them.

`validity` — bookkeeping about the run: `biased_cvs` (the coordinate(s)
currently biased; after `add_cv` several are biased in parallel and
`cv_ranges` gives the range each covered), `cv_label` (the primary
coordinate the surface fields describe — the campaign observable's own
marginal when it is among the biased ones, otherwise the first), `cv_start`,
`cv_min`, `cv_max` (the range the walker visited over the *whole* biased
phase — cumulative, so they go on reporting the widest excursion the
campaign ever made long after the walker has stopped moving), and
`n_fes_estimates`.

`precision` — has the surface stopped moving, all from the deposited bias
(HILLS) integrated into a free-energy surface:

- `fes_drift_kj_per_mol` — how much the surface changed between the half-way
cumulative estimate and the latest one. The standard well-tempered test,
taken over a gap long enough to move.
- `recrossings` — transitions counted with hysteresis between
`recrossing_low` and `recrossing_high` on the task's own states
(`recrossing_basis=task_states`), measured on `recrossing_observable`, which
is usually *not* the CV you are biasing. Fixed for the whole campaign, so the
count is comparable across rounds and across a change of CV. `null` means
the count could not be taken at all; that is not zero crossings.
- `n_basins_fes`, `barrier_kj_per_mol`, `fes_depth_kj_per_mol` — shape of the
surface so far; depth is measured over the visited range only.
- `gate` — true only when drift is below kT (≈2.5 kJ/mol at 300 K) AND
`recrossings >= min_recrossings`, which the report states alongside it: 1
accepts a one-way crossing, 2 requires a full round trip so the reverse
barrier is sampled too. Low drift *alone* is not stability: a walker that
never left its starting basin produces a surface that stops changing
immediately. A true gate says the estimate is stable. It does not say the
estimate is the one intended — that is the correctness block's question.

`correctness.answer` — `delta_g_low_minus_high_kj_per_mol` on the campaign
observable, integrated over the task's two states, reweighted from COLVAR.
Positive means the high state is the more stable one.

`correctness.falsifiers` — the fixed set, every round. Read each one's
`state`, then `magnitude` against `tolerance`, then `inputs`:

- `seed_invariance` — a second walker started in the other state must reach
the same ΔG. `not_applicable` on a single-walker campaign: hysteresis is
untested, and you should say so in `ledger_note` when it matters.
- `estimator_invariance` — the hills marginal and the reweighted surface must
agree, when the observable is among the biased coordinates.
- `time_window_invariance` — ΔG from the last half of the biased phase must
agree with ΔG from all of it. Refuted means the answer is still walking.
- `occupancy_invariance` — under a bias that has flattened the surface the
walker keeps visiting both states. Its `inputs` carry the per-round view the
cumulative ranges cannot give you: `observable_min_this_round` /
`observable_max_this_round`, `confined_to_state` and `rounds_confined` (a
walker parked entirely inside one state), and `ns_since_low_visited` /
`ns_since_high_visited` (a walker roaming without returning to the state it
started in — `start_state`). Its `tolerance` is in nanoseconds and comes from
the task file. Refuted means the biased coordinate cannot bring the system
back; the bias is filling a region the coordinate cannot lead it out of.
- `state_definition_invariance` — ΔG must be stable when each threshold is
moved by half a band. Refuted means the states are not separated by a barrier
on this observable.
- `bias_accounting_invariance` — with the wall's bias accounted for, ΔG must
not depend on whether frames beyond the wall are included. `not_applicable`
without a wall.

`correctness.summary` — `refuted`, `not_evaluable`, `not_applicable` (lists of
names) and `not_refuted` (nothing refuted, nothing not_evaluable).

Decision rule:

- `precision.gate=true` AND `correctness.summary.not_refuted=true` → `stop`.
The loop enforces this: a `stop` against either is converted to an extend
and the refusal is written to your ledger.
- otherwise → `extend`, unless a falsifier says the coordinate itself is the
problem — see the `switch_cv` / `add_cv` sections when they are offered.
This includes `gate=null` and every `not_evaluable`.
- A `refuted` falsifier names what is wrong in its `note`; quote it in
`reason` and record the diagnosis in `ledger_note`. If you cannot act on it
(no revision left), say so — a human reading the ledger can.
- Before treating `recrossings` as evidence about your task's transition,
check `recrossing_low` and `recrossing_high` against the states the task
describes.
