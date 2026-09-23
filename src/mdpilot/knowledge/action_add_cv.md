--- Action `add_cv` (available this round) ---

`add_cv` keeps the coordinate(s) already biased and biases the proposed one
*in parallel* with them (parallel-bias metadynamics: one low-dimensional bias
per coordinate, deposited simultaneously). The hills already on the current
coordinates are kept. It shares the revision allowance with `switch_cv`, so
`cv_switches_remaining` counts both, and the set is capped at three
coordinates.

Choose between the two by what the falsifiers say about the current
coordinate:

- `switch_cv` when the coordinate was *wrong*: its boundaries never separated
the task's states, or `recrossings` was counted between bands that sit on the
same side of them. Nothing deposited on it is worth keeping.
- `add_cv` when the coordinate was *insufficient*: it separates the states —
the walker crossed on it, `recrossings` is non-zero on the task basis — but
cannot bring the system back: `occupancy_invariance` is `refuted` on
`ns_since_*_visited` for the starting state while `fes_depth_kj_per_mol`
keeps rising, or `time_window_invariance` is `refuted` because ΔG keeps
walking in one direction. The barrier that gates the return is then along
some *other* coordinate the bias never touches. Add the one you believe
carries it — the coordinate the transition state is organized around, not a
second global measure of the same progress that will collapse the same
conformations onto the same value.

Say in `reason` which of the two cases the evidence supports, and record in
`ledger_note` what the added coordinate is meant to carry, so the rounds that
follow can judge whether it did.
