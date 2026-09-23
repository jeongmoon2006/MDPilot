--- Action `switch_cv` (available this round) ---

`switch_cv` replaces the biased collective variable and starts a fresh bias on
the new coordinate. The deposited hills on the old CV are kept as a record but
are not carried over — they describe a different coordinate. The walker itself
carries over: the new bias starts from where the system is now, not from the
cached start. The compute already spent is *not* refunded: the biased budget
is cumulative across CVs, so a switch late in a campaign buys little. Use it
when the evidence says the coordinate is *wrong*, not when the surface is
merely still filling:

- `occupancy_invariance` is `refuted` and its `inputs` show the walker never
separated the states on this coordinate: `recrossings` stayed at 0 or the
count was taken between boundaries that sit on the same side of the task's
states, while `fes_depth_kj_per_mol` kept rising. The bias is filling a basin
it cannot escape along this coordinate, and nothing deposited on it is worth
keeping.
- `state_definition_invariance` is `refuted`: the states are not separated by
a barrier on the campaign observable, so no bias on this coordinate can
produce a ΔG that means what the task asks.
- the walker left the region it started in — compare `cv_start` against
`cv_min`/`cv_max` — and `ns_since_*_visited` for the starting state has grown
past the tolerance with no sign of return.

Judge absence in nanoseconds, not rounds — rounds are whatever length the
schedule makes them, and `occupancy_invariance.tolerance` is in ns. A
bounded coordinate does not protect you: a contact count collapses every
disordered conformation onto roughly the same value, so once the system is
disordered the bias fills one degenerate bin and cannot lead it back. Prefer
a replacement that separates the states on the side you are stuck in.

`cv_switches_remaining` says how many revisions the campaign has left. When it
reaches 0 the action disappears from your tool; spend one on a coordinate you
have evidence against, not on a surface that is merely still filling.

Prefer a coordinate that is bounded on both sides when the failure was a
walker that left and did not come back — but do not mistake `rmsd`, `distance`
or `gyration` for bounded. They are unbounded above; what makes `rmsd` a usable
replacement is the upper wall the campaign configures, not the coordinate
itself. Say which it is in `reason`; claiming a coordinate is bounded when it
is not is how the previous CV was chosen. Justify the replacement in `reason` by
naming what the previous CV failed to do, and record the diagnosis in
`ledger_note` — the next rounds will be judged on a different coordinate and
the history has to explain the discontinuity.
