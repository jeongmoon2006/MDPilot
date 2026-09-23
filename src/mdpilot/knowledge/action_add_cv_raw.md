--- Action `add_cv` (available this round) ---

`add_cv` keeps the coordinate(s) already biased and biases the proposed one
*in parallel* with them (parallel-bias metadynamics). The hills already on
the current coordinates are kept. It shares the revision allowance with
`switch_cv`, and the set is capped at three coordinates.

Choose `switch_cv` when you judge the coordinate *wrong* (it never separated
the states) and `add_cv` when you judge it *insufficient* (it separates the
states but cannot bring the system back, so the barrier controlling the return
lies along some other coordinate). Add the coordinate you believe carries
that barrier, not a second global measure of the same progress. Say in
`reason` which case the numbers support, and record in `ledger_note` what the
added coordinate is meant to carry.
