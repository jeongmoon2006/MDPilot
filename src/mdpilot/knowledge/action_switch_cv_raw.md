--- Action `switch_cv` (available this round) ---

`switch_cv` replaces the biased collective variable and starts a fresh bias on
the new coordinate. The deposited hills on the old CV are kept as a record but
are not carried over. The walker itself carries over. The compute already
spent is not refunded: the biased budget is cumulative across CVs, so a switch
late in a campaign buys little. Use it when you judge the coordinate to be
*wrong* — it does not separate the task's states, or the walker left the
region it started in and the numbers say it is not coming back — rather than
when the surface is merely still filling. `cv_switches_remaining` says how
many revisions are left.

Do not mistake `rmsd`, `distance` or `gyration` for bounded; they are unbounded
above, and what makes `rmsd` usable is the wall the campaign configures.
Justify the replacement in `reason` by naming what the previous CV failed to
do, and record the diagnosis in `ledger_note`.
