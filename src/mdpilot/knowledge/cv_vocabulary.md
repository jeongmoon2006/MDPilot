Populate `metad_proposal` when you choose the action that (re)defines the
biased coordinate: `switch_to_metad` in the vanilla phase, `switch_cv` or
`add_cv` in the biased phase. Your tool's `decision` enum says which actions
are available this round; a vanilla round never offers the biased-phase
revisions and a biased round never offers the pivot.

- `cv_type` — one of `distance`, `torsion`, `gyration`, `rmsd`, `contacts`.
Pick the type that matches the physical coordinate you believe is slow. Note
that `distance`, `gyration` and `rmsd` are unbounded above, so a bias on them
can drive the system into an ever-larger disordered space and never return;
`torsion` and `contacts` are bounded on both sides, which removes that
failure mode.

Bounded is not the same as reversible. A coordinate that collapses many
distinct conformations onto roughly the same value — a contact count does
this for every disordered structure — cannot tell those states apart once the
system is there, so the bias fills one degenerate bin without ever pushing
the system back. The `occupancy_invariance` falsifier is what detects that
after the fact; choosing the coordinate is what prevents it. Ask not only
whether a coordinate is bounded but whether it distinguishes states on *both*
sides well enough to lead the system back.
- `selections` — MDTraj selection strings. Arity is type-specific:
- `distance`: 2 selections, each must resolve to exactly 1 atom. Example:
`["name CA and resSeq 1", "name CA and resSeq 10"]`.
- `torsion`: 4 selections, each must resolve to exactly 1 atom. Example:
`["resSeq 2 and name N", "resSeq 2 and name CA", "resSeq 2 and name C",
"resSeq 3 and name N"]`.
- `gyration`: 1 selection, must resolve to ≥2 atoms. Example: `["backbone and
resSeq 1 to 10"]`.
- `rmsd`: 1 selection, must resolve to ≥3 atoms. RMSD to the campaign's
reference structure after optimal superposition — the usual folding order
parameter. Example: `["name CA"]`.
- `contacts`: 1 selection, must resolve to ≥2 atoms. The *fraction* of native
contacts formed among the selected atoms, on [0, 1]: 1 is the reference
structure's full set, 0 is none. Every number you see for it — the biased
coordinate, `cv_min`/`cv_max`, the free-energy axis — is that fraction, not a
count. Pairs closer than 3 residues in sequence are excluded, since those are
formed in any conformation.
Example: `["name CA"]`. For a structural transition this is usually a better
coordinate than `rmsd`: it measures how much of the reference structure is
present rather than how far the whole chain has moved, and because it is
bounded the disordered side cannot run away.
- `label` — short snake_case identifier (e.g. `rg_back`, `d_term`). It MUST
name the coordinate you are actually biasing. Do not name it after a
coordinate you would have preferred but did not select: the label is written
into plumed.dat, HILLS, COLVAR and every downstream report, and a label that
misdescribes the CV makes the run unreadable afterwards. If you want RMSD,
choose `cv_type: rmsd` — do not call a `distance` an rmsd.

Bias parameters (sigma, height, pace) are *not* your concern — a deterministic
helper computes them from the prior trajectory.

When not switching, `metad_proposal` must be null.
