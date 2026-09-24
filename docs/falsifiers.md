# Falsifiers — design note (Step 1 of the validity / precision / correctness reframing)

Status: **Steps 1–5 landed 2026-09-22** (§13–§15 say exactly what is in the
code). §15.3 holds the first measurements; the full-length fault runs are
not done.

The reframing this note adopts, stated once so every later section can lean
on it:

> Without an external reference you cannot confirm that a computed quantity
> is the intended one. You can only refute it. Every correctness check is
> therefore a **falsifier**: a transformation under which the answer must be
> invariant, plus a tolerance. It *fires* when the answer is not invariant. A
> falsifier that does not fire means **not refuted** — never "correct".

---

## 1. Three categories, and why they must not overlap

| category | question | epistemic status | who decides |
|---|---|---|---|
| **VALIDITY** | did the simulation do what it claimed, mechanically? | hard signal: true or false, no tolerance to argue about | code |
| **PRECISION** | has the estimate stopped moving, and how far would it move on a repeat? | deterministic computation with a stated tolerance | code |
| **CORRECTNESS** | is the quantity we computed the quantity we intended? | cannot be confirmed from inside; can only be refuted | code refutes; nobody confirms |

The asymmetry between the last row and the other two is the point.
Validity and precision failures are visible in the numbers themselves: a NaN,
a drift of 30 kJ/mol. A correctness failure is invisible in the numbers — the
surface converges, the count is non-zero, every field is finite — and is only
visible as a *disagreement* between two ways of computing the same thing that
should have agreed. That is why correctness checks take the form they do.

The current code and prompts blur these. §4.4 of `overview.md` calls three
different things "correctness"; `fes_converged` is read by the loop as
permission to stop, which makes a precision statistic carry a correctness
verdict; `check_observable_scale` is a validity check filed under pre-flight
with no category at all. Step 2 audits this; Step 3 regroups the report.

### Vocabulary that the codebase must adopt

- `refuted` / `not_refuted` / `not_evaluable` — the three states of a
  falsifier. `not_evaluable` (the transformation could not be applied: no
  second walker yet, no frames past the wall, fewer than two surface
  estimates) is **not** `not_refuted`. Acceptance requires every falsifier in
  the fixed set to be `not_refuted`; a `not_evaluable` blocks acceptance the
  way a `null` recrossing count already withholds `fes_converged`.
- `verified` — reserved for one thing: a match against the external
  reference at evaluation time (Layer 3 of the verdict). It is never produced
  inside the loop. "Not falsified" and "verified" are different words and
  stay different.
- `converged` — a precision word. It means "stopped moving", nothing more.
- **Banned:** `correct`, `right`, `valid` as a *result* predicate in any
  function name, field name, log line, prompt chunk or docstring. (`validity`
  as a category name is fine; "the surface is correct" is not.) `verdict`
  stays, because the existing `PASS / INCOMPLETE / FAIL` semantics stay.

---

## 2. The form of a falsifier

Every falsifier is one record with the same fields, whatever family it is in:

```
{
  "name":            "seed_invariance",
  "invariance":      "ΔG(low−high) on the campaign observable is independent of where the walker started",
  "transformation":  "second walker started in the other task state, same bias definition, independent hills",
  "statistic":       "|ΔG_A − ΔG_B|",
  "magnitude":       3.1,            # in the statistic's own units (kJ/mol here)
  "tolerance":       2.49,           # 1 kT at 300 K; §8 says where each number comes from
  "state":           "refuted",      # refuted | not_refuted | not_evaluable
  "cost":            "2x biased ns", # what it took to evaluate
  "window":          "cumulative"    # what data it was evaluated over
}
```

`fired` in the task description maps to `state == "refuted"`; the tri-state
is kept because `not_evaluable` has to be distinguishable from silence.

Rules:

1. **Fixed set per method family, in code.** The well-tempered metadynamics
   family gets the set in §5; a future umbrella/REMD family gets its own.
   The LLM never selects, orders, weights or suppresses a falsifier. It reads
   the block. (§9 says which one I would have made adaptive and why it is
   not.)
2. **Every falsifier is expressible as invariance + tolerance.** A check that
   cannot be written that way is a validity check or a precision statistic,
   not a falsifier, and is filed there.
3. **Magnitude is always reported**, even when `not_refuted`. A magnitude of
   0.9 kT is not the same evidence as 0.1 kT, and the ablation harness (Step
   4, arm B) needs the raw number without the flag.
4. **A falsifier that costs GPU time must be justified against its coverage**
   (§7). Today exactly one does.

---

## 3. What is a validity check, not a falsifier

For Step 3's `validity` block. These are hard signals, most of which exist
already in some form; listed so the boundary is explicit.

| check | signal | exists today? |
|---|---|---|
| integrator health | NaN energy or coordinates; PLUMED abort; `plumed sum_hills` non-zero exit | partially (exceptions propagate; no field) |
| thermodynamic state | mean T within ±3 K of target over the round; mean P within noise of target; box volume drift < 1 % between rounds after equilibration | no |
| periodic image contact | minimum heavy-atom distance between the solute and its nearest image ≥ nonbonded cutoff, every frame (F11) | measured post hoc for F11, not in the loop |
| walker inside the intended ensemble | fraction of frames past an upper wall; hill centre inside the PBMETAD grid (a fatal PLUMED error today, so it is "did we crash") | no |
| observable evaluable | series finite, non-constant, on the scale the thresholds are on (F12, F14) | `check_observable_scale`, one direction only |
| structure as declared | residue count vs description (F12) | `check_residue_count` |
| bias definition as intended | SIGMA floored/ceiled flags (F4), wall present for length CVs (F6), RESTART read the hills it should have | flags exist on the single-CV path only |

None of these has a tolerance a scientist would argue about, and none says
anything about whether ΔG is the intended ΔG. They say whether the trajectory
is worth analysing.

---

## 4. What is precision, not correctness

For Step 3's `precision` block — the existing convergence statistics, which
stay exactly as computed:

- vanilla: `sem_blocked`, `plateau_reached`, `statistical_inefficiency_*`,
  `tau_int_frames`, `ess`, `well_sampled`;
- biased: `fes_drift_kj_per_mol`, `n_fes_estimates`, `fes_converged`,
  `recrossings` and its band, `n_basins_fes`, `barrier_kj_per_mol`,
  `fes_depth_kj_per_mol`.

One reclassification matters. `fes_converged = drift < kT AND recrossings ≥
min_recrossings` is a precision gate — it says the estimate stopped moving
and the walker crossed enough times to have sampled both sides. It currently
also functions as the loop's *permission to stop*. Under the reframing, stop
permission requires precision **and** every falsifier `not_refuted`. That is
the one behavioural change this note implies for the loop, and it is the
whole reason to do this.

Where precision meets correctness — `exploring` / bimodality on the vanilla
side — is discussed in Step 2, not here; it is a *decision input* for the
pivot, not a test of the biased result.

---

## 5. The falsifiers — well-tempered metadynamics family

Five families, as specified, plus one reframing of the existing trap
signals so they stop being a list of patches. For each: invariance,
transformation, statistic, tolerance and its source, cost, which recorded
failure it would have caught, and what it cannot catch.

Throughout, ΔG means `delta_g_low_minus_high_kj_per_mol` on the campaign
observable, integrated over the task's two states (§4.4 of `overview.md`).
kT = 2.49 kJ/mol at 300 K.

### (a) `seed_invariance` — initial condition / seed

- **Invariance.** ΔG does not depend on which task state the walker started
  in, or on the seed.
- **Transformation.** A second walker, B, started in the *other* task state
  from walker A, under the same bias definition (same CV set, SIGMA, HEIGHT,
  PACE, γ, wall), with its own independent HILLS and COLVAR. Not a
  multiple-walker shared bias: sharing hills couples the two estimates and
  turns the test into a test of the reweighting only.
- **Statistic.** |ΔG_A − ΔG_B|, each from its own reweighted F(observable)
  over its own cumulative biased phase. Secondary: rms(F_A − F_B) over the
  range both sampled, after re-zeroing each to its own minimum — the same
  comparison the reference match already makes.
- **Tolerance.** 1 kT on ΔG, 1 kT rms on the surface. Source: the reference
  match criterion already uses 1 kT for the same two statistics, and a
  falsifier should not be laxer than the yardstick it feeds. Provisional; §8.
- **Cost.** 2× the biased budget. Minimum walkers: **two** — one per task
  state. A third walker adds a seed repeat of one state and would let the
  tolerance be estimated rather than assumed, but doubles the cost again;
  not proposed for v1.
- **Where walker B's starting structure comes from.** This is the practical
  problem. The task gives one structure, in one state. Options, in order of
  preference: (1) the first frame of walker A's biased trajectory whose
  observable is inside the other state, taken the round it first appears —
  cheap, mechanical, and B then starts one to two rounds after A; (2) a short
  high-temperature unfolding pre-run at pivot time, ~2–5 ns, extra cost and a
  structure from a different ensemble; (3) a user-supplied second structure
  in the task file, which is the honest option for a binding campaign
  (bound/unbound are both known). Proposal: implement (1) with (3) as an
  optional task-file field; never (2). Note that under (1) the two walkers
  are not independent in the strict sense — B's start is a frame A visited
  — but B's *bias* is independent, which is what the test needs.
- **Would have caught.** F4 (hills too narrow: neither walker leaves its
  state, ΔG_A and ΔG_B disagree by the whole basin depth). The "γ too small"
  injected fault, for the same reason. Any hysteresis: a hidden slow degree
  of freedom that lets each walker stay where it began.
- **Cannot catch — and this is the important limit.** A **shared absorbing
  state.** If the coordinate cannot lead the system *back* from one state,
  both walkers end up in that state: A gets driven there, B is already there,
  and their ΔG values agree. F6, F13 and F15 are all of this kind — every
  CLN025 failure on record was a walker that unfolded and never refolded —
  and two walkers would both have reported the unfolded ensemble as stable.
  For those, (c) and the occupancy falsifier below are the ones that fire.
  Seed invariance is the strongest test for hysteresis, not for absorption.
  It is also blind to any error both walkers share by construction: the
  force field, the box (F11), a wrong observable (F12, F14).

### (b) `estimator_invariance` — two estimators, one quantity

Two sub-checks, evaluated whenever their inputs exist.

**(b-i) hills marginal vs reweighted.** Applies when the biased CV *is* the
campaign observable (or, under PBMETAD, when the observable is one of the
biased set — each parallel bias converges to its own marginal).

- **Invariance.** F(observable) from `sum_hills` on that CV's HILLS and from
  reweighting COLVAR by exp(V/kT) are two estimators of the same function and
  must agree.
- **Statistic.** |ΔG_hills − ΔG_rw| and rms over the range both cover.
- **Tolerance.** 1 kT on ΔG, 1 kT rms. Source: measured 1.75 kJ/mol rms over
  the last half of the one campaign where this comparison was possible (D10),
  i.e. inside 1 kT when the run was healthy.
- **Cost.** None; both are already computed every biased round.
- **Would have caught.** F7 (depth measured over grid padding: the reweighted
  surface has no padding, so the two disagree exactly there). F14 (the
  observable normalised twice: the hills marginal is on the fraction PLUMED
  biased, the reweighted one on the doubly-divided series — they disagree by
  a factor of the pair count, which is not within any tolerance).
- **Cannot catch.** Anything both estimators inherit from the same
  trajectory: an absorbing state, a wrong CV, a too-small box. When the
  biased CV is *not* the observable it is `not_evaluable`, which after a
  `switch_cv` to RMSD is most of the time.

**(b-ii) with vs without the c(t) offset.**

- **Invariance.** ΔG from instantaneous-bias reweighting and from
  well-tempered reweighting with the time-dependent offset c(t) must agree
  once the bias has stopped growing.
- **Statistic.** |ΔG_inst − ΔG_c(t)|.
- **Tolerance.** 1 kT. Source: on a 1-D model at convergence the two differed
  by 0.01 kJ/mol; in a one-way (truncated) regime by 0.9 kJ/mol. So this
  sub-check fires only weakly on the trap shape and its main value is as an
  internal consistency check on the reweighting code, not as a trap detector.
  Say so in the doc when it lands.
- **Cost.** None beyond computing c(t) from the HILLS, which `sum_hills`
  already reads.
- **Would have caught.** None of the F-series on record. Worth keeping
  because it is free and because a *large* disagreement is a validity
  signal that the reweighting is not being fed the bias it thinks it is.
- **Cannot catch.** Everything (a) cannot, plus it is `not_evaluable` while
  the bias is still growing fast — which is when precision has already said
  "extend".

### (c) `time_window_invariance` — the estimate over one window vs another

- **Invariance.** ΔG does not depend on which part of the biased phase it is
  computed from, once the bias has stopped growing.
- **Transformation.** Compute over the last half of the biased phase and over
  the whole of it. (The reference generator already does exactly this and
  calls it stationarity.)
- **Statistic.** |ΔG_last_half − ΔG_whole|. Secondary, on the surface rather
  than the integral: rms(F over the last fifth vs the final estimate)
  restricted to the task-relevant range, states ± half a band — again what the
  generator does.
- **Tolerance.** 1 kT on ΔG and 1 kT rms on the restricted surface. Source:
  the generator's own acceptance test, which was set at kT to match the
  drift gate; and D10's record that on the 50 ns parallel-bias reference the
  ΔG statistic passed (−24.006 vs −24.003) while the *surface* statistic did
  not (17 kJ/mol at Q ≈ 0.03) — and the ΔG then walked 12 kJ/mol over the
  next 25 ns. So the ΔG-only version of this falsifier was wrong to stay
  silent at 50 ns, and the surface version is the one to trust. Both are
  reported; refutation is either.
- **Cost.** None.
- **Would have caught.** F6 (ΔG inverted and still moving), F13 (−16 → −27 by
  sum_hills), F15 (walker never came back: the last half is all unfolded),
  F10 (the vacuous drift: last-half-vs-whole is a different comparison and
  is not vacuous at two estimates). The "truncated budget" injected fault.
- **Cannot catch.** A drift slower than the window — the 50 ns example above
  is exactly that on the ΔG statistic. A walker that settled into the wrong
  state *early* and stayed: last half and whole then agree on the wrong
  answer. That is the case (a) cannot catch either, which is why the
  occupancy falsifier below exists.

### (c′) `occupancy_invariance` — the existing trap signals, reframed

The per-round trap signals (`rounds_confined`, `ns_since_*_visited`,
`observable_*_this_round`) were written one at a time as each failure
appeared (F13, F15). They are all one falsifier:

- **Invariance.** Under a bias that has flattened the surface, the walker's
  occupancy of each task state is invariant across time windows — it keeps
  visiting both.
- **Transformation.** Compare state occupancy in the most recent window
  (this round, or the last N ns) against the whole biased phase.
- **Statistic.** For each state, the biased nanoseconds since the walker was
  last inside it (`ns_since_low_visited`, `ns_since_high_visited`); and the
  per-round range's position relative to the bands (`rounds_confined`).
- **Tolerance.** 8 ns absent from the starting state (the current prompt
  trigger, chosen from the measured 5–10 ns round-trip cost on a working
  coordinate). This is the one tolerance in the set that is in time, not
  energy, and the one most obviously system-specific; §8.
- **Cost.** None; already computed.
- **Would have caught.** F13, F15 — it was built from them. F6.
- **Cannot catch.** A coordinate that visits both states but at the wrong
  ratio (the bias flattens a *different* surface than the observable's, and
  the walker crosses freely): occupancy looks healthy, ΔG is wrong. (a) and
  (d) are the ones that see that.

Reframing it this way changes nothing numerically. It makes the in-loop
trap detection a member of the set with the same record shape, and it makes
the 8 ns number a *tolerance* that the calibration in §8 can revisit rather
than a prompt constant.

### (d) `state_definition_invariance` — the thresholds

- **Invariance.** If the two task states are separated by a barrier on the
  campaign observable, ΔG does not depend on exactly where the thresholds are
  drawn.
- **Transformation.** Move each threshold by ±half a band (band = high − low)
  and recompute ΔG on the same surface: four perturbed values.
- **Statistic.** max |ΔG_perturbed − ΔG| over the four.
- **Tolerance.** 1 kT. Source: if the region within half a band of a
  threshold lies ≥ 1 kT above the basin minimum, its population is at most
  e⁻¹ of the basin's per unit length, and moving the boundary through it
  changes the integral by well under kT. A swing above kT means the
  observable has real population at the boundary — the states are not
  separated *on this coordinate*.
- **Cost.** None.
- **Would have caught.** F9 (recrossings counted between migrating bands: the
  same perturbation applied to the *count* would have shown the count is not
  invariant under a small move of the boundary). F12 and F14 (every frame in
  one state: perturbing the threshold changes nothing or everything, so
  either the statistic is huge or ΔG is `not_evaluable` — both block
  acceptance). The "observable does not separate the states" injected fault,
  directly.
- **Cannot catch.** A coordinate that separates the states cleanly but on
  which the *populations* are wrong (absorbing state, hidden DOF): the
  surface has a barrier where the thresholds are, ΔG is stable under
  perturbation, and it is still the wrong ΔG. Also blind to a wrong force
  field or box.
- **Also apply to `recrossings`.** The count under ±half-band perturbed
  thresholds must be invariant to within ±1. This is cheap and is the
  falsifier form of the F9 fix.

### (e) `bias_accounting_invariance` — the wall

Prerequisite fix, which is a validity fix and lands first: print the wall's
bias to COLVAR (`uwall.bias` alongside `metad.bias`) and sum every `.bias`
column in the reweighting. Today frames past the wall are reweighted with the
metadynamics bias alone, so the wall's penalty is treated as physics.

- **Invariance.** Once the wall's bias is accounted for, ΔG does not depend on
  whether frames beyond the wall are included in the reweighting.
- **Transformation.** Compute ΔG (i) reweighting by the total bias over all
  frames and (ii) reweighting by the total bias over frames with the wall's
  bias ≈ 0 only.
- **Statistic.** |ΔG_all − ΔG_inside|, and the fraction of frames past the
  wall (the latter is the validity signal; the former the falsifier).
- **Tolerance.** 1 kT. Source: if the wall sits where the task's states are
  not, the excluded frames carry negligible weight in either state and the
  two agree trivially; disagreement means the wall is inside a region that
  matters to the answer.
- **Cost.** None after the COLVAR fix.
- **Would have caught.** F6 run 2 (walker sat against the 0.8 nm wall; the
  unfolded ensemble the task defines lies partly beyond it). The "wall placed
  inside the relevant range" injected fault, directly.
- **Cannot catch.** Anything on a coordinate with no wall — bounded CVs
  (`contacts`, `torsion`), which is `not_evaluable` there rather than
  `not_refuted`. A wall that is too *loose* (box contact before the wall
  pushes back, F11) is a validity failure, not a bias-accounting one.

---

## 6. Coverage

Falsifier × the recorded findings. ✓ would fire; (✓) fires only on one
statistic or only weakly; — silent; V = a validity check, not a falsifier;
P = a precision statistic already catches it; n/a = not a result defect.

| finding | what it was | (a) seed | (b) estimator | (c) window | (c′) occupancy | (d) states | (e) wall | V / P |
|---|---|---|---|---|---|---|---|---|
| F1 | 5 ns Trp-cage not converged | — | — | — | — | — | — | P (ESS) |
| F2 | resume paid full setup | n/a | | | | | | |
| F3 | stale planted fixtures | n/a | | | | | | |
| F4 | SIGMA sized on thermal jitter, bias never fills | ✓ | — | — | (✓) | — | — | V (floored flag) |
| F5 | HILLS written to CWD, unflushed | — | — | — | — | — | — | V |
| F6 | RMSD one-way, ΔG inverted, walker on the wall | — (shared absorbing state) | — | ✓ | ✓ | (✓) | ✓ | |
| F7 | depth over grid padding; count between migrating bands | — | ✓ | — | — | ✓ (count) | — | |
| F8 | seed does not fix solvation | — (both walkers share it) | — | — | — | — | — | n/a to results; note in (a) |
| F9 | recrossings not comparable across rounds | — | — | — | — | ✓ (count) | — | |
| F10 | drift vacuous at two estimates | — | — | ✓ | — | — | — | P (fixed) |
| F11 | solute meets its periodic image | — | — | — | — | — | — | V (image distance) |
| F12 | wrong molecule; observable off by 10³ | — | — | — | — | ✓ | — | V (preflight) |
| F13 | bounded CV trapped the walker | — (shared absorbing state) | — | ✓ | ✓ | — | n/a | |
| F14 | observable normalised twice | — | ✓ | — | — | ✓ | — | V |
| F15 | walker never came back; misread number | — (shared absorbing state) | — | ✓ | ✓ | — | n/a | citation check |

Two things the table says that are worth saying in words.

1. **No single falsifier covers the failures that actually happened.** The
   three CLN025 traps (F6, F13, F15) are caught by (c) and (c′); the
   estimator and threshold bugs (F7, F14, F9) are caught by (b) and (d); the
   one failure (a) uniquely catches (F4-shaped hysteresis) has not yet
   occurred on this system. That is the argument for a *set* with a fixed
   membership rather than the strongest single test.
2. **The expensive one is the one with the least coverage of the record.**
   (a) costs 2× and would have fired on one of fifteen findings. It earns its
   place on the fault it uniquely detects — a hidden slow degree of freedom
   that produces hysteresis rather than absorption, which is the textbook
   metadynamics failure and the one a reviewer will ask about first — not on
   the history. §7 makes the cost case; the user's call.

And the injected faults planned for Step 5, mapped to what should fire:

| injected fault | expected to fire | expected silent |
|---|---|---|
| unwalled RMSD-only CV | (c), (c′); V: image contact | (e) not evaluable; (a) shared absorbing |
| wall inside the relevant range | (e), (d) | (a) |
| observable that does not separate the states | (d); V: scale | (a), (c) may pass |
| truncated budget | (c), P: drift | (d) |
| walker pinned in one basin | (a), (c′); P: recrossings null | (d) |
| γ too small for the barrier | (a), (c′); P | (b), (d) |

A fault none of the six fires on is the finding that the set is incomplete;
that is what the harness is for.

---

## 7. Cost and scheduling

| falsifier | GPU cost | in-loop? | when evaluable |
|---|---|---|---|
| (a) seed | **2× biased ns** | yes, if walker B runs alongside A from the round its seed frame exists; otherwise end-of-campaign | after both walkers have ≥ 1 recrossing |
| (b-i) estimator | none | yes | every biased round, when the observable is biased |
| (b-ii) c(t) | none | yes | once `n_fes_estimates ≥ 2` |
| (c) window | none | yes | once the biased phase has ≥ 2 rounds |
| (c′) occupancy | none | yes (already) | every biased round |
| (d) states | none | yes | every biased round with a reweighted surface |
| (e) wall | none (after the COLVAR fix) | yes | every biased round on a walled CV |

**Everything except (a) is affordable in-loop** at the cost of a few extra
reweighting passes over COLVAR per round, which is milliseconds against a
round of MD. They should all be in the per-round report so the scientist and
the gate see them at the same time.

**(a) is the only GPU-cost falsifier.** Proposal:

- Minimum two walkers. Walker B starts from the first frame of A that enters
  the other task state (§5a option 1), so B lags A by however long A takes to
  cross once. B runs under its own adapter and its own HILLS in a sibling
  directory; budgets are per walker, so a 100 ns task costs 200 ns.
- Run B **in-loop** rather than after the campaign: the whole value of a
  replica is that a refutation arrives while budget remains to act on it.
  An end-of-campaign replica can only downgrade the verdict.
- The scientist sees (a)'s record like any other falsifier. It does not see
  walker B's report separately, and it does not decide for B: B extends when
  A extends and stops when A stops. One campaign, one decision per round, two
  walkers. This keeps "one LLM call per round" intact.
- If the budget cannot afford two walkers, (a) is `not_evaluable` for the
  whole campaign and the campaign cannot reach `not_refuted` on the full set.
  That is the honest outcome: a single-walker campaign has not been tested
  for hysteresis. The verdict layer should say so in its own words rather
  than folding it into `INCOMPLETE`.

---

## 8. Tolerances — where the numbers come from, and how they get calibrated

Every energy tolerance above is 1 kT. That is not five independent
derivations; it is one choice made once, for two reasons: the precision gate
already uses kT (drift < kT), the reference match already uses 1 kT, and a
falsifier laxer than either would let through what the yardstick would then
reject. Where a measurement exists it is quoted (D10's 1.75 kJ/mol rms;
the c(t) 0.01/0.9 kJ/mol on the model) and it is consistent with kT.

The one time tolerance, 8 ns for (c′), comes from the measured round-trip
cost on CLN025 and is the most system-specific number in the set.

**These are provisional and the harness calibrates them.** Steps 4 and 5
produce, per falsifier and per tolerance value, a false-accept rate on
injected faults and a false-fire rate on the converged reference run. The
tolerance to ship is the one that minimises false accepts subject to a
bounded false-fire rate on the reference — and if no value of the tolerance
separates the two, that falsifier is not doing work and should be reported
as such rather than kept for completeness. The design note should be
updated with the calibrated values when they exist; until then every
tolerance carries `"source": "provisional, 1 kT by convention"` in its
record.

---

## 9. What is fixed, and what I would have made adaptive

Per the constraint, the set is fixed per method family and the LLM never
chooses. Two places where I would otherwise have reached for adaptivity, left
out deliberately:

- **Running walker B only when the campaign "looks like it needs it."** The
  temptation is to spend the 2× only after (c′) has fired once, since that
  is when hysteresis becomes plausible. But (a)'s unique coverage is the
  case where (c′) *does not* fire — both walkers look healthy from their own
  side — so gating B on (c′) removes exactly the coverage B exists for.
  Fixed: B runs whenever the budget allows two walkers, decided once at
  campaign start from the task file, not per round.
- **Choosing the perturbation size in (d) from the surface.** One could
  scale the ±half-band by the local barrier width. Left fixed at half a band
  because the point is that the *task's* definition of the states is under
  test; a perturbation sized from the answer tests the answer against
  itself.

One thing that *is* mechanically conditional and is not adaptivity: which
falsifiers are `evaluable` this round depends on what exists (a second
walker, a walled CV, an observable that is biased). That is a property of the
data, computed in code, and reported as `not_evaluable` rather than omitted.

---

## 10. Decisions (2026-09-22, user)

1. **Walker B is not funded for v1.** (a) `seed_invariance` stays in the
   fixed set and is reported `not_evaluable` on every single-walker
   campaign. A campaign therefore cannot reach `not_refuted` on the full set
   with one walker, and the verdict layer says so in words. The seeding
   design in §5a stands for when it is funded.
2. **Stop permission becomes "precision gate AND every falsifier
   `not_refuted`".** With (a) unfunded, that means a single-walker campaign
   never stops on the scientist's say-so — only on budget. Accepted as the
   honest consequence.
3. **`verified` is reference-only**, produced nowhere inside the loop.
4. **The 8 ns tolerance moves from prompt prose to a task-file field**
   (`done_criterion.absence_tolerance_ns`, name to be settled in Step 3), and
   the report carries it in the falsifier's record. Every other tolerance is
   1 kT by convention and lives in code until §8's calibration says
   otherwise.

## 11. Generality — what is universal and what is per system class

The falsifiers are written against exactly four things and nothing else: a
campaign observable (a 1-D coordinate computable from the topology), two
named states as bands on it, the scalar ΔG(low − high) integrated over those
bands, and the bias V(t) per frame. None of (a)–(e) mentions folding, a
hairpin, or a contact. That is what makes the verifier universal, and it is
the test to keep applying: **no module under `diagnostics/` and no falsifier
may name a CV type, a residue, or a system.** The tolerances are data, not
code — which is why decision 4 above matters more than it looks: a
binding round trip may cost 50 ns where a hairpin's costs 8, and the
falsifier must not change to say so.

What has to be *declared per method family*, not assumed, is which
invariances are expected to hold. Protein–ligand binding from the bound pose
along a distance shows this:

- **(d) breaks by construction on the unbound side.** The unbound state on a
  distance is a diffusive shell, not a basin; ΔG shifts by −kT ln(V_ratio)
  when its outer boundary moves. So (d) applies to the bound threshold only,
  and the standard-state correction is an explicit term in the method's
  definition of ΔG — an omission there is a correctness error **no falsifier
  can catch**, because both estimators share it.
- **(e) changes role.** For binding the wall is not insurance, it *defines*
  the unbound volume (funnel-shaped restraints are the standard). "With vs
  without the wall" then fires by design; the invariance that does hold is
  under moving the wall outward, with the predictable −kT ln(V) shift.
- **(a) becomes cheap to seed.** The unbound start is trivial (place the
  ligand far away), so the awkward seeding of §5a disappears. The absorbing
  state for binding is the ligand leaving and never returning — F6's shape
  exactly — and (c)/(c′) cover it as they do here.

Above the seam, everything is per system class and should be *flexible*:
the setup agent's knowledge chunks (a `binding_role.md` beside
`setup_role.md`), the closed vocabularies (a small-molecule force-field entry
— D9 already names the gap; a group-COM `distance` in `cv_designer`, since
today `distance` requires single atoms), `SystemSpec` (a ligand alongside
the protein), and the bias designer (the wall as part of the family). The
task file is the seam. Adding a system class means adding vocabulary entries
and knowledge chunks and a method-family declaration of expected
invariances; it must never mean touching a falsifier.

One evaluation-side item should move for the same reason: the literature
band in `run_cln025_e2e.py` is CLN025-specific and hard-coded. It belongs in
the task file as an optional `expected_answer` block (value, unit, source)
that **only the verdict reads** — it must be excluded from
`render_task_expectation`, or moving it creates the leak §12 is about.

## 12. Step 2 — audit (read-only, 2026-09-22)

### 12.1 Where each existing statistic belongs

| statistic (report field) | category | note |
|---|---|---|
| `frame_dt_ps`, `trajectory_length_ns`, `note` (too few frames), `n_fes_estimates`, `n_steps`, paths | VALIDITY | bookkeeping that says whether there is enough to analyse. `trajectory_length_ns` is wrong today for any frame spacing other than 1 ps (review, 2026-09-22) |
| `check_residue_count`, `check_observable_scale` (pre-flight) | VALIDITY | filed under pre-flight with no category; scale check is one-directional (F14) |
| `sigma_floored` / `sigma_ceiled` (in `plumed.dat` only), wall notes (ledger only) | VALIDITY | bias-definition-as-intended; reach the scientist only as prose, and not at all on the PBMETAD path |
| `cv_start`, `cv_min`, `cv_max`, `cv_ranges` | VALIDITY | where the walker went; descriptive, cumulative |
| `sem_blocked`, `sem_naive`, `plateau_reached`, `statistical_inefficiency_*`, `tau_int_frames`, `ess`, `well_sampled` | PRECISION | vanilla, unchanged |
| `fes_drift_kj_per_mol` | PRECISION | biased, unchanged |
| `n_basins_fes`, `barrier_kj_per_mol` | PRECISION | shape of the estimate |
| `fes_depth_kj_per_mol` | PRECISION, **double duty** | a depth far past ~γkT (F13: 116 kJ/mol) is a validity signal that the bias is filling a degenerate bin; nothing computes that threshold |
| `recrossings`, `recrossing_low/high`, `recrossing_basis`, `barrier_crossed`, `min_recrossings` | PRECISION, **double duty** | a sampling-adequacy count; the F9 anchoring to task states also makes it the input to (d). Keep the count in precision; (d) tests its invariance |
| `fes_converged` | PRECISION, **double duty** | drift < kT AND recrossings ≥ min. Also the loop's *permission to stop* (`_refuse_premature_stop`), which makes a precision gate carry the correctness verdict. Decision 2 removes that |
| `bimodality_coefficient`, `n_basins`, `minor_basin_occupancy`, `exploring` | PRECISION (sampling adequacy of the unbiased phase), **double duty** | a decision input for the pivot, and read by the prompt as "the system has visited multiple states" — a correctness-flavoured claim it cannot support (skewed-unimodal false positive, review 2026-09-22) |
| `observable_min/max_this_round`, `confined_to_state`, `rounds_confined`, `rounds_since_*`, `ns_since_*` | CORRECTNESS — falsifier (c′) | today the tolerance (8 ns) is prompt prose, so there is no `fired` flag: the only gate is the LLM. A falsifier without a gate |
| `delta_g_low_minus_high_kj_per_mol`, `observable_fes_path` | CORRECTNESS — the answer | the quantity under test, not a test |
| `cv_switches_used/remaining`, `biased_cvs`, `phase` | bookkeeping | action-space state |

Not present anywhere today: (b), (d), (e) as computed statistics; any
thermodynamic-state or image-contact validity field; a `not_evaluable`
state for anything except the null recrossing count.

### 12.2 Where the reference touches the loop

The reference must be an evaluation-only oracle. Checked: every import in
`src/` and `app.py`, the scientist's payload (`report`, `prior_round_summaries`,
`hypothesis_ledger`, `task_expectation`, `max_extra_ns`), the ledger's
writers (the scientist, loop refusals, wall notes), and every `knowledge/`
chunk.

**Direct leak — must be removed.** `knowledge/action_add_cv.md` lines 21–25
tell the scientist: "For a hairpin that is the turn — backbone torsions of
the turn residues, or the cross-strand distance … A reference run on CLN025
that refolded under three parallel coordinates had not refolded under either
of the global ones alone in 50 ns." That is the reference's result, handed to
the agent as CV guidance. It also pre-selects the coordinate the 2026-09-14
journal credits the model with finding unprompted ("the coordinate a person
would pick and which was deliberately not pre-selected"). The lesson that
survives is the general one already in the chunk — add the coordinate you
believe carries the barrier, not a second global measure — with the
system, the reference and its numbers removed.

**Indirect leak — benchmark overfitting, not the reference.**
`cv_vocabulary.md` lines 16–18 and `action_switch_cv.md` lines 33–39 narrate
the evaluated campaigns' own trajectories with their numbers (Q between 0.03
and 0.13; 0.03 and 0.55). The prompt was tuned on the campaigns it is scored
on. Step 3 replaces these anecdotes with the falsifier definitions, which is
the fix; the numbers must not survive the rewrite.

**Indirect influence — allowed, but must be declared.** The task file's force
field (ff99SB-ILDN) and budget (100 ns) were chosen from what the ff14SB
reference runs revealed (task-file comments at `system:` and
`done_criterion:`; 2026-09-14 journal). The agent reads neither comment, but
the campaign's difficulty was calibrated on the oracle. Rule to record: the
reference may inform *task design* — that is what a benchmark is for — and
may never inform the agent's prompt, ledger, or report.

**Not leaks, confirmed.** No `src/` or `app.py` code reads
`benchmarks/data/` or imports `benchmarks.`; the app does not display the
reference; `verdict()` runs only after the campaign; the reference generator
*shares code* with the loop (`bias_designer`, `plumed_writer`, the task
file), which is shared blind spots (§5a) rather than information flow.

**Future leak to prevent now.** If the literature band moves into the task
file (§11), it must be excluded from `render_task_expectation`; the
expectation string is the one task-file view the agent reads.

---

## See also

- `overview.md` §4.4 — the current three-layer description this note splits
- `activity-log.md` — the F-series this note's coverage table indexes
- `architecture.md` — the one-call-per-round constraint §7 preserves

## 13. Step 3 — what landed (2026-09-22)

- `diagnostics/falsifiers.py` — the fixed set for the well-tempered
  metadynamics family, one record per falsifier (`statistic`, `magnitude`,
  `tolerance`, `tolerance_source`, `state`, `fired`, `note`, `inputs`), and
  `summarize`. States are the four in §1 **plus `not_applicable`**: the
  implementation separates "the data cannot yet support the check" from
  "the transformation has nothing to act on in this campaign" (no wall on a
  bounded CV, the observable not among the biased coordinates, no second
  walker). `not_evaluable` blocks a stop; `not_applicable` does not.
- **Deviation from decision 1, flagged.** `seed_invariance` on a single
  walker is `not_applicable`, not `not_evaluable`. Under the strict reading no
  campaign could ever stop on the scientist's say-so, `scientist_said_stop`
  becomes unreachable, and Step 5's false-accept rate is zero by construction
  for every arm. The constant `_SINGLE_WALKER_STATE` in `falsifiers.py` is
  the one place to flip it. The verdict layer should still say "hysteresis
  untested" for `n_walkers=1`; that is Step 4/5 work.
- `diagnostics/report.py` — `group_report` files every existing number
  under `validity` / `precision` / `correctness` / `action_space` (an unfiled
  key raises), adds `precision.gate`, and `report_field` /
  `has_report_field` read a grouped or a legacy flat report by leaf name or
  dotted path. `make_report` and `metad_report` are unchanged.
- `orchestrator/loop.py` — `_round_report` returns the grouped report;
  the answer is reweighted with **every** `.bias` column summed (walls
  included — the (e) prerequisite); (b-i), (c), (c′), (d), (e) are computed
  each biased round, (a) is recorded `not_applicable` with `n_walkers=1`;
  `_refuse_premature_stop` honours a stop only when `precision.gate` is true
  and `correctness.summary.not_refuted` is true (decision 2); the
  occupancy tolerance is `run_campaign(absence_tolerance_ns=)`, locked in the
  config with a legacy default of 8.0 (decision 4). The (c) *surface*
  statistic (last fifth vs final over the task range) is deferred: the
  precision drift already blocks the case it would catch, and the two would
  be redundant until drift is restricted to the task range.
- `adapters/plumed_writer.py` — each wall is labelled `uwall_<cv>` and its
  bias is printed to COLVAR; a reused action label is refused at render time
  (the review's high finding).
- `task_file.py` / `setup_agent.py` — `done_criterion.absence_tolerance_ns`
  is a mapped field; the setup schema requires it and carries `cannot_run`,
  which raises `SetupRefused` instead of writing a task file.
- `knowledge/` — every chunk rewritten to the three-block vocabulary; the
  reference-run leak and the benchmark anecdotes (§12.2) are gone; the words
  `correct` / `right` / `valid` / `verified` appear only in the sentence that
  forbids them.
- `scientist.py` — citations resolve leaf names and dotted paths through
  `report_field`.

## 14. Step 4 — the verifier ablation (2026-09-22)

`mdpilot/ablation.py`, tests in `tests/unit/test_ablation.py`.

**Design: replay, not three live runs.** The LLM's decisions (`extend`
lengths, `switch_cv`, `add_cv`) drive the simulation, so three arms each
running the loop would judge three different trajectories. Instead one
finished campaign on disk *is* the simulation; each round's three-block
report is recomputed once from the per-round HILLS/COLVAR snapshots
(`replay_report`, read-only with respect to the campaign); and each arm is a
verifier that reads that report in its own view and returns a per-round
verdict — `accept` / `continue` / `revise`. Identical simulation, bias design
and seed are then true by construction, and `Comparison` still asserts it:
every arm's stored campaign config must be identical field for field
(`store.config_differences`, legacy defaults applied), the round table must
be identical (index, steps, phase, decision, proposal), and the model must be
the same. Any difference raises `ArmsNotComparable` naming the field.

**The arms and what each sees.**

| arm | view | verdict |
|---|---|---|
| `gates_only` | nothing shown to a model | rule: biased — accept on `precision.gate ∧ not_refuted`, revise on any refuted, else continue; unbiased — accept on the gate for a pure convergence task, revise when pinned against a task that needs a transition |
| `lm_only` | `raw_view`: every magnitude and input, **no** `gate`, `fes_converged`, `plateau_reached`, `well_sampled`, `exploring`, `barrier_crossed`, `min_recrossings`; falsifiers reduced to name / statistic / magnitude / inputs; no summary. Prompt from the `*_raw` chunks, which name no threshold; a test greps the assembled raw prompt for `refuted`, `tolerance`, `gate`, `kT`, `8 ns` and the like | the model's decision, unfiltered |
| `lm_plus_gates` | the full report and the campaign's prompt | the model's decision, then the loop's `_refuse_premature_stop`; the model's own verdict is logged beside the final one |

Each LLM arm keeps its own ledger and its own prior-round summaries (the
raw arm's priors are stripped of `precision_gate`, `not_refuted` and the
falsifier lists), so an arm's memory is what it wrote, not what another arm
saw.

**What is logged.** `out/<arm>/verdicts.jsonl`: per round — verdict, the
LLM's own verdict, reason, override note, ΔG, the precision gate, the refuted
and not-evaluable lists, cumulative biased ns. `out/summary.json`: per arm —
whether and when it accepted, its first non-continue verdict (round and ns:
the arm's detection latency), and `final_delta_g` = ΔG at the round it
accepted, or at the last round if it never did. `gates_first_fire` is the
first round any falsifier refuted, arm-independent. `out/reports/` holds the
recomputed reports. Ground truth and the false-accept rate belong to Step 5;
this records, it does not score.

**Two calibrations the first replay forced** (`campaigns/cln025_e2e_v2`,
the 2026-09-14 campaign, `gates_only`):

- `estimator_invariance` refuted in every biased round on its rms statistic
  (4–5 kJ/mol) while the ΔG gap was 0.3–2.6 kJ/mol. The whole-range rms was
  dominated by the far edge of the contact fraction, the one bin a
  still-filling bias keeps deepening — a precision symptom the drift already
  reports, not a disagreement about the answer. The rms is now taken over the
  task range (states ± half a band), as the reference generator's own drift
  test is.
- `state_definition_invariance` was `not_evaluable` in every round because
  `high + band/2 = 0.9` lies past the largest contact fraction the walker
  ever reached (0.89). An outward perturbation past the sampled range has
  nothing to enclose; it is skipped and named in `inputs`, and the two inward
  perturbations — the ones that test the boundary between the states — are
  always required. With that, the replay shows the inward move of the high
  threshold shifting ΔG by ~5 kJ/mol at rounds 8–10: on this force field's
  surface the task's states are **not separated by a barrier on the contact
  fraction**, which is a finding about the task file, and one no previous
  check could make.

Both are tolerances being calibrated by data, which is §8's plan; neither
changed a tolerance value, only the range a statistic is taken over.

**Not done in this step.** No LLM arm has been run live (the harness takes
`--arms` and a model; the raw and gated arms need an API key and cost one
call per round per arm). Running `lm_only` and `lm_plus_gates` over the two
2026-09-14 campaigns is the first thing Step 5 should do, since those two are
the only campaigns with a known outcome.

## 15. Step 5 — fault injection (2026-09-22)

`mdpilot/faults.py`, `benchmarks/run_faults.py`, tests in
`tests/unit/test_faults.py`.

### 15.1 The registry

Six faults, each a set of `run_campaign` overrides on top of an ordinary task
file plus the coordinate a scripted policy pivots to, registered with the
falsifier(s) expected to fire and a statement of why the answer is wrong by
construction. A fault that names no expected falsifier is refused at
construction: a fault the set cannot catch is a finding and has to be
written as one, not registered quietly.

| fault | what breaks | expected to fire |
|---|---|---|
| `rmsd_no_configured_wall` | CA-RMSD biased, `cv_upper_wall_nm=None` (only the box-derived bound remains) | occupancy, time-window |
| `wall_inside_transition_region` | CA-RMSD biased, wall at 0.3 nm, inside the unfolding path | bias-accounting, state-definition |
| `thresholds_inside_one_basin` | states at Q = 0.75 / 0.85, both in the folded basin | state-definition |
| `truncated_budget` | contacts bias, 2 ns budget | time-window |
| `walker_pinned_weak_bias` | contacts bias, one hill per 50 000 steps | occupancy |
| `gamma_too_small` | contacts bias, γ = 1.5 | occupancy |

The overrides are locked into each campaign's config, so a resume cannot
quietly repair a fault.

### 15.2 How a broken campaign is made, and scored

The campaign is driven by `scripted_policy`, not the LLM: pivot to the
fault's coordinate at round 1, then `extend` to the budget whatever the
report says. The policy is the *cause* of the fault and never a verifier of
it, so the finished record is one uninterrupted bias for every arm to judge —
the same separation the ablation rests on. `run_campaign` grew a `decide_fn`
hook for this; left unset, the LLM decides as always.

`benchmarks/run_faults.py` builds each fault from the task file into its own
work_dir, runs it, replays the requested arms over it with
`mdpilot.ablation`, and calls `faults.score`, which reads every fault's
record and ablation output and reports:

- per arm: `n_faults`, `false_accepts`, **`false_accept_rate`** (fraction of
  broken campaigns the arm declared done), and `latency_ns` per fault (biased
  ns until the arm's first non-`continue` verdict);
- per fault: `fired_at_ns` for each expected falsifier (first `refuted`),
  `blocked_at_ns` (first `refuted` *or* `not_evaluable` — an unevaluable
  check blocks a stop even though it does not fire), `detected`,
  `unexpected_fires`, `expected_silent_but_fired`, and each arm's acceptance.

The LLM arms are opt-in on the command line; the deterministic arm is enough
to learn whether the falsifiers fire, which is the first question.

### 15.3 First measurements (2026-09-22/23)

**The three arms over the two campaigns with a known outcome.** Both ran on
2026-09-14 on the same task and seed, 12 rounds, 20 ns biased; both ended on
budget with a surface the reference disagrees with (v1 wrong-signed, +44.5
kJ/mol against −35.8; v2 right-signed, −16.5, three times closer). Neither
should be accepted. The LLM arms ran live on `claude-sonnet-4-6`, one call
per round per arm; every arm judged the same recomputed reports.

| campaign | arm | verdicts, rounds 1–12 (A accept · continue R revise) | accepted | first biased non-continue |
|---|---|---|---|---|
| v2 | gates_only | R R R R R R R R R R R R | no | 1 ns |
| v2 | lm_only | R · · · · · · · · · R · | no | 18 ns (revise) |
| v2 | lm_plus_gates | R · · · · · · · · · R · | no | 18 ns (revise) |
| v1 | gates_only | R R R R R R · · · · · R | no | 1 ns |
| v1 | **lm_only** | R · · · · · R · · · **A A** | **yes, at 18 ns** | 11 ns (revise) |
| v1 | lm_plus_gates | R · · · · · · · · · · R | no | 20 ns (revise) |

Three things the table says.

1. **The false accept is real, and it is the arm without gates.** On v1 the
   raw-statistics LLM declared the campaign done at round 11 — its own
   `reason` says the walker "has not visited the low state for 9.0 ns" and
   accepts anyway — and again at round 12 on the strength of a fifth
   recrossing. The final surface at that point is the +44.5 kJ/mol one. The
   same model reading the same numbers *with* the flags and the stop rule
   never accepted; neither did the rule alone. Across the two campaigns:
   false-accept rate `lm_only` 1/2, `gates_only` 0/2, `lm_plus_gates` 0/2.
2. **The gates are earlier; the LLM is more selective.** `gates_only`
   revises from the first biased round on both campaigns because
   `estimator_invariance` refutes on every filling surface (§14) — it is
   never wrong to say "not done" there, but it is not detection either. The
   LLM arms revise once, at the round a human would: v2 round 11 on the
   occupancy refutation (absent 9 ns from the folded state), v1 round 7 on
   the same signal in the raw numbers, v1 round 12 (gated arm) on the
   state-definition refutation. `lm_plus_gates` also cites the falsifier's
   magnitude, tolerance and state by name.
3. **Both LLM arms quote the trajectory-length bug** in round 1 ("only 0.199
   ns of vanilla MD" for a 1 ns round — the mdtraj time-axis defect from the
   2026-09-22 review). It did not change the pivot, but it is in the reasoning
   of every campaign and should be fixed before any latency number is taken
   seriously.

**The six faults, dry run** (`benchmarks/run_faults.py --dry-run`, 0.05 ns
opening, 0.1 ns biased cap, `gates_only`; ~3 min each on the GTX 1660,
`campaigns/faults_dryrun/`). A dry run proves the plumbing — every fault
builds, runs under PLUMED, replays, scores — and it also says which faults
are visible at 100 ps:

| fault | detected at 0.1 ns | by |
|---|---|---|
| `thresholds_inside_one_basin` | **yes, at 0.1 ns** | `state_definition_invariance` |
| `rmsd_no_configured_wall` | no | (needs the walker to leave and not return: ≥ 8 ns) |
| `wall_inside_transition_region` | no | (needs frames at the wall) |
| `truncated_budget` | no | (needs two windows with both states populated) |
| `walker_pinned_weak_bias` | no | (needs ≥ 8 ns absent or confined) |
| `gamma_too_small` | no | (same) |

`gates_only` false-accept rate 0/6, trivially: at 100 ps the precision gate
is false everywhere. The number that matters — the false-accept rate per
arm on full 20 ns broken campaigns, and each arm's detection latency — needs
the full-length runs: six campaigns of ~5 h GPU each, plus the two LLM arms
at one call per round per arm. Not launched in this session; the command is
`python -m benchmarks.run_faults --arms gates_only,lm_only,lm_plus_gates`.
One fault is already informative: a task whose two states sit in one basin
is refuted within the first biased round by the threshold perturbation,
which is the cheapest possible check and the one no earlier version of the
agent could make.

Also found by these runs and fixed in the same session: `switch_cv` with a
null proposal was a bare `RuntimeError` out of the parser (it ended the v1
replay); it is now `MalformedDecision`, retried like a misquote and converted
to an extend after three attempts. A header-only HILLS (a first biased round
shorter than one deposition interval) made `plumed sum_hills` segfault; it
now yields a report with `n_fes_estimates=0` and every surface statistic
`None`.
