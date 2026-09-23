"""Generate the CLN025 reference free-energy surface.

The campaign benchmark asks whether the scientist's biased phase recovered the
right surface. "Right" needs a yardstick that does not depend on which CV the
scientist happened to bias, so this runs a long well-tempered metadynamics
simulation through the same adapter, force field, bias designer and PLUMED
writer a campaign uses — only the length and the seed differ — and reports the
surface *along the campaign observable*, the fraction of native CA contacts
the task file declares. A campaign whose surface on that observable matches
this one found the surface the method converges to, which is the agent's job;
whether the force field's surface is nature's is a separate question the
literature band in `run_cln025_e2e.py` addresses.

`--bias` picks what the reference biases. `observable` biases the contact
fraction itself, so the surface is `sum_hills` on it directly. `rmsd` biases
CA-RMSD to the native structure under the task's upper wall — the coordinate
the scientist switches to when contacts trap the walker (F13) — and the
surface on the observable is reweighted from the column PLUMED prints for it,
exactly as a campaign's is. `pbmetad` biases RMSD (walled), the contact
fraction and the radius of gyration in parallel (PLUMED PBMETAD), each bias
converging to its own marginal, so the surface on the observable is
`sum_hills` on the contact-fraction hills and the reweighting is a
cross-check. The first 50 ns reference on `observable` reproduced F13 on
itself: unfolded at 2.5 ns, five nanoseconds parked at Q ≈ 0.05 under
136 kJ/mol of bias, then not one folded frame in its last 25 ns; `rmsd` sat
unfolded from 1 ns under 115 kJ/mol. A 1-D bias on a coordinate orthogonal to
the barrier that gates refolding piles up without lowering it, however long
it runs — which is what covering several coordinates at once is for.

The reference validates itself before it is written: the well-tempered drift
between the half-way and final estimates must be below kT and the walker must
have made at least `--min-recrossings` transitions between the task's two
states, or the run is reported as unconverged and no `reference_fes.dat` is
produced. A reference that is itself noise would make every comparison pass.

Resumable: the run is chunked, each chunk ends in a checkpoint, and a restart
picks up from the last one with PLUMED's RESTART enabled so HILLS and COLVAR
continue rather than starting over. Run inside the conda environment via
`micromamba run` so `plumed` is on PATH for the final integration:

    export MAMBA_ROOT_PREFIX=$HOME/.micromamba
    ~/.local/bin/micromamba run -n mdpilot \\
        python -m benchmarks.generate_cln025_reference --dry-run   # minutes
    ~/.local/bin/micromamba run -n mdpilot \\
        python -m benchmarks.generate_cln025_reference             # ~12 h on a GTX 1660

Output, under `benchmarks/data/cln025/<forcefield>/` (gitignored, regenerable;
the force field is part of the reference's identity):
    reference_fes.dat   the final sum_hills surface, F(Q) in kJ/mol
    reference.json      what it is: length, seed, SIGMA, drift, recrossings,
                        ΔG between the task's states, and the pass verdict
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import mdtraj as md

from mdpilot.adapters.openmm_adapter import OpenMMAdapter
from mdpilot.adapters.plumed_writer import ParallelBias, PlumedInput, enable_restart
from mdpilot.diagnostics.free_energy import (
    _KB_KJ_PER_MOL_K,
    _baseline_index,
    count_recrossings,
    delta_g_kj_per_mol,
    fes_drift_kj_per_mol,
    load_colvar,
    load_fes,
    sum_hills,
)
from mdpilot.diagnostics.free_energy import (
    FreeEnergySurface,
    reweighted_profile,
    write_fes,
)
from mdpilot.observables import (
    COLVAR_OBSERVABLE_LABEL,
    colvar_to_observable_factor,
    observable_cv_proposal,
)
from mdpilot.orchestrator.loop import steps_per_ns_for
from mdpilot.sampling.bias_designer import design_bias, design_upper_wall
from mdpilot.sampling.cv_designer import CVProposal, design_cv
from mdpilot.task_file import load_task_file

_TASK_FILE = Path("benchmarks/tasks/cln025_contacts.yaml")
_DATA_DIR = Path("benchmarks/data/cln025")


def forcefield_slug(forcefield: str) -> str:
    return forcefield.replace("/", "_")


def reference_dir(forcefield: str, data_dir: Path = _DATA_DIR) -> Path:
    """Where a reference for this force field lives. A reference is a property
    of the force field as much as of the system; one per pair, never shared."""
    return data_dir / forcefield_slug(forcefield)
_REFERENCE_SEED = 7          # not the campaign's 42: an independent realization
_VANILLA_NS = 0.2            # what SIGMA is sized from, as at a campaign's pivot
_FRAME_PS = 20.0             # trajectory kept sparse; COLVAR carries the observable


def _log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _fmt_time(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.1f} min"


def build_reference(
    *,
    total_ns: float,
    chunk_ns: float,
    seed: int,
    min_recrossings: int,
    bias_cv: str = "observable",
    check_every: int = 10,
    data_dir: Path | None = None,
    dry_run: bool = False,
    task_file: Path = _TASK_FILE,
) -> dict[str, Any]:
    task = load_task_file(task_file)
    if data_dir is None:
        # Namespaced by force field even for a dry run: the adapter reuses a
        # cached System in its work dir without checking what built it.
        data_dir = reference_dir(task.spec.forcefield)
        if dry_run:
            data_dir = data_dir / "dryrun"
    observable = task.campaign.get("observable")
    assert observable is not None, "the task file must declare the observable"
    low, high = task.campaign["state_thresholds"]

    work_dir = data_dir / ("reference" if bias_cv == "observable" else f"reference_{bias_cv}")
    chunks_dir = work_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    temperature_k = task.spec.ensemble.temperature_k

    # --- vanilla: build the system, equilibrate, and sample the folded basin ---
    vanilla = task.build_adapter(work_dir, seed=seed)
    steps_per_ns = steps_per_ns_for(vanilla)
    frame_steps = max(int(_FRAME_PS * steps_per_ns / 1000), 1)
    vanilla_dcd = work_dir / "vanilla.dcd"
    t0 = time.monotonic()
    vanilla.prepare()
    vanilla.start()
    _log(f"system built ({_fmt_time(time.monotonic() - t0)}); platform {vanilla.platform_name}")
    if not vanilla_dcd.exists():
        t0 = time.monotonic()
        vanilla.run_steps(
            int(_VANILLA_NS * steps_per_ns), trajectory_path=vanilla_dcd,
            report_interval_steps=max(int(0.001 * steps_per_ns), 1),   # 1 ps: SIGMA wants frames
        )
        _log(f"vanilla {_VANILLA_NS} ns done ({_fmt_time(time.monotonic() - t0)})")

    # --- the bias, exactly as a campaign builds it at the pivot ---
    reference = md.load(str(vanilla.topology_path))
    def resolve(proposal: CVProposal):
        return design_cv(proposal, reference.topology, reference=reference, output_dir=work_dir)

    def sized(cv):
        return design_bias(cv, vanilla_dcd, vanilla.topology_path, temperature_k=temperature_k)

    rmsd_proposal = CVProposal(cv_type="rmsd", selections=("name CA",), label="rmsd_ca")
    walls: tuple = ()
    if bias_cv == "observable":
        # The observable itself is biased: its column already is the observable,
        # and adding a second copy would change COLVAR's layout under a run
        # that may be resuming.
        cv = resolve(CVProposal(
            cv_type=observable.cv_type, selections=tuple(observable.selections),
            label=observable.name,
        ))
        bias = sized(cv)
        observable_cv, observable_column = cv, cv.label
        cvs: tuple[Any, ...] = (cv,)
        hills_for_surface = bias.hills_file
    elif bias_cv == "rmsd":
        # Something else is biased; the observable rides along unbiased in
        # COLVAR, so the surface on it comes from where a campaign's does.
        cv = resolve(rmsd_proposal)
        bias = sized(cv)
        observable_cv = resolve(observable_cv_proposal(observable))
        observable_column = COLVAR_OBSERVABLE_LABEL
        cvs = (cv, observable_cv)
        hills_for_surface = None
    elif bias_cv == "pbmetad":
        rmsd_cv = resolve(rmsd_proposal)
        observable_cv = resolve(observable_cv_proposal(observable))
        rg_cv = resolve(CVProposal(cv_type="gyration", selections=("name CA",), label="rg_ca"))
        per_cv = [sized(c) for c in (rmsd_cv, observable_cv, rg_cv)]
        cv = rmsd_cv
        bias = ParallelBias(
            cv_labels=(rmsd_cv.label, observable_cv.label, rg_cv.label),
            sigma=tuple(b.sigma[0] for b in per_cv),
            height=per_cv[0].height, pace=per_cv[0].pace,
            bias_factor=per_cv[0].bias_factor, temperature_k=temperature_k,
            # Generous bounds: the wall turns RMSD around near 0.9 nm, Rg has
            # reached 1.05 nm, and a hill centre outside the grid is a fatal
            # PLUMED error.
            grid=((-0.2, 1.6), (-0.3, 1.3), (0.1, 2.0)),
        )
        observable_column = COLVAR_OBSERVABLE_LABEL
        cvs = (rmsd_cv, observable_cv, rg_cv)
        # Each parallel bias converges to its CV's own marginal, so the
        # contact-fraction hills integrate straight to the surface wanted.
        hills_for_surface = f"{bias.hills_prefix}.{observable_cv.label}"
    else:
        raise ValueError(f"--bias must be observable, rmsd or pbmetad; got {bias_cv!r}")
    if bias_cv != "observable":
        wall = design_upper_wall(
            cv, task.campaign.get("cv_upper_wall_nm"),
            trajectory_path=vanilla_dcd, topology_path=vanilla.topology_path,
        )
        walls = (wall,) if wall is not None else ()
    plumed_input = PlumedInput(
        cvs=cvs, bias=bias, walls=walls, output_dir=work_dir.resolve(),
    ).render()
    (work_dir / "plumed.reference.dat").write_text(plumed_input)
    _log(f"bias on {','.join(bias.cv_labels)}: SIGMA={','.join(f'{s:.3g}' for s in bias.sigma)} "
         f"HEIGHT={bias.height:.3g} PACE={bias.pace} gamma={bias.bias_factor:g}"
         + (f"; wall at {walls[0].at:g}" if walls else ""))

    # --- integrate and judge, at checkpoints and at the end ---
    kt = _KB_KJ_PER_MOL_K * temperature_k
    band = high - low
    # Convergence is judged where the task lives — between the states and
    # half a band beyond each — not over the whole grid. The far edge of the
    # unfolded ensemble on a contact fraction is one bin that deepens for as
    # long as the walker sits in it, and a whole-range max |ΔF| never settles
    # there; the reference's job is the region the campaign is scored on.
    judge_lo, judge_hi = low - 0.5 * band, high + 0.5 * band
    factor = colvar_to_observable_factor(observable, observable_cv)

    def judge(ns_done: float) -> tuple[dict[str, Any], Any]:
        # For `rmsd` the biased CV's own hills carry the drift test; the
        # surface on the observable comes from reweighting instead.
        hills = work_dir / (hills_for_surface or bias.hills_file)
        n_hills = sum(
            1 for line in hills.read_text().splitlines() if line and not line.startswith("#")
        )
        stride = max(10, n_hills // 500)
        surfaces = sum_hills(hills, work_dir / "fes", stride=stride)
        final_biased = load_fes(surfaces[-1])
        baseline = (
            load_fes(surfaces[_baseline_index(len(surfaces))]) if len(surfaces) >= 2 else None
        )
        tail = load_fes(surfaces[int(0.8 * len(surfaces)) - 1]) if len(surfaces) >= 5 else None
        colvar_columns = load_colvar(work_dir / "COLVAR")
        series = colvar_columns[observable_column] * factor
        bias_series = colvar_columns[bias.bias_value]
        if hills_for_surface is not None:
            final = _in_observable_units(final_biased, factor)
            # Drift on the observable's own marginal, in its units, over the
            # task-relevant range.
            crop = lambda f: _in_observable_units(f, factor).restricted_to(judge_lo, judge_hi)  # noqa: E731
            drift = fes_drift_kj_per_mol(crop(baseline), crop(final_biased)) if baseline else None
            drift_tail = fes_drift_kj_per_mol(crop(tail), crop(final_biased)) if tail else None
        else:
            final = reweighted_profile(series, bias_series, temperature_k, label=observable.name)
            drift = fes_drift_kj_per_mol(baseline, final_biased) if baseline else None
            drift_tail = fes_drift_kj_per_mol(tail, final_biased) if tail else None
        recrossings = count_recrossings(series, low, high)
        delta_g = delta_g_kj_per_mol(final, low, high, temperature_k)
        # Stationarity of the reweighting: the last half against the whole.
        half = series.size // 2
        late = reweighted_profile(
            series[half:], bias_series[half:], temperature_k, label=observable.name,
        )
        delta_g_late = delta_g_kj_per_mol(late, low, high, temperature_k)
        dg_gap = (
            abs(delta_g - delta_g_late)
            if delta_g is not None and delta_g_late is not None else None
        )
        converged = (
            drift_tail is not None and drift_tail < kt
            and dg_gap is not None and dg_gap < kt
            and recrossings >= min_recrossings
        )
        summary: dict[str, Any] = {
            "task_file": str(task_file),
            "task_sha256": task.sha256,
            "forcefield": task.spec.forcefield,
            "observable": observable.name,
            "bias_cv": ",".join(bias.cv_labels),
            "state_thresholds": [low, high],
            "judged_range": [judge_lo, judge_hi],
            "seed": seed,
            "biased_ns": ns_done,
            "sigma": list(bias.sigma),
            "sigma_floored": getattr(bias, "sigma_floored", None),
            "height_kj_per_mol": bias.height,
            "pace": bias.pace,
            "bias_factor": bias.bias_factor,
            "temperature_k": temperature_k,
            "n_hills": n_hills,
            "n_fes_estimates": len(surfaces),
            "fes_drift_kj_per_mol": drift,
            "fes_drift_last_fifth_kj_per_mol": drift_tail,
            "recrossings": recrossings,
            "min_recrossings": min_recrossings,
            # F(unfolded) - F(folded): positive means the hairpin is the stable state.
            "delta_g_low_minus_high_kj_per_mol": delta_g,
            "delta_g_low_minus_high_last_half_kj_per_mol": delta_g_late,
            "observable_min": float(series.min()),
            "observable_max": float(series.max()),
            "converged": converged,
            "generated": datetime.now().isoformat(timespec="seconds"),
        }
        _log(f"judged at {ns_done:g} ns: drift(last fifth, task range)={drift_tail} kJ/mol "
             f"(kT={kt:.2f}), dG={delta_g} vs last-half {delta_g_late}, "
             f"recrossings={recrossings} (need {min_recrossings}) -> "
             f"{'CONVERGED' if converged else 'not yet'}")
        return summary, final

    # --- biased: chunked, checkpointed, resumable; judged every `check_every` ---
    done = sorted(chunks_dir.glob("chunk_*.chk"))
    n_done = len(done)
    chunk_steps = int(chunk_ns * steps_per_ns)
    n_chunks = int(round(total_ns / chunk_ns))
    summary: dict[str, Any] | None = None
    final = None
    if n_done >= n_chunks:
        _log(f"all {n_chunks} chunks already on disk; integrating")
    else:
        biased = OpenMMAdapter(
            work_dir=work_dir, seed=seed, spec=task.spec,
            plumed_input=enable_restart(plumed_input) if n_done else plumed_input,
        )
        biased.prepare()
        biased.start()
        if n_done:
            biased.load_checkpoint(done[-1])
            _log(f"resumed from {done[-1].name} ({n_done}/{n_chunks} chunks, RESTART on)")
        for i in range(n_done, n_chunks):
            t0 = time.monotonic()
            biased.run_steps(
                chunk_steps,
                trajectory_path=chunks_dir / f"chunk_{i + 1:03d}.dcd",
                report_interval_steps=frame_steps,
            )
            biased.save_checkpoint(chunks_dir / f"chunk_{i + 1:03d}.chk")
            elapsed = time.monotonic() - t0
            remaining = (n_chunks - i - 1) * elapsed
            _log(f"chunk {i + 1}/{n_chunks} ({(i + 1) * chunk_ns:g} ns) in "
                 f"{_fmt_time(elapsed)}; ~{_fmt_time(remaining)} left")
            if check_every and (i + 1) % check_every == 0 and (i + 1) < n_chunks:
                summary, final = judge((i + 1) * chunk_ns)
                if summary["converged"]:
                    _log("stopping early: the reference has converged")
                    break
    if summary is None or not summary["converged"]:
        n_have = len(sorted(chunks_dir.glob("chunk_*.chk")))
        summary, final = judge(n_have * chunk_ns)

    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "reference.json").write_text(json.dumps(summary, indent=2))
    if summary["converged"]:
        write_fes(final, data_dir / "reference_fes.dat")
        _log(f"reference written at {summary['biased_ns']:g} ns: "
             f"recrossings={summary['recrossings']}, "
             f"dG(unfolded - folded)={summary['delta_g_low_minus_high_kj_per_mol']} kJ/mol")
    else:
        (data_dir / "reference_fes.dat").unlink(missing_ok=True)
        _log("NOT converged; no reference written. Re-run with a larger --ns to "
             "continue from the last chunk.")
    return summary


def _in_observable_units(fes: FreeEnergySurface, factor: float) -> FreeEnergySurface:
    return FreeEnergySurface(fes.cv_label, fes.cv * factor, fes.free_energy, fes.periodic)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", type=float, default=200.0,
                        help="ceiling on biased length; stops early once converged")
    parser.add_argument("--check-every", type=int, default=10,
                        help="judge convergence every N chunks")
    parser.add_argument("--chunk-ns", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=_REFERENCE_SEED)
    parser.add_argument("--min-recrossings", type=int, default=4,
                        help="transitions the reference itself must show")
    parser.add_argument("--bias", choices=["observable", "rmsd", "pbmetad"], default="observable",
                        help="what to bias; the surface is on the observable either way")
    parser.add_argument("--dry-run", action="store_true",
                        help="0.05 ns in 0.01 ns chunks into a scratch dir; proves the "
                             "pipeline, produces no reference")
    args = parser.parse_args(argv)

    if args.dry_run:
        summary = build_reference(
            total_ns=0.05, chunk_ns=0.01, seed=args.seed, min_recrossings=1,
            bias_cv=args.bias, check_every=0, dry_run=True,
        )
    else:
        summary = build_reference(
            total_ns=args.ns, chunk_ns=args.chunk_ns, seed=args.seed,
            min_recrossings=args.min_recrossings, bias_cv=args.bias,
            check_every=args.check_every,
        )
    print(json.dumps(summary, indent=2))
    return 0 if summary["converged"] or args.dry_run else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
