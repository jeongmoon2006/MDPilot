"""Generate the CLN025 reference free-energy surface.

The campaign benchmark asks whether the scientist's biased phase recovered the
right surface. "Right" needs a yardstick that does not depend on which CV the
scientist happened to bias, so this runs a long well-tempered metadynamics
simulation *directly on the campaign observable* — the fraction of native CA
contacts the task file declares — through the same adapter, force field, bias
designer and PLUMED writer a campaign uses. Only the length and the seed
differ. A campaign whose reweighted surface on that observable matches this one
found the surface the method converges to, which is the agent's job; whether
the force field's surface is nature's is a separate question the literature
band in `run_cln025_e2e.py` addresses.

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

Output, under `benchmarks/data/cln025/` (gitignored, regenerable):
    reference_fes.dat   the final sum_hills surface, F(Q) in kJ/mol
    reference.json      what it is: length, seed, SIGMA, drift, recrossings,
                        ΔG between the task's states, and the pass verdict
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import mdtraj as md
import numpy as np

from mdpilot.adapters.openmm_adapter import OpenMMAdapter
from mdpilot.adapters.plumed_writer import PlumedInput, enable_restart
from mdpilot.diagnostics.free_energy import (
    _KB_KJ_PER_MOL_K,
    _baseline_index,
    count_recrossings,
    fes_drift_kj_per_mol,
    load_colvar,
    load_fes,
    sum_hills,
)
from mdpilot.orchestrator.loop import steps_per_ns_for
from mdpilot.sampling.bias_designer import design_bias
from mdpilot.sampling.cv_designer import CVProposal, design_cv
from mdpilot.task_file import load_task_file

_TASK_FILE = Path("benchmarks/tasks/cln025_contacts.yaml")
_DATA_DIR = Path("benchmarks/data/cln025")
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
    data_dir: Path = _DATA_DIR,
    task_file: Path = _TASK_FILE,
) -> dict[str, Any]:
    task = load_task_file(task_file)
    observable = task.campaign.get("observable")
    assert observable is not None, "the task file must declare the observable"
    low, high = task.campaign["state_thresholds"]

    work_dir = data_dir / "reference"
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
    cv = design_cv(
        CVProposal(
            cv_type=observable.cv_type,
            selections=tuple(observable.selections),
            label=observable.name,
        ),
        reference.topology,
        reference=reference,
        output_dir=work_dir,
    )
    bias = design_bias(cv, vanilla_dcd, vanilla.topology_path, temperature_k=temperature_k)
    plumed_input = PlumedInput(cvs=(cv,), bias=bias, output_dir=work_dir.resolve()).render()
    (work_dir / "plumed.reference.dat").write_text(plumed_input)
    _log(f"bias: SIGMA={bias.sigma[0]:.4g} (floored={bias.sigma_floored}) "
         f"HEIGHT={bias.height:.3g} PACE={bias.pace} gamma={bias.bias_factor:g}")

    # --- biased: chunked, checkpointed, resumable ---
    done = sorted(chunks_dir.glob("chunk_*.chk"))
    n_done = len(done)
    chunk_steps = int(chunk_ns * steps_per_ns)
    n_chunks = int(round(total_ns / chunk_ns))
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

    # --- integrate and judge ---
    hills = work_dir / bias.hills_file
    colvar = work_dir / "COLVAR"
    n_hills = sum(1 for line in hills.read_text().splitlines() if line and not line.startswith("#"))
    stride = max(10, n_hills // 500)
    surfaces = sum_hills(hills, work_dir / "fes", stride=stride)
    final = load_fes(surfaces[-1])
    baseline = load_fes(surfaces[_baseline_index(len(surfaces))]) if len(surfaces) >= 2 else None
    drift = fes_drift_kj_per_mol(baseline, final) if baseline is not None else None
    series = load_colvar(colvar)[final.cv_label]
    recrossings = count_recrossings(series, low, high)
    kt = _KB_KJ_PER_MOL_K * temperature_k
    delta_g = _delta_g(final, low, high, kt)

    converged = drift is not None and drift < kt and recrossings >= min_recrossings
    summary: dict[str, Any] = {
        "task_file": str(task_file),
        "task_sha256": task.sha256,
        "observable": observable.name,
        "state_thresholds": [low, high],
        "seed": seed,
        "biased_ns": n_chunks * chunk_ns,
        "sigma": bias.sigma[0],
        "sigma_floored": bias.sigma_floored,
        "height_kj_per_mol": bias.height,
        "pace": bias.pace,
        "bias_factor": bias.bias_factor,
        "temperature_k": temperature_k,
        "n_hills": n_hills,
        "n_fes_estimates": len(surfaces),
        "fes_drift_kj_per_mol": drift,
        "recrossings": recrossings,
        "min_recrossings": min_recrossings,
        "delta_g_unfolded_minus_folded_kj_per_mol": delta_g,
        "observable_min": float(series.min()),
        "observable_max": float(series.max()),
        "converged": converged,
        "generated": datetime.now().isoformat(timespec="seconds"),
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "reference.json").write_text(json.dumps(summary, indent=2))
    if converged:
        shutil.copy2(surfaces[-1], data_dir / "reference_fes.dat")
        _log(f"reference written: drift={drift:.2f} kJ/mol, recrossings={recrossings}, "
             f"dG(unfolded - folded)={delta_g:.2f} kJ/mol")
    else:
        (data_dir / "reference_fes.dat").unlink(missing_ok=True)
        _log(f"NOT converged: drift={drift} kJ/mol (kT={kt:.2f}), "
             f"recrossings={recrossings} (need {min_recrossings}); no reference written. "
             f"Re-run with a larger --ns to continue from the last chunk.")
    return summary


def _delta_g(fes, low: float, high: float, kt: float) -> float | None:
    """F(unfolded) - F(folded) from the surface, by integrating the populations.

    Positive means the folded state (observable above `high`) is more stable.
    Returns None when either state has no grid points in the sampled surface.
    """
    p = np.exp(-(fes.free_energy - fes.free_energy.min()) / kt)
    folded = p[fes.cv >= high].sum()
    unfolded = p[fes.cv <= low].sum()
    if folded <= 0 or unfolded <= 0:
        return None
    return float(-kt * np.log(unfolded / folded))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", type=float, default=50.0, help="total biased length")
    parser.add_argument("--chunk-ns", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=_REFERENCE_SEED)
    parser.add_argument("--min-recrossings", type=int, default=4,
                        help="transitions the reference itself must show")
    parser.add_argument("--dry-run", action="store_true",
                        help="0.05 ns in 0.01 ns chunks into a scratch dir; proves the "
                             "pipeline, produces no reference")
    args = parser.parse_args(argv)

    if args.dry_run:
        summary = build_reference(
            total_ns=0.05, chunk_ns=0.01, seed=args.seed, min_recrossings=1,
            data_dir=_DATA_DIR / "dryrun",
        )
    else:
        summary = build_reference(
            total_ns=args.ns, chunk_ns=args.chunk_ns, seed=args.seed,
            min_recrossings=args.min_recrossings,
        )
    print(json.dumps(summary, indent=2))
    return 0 if summary["converged"] or args.dry_run else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
