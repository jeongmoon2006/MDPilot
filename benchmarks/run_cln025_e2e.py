"""The CLN025 end-to-end case: launch, detect, reason, correct, recover the surface.

The one campaign MDPilot exists for, run as a benchmark. The scientist starts
unbiased MD on chignolin, reads the bimodality gate (`exploring=false` against
a task that demands a transition), pivots to metadynamics on a CV of its own
choosing, watches the biased phase — and if the coordinate cannot bring the
walker back, says so and replaces it. The campaign is then scored on the
surface it recovered *along the campaign observable*, whatever CV it ended up
biasing, against two yardsticks:

- the in-pipeline reference (`benchmarks/generate_cln025_reference.py`): a
  much longer, independently seeded run on the same force field. Matching it
  means the agent found the surface this method converges to, which is the
  agent's job;
- the literature: CLN025 is a stable hairpin at 300 K — Tm 343 K, ~90% folded,
  ΔG_fold ≈ 0.3–0.5 kcal/mol by CD/NMR (Honda et al., JACS 130:15327, 2008).
  Force fields disagree with each other on chignolin (Kührová et al., Biophys
  J 102:1897, 2012), and ff14SB/TIP3P in particular folds it poorly — an
  aggregated native population of 7 ± 14% at 277 K over microseconds (Pang,
  Proteins 84:1490, 2016). So the literature is *reported*, not gated: it says
  whether the force field's surface is nature's, which is not the agent's
  doing. The pass line is the in-pipeline reference.

Self-correction is *recorded*, not required. The scientist chooses the first
CV; native contacts worked on one real campaign and trapped the walker on two
others (F13). Forcing the mistake would mean pre-selecting a bad coordinate,
which is the anti-goal. What is required is that the campaign pivots and that
the surface it ends with is right.

    export MAMBA_ROOT_PREFIX=$HOME/.micromamba
    ~/.local/bin/micromamba run -n mdpilot python -m benchmarks.run_cln025_e2e            # ~5 h
    ~/.local/bin/micromamba run -n mdpilot python -m benchmarks.run_cln025_e2e --dry-run  # minutes
    python -m benchmarks.run_cln025_e2e --verdict-only --work-dir campaigns/<existing>  # score only

Writes into the work dir: `trace.jsonl` (every campaign event), `verdict.json`
and `case_study.md` (the round-by-round account).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from mdpilot.diagnostics.free_energy import (
    _KB_KJ_PER_MOL_K,
    FreeEnergySurface,
    delta_g_kj_per_mol,
    load_colvar,
    load_fes,
    reweighted_profile,
)
from mdpilot.memory import store
from mdpilot.orchestrator.loop import (
    observable_surface_from_colvar,
    run_campaign,
    steps_per_ns_for,
)
from mdpilot.task_file import TaskFile, load_task_file

_TASK_FILE = Path("benchmarks/tasks/cln025_contacts.yaml")
_REFERENCE = Path("benchmarks/data/cln025/reference_fes.dat")
_REFERENCE_META = Path("benchmarks/data/cln025/reference.json")

# ΔG(unfolded) - ΔG(folded) the literature supports at 300 K, kJ/mol. The
# lower bound is the sign — the hairpin is the stable state. The upper bound
# is generous: ~90% folded is 5.5 kJ/mol, and the CD/NMR numbers are smaller
# still; 10 kJ/mol (~4 kT, 98% folded) is where "stable hairpin" becomes
# "rigid", and the trapped campaigns reported over 100.
_LITERATURE_DG_KJ = (0.0, 10.0)
_LITERATURE = [
    "Honda et al., J. Am. Chem. Soc. 130:15327 (2008): CLN025 Tm = 343 K, "
    "~90% folded at 300 K, ΔG_fold 0.26-0.45 kcal/mol at 298 K (CD, NMR)",
    "Kührová et al., Biophys. J. 102:1897 (2012): chignolin folding and "
    "misfolding are force-field dependent",
    "Pang, Proteins 84:1490 (2016): under ff14SB/TIP3P at 277 K, CLN025 "
    "aggregated native-state population 7 ± 14%, folding time ~1 µs",
]
# How closely the campaign's surface must track the reference over the range
# both sampled: kT on the root-mean-square, the same scale the well-tempered
# drift test uses. The maximum deviation is reported but not gated — the
# edges of a 20 ns surface are the least sampled points on it.
_MATCH_RMS_KT = 1.0
_MATCH_DG_KT = 1.0

_EVENTS = {
    "campaign_start", "preflight_ok", "round_start", "simulated", "report",
    "decision", "override", "pivot", "campaign_end",
}


class _Trace:
    """Every event to `trace.jsonl`, and a one-line log for the terminal."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "a")   # noqa: SIM115 - lives as long as the campaign

    def __call__(self, name: str, payload: dict[str, Any]) -> None:
        stamp = datetime.now().isoformat(timespec="seconds")
        self._file.write(json.dumps({"t": stamp, "event": name, **payload}, default=str) + "\n")
        self._file.flush()
        if name == "report":
            return
        body = "  ".join(f"{k}={str(v)[:120]}" for k, v in payload.items())
        print(f"[{stamp[11:]}] {name:<14} {body}", flush=True)


def run(task: TaskFile, work_dir: Path, *, dry_run: bool, max_rounds: int) -> None:
    adapter = task.build_adapter(work_dir)
    steps_per_ns = steps_per_ns_for(adapter)
    overrides: dict[str, Any] = {
        "max_rounds": max_rounds,
        "max_extra_ns": 2.0,
        "initial_steps": int((0.05 if dry_run else 1.0) * steps_per_ns),
        "report_interval_steps": max(int(0.005 * steps_per_ns), 1),   # 5 ps/frame
    }
    if dry_run:
        overrides["max_biased_ns"] = 0.1
    run_campaign(
        work_dir=work_dir, adapter=adapter, on_event=_Trace(work_dir / "trace.jsonl"),
        **task.run_kwargs(**overrides),
    )


# --------------------------------------------------------------------------
# The verdict. Reads a finished (or interrupted) campaign off disk.
# --------------------------------------------------------------------------

def verdict(
    work_dir: Path,
    task: TaskFile,
    *,
    reference_path: Path = _REFERENCE,
    reference_meta: Path = _REFERENCE_META,
) -> dict[str, Any]:
    rows = store.list_rounds(work_dir)
    notes = store.list_ledger_notes(work_dir)
    low, high = task.campaign["state_thresholds"]
    temperature_k = task.spec.ensemble.temperature_k
    kt = _KB_KJ_PER_MOL_K * temperature_k

    biased = [r for r in rows if r.plumed_dat_path is not None]
    pivot = next((r for r in rows if r.decision == "switch_to_metad"), None)
    switches = [r for r in rows if r.decision == "switch_cv"]

    out: dict[str, Any] = {
        "work_dir": str(work_dir),
        "task_sha256": task.sha256,
        "n_rounds": len(rows),
        "n_biased_rounds": len(biased),
        "biased_ns": None,
        "stop": rows[-1].decision if rows else None,
    }
    steps_per_ns = 1e6 / task.spec.ensemble.timestep_fs
    out["biased_ns"] = sum(r.n_steps for r in biased) / steps_per_ns

    # 1. The gate: what the vanilla report said when the scientist pivoted.
    out["pivoted"] = pivot is not None
    if pivot is not None:
        rep = pivot.report
        out["pivot"] = {
            "round": pivot.round_index,
            "exploring": rep.get("exploring"),
            "n_basins": rep.get("n_basins"),
            "bimodality_coefficient": rep.get("bimodality_coefficient"),
            "ess": rep.get("ess"),
            "plateau_reached": rep.get("plateau_reached"),
            "cv": pivot.metad_proposal,
            "reason": pivot.reason,
        }

    # 2. Self-correction: did it change the plan, and on what evidence?
    out["self_corrected"] = bool(switches)
    out["cv_switches"] = [
        {
            "round": r.round_index,
            "from": r.report.get("cv_label"),
            "to": r.metad_proposal,
            "rounds_since_high_visited": r.report.get("rounds_since_high_visited"),
            "rounds_since_low_visited": r.report.get("rounds_since_low_visited"),
            "rounds_confined": r.report.get("rounds_confined"),
            "fes_depth_kj_per_mol": r.report.get("fes_depth_kj_per_mol"),
            "recrossings": r.report.get("recrossings"),
            "reason": r.reason,
        }
        for r in switches
    ]
    # ... and the corrections the loop made on its behalf.
    out["loop_refusals"] = [
        {"round": n.round_index, "note": n.text}
        for n in notes
        if n.text.startswith(("stop refused", "decision refused", "proposal refused"))
    ]

    # 3. The surface on the campaign observable, from the last biased round.
    surface = _observable_surface(work_dir, biased, task)
    out["observable"] = task.observable_name
    out["observable_surface"] = None
    if surface is not None:
        dg = delta_g_kj_per_mol(surface, low, high, temperature_k)
        out["observable_surface"] = {
            "sampled_range": [float(surface.cv.min()), float(surface.cv.max())],
            "delta_g_low_minus_high_kj_per_mol": dg,
            "n_basins": len(surface.minima(temperature_k)),
            "barrier_kj_per_mol": surface.barrier_kj_per_mol(temperature_k),
        }
        out["literature"] = {
            "band_kj_per_mol": list(_LITERATURE_DG_KJ),
            "sources": _LITERATURE,
            "consistent": (
                dg is not None and _LITERATURE_DG_KJ[0] < dg < _LITERATURE_DG_KJ[1]
            ),
            "note": (
                "Reported, not gated: whether the force field's surface agrees "
                "with experiment is not the agent's doing."
            ),
        }

    # 4. Against the in-pipeline reference.
    out["reference"] = None
    if surface is not None and reference_path.exists():
        reference = load_fes(reference_path)
        meta = json.loads(reference_meta.read_text()) if reference_meta.exists() else {}
        dev = surface_deviation(surface, reference)
        dg_ref = delta_g_kj_per_mol(reference, low, high, temperature_k)
        dg_campaign = out["observable_surface"]["delta_g_low_minus_high_kj_per_mol"]
        dg_gap = (
            abs(dg_campaign - dg_ref)
            if dg_campaign is not None and dg_ref is not None
            else None
        )
        converged = bool(meta.get("converged", True))
        out["reference"] = {
            "path": str(reference_path),
            "biased_ns": meta.get("biased_ns"),
            "bias_cv": meta.get("bias_cv"),
            "seed": meta.get("seed"),
            "recrossings": meta.get("recrossings"),
            # The generator only writes `reference_fes.dat` once its own
            # stationarity checks pass; a surface handed in by path with a
            # `reference.json` that says otherwise is compared for the record
            # and cannot be matched against.
            "converged": converged,
            "delta_g_low_minus_high_kj_per_mol": dg_ref,
            **dev,
            "delta_g_gap_kj_per_mol": dg_gap,
            "matches": (
                converged
                and dev["rms_kj_per_mol"] is not None
                and dev["rms_kj_per_mol"] <= _MATCH_RMS_KT * kt
                and dg_gap is not None
                and dg_gap <= _MATCH_DG_KT * kt
            ),
        }

    # The agent is judged against the surface the method converges to. Without
    # a reference there is nothing to judge it against, and the verdict says
    # so rather than passing on the pivot alone.
    out["reference_available"] = out["reference"] is not None
    out["passed"] = bool(
        out["pivoted"] and out["reference"] is not None and out["reference"]["matches"]
    )
    out["verdict"] = (
        "PASS" if out["passed"]
        else "INCOMPLETE (no reference surface)" if out["pivoted"] and out["reference"] is None
        else "INCOMPLETE (reference unconverged)"
        if out["pivoted"] and not out["reference"]["converged"]
        else "FAIL"
    )
    return out


def surface_deviation(a: FreeEnergySurface, b: FreeEnergySurface) -> dict[str, Any]:
    """How far two surfaces sit apart over the range both sampled.

    Each is re-referenced to its own minimum over the overlap first, as
    `fes_drift_kj_per_mol` does, so an offset in where `--mintozero` put zero
    is not counted as disagreement.
    """
    lo = max(float(a.cv.min()), float(b.cv.min()))
    hi = min(float(a.cv.max()), float(b.cv.max()))
    if not hi > lo:
        return {"overlap": None, "rms_kj_per_mol": None, "max_kj_per_mol": None}
    grid = np.linspace(lo, hi, 200)
    fa = np.interp(grid, a.cv, a.free_energy)
    fb = np.interp(grid, b.cv, b.free_energy)
    fa, fb = fa - fa.min(), fb - fb.min()
    diff = fa - fb
    return {
        "overlap": [lo, hi],
        "rms_kj_per_mol": float(np.sqrt(np.mean(diff**2))),
        "max_kj_per_mol": float(np.abs(diff).max()),
    }


def _observable_surface(
    work_dir: Path, biased: list[store.RoundRow], task: TaskFile
) -> FreeEnergySurface | None:
    """The reweighted surface on the observable after the last biased round.

    Read from the round's report when the loop wrote one; otherwise rebuilt
    from COLVAR. A campaign recorded before the loop printed the observable
    there can still be scored when the CV it biased *is* the observable —
    the column is then under the biased CV's own label.
    """
    if not biased:
        return None
    last = biased[-1]
    path = last.report.get("observable_fes_path")
    if path and Path(path).exists():
        return load_fes(Path(path))
    colvar_path = last.plumed_dat_path.parent / "COLVAR"
    if not colvar_path.exists():
        return None
    surface = observable_surface_from_colvar(
        colvar_path, work_dir / "topology.pdb", task.campaign.get("observable"),
        task.spec.ensemble.temperature_k,
    )
    if surface is not None:
        return surface
    colvar = load_colvar(colvar_path)
    bias_column = next((k for k in colvar if k.endswith(".bias")), None)
    column = last.report.get("cv_label")
    if bias_column is None or column not in colvar or column != task.observable_name:
        return None
    return reweighted_profile(
        colvar[column], colvar[bias_column], task.spec.ensemble.temperature_k,
        label=task.observable_name,
    )


# --------------------------------------------------------------------------
# The account a human reads.
# --------------------------------------------------------------------------

def case_study(work_dir: Path, task: TaskFile, result: dict[str, Any]) -> str:
    rows = store.list_rounds(work_dir)
    notes = store.list_ledger_notes(work_dir)
    steps_per_ns = 1e6 / task.spec.ensemble.timestep_fs
    lines = [
        f"# {task.name}: end-to-end case",
        "",
        f"Campaign `{work_dir}`, task `{task.path}` (sha {task.sha256[:12]}). "
        f"{len(rows)} rounds, {result['biased_ns']:.1f} ns biased. "
        f"Verdict: **{result['verdict']}** "
        f"(pivoted={result['pivoted']}, self_corrected={result['self_corrected']}, "
        f"reference={result['reference']['matches'] if result['reference'] else 'n/a'}, "
        f"literature={result.get('literature', {}).get('consistent')} — reported, not gated).",
        "",
        "| round | phase | ns | what the report said | decision | CV |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        rep = r.report
        ns = r.n_steps / steps_per_ns
        if r.plumed_dat_path is None:
            said = (
                f"exploring={rep.get('exploring')} n_basins={rep.get('n_basins')} "
                f"BC={_fmt(rep.get('bimodality_coefficient'))} ess={_fmt(rep.get('ess'))}"
            )
        else:
            said = (
                f"drift={_fmt(rep.get('fes_drift_kj_per_mol'))} "
                f"recross={rep.get('recrossings')} "
                f"range=[{_fmt(rep.get('observable_min_this_round'))}, "
                f"{_fmt(rep.get('observable_max_this_round'))}] "
                f"since_high={rep.get('rounds_since_high_visited')} "
                f"depth={_fmt(rep.get('fes_depth_kj_per_mol'))} "
                f"converged={rep.get('fes_converged')}"
            )
        cv = (r.metad_proposal or {}).get("label", "") if r.metad_proposal else ""
        lines.append(
            f"| {r.round_index} | {'metad' if r.plumed_dat_path else 'vanilla'} | "
            f"{ns:.2f} | {said} | {r.decision} | {cv} |"
        )
    lines += ["", "## What the scientist said", ""]
    for r in rows:
        lines.append(f"**R{r.round_index} → {r.decision}.** {r.reason}")
        lines.append("")
    if notes:
        lines += ["## Ledger", ""]
        lines += [f"- R{n.round_index}: {n.text}" for n in notes]
        lines.append("")
    lines += ["## Verdict", "", "```json", json.dumps(result, indent=2), "```", ""]
    return "\n".join(lines)


def _fmt(v: Any) -> str:
    return f"{v:.3g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="0.05 ns opening, 0.1 ns biased cap")
    parser.add_argument("--verdict-only", action="store_true", help="score an existing campaign")
    parser.add_argument("--reference", type=Path, default=_REFERENCE)
    args = parser.parse_args(argv)

    task = load_task_file(_TASK_FILE)
    work_dir = args.work_dir or Path(
        "campaigns/cln025_e2e_dryrun" if args.dry_run else "campaigns/cln025_e2e"
    )
    if not args.verdict_only:
        run(task, work_dir, dry_run=args.dry_run, max_rounds=args.max_rounds)

    result = verdict(work_dir, task, reference_path=args.reference)
    (work_dir / "verdict.json").write_text(json.dumps(result, indent=2))
    (work_dir / "case_study.md").write_text(case_study(work_dir, task, result))
    print(json.dumps(result, indent=2))
    print(f"\ncase study: {work_dir / 'case_study.md'}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
