"""Run the fault-injection suite: broken campaigns, three verifier arms, a score.

For each registered fault (`mdpilot.faults.FAULTS`) build the task file's
system, break the campaign the way the fault says, drive it with the scripted
policy to its budget, replay the verifier arms over the finished record
(`mdpilot.ablation`), and score the lot: false-accept rate and detection
latency per arm, and per fault whether the falsifiers registered against it
fired, and when.

    export MAMBA_ROOT_PREFIX=$HOME/.micromamba
    ~/.local/bin/micromamba run -n mdpilot python -m benchmarks.run_faults --dry-run     # minutes
    ~/.local/bin/micromamba run -n mdpilot python -m benchmarks.run_faults               # ~6 x 20 ns
    ~/.local/bin/micromamba run -n mdpilot python -m benchmarks.run_faults --faults gamma_too_small
    python -m benchmarks.run_faults --score-only                                        # re-score

The LLM arms are opt-in (`--arms gates_only,lm_only,lm_plus_gates`): each
costs one call per round per arm. The default runs the deterministic arm
only, which is enough to see whether the falsifiers fire.

Outputs, under `campaigns/faults/` (or `campaigns/faults_dryrun/`):
    <fault>/                the broken campaign, an ordinary work_dir
    <fault>/fault.json      the fault record: overrides, expected falsifiers, why it is wrong
    <fault>/ablation/       the arms' per-round verdicts and the recomputed reports
    fault_report.json       the score
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from mdpilot.ablation import ARMS, ArmConfig, Comparison, run_comparison
from mdpilot.faults import ABLATION_DIR, FAULTS, Fault, score, scripted_policy, write_fault_record
from mdpilot.orchestrator.loop import run_campaign, steps_per_ns_for
from mdpilot.task_file import TaskFile, load_task_file

_TASK_FILE = Path("benchmarks/tasks/cln025_contacts.yaml")
_DRY_RUN_BIASED_NS = 0.1


def _log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


class _Trace:
    def __init__(self, path: Path) -> None:
        self._path = path

    def __call__(self, name: str, payload: dict[str, Any]) -> None:
        row = {"t": datetime.now().isoformat(timespec="seconds"), "event": name, **payload}
        with self._path.open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")


def run_fault(
    task: TaskFile,
    fault: Fault,
    work_dir: Path,
    *,
    dry_run: bool,
    max_rounds: int,
    extend_ns: float,
    biased_ns: float,
    arms: tuple[str, ...],
    model: str,
) -> dict[str, Any]:
    """Produce one broken campaign and replay the arms over it."""
    work_dir.mkdir(parents=True, exist_ok=True)
    write_fault_record(work_dir, fault)
    adapter = task.build_adapter(work_dir)
    steps_per_ns = steps_per_ns_for(adapter)
    overrides: dict[str, Any] = {
        "max_rounds": max_rounds,
        "max_extra_ns": extend_ns,
        "initial_steps": int((0.05 if dry_run else 1.0) * steps_per_ns),
        "report_interval_steps": max(int(0.005 * steps_per_ns), 1),   # 5 ps/frame
        "max_biased_ns": biased_ns,
        **fault.overrides,
    }
    if dry_run:
        # A dry run only proves the plumbing; every fault is "truncated" then.
        overrides["max_biased_ns"] = min(_DRY_RUN_BIASED_NS, overrides["max_biased_ns"])
    t0 = time.monotonic()
    result = run_campaign(
        work_dir=work_dir, adapter=adapter, on_event=_Trace(work_dir / "trace.jsonl"),
        decide_fn=scripted_policy(fault, extend_ns=extend_ns),
        **task.run_kwargs(**overrides),
    )
    _log(f"{fault.name}: {len(result.rounds)} rounds, {result.stop_reason}, "
         f"{(time.monotonic() - t0) / 60:.1f} min")
    comparison = Comparison(tuple(ArmConfig(a, work_dir, model) for a in arms))  # type: ignore[arg-type]
    return run_comparison(comparison, out_dir=work_dir / ABLATION_DIR)


def _table(report: dict[str, Any]) -> str:
    lines = ["arm             faults  false-accepts  rate    latency ns (per fault)"]
    for arm, s in report["arms"].items():
        if not s["n_faults"]:
            continue
        rate = f"{s['false_accept_rate']:.2f}" if s["false_accept_rate"] is not None else "-"
        lat = " ".join(
            f"{k}={'-' if v is None else f'{v:g}'}" for k, v in s["latency_ns"].items()
        )
        lines.append(f"{arm:15s} {s['n_faults']:6d}  {s['false_accepts']:13d}  {rate:6s}  {lat}")
    lines.append("")
    lines.append("fault                            detected  fired at ns                     unexpected")
    for name, f in report["faults"].items():
        fired = " ".join(f"{k}={'-' if v is None else f'{v:g}'}" for k, v in f["fired_at_ns"].items())
        lines.append(f"{name:32s} {str(f['detected']):8s}  {fired:31s}  {f['unexpected_fires']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, default=_TASK_FILE)
    parser.add_argument("--faults", default=",".join(FAULTS), help="comma-separated fault names")
    parser.add_argument("--dry-run", action="store_true", help="0.05 ns opening, 0.1 ns biased cap")
    parser.add_argument("--max-rounds", type=int, default=40)
    parser.add_argument("--extend-ns", type=float, default=2.0)
    parser.add_argument("--biased-ns", type=float, default=20.0,
                        help="biased budget per fault unless the fault overrides it")
    parser.add_argument("--arms", default="gates_only",
                        help="comma-separated subset of " + ",".join(ARMS))
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--score-only", action="store_true", help="score existing fault campaigns")
    args = parser.parse_args(argv)

    out = args.out or Path("campaigns/faults_dryrun" if args.dry_run else "campaigns/faults")
    names = [n.strip() for n in args.faults.split(",") if n.strip()]
    unknown = sorted(set(names) - set(FAULTS))
    if unknown:
        parser.error(f"unknown fault(s) {unknown}; registered: {sorted(FAULTS)}")
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())

    if not args.score_only:
        task = load_task_file(args.task)
        for name in names:
            _log(f"--- {name}: {FAULTS[name].description}")
            try:
                run_fault(
                    task, FAULTS[name], out / name, dry_run=args.dry_run,
                    max_rounds=args.max_rounds, extend_ns=args.extend_ns,
                    biased_ns=args.biased_ns, arms=arms, model=args.model,
                )
            except Exception as exc:  # noqa: BLE001 - one broken fault must not hide the others
                _log(f"{name}: FAILED {type(exc).__name__}: {exc}")
                (out / name / "error.txt").parent.mkdir(parents=True, exist_ok=True)
                (out / name / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n")

    done = [out / n for n in names if (out / n / ABLATION_DIR / "summary.json").exists()]
    if not done:
        _log("nothing to score")
        return 1
    report = score(done)
    report["generated"] = datetime.now().isoformat(timespec="seconds")
    report["dry_run"] = args.dry_run
    (out / "fault_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(_table(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
