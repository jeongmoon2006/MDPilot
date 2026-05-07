# MDPilot

A closed-loop scientific reasoning agent for molecular dynamics simulations.
The outer loop on top of MD operator agents like MDCrow: judge convergence,
decide what to simulate next.

> An MD operator runs the simulation you describe.
> An MD scientist decides what to simulate next.
> MDPilot is the latter.

- For *what* MDPilot is and why it exists, see [`docs/architecture.md`](docs/architecture.md).
- For *what to build next*, see [`ROADMAP.md`](ROADMAP.md).
- For *how to work on this codebase*, see [`CLAUDE.md`](CLAUDE.md).
- For *positioning vs related work*, see [`docs/related_work.md`](docs/related_work.md).

## Status

Milestone 1 (closed loop on Trp-cage convergence) — in progress.

## Setup

Requires Python 3.10+ and an Anthropic API key. OpenMM CPU build is sufficient
for Milestone 1 (Trp-cage in TIP3P, ~6500 atoms).

```sh
python3 -m venv .venv
.venv/bin/pip install -e .
cp .env.example .env  # then edit, setting ANTHROPIC_API_KEY
```

## Run a single round on Trp-cage

```python
from pathlib import Path
from mdpilot.orchestrator.loop import run_campaign

result = run_campaign(
    work_dir=Path("campaigns/demo"),
    initial_steps=10_000,   # 20 ps at 2 fs
    max_rounds=1,
)
print(result.rounds[0].decision)
```

The first call downloads PDB 1L2Y via PDBFixer, solvates and minimises
(~60 s on CPU), then runs the requested simulation. Per-round artifacts
land in `campaigns/demo/rounds/`.

## Tests

```sh
pytest tests/unit                 # ~3 s — pure Python, no API or MD
pytest tests/integration          # ~30 s + Anthropic API cost; auto-skipped if ANTHROPIC_API_KEY is unset
```

The Milestone-1 done-criterion test (`tests/integration/test_milestone1_live.py`)
needs the planted reference trajectories to be on disk:

```sh
python -m benchmarks.generate_trpcage_planted --traj under       # 50 ps,  ~6 min CPU
python -m benchmarks.generate_trpcage_planted --traj converged   # 5 ns,  ~9 hours CPU (overnight)
```

Trajectories land under `benchmarks/data/trpcage/` (gitignored — they are
deterministic given the same seed).

## License

See [`LICENSE`](LICENSE).
