# Related Work

## Reference table

| Work | Why it matters for MDPilot |
|---|---|
| MDCrow (Campbell et al., arXiv 2502.09565) | Closest precursor; we delegate setup to it via `adapters/mdcrow_adapter.py` |
| DynaMate (arXiv 2512.10034) | Protein-ligand multi-agent; cleanest articulation of the three-stage planner/executor/analyzer pipeline |
| MDAgent2 / MD-GRPO (arXiv 2601.02075) | Closed-loop RL on LAMMPS code generation — different angle, watch for framing overlap when writing the paper |
| HTC-Claw (arXiv 2604.06076) | One of the few existing closed-loop campaign systems (materials, not biomolecular) |
| MatClaw (arXiv 2604.02688) | Architectural argument for single-agent over multi-agent for long-horizon work |
| NAMD-Agent (arXiv 2507.07887) | CHARMM-GUI / NAMD pipeline reference |
| PolyJarvis | First MCP-based MD agent; useful pattern for cross-engine adapters |
| MatSciAgent (Nature Comms Materials, 2026) | Multi-task materials agent; hierarchical dispatch pattern |
| ToPolyAgent (arXiv 2510.12091) | Coarse-grained topological polymers; example of going beyond all-atom |
| ChemCrow, Coscientist | Earlier chemistry agent precursors that established the design space |

---

## How MDPilot positions itself

MDPilot is **not** another setup-and-run agent. It builds on top of those.

- **Vs. MDCrow / DynaMate / NAMD-Agent / PolyJarvis:** they handle the inner loop (setup → run → analyze). MDPilot handles the outer loop (judge → decide → replan) and calls them as tools.
- **Vs. MDAgent2:** they do closed-loop RL on code generation (training-time feedback to make a model that writes better LAMMPS). MDPilot does closed-loop reasoning on simulation results (inference-time decisions about what to simulate). Different layer of the stack; not competitors.
- **Vs. HTC-Claw:** thematically closest — they have a "submit–monitor–analyze–report" loop for materials high-throughput campaigns. MDPilot brings outer-loop reasoning to biomolecular MD with deeper convergence and sampling judgment.
- **Vs. MatClaw:** shares the single-agent design philosophy. MatClaw is code-first and general-purpose for materials; MDPilot is tool-calling and specialized for MD scientific reasoning.

---

## What's missing in the field that MDPilot targets

From reading the above, the consistent gaps:

1. **Convergence judgment.** Most agents run for the time the user requested and stop. None reliably says "this isn't converged, extend" or "this won't ever converge with vanilla MD, switch methods."
2. **Enhanced sampling decisions.** CV selection for metadynamics, REMD ladder design, umbrella window placement — almost no agent attempts these.
3. **Hypothesis-driven follow-up.** No existing agent says "the binding pose is ambiguous; let me run alchemical FEP to disambiguate" without the user prompting that explicitly.
4. **Cross-trajectory ensemble reasoning.** Each existing agent reasons one trajectory at a time.
5. **Standard benchmarks.** Every system invents its own evaluation; there is no SWE-bench equivalent for MD agents.

MDPilot's milestones map directly onto gaps 1–4. Gap 5 is Milestone 6.
