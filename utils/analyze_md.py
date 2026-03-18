import os
import logging
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import rms

logger = logging.getLogger(__name__)

TOPOLOGY_EXTENSIONS = ['.gro', '.pdb', '.psf', '.prmtop', '.top']
TRAJECTORY_EXTENSIONS = ['.xtc', '.dcd', '.trr', '.nc', '.ncdf']


def find_simulation_files(work_dir):
    """Find topology and trajectory files in the work directory."""
    topology = None
    trajectory = None
    for fname in sorted(os.listdir(work_dir)):
        fpath = os.path.join(work_dir, fname)
        ext = os.path.splitext(fname)[1].lower()
        if ext in TOPOLOGY_EXTENSIONS and topology is None:
            topology = fpath
        if ext in TRAJECTORY_EXTENSIONS and trajectory is None:
            trajectory = fpath
    return topology, trajectory


def analyze_md(work_dir, analysis_targets):
    """
    Run MDAnalysis on files in work_dir for the specified analysis targets.
    Returns a dict of human-readable text summaries (no raw arrays sent to LLM).
    """
    topology, trajectory = find_simulation_files(work_dir)

    if topology is None:
        return {"error": f"No topology file found in {work_dir}. Expected extensions: {TOPOLOGY_EXTENSIONS}"}

    logger.info(f"Loading topology: {topology}")
    logger.info(f"Loading trajectory: {trajectory}")

    if trajectory:
        u = mda.Universe(topology, trajectory)
    else:
        u = mda.Universe(topology)

    results = {}
    results["structure"] = _get_structure_info(u)

    for target in analysis_targets:
        if target == "rmsd":
            results["rmsd"] = _calc_rmsd(u, trajectory)
        elif target == "rmsf":
            results["rmsf"] = _calc_rmsf(u)
        elif target == "radius_of_gyration":
            results["radius_of_gyration"] = _calc_rg(u)
        elif target == "energy":
            results["energy"] = _read_energy(work_dir)

    return results


def _get_structure_info(u):
    lines = [
        f"Total atoms: {len(u.atoms)}",
        f"Total residues: {len(u.residues)}",
        f"Segments/chains: {len(u.segments)}",
    ]
    if hasattr(u, 'trajectory') and len(u.trajectory) > 1:
        lines.append(f"Trajectory frames: {len(u.trajectory)}")
        lines.append(f"Timestep: {u.trajectory.dt:.2f} ps")
        lines.append(f"Total simulation time: {u.trajectory.totaltime:.2f} ps")
    return "\n".join(lines)


def _calc_rmsd(u, trajectory):
    if trajectory is None:
        return "No trajectory file available for RMSD calculation."
    protein_ca = u.select_atoms("protein and name CA")
    if len(protein_ca) == 0:
        protein_ca = u.select_atoms("name CA")
    if len(protein_ca) == 0:
        return "No Cα atoms found. Cannot compute RMSD."

    R = rms.RMSD(protein_ca, ref_frame=0)
    R.run()
    rmsd_data = R.results.rmsd[:, 2]

    first_q = rmsd_data[:len(rmsd_data) // 4].mean()
    last_q = rmsd_data[3 * len(rmsd_data) // 4:].mean()
    drift = last_q - first_q

    lines = [
        f"RMSD analysis (Cα atoms, n={len(protein_ca)}):",
        f"  Min:   {rmsd_data.min():.3f} Å",
        f"  Max:   {rmsd_data.max():.3f} Å",
        f"  Mean:  {rmsd_data.mean():.3f} Å",
        f"  Final: {rmsd_data[-1]:.3f} Å",
    ]
    if abs(drift) > 1.0:
        lines.append(f"  WARNING: RMSD drift detected ({drift:+.3f} Å from first to last quarter of trajectory)")
    return "\n".join(lines)


def _calc_rmsf(u):
    protein_ca = u.select_atoms("protein and name CA")
    if len(protein_ca) == 0:
        protein_ca = u.select_atoms("name CA")
    if len(protein_ca) == 0:
        return "No Cα atoms found. Cannot compute RMSF."

    rmsf_calc = rms.RMSF(protein_ca)
    rmsf_calc.run()
    rmsf_data = rmsf_calc.results.rmsf

    top5_idx = np.argsort(rmsf_data)[-5:][::-1]
    lines = [
        f"RMSF analysis (Cα atoms, n={len(protein_ca)}):",
        f"  Mean RMSF: {rmsf_data.mean():.3f} Å",
        f"  Max RMSF:  {rmsf_data.max():.3f} Å",
        "  Most flexible residues (top 5):",
    ]
    for idx in top5_idx:
        resname = protein_ca.residues[idx].resname
        resid = protein_ca.residues[idx].resid
        lines.append(f"    {resname}{resid}: {rmsf_data[idx]:.3f} Å")
    return "\n".join(lines)


def _calc_rg(u):
    protein = u.select_atoms("protein")
    if len(protein) == 0:
        return "No protein atoms found. Cannot compute radius of gyration."

    rg_values = np.array([protein.radius_of_gyration() for _ in u.trajectory])
    lines = [
        "Radius of gyration (protein):",
        f"  Min:  {rg_values.min():.3f} Å",
        f"  Max:  {rg_values.max():.3f} Å",
        f"  Mean: {rg_values.mean():.3f} Å",
    ]
    return "\n".join(lines)


def _read_energy(work_dir):
    """Look for GROMACS .xvg energy files."""
    for fname in sorted(os.listdir(work_dir)):
        if fname.endswith('.xvg'):
            try:
                return _parse_xvg(os.path.join(work_dir, fname))
            except Exception as e:
                logger.warning(f"Failed to parse {fname}: {e}")
    return (
        "No .xvg energy file found. For GROMACS, export energy first:\n"
        "  gmx energy -f ener.edr -o energy.xvg"
    )


def _parse_xvg(filepath):
    labels = []
    rows = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('@ s') and 'legend' in line:
                labels.append(line.split('"')[1])
            if not line.startswith(('#', '@')):
                parts = line.split()
                try:
                    rows.append([float(x) for x in parts])
                except ValueError:
                    pass

    if not rows:
        return f"Energy file {os.path.basename(filepath)} found but could not parse numeric values."

    data = np.array(rows)
    lines = [f"Energy data from {os.path.basename(filepath)}:"]
    for i, label in enumerate(labels[:3]):
        col = i + 1
        if col < data.shape[1]:
            lines.append(f"  {label}: mean={data[:, col].mean():.2f}, min={data[:, col].min():.2f}, max={data[:, col].max():.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python analyze_md.py <work_dir> [targets...]")
        print("Example: python analyze_md.py ./sim rmsd rmsf")
        sys.exit(1)
    work_dir = sys.argv[1]
    targets = sys.argv[2:] if len(sys.argv) > 2 else ["rmsd", "rmsf", "radius_of_gyration"]
    results = analyze_md(work_dir, targets)
    for key, val in results.items():
        print(f"\n=== {key.upper()} ===")
        print(val)
