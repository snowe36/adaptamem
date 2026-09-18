# adaptamem

Generic engine for **membrane-protein molecular dynamics**. A system is a YAML recipe. The core does not know DltB, β2AR, or AftD.

Objective: **scientific information per GPU-hour**, not nanoseconds per day of one trajectory.

## What it is for

Any integral membrane protein you can point a PDB/CIF at:

1. **Inner loop** — OpenMM, HMR, 4 fs in equilibration and production, compact box, sparse frames.
2. **Adaptive sampling** — user-defined CVs, many short walkers, stop when those CVs have converged.
3. **Hybrid resolution** (later) — atomistic protein + first lipid shells; cheaper bulk. Only if the question still needs those shells.

Protein-family science (landmarks, activation switches, catalytic residues) belongs in the recipe’s `cvs:` block, not in Python.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

OpenMM is an extra (`pip install -e ".[dev,sim]"`) and is not required to validate recipes.

## Commands

```bash
adaptamem init my_protein.yaml
adaptamem validate my_protein.yaml
adaptamem validate examples/dltb.yaml
```

`bench` (ns/day probe) and `sample` (walker farm) are next.

## Recipe shape

See `src/adaptamem/resources/system.template.yaml`. Examples in `examples/` are illustrations, not defaults.

## Not this repo

CHARMM-GUI web automation, ACEMD/GPCRmd, or a one-protein campaign (that lives in its own project and can *call* this engine).
