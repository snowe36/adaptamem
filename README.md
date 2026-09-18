# Adaptamem

Given a protein structure and a scientific objective, construct the **cheapest simulation that is sufficient to answer it**.

Not a YAML-driven MD runner. A decision engine.

```
PDB / CIF
    → Structure Doctor
    → cheapest sufficient box
    → short pilot
    → dynamical cartography (what actually moves)
    → spend GPU on uncertain states, drop redundant ones
    → stop when the objective’s uncertainty is below the asked precision
```

Walker count and nanoseconds are **outputs**, not protocol knobs.

## What exists now

| Command | Does |
|---------|------|
| `adaptamem doctor protein.pdb` | Structure report (TM spans, missing bits, ligands, clashes, protonation flags) |
| `adaptamem plan protein.pdb --objective discover-states` | Doctor + atom-count-minimizing box + experiment sketch |
| `adaptamem validate recipe.yaml` | Load an optional recipe (hints, not a full MD deck) |
| `adaptamem run …` | Prints the plan; physics engine is not built |

## Objective, not a recipe

```yaml
objective:
  type: discover_states          # or conformational_shift | conventional
  discover_cvs: true
  observables:                   # optional named questions
    - name: helix_rmsd
      kind: rmsd
      selection: "name CA and resid 20-40"
      precision: 0.5
```

If you already know the question, name observables and precisions. If you do not, `discover_states` runs a pilot and finds slow coordinates. `conventional` is the escape hatch: one trajectory, no scheduler.

## Metric

**Scientific information per GPU-hour**, against a fixed-budget baseline (1×100 ns vs naive *N* short runs vs Adaptamem). The paper is that comparison, not ns/day.

## Roadmap

See [docs/architecture.md](docs/architecture.md). Phase 1 is trustworthy PDB → membrane MD. The first novel algorithm is Phase 3 (adaptive sampler). Hybrid resolution is Phase 5–6, after the scheduler has somewhere useful to point.

## Install

```bash
uv venv --python python3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
adaptamem doctor path/to/protein.pdb
```
