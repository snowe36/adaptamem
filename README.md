# Adaptamem

Automatically choose the physical resolution, computational configuration, and sampling strategy required to answer a membrane-protein question at minimum cost.

Three axes, never mixed in a benchmark:

1. **Cheaper timestep** — same Hamiltonian, less work per step.
2. **Fewer timesteps** — stop, branch, or skip work that does not reduce uncertainty in the objective.
3. **Fewer expensive atoms** — fewer particles on the atomistic force field (a *different* approximation the moment lipids or water are no longer AA).

```
PDB / CIF + objective + GPU-hour budget
        → Structure Doctor
        → cheapest sufficient box
        → STRATEGY (same_physics | approximation)
        → pilot → allocate compute → stop
        or REFUSE
```

## What exists now

| Command | Does |
|---------|------|
| `adaptamem doctor protein.pdb` | Structure report |
| `adaptamem plan protein.pdb --objective discover-states` | Doctor, box, **strategy on the three axes** |
| `adaptamem validate recipe.yaml` | Optional hints |
| `adaptamem run …` | Plan only; no MD yet |

```bash
adaptamem plan protein.pdb --objective membrane-environment --budget-hours 12
```

## Objective

```yaml
objective:
  type: discover_states    # conventional | conformational_shift | discover_states
                           # comparison | membrane_environment
  discover_cvs: true
  observables:
    - name: helix_rmsd
      kind: rmsd
      selection: "name CA and resid 20-40"
      precision: 0.5
```

Named observables if you know the question. `discover_states` if you do not. `comparison` optimizes U(Δ) across systems. `membrane_environment` forbids cheapening first-shell lipids.

## Metric

Information per GPU-hour, **inside one physics class**. A hybrid membrane that runs faster is not “higher ns/day.”

Design: [docs/architecture.md](docs/architecture.md).

## Install

```bash
uv venv --python python3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```
