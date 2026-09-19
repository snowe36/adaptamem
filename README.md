# adaptamem

**The cheapest sufficient membrane-protein simulation is a function of the question and the GPU-hour budget, not a 100 ns CHARMM-GUI default.**

Adaptamem chooses physical resolution, box geometry, and sampling for a membrane-protein objective, then runs a CHARMM36 OpenMM path when that is the cheapest sufficient answer. It is not a GPCR-only pipeline and it is not an AftD/TmaT campaign manager. `ns/day` comparisons are legal only inside one physics class.

[![CI](https://github.com/snowe36/adaptamem/actions/workflows/ci.yml/badge.svg)](https://github.com/snowe36/adaptamem/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

Repo: [github.com/snowe36/adaptamem](https://github.com/snowe36/adaptamem)

---

## The problem

A membrane protein PDB plus “run 100 ns” looks like a recipe. The expensive part is usually the wrong box, a 2 fs equilibration pin, and quoting a hybrid membrane’s throughput against full atomistic CHARMM36.

**Given a structure, a scientific objective, and a GPU-hour budget, what is the cheapest simulation that can actually answer the question — and when should the tool refuse?**

Three axes, never mixed in a benchmark:

1. **Cheaper timestep** — same Hamiltonian, less work per step (HMR 4 fs, platform, cutoff).
2. **Fewer timesteps** — stop, branch, or skip work that does not reduce uncertainty in the objective.
3. **Fewer expensive atoms** — fewer particles on the atomistic force field (an *approximation* the moment lipids or water leave AA).

---

## What this repo builds

1. **Doctor** the PDB (missing loops, TM spans, clashes) — human veto on ACTION items
2. **Size** a geometry-minimized bilayer, not a cubic CHARMM-GUI default
3. **Choose** a strategy on the three axes (`same_physics` | `approximation`) or **REFUSE**
4. **Orient** the TM axis to z (`auto`, or local PPM 3.0 `immers` when requested)
5. **Assemble** CHARMM36 + compact OpenMM `addMembrane` (mixed POPE:POPG by swap)
6. **Equilibrate** at 4 fs and stop on a membrane QC scorecard
7. **Bench** throughput (`ns/day`, steps/s, atom count, GPU, physics) — not science
8. **Produce** production MD with streaming observables (XTC is compatibility)
9. **Sample** iteratively (`initialize → pilot → chunk → analyze → select → branch → stop`) against a held-out AA oracle

Walker count and nanoseconds are scheduler **outputs**. Design: [docs/architecture.md](docs/architecture.md).

Layers stay separate: physics validity, performance (`bench`), inference (`sample`), decision (`select` / `REFUSE`).

---

## Key results

| Check | Result |
|-------|--------|
| Inner-loop protocol | **4 fs**, HMR **4 amu**, cutoff **1.0 nm**, CHARMM36/TIP3P, 310 K |
| CPU path (CI) | `doctor` + `plan` + unit tests; **OpenMM is not installed on CI** |
| Helix fixture (`tests/fixtures/helix.pdb`) | 40 LEU; doctor finds **≥1** TM span |
| OpenMM extra | optional `[sim]`; assemble/eq/bench skipped in CI |
| Public demo protein | Glycophorin A (PDB 1AFO), not AftD/TmaT |
| GPU bench (pod `i3ha8sl8wymclh`) | Secure RTX 4090, EU-RO-1, **$0.74/hr**. CHARMM36 HMR 4 fs, 31213 atoms: **1052.2 ns/day** CUDA in 1.31 s (`runs/gpu/bench.json`) |

A local OpenCL bench of one assembled helix is not a product result and is not quoted here.

---

## Quick start

Requires **Python 3.11+** and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/snowe36/adaptamem.git && cd adaptamem
uv venv --python 3.11
uv sync --extra dev
make test
# or: bash scripts/reproduce.sh
```

CPU story (what CI runs):

```text
adaptamem doctor tests/fixtures/helix.pdb
adaptamem plan  tests/fixtures/helix.pdb --objective discover-states
```

Optional physics: `uv sync --extra dev --extra sim` then `adaptamem assemble …` / `run`.

GPU throughput on a CUDA box (CPU-minimize, then time CUDA; caches the assembled system so a second run is the timed loop). On Linux GPU hosts install `openmm[cuda12]`, not the CPU-only `openmm` wheel:

```bash
pip install 'openmm[cuda12]' pdbfixer
python scripts/gpu_job.py --out runs/gpu --steps 4000
# quote ns/day only from runs/gpu/bench.json
```

---

## Objective

Named observables if you know the question. `discover_states` if you do not. `comparison` optimizes U(Δ) across systems. `membrane_environment` forbids cheapening first-shell lipids.

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

```bash
adaptamem plan protein.pdb --objective membrane-environment --budget-hours 12
```

Recipes in [`examples/`](examples/) are optional hints, not the engine.

---

## Metric

**Uncertainty per GPU-hour** (`CI width / GPU-hours`), inside one physics class, scored against a held-out AA oracle. `bench` is throughput only. A hybrid membrane that runs faster is not “higher ns/day.”

---

## Limitations

- Cold-start TM detection is Kyte–Doolittle; β-barrels and interfacial helices are missed.
- `auto` orientation is not PPM; `method: ppm` refuses unless a local `immers` binary is on `ADAPTAMEM_PPM_DIR` or PATH.
- Mixed membranes: POPE:POPG swap after a POPE `addMembrane`. Other mixes refuse.
- Charged (POPG) bilayers in a tiny XY box refuse — finite-size electrostatics.
- Adaptive sampling is an iteration with a pluggable `select`; walker counts are outputs.
- Hybrid AA/CG is a labeled **approximation** and is not executed as a Martini engine in this version.
- Apple Silicon OpenCL is not a 4090.
- Public demo protein is glycophorin A (1AFO), not AftD/TmaT.

---

## Future directions

- Broader lipid mixes (POPC:POPE, cardiolipin) without dropping to 100% majority
- PPM as default when `immers` is present
- Paired Δ sampling for `comparison`
- Executed CG bulk with AA promotion (backmapping + re-eq)

---

## How to reproduce (detail)

| Command | Needs | What it proves |
|---------|-------|----------------|
| `make test` / `uv run pytest -q` | `[dev]` | doctor, box, strategy, mix, diagnostics, sample loop, oracle/compare |
| `bash scripts/reproduce.sh` | uv or venv | lint + tests + doctor/plan on the helix fixture |
| `make gpu` / `python scripts/gpu_job.py` | CUDA + `openmm`/`pdbfixer` | CHARMM36 throughput JSON; no repo install |
| `adaptamem assemble --out runs/job` | `[sim]` (OpenMM) | compact bilayer + 4 fs HMR system |
| `adaptamem equilibrate runs/job` | assembled system | QC scorecard (APL, thickness) |
| `adaptamem bench runs/job` | assembled system | throughput JSON (not U/GPU-hour) |
| `adaptamem produce runs/job --ns 1` | eq system | streaming CVs + XTC/checkpoints |
| `adaptamem sample … --select random` | YAML CVs | iterative loop; `--execute` runs chunks |
| `adaptamem oracle-freeze` / `compare` | traces + oracle | held-out recovery; equal hours and equal precision |
| `adaptamem reproduce <id>` | campaign.json | hashes + protocol replay |
| `adaptamem hybrid protein.pdb` | CPU | hybrid plan; refuses implicit annular lipids |

CI: `.github/workflows/ci.yml` — Python 3.11 and 3.12, `uv sync --extra dev`, ruff, pytest. No OpenMM.

---

## Project layout

```text
src/adaptamem/     engine (doctor, strategy, orient, assemble, sample, hybrid)
src/adaptamem/resources/  protocol.yaml, system.template.yaml
tests/             unit tests (OpenMM skipped if missing)
tests/fixtures/    helix.pdb
examples/          optional YAML hints
docs/architecture.md
scripts/gpu_job.py standalone CUDA throughput
scripts/reproduce.sh
```

---

## Acknowledgments

CHARMM36 and OpenMM `addMembrane` are the inner-loop substrate. PPM 3.0 (`immers`) is optional and not vendored.

---

## License

MIT. See [LICENSE](LICENSE).
