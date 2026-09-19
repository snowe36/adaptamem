# adaptamem

**MD is the teacher, not the engine.** The cheapest sufficient membrane-protein *answer* is the one that matches a long conventional-MD oracle at a predefined error, using 10–100× fewer GPU-hours — or no production MD at all.

Adaptamem still builds a CHARMM36 OpenMM path so the teacher is trustworthy. It is not a GPCR-only pipeline and it is not an AftD/TmaT campaign manager. `ns/day` is a clock, not science.

[![CI](https://github.com/snowe36/adaptamem/actions/workflows/ci.yml/badge.svg)](https://github.com/snowe36/adaptamem/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.org/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.org/badge/python-3.11%2B-blue.svg)

Repo: [github.com/snowe36/adaptamem](https://github.com/snowe36/adaptamem)

---

## The problem

A membrane protein PDB plus “run 100 ns” is a recipe. Adaptive walker allocation is still that recipe with a stopping rule. The 5EH4 GxxxG campaign proved it: one packed well, `select` STOP, **no acceleration**.

**What information from a tiny amount of trustworthy MD can be exploited to predict the long-time ensemble, transitions, or observable without integrating the missing trajectories?**

Conventional MD is the oracle. GPU-hours to match that oracle within ε is the score. More MD that stops sooner is a failure mode.

---

## What this repo builds

1. **prepare** (CPU) — doctor, orient, assemble
2. **features** (CPU) — observables / latent representation from teacher traces
3. **compress** (CPU) — MSM (built); other named plugs REFUSE as controls
4. **infer** (CPU) — predict the ensemble; list under-sampled bins
5. **oracle** (GPU, explicit) — short trustworthy MD on a shipped system. Not a campaign.
6. **analyze** (CPU) — compression = baseline MD avoided / GPU oracle spent

`adaptamem run` refuses. Adaptive `sample` is a **baseline**, not the goal. Long production MD is what this project is trying to eliminate.

Design: [docs/architecture.md](docs/architecture.md).

---

## Key results

| Check | Result |
|-------|--------|
| Inner-loop protocol | **4 fs**, HMR **4 amu**, cutoff **1.0 nm**, CHARMM36/TIP3P, 310 K |
| CPU path (CI) | `doctor` + `plan` + unit tests; **OpenMM is not installed on CI** |
| Helix fixture (`tests/fixtures/helix.pdb`) | 40 LEU, twisted+centered; doctor finds **≥1** TM span |
| Primary metric | **compression = baseline MD avoided / GPU oracle spent**. Acceleration = **≥10×** at matched ε, or CPU-only inference that matches. Lower CI is not a claim. No oracle leakage. |
| Easy-well control (5EH4 G83) | **no acceleration claim.** Clock 1085.7 ns/day, 30 202 atoms, pod `ieh156d6psdtwu`. Oracle μ=0.457 nm. Adaptive STOP'd on low uniqueness. Wrong target. |
| Compression target | **β2AR** inactive **2RH1** vs active **3SN6** (chain R), TM6 IC CA 131–272 (`tm6_ic`, ε=0.2 nm). Crystal span 0.70 nm. **3P0G aborted** (span vs 3SN6 0.128 nm). |

Helix **1052.2 ns/day** (pod `i3ha8sl8wymclh`) is a conventional clock, not a protein result.

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

Optional physics: `uv sync --extra sim` then `adaptamem prepare …`.

CPU compression loop:

```text
adaptamem features traces.json --out features.json
adaptamem compress features.json --kind msm --out model.json
adaptamem infer model.json --out pred.json
adaptamem analyze --oracle oracle.json --method msm=pred.json --method single_long=long.json
```

GPU oracle (ship an assembled system first; not a campaign):

```bash
# on the pod, assembled.pdb + system.xml already present
export ADAPTAMEM_WORKDIR=runs/oracle ORACLE_NS=10
bash scripts/runpod_boot.sh
```

---

## Objective

Named observables if you know the slow coordinate. `discover_states` if you do not. Do not pick a CV that 20 ns of conventional MD already pins.

```yaml
objective:
  type: conformational_shift
  observables:
    - name: tm6_ic
      kind: distance
      selection: "name CA and resid 131 ; name CA and resid 272"
      precision: 0.2
```

Recipes in [`examples/`](examples/): `b2ar.yaml` is the compression target; `glycophorin.yaml` is the easy-well negative control.

---

## Metric

**compression = baseline MD compute avoided / GPU oracle spent.**

**speedup = baseline production MD / (CPU inference + oracle).**

Bar: 10× at matched ε, or CPU-only inference that matches. Lower CI is not a claim. `bench` is throughput only.

---

## Limitations

- Compression plugs (`msm`, `learned_propagator`, …) **REFUSE** until implemented — they must not fall through to another production run.
- Cold-start TM detection is Kyte–Doolittle; β-barrels and interfacial helices are missed.
- `auto` orientation is not PPM.
- Mixed membranes: POPE:POPG swap only.
- Adaptive `select` is a baseline that failed on 5EH4; do not tune it as the product.
- Hybrid AA/CG is a labeled **approximation**.
- Apple Silicon OpenCL is not a 4090.

---

## Future directions

Proposals (no GPU yet): [docs/compression-proposals.md](docs/compression-proposals.md) — MSM from short shots, latent dynamics + equilibrium generator, uncertainty-gated hybrid. Adaptive `select` is not a candidate.

---

## How to reproduce (detail)

| Command | Needs | What it proves |
|---------|-------|----------------|
| `make test` / `uv run pytest -q` | `[dev]` | doctor, box, strategy, compare `oracle_compression`, compress REFUSE |
| `bash scripts/reproduce.sh` | uv or venv | lint + tests + doctor/plan on the helix fixture |
| `make gpu` / `python scripts/gpu_job.py` | CUDA + `openmm`/`pdbfixer` | CHARMM36 throughput JSON; no repo install |
| `adaptamem assemble --out runs/job` | `[sim]` | compact bilayer + 4 fs HMR system |
| `adaptamem produce runs/job --ns 1` | eq system | teacher traces |
| `adaptamem compare --oracle … --method …` | traces | GPU-hours to oracle ε |
| `adaptamem reproduce <id>` | campaign.json | hashes + protocol replay |

CI: `.github/workflows/ci.yml` — Python 3.11 and 3.12, `uv sync --extra dev`, ruff, pytest. No OpenMM.

---

## Project layout

```text
src/adaptamem/     engine (prepare, features, compress, infer, oracle, analyze)
src/adaptamem/compress.py  MSM is built; other named plugs REFUSE as controls
tests/             unit tests (OpenMM skipped if missing)
examples/          b2ar.yaml (hard CV), glycophorin.yaml (easy-well control)
docs/architecture.md
docs/compression-proposals.md  MSM / latent / active-learning; 10× budgets
scripts/gpu_oracle.py   short MD teacher only; system must already be assembled
scripts/gpu_campaign.py 5EH4 negative-control baseline; not the boot path
scripts/runpod_watchdog.py  terminate the pod when the job exits
```

---

## Acknowledgments

CHARMM36 and OpenMM `addMembrane` are the teacher substrate. PPM 3.0 (`immers`) is optional and not vendored.

---

## License

MIT. See [LICENSE](LICENSE).
