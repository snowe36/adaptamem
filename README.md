# adaptamem

**What can we learn about this membrane protein without a long MD simulation?**

MD is the teacher, not the engine. Use it to learn what membrane-protein dynamics looks like, then **amortize that knowledge** across new proteins. Adaptamem predicts long-timescale *observables* (ensembles, CVs, uncertainty) — not every atom's trajectory — and runs conventional MD only where the prior is untrustworthy.

It still builds a CHARMM36 OpenMM path so the teacher is trustworthy. It is not a GPCR-only pipeline and it is not an AftD/TmaT campaign manager. `ns/day` is a clock, not science.

[![CI](https://github.com/snowe36/adaptamem/actions/workflows/ci.yml/badge.svg)](https://github.com/snowe36/adaptamem/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)

Repo: [github.com/snowe36/adaptamem](https://github.com/snowe36/adaptamem)

---

## The problem

A membrane protein PDB plus “run 100 ns” is a recipe. Adaptive walker allocation is still that recipe with a stopping rule. The 5EH4 GxxxG campaign proved it: one packed well, `select` STOP, **no acceleration**.

**Can a small amount of trustworthy MD, combined with structural priors, predict the long-time observable without integrating the missing trajectory?**

The destination is a transferable dynamics prior: trained on other proteins, evaluated on a **held-out** protein, with calibrated uncertainty, at substantially less compute than conventional MD. Three demonstrations: (1) mechanistic teacher selection on β2AR, (2) leave-one-protein-out crystal prior (`adaptamem loo --hold-out adrb2`) — TM6 transfers on 9/10 catalog proteins at 0 GPU-h; **5-HT2A is the first TM6 miss**; ionic lock fails on β2AR/A2A/M2; packing is trivial except κOR — (3) tiny MD only where that prior is wrong, scored as error reduction per GPU-hour vs a frozen conventional oracle (`adaptamem correct`). GPU is off until a named shot has `eq.pdb`.

Do not make “generate a realistic trajectory” the first target. Predict the CV / ensemble / contacts plus uncertainty. Conventional MD remains the held-out reference. More MD that stops sooner is a failure mode.

---

## Three modes

| Mode | Engine | When |
|------|--------|------|
| `cpu_only` | structure → CPU infer | no MD |
| `cpu_first` | CPU infer, tiny MD teacher only if uncertain | default |
| `conventional` | long MD, frozen | scoring reference, not the product |

---

## What this repo builds

1. **prepare** (CPU) — doctor, orient, assemble
2. **features** (CPU) — observables from crystals or teacher traces
3. **compress** (CPU) — structure prior (`latent_dynamics`) or MSM from traces
4. **infer** (CPU) — predict the ensemble; list what is identified vs not
5. **oracle** (GPU, explicit) — short trustworthy MD after `eq.pdb`. Not a campaign.
6. **analyze** (CPU) — compression = baseline MD avoided / GPU oracle spent

`adaptamem run` refuses. Adaptive `sample` is a **baseline**, not the goal. Sequence-in without a structure is `NOT_READY`.

Design: [docs/architecture.md](docs/architecture.md). Six-week ladder: [docs/compression-proposals.md](docs/compression-proposals.md).

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

CPU-only (no MD):

```text
adaptamem features --crystal 2RH1.pdb:A --crystal 3SN6_R.pdb:R --out features.json
adaptamem compress features.json --kind latent_dynamics --out model.json
adaptamem infer model.json --mode cpu_only --out pred.json
```

CPU compression loop with a teacher:

```text
adaptamem features traces.json --out features.json
adaptamem compress features.json --kind msm --out model.json
adaptamem infer model.json --mode cpu_first --out pred.json
adaptamem analyze --oracle oracle.json --method msm=pred.json --method single_long=long.json
```

GPU oracle (ship **eq.pdb** + assembled system; not a campaign):

```bash
# on the pod: assembled.pdb + system.xml + eq.pdb already present
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

- Compression plugs (`learned_propagator`, …) **REFUSE** until implemented — they must not fall through to another production run.
- Crystal/short-teacher inference does **not** claim rates, pathways, or free energies (`NOT_IMPLEMENTED`).
- Sequence → structure is `NOT_READY`.
- Cold-start TM detection is Kyte–Doolittle; β-barrels and interfacial helices are missed.
- `auto` orientation is not PPM.
- Mixed membranes: POPE:POPG swap only.
- Adaptive `select` is a baseline that failed on 5EH4; do not tune it as the product.
- Hybrid AA/CG is a labeled **approximation**.
- Apple Silicon OpenCL is not a 4090.

---

## Future directions

Proposals: [docs/compression-proposals.md](docs/compression-proposals.md) — MSM from short shots, latent dynamics + equilibrium generator, uncertainty-gated hybrid. Adaptive `select` is not a candidate. GPU teacher for β2AR is Week 3 of that ladder, after `eq.pdb`.

---

## How to reproduce (detail)

| Command | Needs | What it proves |
|---------|-------|----------------|
| `make test` / `uv run pytest -q` | `[dev]` | doctor, box, strategy, compare `oracle_compression`, compress REFUSE |
| `bash scripts/reproduce.sh` | uv or venv | lint + tests + doctor/plan on the helix fixture |
| `make gpu` / `python scripts/gpu_job.py` | CUDA + `openmm`/`pdbfixer` | CHARMM36 throughput JSON; no repo install |
| `adaptamem assemble --out runs/job` | `[sim]` | compact bilayer + 4 fs HMR system |
| `adaptamem produce runs/job --ns 1` | `eq.pdb` | teacher traces |
| `adaptamem compare --oracle … --method …` | traces | GPU-hours to oracle ε |
| `adaptamem reproduce <id>` | campaign.json | hashes + protocol replay |

CI: `.github/workflows/ci.yml` — Python 3.11 and 3.12, `uv sync --extra dev`, ruff, pytest. No OpenMM.

---

## Project layout

```text
src/adaptamem/     engine (prepare, features, compress, infer, oracle, analyze)
src/adaptamem/compress.py  MSM + latent prior; other named plugs REFUSE
src/adaptamem/ladder.py    scored rungs vs NOT_IMPLEMENTED
src/adaptamem/gpu_contract.py  eq.pdb, CUDA 12.8, no CPU fallback
tests/             unit tests (OpenMM skipped if missing)
examples/          b2ar.yaml (hard CV), glycophorin.yaml (easy-well control)
docs/architecture.md
docs/compression-proposals.md  MSM / latent / active-learning; 6-week ladder
scripts/gpu_oracle.py   short MD teacher only; eq.pdb required
scripts/gpu_campaign.py 5EH4 negative-control baseline; not the boot path
scripts/runpod_watchdog.py  terminate the pod when the job exits
```

---

## Acknowledgments

CHARMM36 and OpenMM `addMembrane` are the teacher substrate. PPM 3.0 (`immers`) is optional and not vendored.

---

## License

MIT. See [LICENSE](LICENSE).
