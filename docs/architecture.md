# Architecture

Adaptamem is a **CPU-first system that predicts long-timescale observables of membrane proteins**. It uses uncertainty to decide when a learned dynamics prior is inadequate, and invokes conventional MD only as an adaptive teacher.

Not CPU MD. Not faster OpenMM. Not an MD foundation model as the product.

**Use MD once as a teacher to learn what molecular dynamics looks like, then amortize that knowledge across new proteins.**

The target is a transferable representation of membrane-protein dynamics: sequence / structure / membrane in → ensemble, CVs, and calibrated uncertainty out — **without integrating every timestep**. Do not make “generate a realistic trajectory” the first target. Predict the scientific quantity (TM6 opening, pocket accessibility, contact occupancy, state populations, later rates) plus whether that prediction is trustworthy.

```
        LARGE OFFLINE MD CORPUS          this protein
                  │                           │
                  ▼                           ▼
         learned dynamics prior ← sequence + structure + membrane
                  │
       ┌──────────┴──────────┐
       ▼                     ▼
 predicted ensemble     predicted CVs + uncertainty
       │                     │
       └──────────┬──────────┘
                  ▼
           tiny MD teacher
                  │
           disagreement / error
                  │
       ┌──────────┴──────────┐
       ▼                     ▼
    trust prior          update locally
    stop MD
```

GPU/MD is the exception, not the computational substrate.

## Phases

| Phase | What | MD's job |
|-------|------|----------|
| 1 | Structure + cheap physics → decide whether / where MD is needed | none, or a named shot |
| 2 | Tiny MD on this protein → predict the long-time observable | teacher |
| 3 | Many proteins + many tiny teachers → transferable dynamics prior | corpus |
| 4 | New protein → CPU prediction → MD only when uncertainty is high | exception |

Current work is Phase 1–2 (β2AR 2RH1 / 3SN6, `tm6_ic`, ε = 0.2 nm). Phase 3–4 is the destination. A corpus trained on *other* proteins can learn conserved microswitches, membrane coupling, and collective motions that a tiny β2AR trajectory will never show. The constraint is **validation**, not whether transfer is conceivable:

> Can a dynamics prior trained on other proteins predict a genuinely held-out protein's long-timescale behavior, with calibrated uncertainty, using substantially less computation than conventional MD?

The large corpus is training data. A held-out conventional trajectory on the evaluation protein remains ground truth. Training-set memorization is not transfer. Passing the evaluation protein's oracle π or mean into inference is still `LEAKAGE`.

Three operating modes (Phase 2 default: `cpu_first`):

| Mode | What runs | Role |
|------|-----------|------|
| `cpu_only` | structure (later sequence) → CPU infer | moonshot; no MD |
| `cpu_first` | CPU infer, then tiny MD only where uncertain | near-term default |
| `conventional` | long MD, frozen as a held-out reference | scoring wall, not the product |

`adaptamem run` is a REFUSE. Trajectory input is optional. Sequence-in without a structure is `NOT_READY`.

**GPU teaches. CPU predicts. GPU is called only when CPU does not know.**

| Stage | Default | Purpose |
|-------|---------|---------|
| `prepare` | CPU | acquire inputs, orient, assemble |
| `features` | CPU | observables from crystals or teacher traces |
| `compress` | CPU | occupancy / transitions / sufficient statistics |
| `infer` | CPU | ensemble + uncertainty; no MD |
| `decide` | CPU | COVERAGE / BRIDGE / MECHANISM; no MD |
| `analyze` | CPU | compression = MD avoided / oracle spent |
| `oracle` | GPU, explicit | short trustworthy labels after `eq.pdb` |

Loop: structure → CPU prior → infer → if uncertain, tiny MD teacher → compress → infer; else DONE.

`adaptamem decide` sits on that loop: **COVERAGE** (enough local observations?), **BRIDGE** (evidence connecting known wells?), **MECHANISM** (which coordinate/start would distinguish pathways?). The 2RH1/3SN6 teacher is two wells and no transition: no extra 1D TM6 shot, `BRIDGE` refuse, `MECHANISM` request. GPU stays off until a start exists that can tell the hypotheses apart.

## Three demonstrations

| | Question | Pass |
|--|----------|------|
| 1 Mechanistic teacher | Two wells vs missing path vs 1D coverage | `decide` on frozen β2AR teacher |
| 2 Zero-shot prior | Other proteins' structures predict a **held-out** protein's activation coordinates | `loo`; hold-out traces are `LEAKAGE` |
| 3 Adaptive correction | Tiny MD only where a frozen CPU prior is wrong (5-HT2A TM6, M2 lock); β2AR TM6 is the no-teacher control | `prior_correction_v1`; GPU off until `eq.pdb` |

Experiment 2 starts from crystals, not trajectories. `adaptamem loo` on a 10-protein catalog (β2AR, rhodopsin, A2A, μOR, M2, β1AR, κOR, 5-HT2A, CB1, D2). TM6 reaches ε on every fold except **5-HT2A**. Ionic lock fails on β2AR, A2A, and M2. Packing is hidden except κOR. Hold-out traces remain `LEAKAGE`. Experiment 3 is not “run 1 μs and match TM6.” It is: the CPU prior predicts a held-out protein; tiny MD runs only where the prior is unconfident or wrong (β2AR lock, M2 invented lock, 5-HT2A TM6, `pack_in_inactive_tm6`); the score is error reduction per GPU-hour vs a conventional oracle the method never sees. GPU stays off until a named shot has `eq.pdb`.

Packing (`assembled.pdb`) is not a teacher. `oracle` requires `eq.pdb`. Image: CUDA **12.8** + conda-forge OpenMM with `cuda-version=12.8`. No CPU fallback on a billed GPU.

Primary score:

\[
\mathrm{compression} = \frac{\text{baseline MD compute avoided}}{\text{GPU oracle compute spent}}
\]

\[
\mathrm{speedup} = \frac{\text{baseline production MD cost}}{\text{CPU inference cost} + \text{oracle cost}}
\]

Acceleration is ≥10× at matched ε, or CPU-only inference that matches. Lower CI is not a claim. The method must not see future oracle frames, oracle populations, or the oracle mean at inference (`LEAKAGE`). `ns/day` is a hardware clock for converting ns to GPU-hours.

## Research ladder

Score easy rungs first. Stub the rest as `NOT_IMPLEMENTED` rather than inventing them.

| Rung | Now |
|------|-----|
| structural observables | scored (`tm6_ic` support) |
| flexibility / RMSF placeholder | scored (B-factors) |
| contact probabilities | scored (crystal contact differences) |
| conformational heterogeneity | scored (crystal span) |
| state populations | occupancy of visited support; π only if the teacher mixes |
| equilibrium distributions | resampled support |
| free-energy differences | `NOT_IMPLEMENTED` |
| transition probabilities / rates | `NOT_IMPLEMENTED` |
| rare-state occupancy | `NOT_IMPLEMENTED` |
| pathways / timescales | `NOT_IMPLEMENTED` |
| mutation / ligand / environment Δ | `NOT_IMPLEMENTED` |

MSM is the first dynamics plug. It is not the product.

## Objective types

| type | Question shape | Do not |
|------|----------------|--------|
| `conventional` | one trajectory | pretend it is adaptive |
| `conformational_shift` | P(state) or a named CV | skip “boring” fluctuations — they *are* the ensemble |
| `discover_states` | find slow coordinates from a pilot | require the user to know the CV |
| `comparison` | Δ between systems (apo vs complex, mutant, lipid) | two independent 100 ns movies |
| `membrane_environment` | local lipids, thickness, defects | CG/implicit the first lipid shells |

## Tiers

**1 — same physics, teacher.** Geometry box, water pad, GPU bench, 4 fs HMR, short trustworthy MD used as the teacher set.

**2 — compression.** MSM / milestoning / weighted ensemble from short shots, learned propagator or latent dynamics, generative equilibrium sampling, active MD only when the model is uncertain, hybrid CPU surrogate + sparse GPU correction. Goal: 10–100× fewer GPU-hours at matched oracle error, or CPU-only inference.

**3 — research ceiling.** Moving AA/CG membrane, effective membrane potentials, particle–continuum boundaries — after compression is real.

Phase order: crystal gate → CPU-only prior → tiny teacher if uncertain → compress against a held-out long AA oracle on a slow CV. Do not spend tier 2 on an easy well.

A faster timestep that changes the ensemble is not the same as a smaller box. `ns/day` comparisons are only legal inside **same_physics**. Approximations are labeled.

## Other constraints (easy to miss)

- **Refuse.** A real chooser can say the budget cannot answer this objective.
- **Provenance.** Every choice (box, Δt, lipids, fidelity) is a logged decision with evidence, or the tool is a black box.
- **Policy version.** Papers cite `adaptamem` policy 0.x, not just a git hash of YAML.
- **Finite-size is physics** for membrane deformation; a minimum box can be the wrong ensemble.
- **CG → AA is backmapping**, not a restart. Promotion needs re-eq.
- **Metastability.** A tight CI from one well is not convergence.
- **Paired Δ.** Apo vs complex share a state representation; lipid composition must not confound the protein effect.
- **Engine choice.** OpenMM vs GROMACS is a strategy output (large AA boxes often favor GROMACS).
- **Cold start.** The first protein has no library; TM detection via Kyte–Doolittle misses β-barrels and interfacial helices.
- **Charged membranes + PBC.** POPG/CL boxes are electrostatic-size sensitive.
- **Human veto** on doctor ACTION items (missing loops, protonation).
- **Cost C(x)** includes rebuild time, storage, and operator attention, not only GPU-hours.
- **Two clocks.** Campaign wall-clock ≠ replica ns/day.
- **Do not saturate-batch** a 600k-atom complex; batching is for small walkers.

Implemented now: doctor, geometry box, CHARMM36 HMR 4 fs teacher path, held-out oracle, `oracle_compression` (GPU-hours to ε), crystal-only `latent_dynamics` prior, MSM from teacher traces, adaptive `sample` as a baseline, named `compress` plugs that REFUSE until they learn from short MD. Martini execution is not built.

Layers: **physics** (is the teacher valid?) → **performance** (`bench`) → **compression** (predict without integrating missing time) → **decision** (`REFUSE`).
