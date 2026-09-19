# Architecture

Adaptamem automatically chooses the **physical resolution, computational configuration, and sampling strategy** needed to answer a membrane-protein question at minimum cost.

Three independent attacks. A faster timestep that changes the ensemble is not the same as a smaller box.

| Axis | What it reduces | Same physics? |
|------|-----------------|---------------|
| Cheaper timestep | cost of one integration step (HMR, MTS, GPU, precision, cutoff) | only if the force field and validated Δt stay the same |
| Fewer timesteps | steps you bother to take (eq scorecard, branching, early stop, CV-guided sampling) | yes, if you still integrate the same Hamiltonian |
| Fewer expensive atoms | particles on the AA Hamiltonian (box, water, hybrid membrane, implicit bulk) | **no**, the moment lipids/water are CG or implicit |

Phase order: trustworthy PDB→MD **teacher** (tier 1) → **oracle-compression** against a long held-out AA trajectory on a *slow* CV (tier 2: MSM, learned propagator, latent dynamics, generative eq, hybrid CPU + sparse GPU) → only then hybrid resolution (tier 3). Adaptive walker allocation is a baseline. It is not tier 2.

`ns/day` comparisons are only legal inside **same_physics**. Approximations are labeled.

**GPU teaches. CPU predicts. GPU is called only when CPU does not know.**

| Stage | Default | Purpose |
|-------|---------|---------|
| `prepare` | CPU | acquire inputs, orient, assemble |
| `features` | CPU | observables / latent representation |
| `compress` | CPU | learn occupancy / transitions / sufficient statistics |
| `infer` | CPU | production-scale search |
| `analyze` | CPU | compression = MD avoided / oracle spent |
| `oracle` | GPU, explicit | short trustworthy labels |

`adaptamem run` is a REFUSE. The loop is prepare → features → infer → uncertain regions → oracle on a tiny subset → compress → infer.

```
          short MD shots (teacher)
                   │
                   ▼
          learned representation
                   │
          ┌────────┴─────────┐
          ▼                  ▼
    latent dynamics      equilibrium
       model              generator
          │                  │
          └────────┬─────────┘
                   ▼
             cheap CPU
             exploration
                   │
            uncertainty
                   │
            ┌──────┴──────┐
            │             │
          confident     uncertain
            │             │
            ▼             ▼
        CPU result     sparse MD
                          │
                          └──► update model
```

Primary score:

\[
\mathrm{compression} = \frac{\text{baseline MD compute avoided}}{\text{GPU oracle compute spent}}
\]

\[
\mathrm{speedup} = \frac{\text{baseline production MD cost}}{\text{CPU inference cost} + \text{oracle cost}}
\]

Acceleration is ≥10× at matched ε, or CPU-only inference that matches. Lower CI is not a claim. The method must not see future oracle frames, oracle populations, or the oracle mean at inference (`LEAKAGE`). Adaptive walker allocation is a baseline control. It is not the product.

## Objective types

| type | Question shape | Do not |
|------|----------------|--------|
| `conventional` | one trajectory | pretend it is adaptive |
| `conformational_shift` | P(state) or a named CV | skip “boring” fluctuations — they *are* the ensemble |
| `discover_states` | find slow coordinates from a pilot | require the user to know the CV |
| `comparison` | Δ between systems (apo vs complex, mutant, lipid) | two independent 100 ns movies |
| `membrane_environment` | local lipids, thickness, defects | CG/implicit the first lipid shells |

## Tiers

**1 — same physics, teacher.** Geometry box, water pad, GPU bench, 4 fs HMR, short trustworthy MD used as the oracle's teacher set.

**2 — compression.** MSM / milestoning / weighted ensemble from short shots, learned propagator or latent dynamics, generative equilibrium sampling, active MD only when the model is uncertain, hybrid CPU surrogate + sparse GPU correction. Goal: 10–100× fewer GPU-hours at matched oracle error, or CPU-only inference.

**3 — research ceiling.** Moving AA/CG membrane, effective membrane potentials, particle–continuum boundaries — after compression is real.

Phase order: trustworthy PDB→MD teacher (tier 1) → compress against a held-out long AA oracle on a slow CV (tier 2) → hybrid resolution only where the model already knows it is cheap (tier 3). Do not spend tier 2 on an easy well.

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

Implemented now: doctor, geometry box, CHARMM36 HMR 4 fs teacher path, held-out oracle, `oracle_compression` (GPU-hours to ε), adaptive `sample` as a baseline, named `compress` plugs that REFUSE until they learn from short MD. Martini execution is not built.

Layers: **physics** (is the teacher valid?) → **performance** (`bench`) → **compression** (predict without integrating missing time) → **decision** (`REFUSE`).
