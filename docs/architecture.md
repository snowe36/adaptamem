# Architecture

Adaptamem automatically chooses the **physical resolution, computational configuration, and sampling strategy** needed to answer a membrane-protein question at minimum cost.

Three independent attacks. A faster timestep that changes the ensemble is not the same as a smaller box.

| Axis | What it reduces | Same physics? |
|------|-----------------|---------------|
| Cheaper timestep | cost of one integration step (HMR, MTS, GPU, precision, cutoff) | only if the force field and validated Δt stay the same |
| Fewer timesteps | steps you bother to take (eq scorecard, branching, early stop, CV-guided sampling) | yes, if you still integrate the same Hamiltonian |
| Fewer expensive atoms | particles on the AA Hamiltonian (box, water, hybrid membrane, implicit bulk) | **no**, the moment lipids/water are CG or implicit |

`ns/day` comparisons are only legal inside **same_physics**. Approximations are labeled and have their own accuracy bar.

```
PDB / CIF  +  objective  +  GPU-hour budget
                    │
                    ▼
            Structure Doctor
                    │
                    ▼
         cheapest sufficient box
                    │
                    ▼
          STRATEGY (three axes)
                    │
         ┌──────────┼──────────┐
         ▼          ▼          ▼
      cheaper    fewer      fewer
      timestep   steps      expensive
                            atoms
                    │
                    ▼
     same_physics  or  approximation
                    │
                    ▼
     pilot → cartography → allocate → stop
     or REFUSE (question cannot be answered at this budget)
```

Walker count and nanoseconds are scheduler **outputs**.

## Objective types

| type | Question shape | Do not |
|------|----------------|--------|
| `conventional` | one trajectory | pretend it is adaptive |
| `conformational_shift` | P(state) or a named CV | skip “boring” fluctuations — they *are* the ensemble |
| `discover_states` | find slow coordinates from a pilot | require the user to know the CV |
| `comparison` | Δ between systems (apo vs complex, mutant, lipid) | two independent 100 ns movies |
| `membrane_environment` | local lipids, thickness, defects | CG/implicit the first lipid shells |

## Tiers

**1 — same physics, build now.** Geometry box, water pad, GPU bench, batched short replicas, eq scorecard, adaptive lengths, sparse I/O, 4 fs HMR with validation.

**2 — differentiation.** Branching walkers, automatic CVs, objective-driven U(Δ), multi-fidelity promotion, cross-system transfer, adaptive box growth, platform selection.

**3 — research ceiling.** Moving AA/CG membrane, effective membrane potentials, particle–continuum boundaries, ML bulk forces, selective PME/precision, reduced-DOF protein.

Phase order: trustworthy PDB→MD (tier 1) → adaptive sampler proved against a held-out 100 ns AA trajectory (tier 2) → only then point hybrid resolution at regions the sampler already knows are useful (tier 3).

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

Implemented now: doctor, geometry box, strategy labeler. Physics engine is not built.
