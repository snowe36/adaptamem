# Compression mechanisms (before GPU spend)

Do not launch a β2AR GPU job until the crystal gate passes. Adaptive walker
heuristics are not on the menu. Packing, eq, and RunPod robustness are not the
product.

**Claim.** A method is an acceleration only if it matches the held-out oracle
observable to ±ε using ≥10× fewer GPU-hours than conventional MD. Lower CI or
earlier stopping is not a claim. Report CPU-hours separately. No future oracle
frames, oracle populations, or oracle mean at inference.

Planning clock (replace with a bench): **350 ns/day** on a 4090-class GPU for
β2AR+POPC. Then 1 μs conventional = **69 GPU-hours**. The 10× bar is **≤6.9 GPU-hours
≈ 100 ns of teacher MD**. CPU inference after that is the stronger outcome.

## What broke

| Failure | Cause | Fix |
|---------|--------|-----|
| No scientific answer from 3P0G GPU | T4L `tm6_ic` already sits near the active crystal; span vs 3SN6 is **0.128 nm < ε=0.2** | **ABORT 3P0G.** Target 2RH1 vs 3SN6 (span **0.70 nm**) |
| 4 fs produce NaN on the first chunk | `gpu_oracle` integrated addMembrane packing | Oracle requires `eq.pdb`. Packing is not a teacher |
| CUDA illegal address / PTX 222 / CUDA 999 | Image OpenMM ≠ XML OpenMM; pip CUDA 12.9 PTX on a 12.8 host; one community box with dead CUDA runtime | Pin **one** OpenMM+CUDA pair; conda-forge `cuda-version=12.8`; terminate dead hosts; **no CPU fallback on a billed GPU** |
| Crash-loop billing | Boot consumed a partial ship tarball | Sentinel `ship.ready` after a complete archive; or SSH+scp then run. Watchdog kills crash loops |

CPU-only synthetic two-basin already hits the claim: mixed-teacher MSM reaches ε=0.1 at **10×** vs 69 h; unmixed teacher does not invent the other well.

## Scientific gate (CPU, before prepare)

X-ray, one biological copy. Measure the named CV on **inactive and active crystals**. No membrane, no GPU.

Measured `tm6_ic` = CA 131–272, ε = 0.2 nm:

| structure | role | nm |
|-----------|------|----|
| 2RH1 chain A | inactive X-ray | 0.840 |
| 3P0G chain A | inactive T4L fusion | 1.413 |
| 3SN6 chain R | active Gs | 1.541 |

| Gate | Action |
|------|--------|
| inactive–active span < ε | **abort that system.** Easy well or a fusion-distorted CV |
| 3P0G vs 3SN6 = 0.128 nm | abort. Do not assemble, eq, or oracle 3P0G |
| 2RH1 vs 3SN6 = 0.70 nm | this is the GPCR question: seed **both** basins as teacher *and* oracle starts |

Do not pretend a single inactive trajectory discovered the active basin. Do not buy a 1 μs 3P0G oracle.

## Three mechanisms

All three consume the same teacher budget. They are not walker-allocation variants.

### 1. MSM from short shots

Discrete kinetics. Cluster teacher frames on TM6, count transitions at lag τ,
read π and ⟨tm6_ic⟩ from the implied equilibrium. CPU: seconds.

| | |
|--|--|
| GPU to 10× | ≤100 ns as many 2 ns shots, not one 100 ns movie |
| CPU | MSM eigen-solve; report hours even if ~0 |
| Works when | teacher **covers** the oracle support; lag is Markov |
| Dies when | disconnected bins, one start, T4L-locked TM6 |
| Leakage | do not use oracle bin counts as a prior |

### 2. Latent dynamics + equilibrium generator

Encoder on 7TM Cα / contacts from teacher frames → cheap latent dynamics **and**
a generator of equilibrium-like latent states. Decode to `tm6_ic`. GPU after
training: **none**, unless a decoded state is OOD.

| | |
|--|--|
| GPU to 10× | 40–80 ns teacher; then CPU-only inference |
| CPU | must be reported; this is where mixing happens |
| Dies when | generating the active basin from inactive-only data |

### 3. Uncertainty-gated hybrid (GPU as active-learning oracle)

Run (1) or (2). If a bin is hungry, buy one short GPU shot there. If everything
is uncertain, **REFUSE COVERAGE** — do not become conventional MD.

This is the only method allowed to call the GPU after the teacher set.

## GPU contract (when a shot is named)

1. Crystal gate already passed.
2. CPU `prepare` (assemble) on laptop. Ship `assembled.pdb` + `system.xml` + `assemble.json`.
3. Equilibrate on the **same** OpenMM/CUDA pair that will produce. Write `eq.pdb`.
4. `gpu_oracle` runs produce only. No fetch, assemble, scout, or CPU fallback.
5. Image: official `runpod/pytorch` CUDA **12.8** + conda-forge `openmm` with `cuda-version=12.8` (tiny CUDA context must succeed before the shot).
6. Watchdog: DONE / FAIL / crash loop / GPU idle / 4 h wall → terminate. No second pod to “retry packing.”

## Order of work

1. Crystal gate. **3P0G is done: abort.**
2. Keep synthetic two-basin as the method CI (MSM 10× already).
3. CPU features on **2RH1 + 3SN6** crystals (two-frame teacher). Compress/infer: coverage of the jump is the named question, not another 3P0G pod.
4. Only then: CPU assemble of **one** 2RH1 copy, matching-OpenMM CUDA eq, independent short teachers from **both** crystals. Held-out 1 μs oracle is a later purchase, not step 4.
5. MSM on those shots. Latent on the same traces. Active learning only if bins are hungry.

Do not implement a fourth walker policy in the gap.
