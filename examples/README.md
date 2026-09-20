Recipes are optional **hints**. The engine can start from a naked PDB plus `--objective`.

| File | Protein | Role |
|------|---------|------|
| `b2ar.yaml` | β2AR 2RH1 inactive, TM6 IC | **compression target** — crystal span ≥ ε |
| `b2ar_3sn6.yaml` | β2AR 3SN6 chain R active start | second teacher/reference start |
| `dltb.yaml` | DltB (6BUG) | `discover_states` |
| `glycophorin.yaml` | Glycophorin A (5EH4) | **negative control** — easy GxxxG well; adaptive did not accelerate |

Do not import these from `src/adaptamem`.
