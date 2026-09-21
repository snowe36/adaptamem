Recipes are optional **hints**. The engine can start from a naked PDB plus `--objective`.

| File | Protein | Role |
|------|---------|------|
| `b2ar.yaml` | β2AR 2RH1 inactive, TM6 IC | **compression target** — crystal span ≥ ε |
| `b2ar_3sn6.yaml` | β2AR 3SN6 chain R active start | second teacher/reference start |
| `dltb.yaml` | DltB (6BUG) | `discover_states` |
| `htr2a.yaml` | 5-HT2A 6A94 inactive | **hard hold-out** — TM6 zero-shot misses ε |
| `htr2a_6wha.yaml` | 5-HT2A 6WHA chain A active start | second basin for the 5-HT2A teacher |
| `acm2.yaml` | M2 3UON inactive | **adversarial** — prior invents an ionic-lock opening |
| `acm2_4mqs.yaml` | M2 4MQS chain A active start | second basin for the M2 lock teacher |

Do not import these from `src/adaptamem`.
