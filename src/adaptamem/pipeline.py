"""GPU teaches. CPU predicts. GPU is called only when CPU does not know."""

CPU_STAGES = ("prepare", "features", "compress", "infer", "analyze")
GPU_STAGES = ("oracle",)
STAGES = CPU_STAGES + GPU_STAGES

LOOP = (
    "prepare → features → infer → uncertain regions → "
    "oracle on a tiny subset → compress → infer"
)


def run_is_not_a_campaign() -> str:
    return (
        "REFUSE  mega-run. GPU teaches; CPU predicts.\n"
        "  " + " | ".join(f"adaptamem {s}" for s in STAGES) + "\n"
        "  loop: " + LOOP
    )
