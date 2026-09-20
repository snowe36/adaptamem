"""MD inference engine. Trajectory input is optional. GPU is a teacher budget."""

CPU_STAGES = ("prepare", "features", "compress", "infer", "decide", "analyze")
GPU_STAGES = ("oracle",)
STAGES = CPU_STAGES + GPU_STAGES

MODES = ("cpu_only", "cpu_first", "conventional")
EXPERIMENTS = (
    "mechanistic_teacher",
    "zero_shot_prior",
    "adaptive_correction",
)

LOOP = (
    "structure → CPU prior → infer → "
    "if uncertain: tiny MD teacher → compress → infer; else DONE"
)


def run_is_not_a_campaign() -> str:
    return (
        "REFUSE  mega-run. What can we learn without a long trajectory?\n"
        "  modes: " + " | ".join(MODES) + "\n"
        "  " + " | ".join(f"adaptamem {s}" for s in STAGES) + "\n"
        "  loop: " + LOOP
    )
