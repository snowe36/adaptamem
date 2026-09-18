from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from adaptamem.objective import Objective, parse_objective

_RESOURCE_DIR = Path(__file__).parent / "resources"


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping")
    return data


def default_protocol_path() -> Path:
    return _RESOURCE_DIR / "protocol.yaml"


def default_system_template_path() -> Path:
    return _RESOURCE_DIR / "system.template.yaml"


@dataclass
class Membrane:
    lipids: dict[str, float] = field(default_factory=lambda: {"POPC": 1.0})
    water_pad_nm: float = 1.2
    ionic_strength_M: float = 0.15
    optimize_size: bool = True
    safety_margin_nm: float = 1.2


@dataclass
class Orientation:
    method: str = "auto"
    topology: str = "in"


@dataclass
class System:
    name: str
    structure: Path
    orientation: Orientation
    membrane: Membrane
    objective: Objective
    source: Path | None = None

    @property
    def adaptive(self) -> bool:
        return self.objective.adaptive


@dataclass
class Protocol:
    timestep_fs: float
    hydrogen_mass_amu: float
    nonbonded_cutoff_nm: float
    temperature_K: float
    eq_timestep_fs: float
    save_interval_ps: float
    raw: dict[str, Any]


def load_protocol(path: Path | None = None) -> Protocol:
    p = path or default_protocol_path()
    raw = _read_yaml(p)
    eq = raw.get("equilibration") or {}
    traj = raw.get("trajectory") or {}
    return Protocol(
        timestep_fs=float(raw["timestep_fs"]),
        hydrogen_mass_amu=float(raw["hydrogen_mass_amu"]),
        nonbonded_cutoff_nm=float(raw["nonbonded_cutoff_nm"]),
        temperature_K=float(raw["temperature_K"]),
        eq_timestep_fs=float(eq.get("timestep_fs", raw["timestep_fs"])),
        save_interval_ps=float(traj.get("save_interval_ps", 100)),
        raw=raw,
    )


def _lipids(spec: Any) -> dict[str, float]:
    if isinstance(spec, str):
        return {spec: 1.0}
    if isinstance(spec, dict):
        out = {str(k): float(v) for k, v in spec.items()}
        if sum(out.values()) <= 0:
            raise ValueError("membrane.lipids fractions must sum to > 0")
        return out
    raise ValueError("membrane.lipids must be a residue name or a fraction mapping")


def load_system(path: Path) -> System:
    raw = _read_yaml(path)
    if "name" not in raw or "structure" not in raw:
        raise ValueError(f"{path} needs name and structure")

    ori = raw.get("orientation") or {}
    mem = raw.get("membrane") or {}
    # Legacy recipes used top-level cvs: — fold into the objective.
    obj_raw = dict(raw.get("objective") or {})
    if not obj_raw.get("observables") and raw.get("cvs"):
        obj_raw.setdefault("type", "conformational_shift")
        obj_raw["observables"] = raw["cvs"]

    structure = Path(raw["structure"])
    if not structure.is_absolute():
        structure = (path.parent / structure).resolve()

    return System(
        name=str(raw["name"]),
        structure=structure,
        orientation=Orientation(
            method=str(ori.get("method", "auto")),
            topology=str(ori.get("topology", "in")),
        ),
        membrane=Membrane(
            lipids=_lipids(mem.get("lipids", "POPC")),
            water_pad_nm=float(mem.get("water_pad_nm", 1.2)),
            ionic_strength_M=float(mem.get("ionic_strength_M", 0.15)),
            optimize_size=bool(mem.get("optimize_size", True)),
            safety_margin_nm=float(mem.get("safety_margin_nm", 1.2)),
        ),
        objective=parse_objective(obj_raw),
        source=path,
    )
