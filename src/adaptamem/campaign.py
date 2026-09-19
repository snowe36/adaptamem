"""Campaign identity. Reproduce means replay hashes + protocol, not a hidden MD path."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from adaptamem.errors import RefuseError
from adaptamem.schema import Protocol

POLICY = "0.2"


def new_campaign_id() -> str:
    return uuid.uuid4().hex[:12]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def protocol_hash(protocol: Protocol) -> str:
    blob = json.dumps(protocol.raw, sort_keys=True, default=str)
    return sha256_text(blob)


def forcefield_hash(protocol: Protocol) -> str:
    return sha256_text(f"{protocol.force_field}|{protocol.water}")


def software_version() -> str:
    try:
        return version("adaptamem")
    except PackageNotFoundError:
        return "0.1.0"


def stamp(
    *,
    campaign_id: str | None = None,
    structure: Path | None = None,
    protocol: Protocol | None = None,
    gpu: str | None = None,
    seed: int = 42,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cid = campaign_id or new_campaign_id()
    proto = protocol
    payload: dict[str, Any] = {
        "campaign_id": cid,
        "policy": POLICY,
        "created_utc": datetime.now(UTC).isoformat(),
        "software_version": software_version(),
        "gpu": gpu,
        "random_seed": int(seed),
        "system_hash": sha256_file(structure) if structure and Path(structure).is_file() else None,
        "protocol_hash": protocol_hash(proto) if proto is not None else None,
        "forcefield_hash": forcefield_hash(proto) if proto is not None else None,
        "forcefield": proto.force_field if proto is not None else None,
        "water": proto.water if proto is not None else None,
    }
    if extra:
        payload.update(extra)
    return payload


def write_campaign(workdir: Path, payload: dict[str, Any]) -> Path:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / "campaign.json"
    existing: dict[str, Any] = {}
    if path.is_file():
        existing = json.loads(path.read_text())
    existing.update(payload)
    path.write_text(json.dumps(existing, indent=2) + "\n")
    return path


def load_campaign(workdir: Path) -> dict[str, Any]:
    path = Path(workdir) / "campaign.json"
    if not path.is_file():
        raise RefuseError(f"no campaign.json in {workdir}", code="NOT_READY")
    return json.loads(path.read_text())


def find_campaign(target: str, *, search: Path | None = None) -> Path:
    """Resolve a campaign id or workdir to campaign.json."""
    p = Path(target)
    if p.is_dir() and (p / "campaign.json").is_file():
        return p / "campaign.json"
    if p.is_file() and p.name == "campaign.json":
        return p
    root = search or Path("runs")
    if root.is_dir():
        for cand in root.rglob("campaign.json"):
            data = json.loads(cand.read_text())
            if str(data.get("campaign_id")) == target:
                return cand
    cwd = Path.cwd() / "campaign.json"
    if cwd.is_file() and json.loads(cwd.read_text()).get("campaign_id") == target:
        return cwd
    raise RefuseError(
        f"campaign {target!r} not found (pass a workdir or an id under runs/)",
        code="NOT_READY",
    )


def reproduce_report(
    target: str, *, protocol: Protocol | None = None, search: Path | None = None
) -> str:
    path = find_campaign(target, search=search)
    data = json.loads(path.read_text())
    lines = [
        f"REPRODUCE  {data.get('campaign_id')}",
        f"file            {path}",
        f"software        {data.get('software_version')}",
        f"gpu             {data.get('gpu')}",
        f"seed            {data.get('random_seed')}",
        f"system_hash     {data.get('system_hash')}",
        f"forcefield_hash {data.get('forcefield_hash')}",
        f"protocol_hash   {data.get('protocol_hash')}",
        f"forcefield      {data.get('forcefield')} / {data.get('water')}",
    ]
    if protocol is not None:
        now = protocol_hash(protocol)
        match = now == data.get("protocol_hash")
        lines.append(f"protocol_now    {now}  {'match' if match else 'DRIFT'}")
        if not match:
            lines.append("NOTE  current protocol.yaml does not match this campaign")
    return "\n".join(lines)
