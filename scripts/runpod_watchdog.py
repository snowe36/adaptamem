#!/usr/bin/env python3
"""Kill a Runpod when the GPU job finishes, fails, or hits the wall.

On every Runpod, RUNPOD_POD_ID is set. Self-kill also needs RUNPOD_API_KEY
in the pod env. Without the key this prints WATCHDOG_CANNOT_KILL — the
agent must then terminate from outside. Never sleep the container to keep
logs; that just keeps billing.
"""

from __future__ import annotations

import argparse
import os
import urllib.error
import urllib.request


API = "https://rest.runpod.io/v2/pods"


def terminate_self(reason: str, *, pod_id: str | None = None, api_key: str | None = None) -> bool:
    pod_id = pod_id or os.environ.get("RUNPOD_POD_ID")
    api_key = api_key or os.environ.get("RUNPOD_API_KEY")
    print(f"watchdog  {reason}", flush=True)
    if not pod_id or not api_key:
        print("WATCHDOG_CANNOT_KILL missing RUNPOD_POD_ID or RUNPOD_API_KEY", flush=True)
        return False
    req = urllib.request.Request(
        f"{API}/{pod_id}",
        method="DELETE",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            print(f"watchdog  terminated {pod_id} HTTP {resp.status}", flush=True)
            return True
    except urllib.error.HTTPError as exc:
        print(f"watchdog  terminate failed HTTP {exc.code} {exc.reason}", flush=True)
        return False
    except urllib.error.URLError as exc:
        print(f"watchdog  terminate failed {exc.reason}", flush=True)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Terminate this Runpod.")
    parser.add_argument("--kill", action="store_true", help="DELETE the pod in RUNPOD_POD_ID")
    parser.add_argument("--reason", default="manual")
    parser.add_argument("--pod-id")
    args = parser.parse_args()
    if not args.kill:
        parser.print_help()
        return 2
    ok = terminate_self(args.reason, pod_id=args.pod_id)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
