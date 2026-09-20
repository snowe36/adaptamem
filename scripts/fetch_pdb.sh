#!/usr/bin/env bash
# Fetch a public PDB into gitignored data/structures/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
id="${1:?usage: fetch_pdb.sh PDBID [out.pdb]}"
id="$(echo "$id" | tr '[:lower:]' '[:upper:]')"
out="${2:-$ROOT/data/structures/${id}.pdb}"
mkdir -p "$(dirname "$out")"
url="https://files.rcsb.org/download/${id}.pdb"
if command -v curl >/dev/null 2>&1; then
  curl -fsSL -A "adaptamem/0.1 (https://github.com/snowe36/adaptamem)" "$url" -o "$out"
else
  wget -q --user-agent="adaptamem/0.1" "$url" -O "$out"
fi
echo "$out"
