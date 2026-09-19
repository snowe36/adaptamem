"""Chooser / pipeline stop. Not a bug — the job cannot proceed as asked."""

from __future__ import annotations

# Typed refusals. Pretty numbers from inadequate sampling are worse than these.
BUDGET = "BUDGET"
COVERAGE = "COVERAGE"
MEMBRANE_QC = "MEMBRANE_QC"
MISSING_ENGINE = "MISSING_ENGINE"
PPM_MISSING = "PPM_MISSING"
NOT_MEMBRANE = "NOT_MEMBRANE"
NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
NOT_READY = "NOT_READY"
STRUCTURE = "STRUCTURE"


class RefuseError(Exception):
    def __init__(self, message: str, code: str = "REFUSED"):
        super().__init__(message)
        self.message = message
        self.code = code

    def format(self) -> str:
        return f"REFUSE  {self.code}\n{self.message}"
