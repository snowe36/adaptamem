"""Chooser / pipeline stop. Not a bug — the job cannot proceed as asked."""


class RefuseError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
