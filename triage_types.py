"""Policy literals shared by core, rules, and the eval scripts.

Not named `types.py`: that would shadow the stdlib `types` module.
"""

from __future__ import annotations

from typing import Literal

Decision = Literal["READY", "NEEDS_FOLLOW_UP", "NOT_CLEARED"]
ProcedureRisk = Literal["LOW", "MODERATE", "HIGH"]
IssueCategory = Literal[
    "REQUIRED_DOCUMENTATION",
    "REQUIRED_TESTING",
    "ANTICOAGULATION_MANAGEMENT",
    "ACUTE_SAFETY_EXCLUSION",
    "MISSING_REQUIRED_DATA",
]
