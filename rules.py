"""Deterministic policy rules (Rules 1-4) over normalized records and grounded facts.

No I/O and no LLM calls: `evaluate` is a pure function of `CaseFacts`. Upstream,
`normalize.py` turns structured submission fields into these types and `extract.py`
turns grounded note facts into them.
"""

from __future__ import annotations

import datetime as dt
from functools import cmp_to_key
from typing import Literal

from pydantic import BaseModel, Field

from triage_types import Decision, IssueCategory, ProcedureRisk

# -------------------------
# Policy constants
# -------------------------

HP_WINDOW_DAYS = 30
REQUIRED_TESTS: dict[ProcedureRisk, list[tuple[str, int]]] = {
    "LOW": [("CBC", 30)],
    "MODERATE": [("CBC", 30)],
    "HIGH": [("CBC", 14), ("CMP", 14)],
}
SBP_LIMIT = 180  # >= is an exclusion
DBP_LIMIT = 110  # >= is an exclusion
TEMP_LIMIT_F = 100.4  # > is an exclusion

# -------------------------
# Input types
# -------------------------

SourceKind = Literal["document", "lab", "vital", "medication", "procedure"]


class Source(BaseModel):
    """Pointer a nurse can follow back to the exact record."""

    kind: SourceKind
    id: str | None = None  # doc_id, lab id, case_id, or med list name
    type: str | None = None  # doc type, lab code, vital type
    date: str | None = None  # ISO date or timestamp of the record
    value: str | None = None  # normalized value, e.g. "BP 157/117"


class HPFact(BaseModel):
    source: Source
    date: dt.date | None
    is_preop_hp: bool


class ConsentFact(BaseModel):
    source: Source
    date: dt.date | None = None
    signed: bool | None  # None = unclear
    matches_scheduled_procedure: bool | None


class LabRecord(BaseModel):
    source: Source
    test: str  # normalized code, e.g. "CBC", "CMP", "BMP"
    effective_at: dt.date
    status: str | None = None


class VitalReading(BaseModel):
    kind: Literal["bp", "temp"]
    source: Source
    date: dt.date
    time: dt.time | None = None
    from_note: bool = False
    note_order: int = 0  # position within its note; a later recheck is higher
    systolic: float | None = None
    diastolic: float | None = None
    temp_f: float | None = None


class AnticoagEvidence(BaseModel):
    """One statement about an anticoagulant: a med list row or a note mention."""

    drug: str  # generic name
    status: Literal["current", "discontinued", "unknown"]
    from_med_list: bool
    source: Source


class AnticoagPlan(BaseModel):
    drug: str  # generic name
    has_pre_op_instruction: bool
    has_post_op_instruction: bool
    deferred_to_other_team: bool
    source: Source


class CaseFacts(BaseModel):
    procedure_source: Source
    procedure_date: dt.date | None
    procedure_risk: ProcedureRisk | None
    hps: list[HPFact] = Field(default_factory=list)
    consents: list[ConsentFact] = Field(default_factory=list)
    labs: list[LabRecord] = Field(default_factory=list)
    vitals: list[VitalReading] = Field(default_factory=list)
    anticoag: list[AnticoagEvidence] = Field(default_factory=list)
    anticoag_plans: list[AnticoagPlan] = Field(default_factory=list)
    documents_searched: int = 0  # all submission documents, for absence evidence

    @property
    def structured_vitals_searched(self) -> int:
        return sum(1 for v in self.vitals if not v.from_note)


# -------------------------
# Output types
# -------------------------

CheckStatus = Literal["PASS", "FAIL", "UNKNOWN", "SKIPPED"]


class Check(BaseModel):
    rule: str
    category: IssueCategory
    status: CheckStatus
    evidence: str
    sources: list[Source] = Field(default_factory=list)


class RuleIssue(BaseModel):
    category: IssueCategory
    rule: str
    evidence: str
    sources: list[Source]


class TriageResult(BaseModel):
    decision: Decision
    issues: list[RuleIssue]
    checks: list[Check]
    explanation: str


# -------------------------
# Rules
# -------------------------


def _days_before(procedure_date: dt.date, record_date: dt.date) -> int:
    return (procedure_date - record_date).days


def check_procedure_fields(facts: CaseFacts) -> list[Check]:
    checks: list[Check] = []
    src = facts.procedure_source
    for rule, value in (
        ("0.1 Procedure date", facts.procedure_date),
        ("0.2 Procedure risk", facts.procedure_risk),
    ):
        if value is None:
            checks.append(
                Check(
                    rule=rule,
                    category="MISSING_REQUIRED_DATA",
                    status="UNKNOWN",
                    evidence=f"{rule.split(' ', 1)[1]} is blank; dependent rules not evaluated.",
                    sources=[src],
                )
            )
        else:
            checks.append(
                Check(
                    rule=rule,
                    category="MISSING_REQUIRED_DATA",
                    status="PASS",
                    evidence=f"{rule.split(' ', 1)[1]}: {value}.",
                    sources=[src],
                )
            )
    return checks


def check_hp(facts: CaseFacts) -> Check:
    rule = f"1.1 H&P within {HP_WINDOW_DAYS} days"
    category: IssueCategory = "REQUIRED_DOCUMENTATION"
    preop = sorted(
        (h for h in facts.hps if h.is_preop_hp and h.date is not None),
        key=lambda h: h.date,
        reverse=True,
    )

    if not preop:
        others = [h.source for h in facts.hps]
        return Check(
            rule=rule,
            category=category,
            status="FAIL",
            evidence=f"No dated pre-op H&P found among {facts.documents_searched} documents"
            + (
                f" ({len(others)} H&P document(s) not pre-op or undated)." if others else "."
            ),
            sources=others or [facts.procedure_source],
        )

    pd = facts.procedure_date
    if pd is None:
        return Check(
            rule=rule,
            category=category,
            status="SKIPPED",
            evidence=f"Pre-op H&P dated {preop[0].date}; window not checked (no procedure date).",
            sources=[preop[0].source],
        )

    for hp in preop:
        days = _days_before(pd, hp.date)
        if 0 <= days <= HP_WINDOW_DAYS:
            return Check(
                rule=rule,
                category=category,
                status="PASS",
                evidence=f"Pre-op H&P {hp.date} is {days} days before {pd}.",
                sources=[hp.source],
            )

    latest = preop[0]
    days = _days_before(pd, latest.date)
    if days < 0:
        detail = f"is dated after the procedure {pd}"
    else:
        detail = f"is {days} days before {pd} (limit {HP_WINDOW_DAYS})"
    return Check(
        rule=rule,
        category=category,
        status="FAIL",
        evidence=f"Most recent pre-op H&P {latest.date} {detail}.",
        sources=[latest.source],
    )


def check_consent(facts: CaseFacts) -> Check:
    rule = "1.2 Signed surgical consent"
    category: IssueCategory = "REQUIRED_DOCUMENTATION"
    valid = [
        c for c in facts.consents if c.signed is True and c.matches_scheduled_procedure is True
    ]
    if valid:
        best = max(valid, key=lambda c: c.date or dt.date.min)
        return Check(
            rule=rule,
            category=category,
            status="PASS",
            evidence=f"Signed consent for the scheduled procedure ({best.date or 'undated'}).",
            sources=[best.source],
        )

    if not facts.consents:
        return Check(
            rule=rule,
            category=category,
            status="FAIL",
            evidence=f"No surgical consent found among {facts.documents_searched} documents.",
            sources=[facts.procedure_source],
        )

    def describe(c: ConsentFact) -> str:
        signed = {True: "signed", False: "not signed", None: "signature unclear"}[c.signed]
        if c.matches_scheduled_procedure is False:
            signed += ", different procedure"
        elif c.matches_scheduled_procedure is None:
            signed += ", procedure unclear"
        return f"{c.date or 'undated'} {signed}"

    return Check(
        rule=rule,
        category=category,
        status="FAIL",
        evidence="No signed consent for the scheduled procedure: "
        + "; ".join(describe(c) for c in facts.consents)
        + ".",
        sources=[c.source for c in facts.consents],
    )


def check_labs(facts: CaseFacts) -> list[Check]:
    category: IssueCategory = "REQUIRED_TESTING"
    pd, risk = facts.procedure_date, facts.procedure_risk
    if pd is None or risk is None:
        missing = "procedure date" if pd is None else "procedure risk"
        return [
            Check(
                rule="2 Required testing",
                category=category,
                status="SKIPPED",
                evidence=f"Not evaluated (no {missing}).",
                sources=[facts.procedure_source],
            )
        ]

    checks: list[Check] = []
    for test, limit in REQUIRED_TESTS[risk]:
        rule = f"2 {test} within {limit} days ({risk} risk)"
        results = [lab for lab in facts.labs if lab.test == test]
        valid = [lab for lab in results if lab.status != "entered-in-error"]
        ignored = len(results) - len(valid)
        note = f" ({ignored} entered-in-error result ignored)" if ignored else ""

        if not valid:
            checks.append(
                Check(
                    rule=rule,
                    category=category,
                    status="FAIL",
                    evidence=f"No {test} found among {len(facts.labs)} lab results{note}.",
                    sources=[facts.procedure_source],
                )
            )
            continue

        latest = max(valid, key=lambda lab: lab.effective_at)
        days = _days_before(pd, latest.effective_at)
        if 0 <= days <= limit:
            status: CheckStatus = "PASS"
            detail = f"is {days} days before {pd}"
        elif days < 0:
            status = "FAIL"
            detail = f"is dated after the procedure {pd}"
        else:
            status = "FAIL"
            detail = f"is {days} days before {pd} (limit {limit})"
        checks.append(
            Check(
                rule=rule,
                category=category,
                status=status,
                evidence=f"Most recent {test} {latest.effective_at} {detail}{note}.",
                sources=[latest.source],
            )
        )
    return checks


def check_anticoag(facts: CaseFacts) -> list[Check]:
    category: IssueCategory = "ANTICOAGULATION_MANAGEMENT"
    if not facts.anticoag:
        return [
            Check(
                rule="3 Anticoagulation",
                category=category,
                status="PASS",
                evidence="No anticoagulant on the med list or reported in notes.",
            )
        ]

    checks: list[Check] = []
    for drug in sorted({e.drug for e in facts.anticoag}):
        rule = f"3 Anticoagulation plan: {drug}"
        evidence = [e for e in facts.anticoag if e.drug == drug]
        sources = [e.source for e in evidence]
        current = any(e.status == "current" for e in evidence)
        unclear = [e for e in evidence if e.status == "unknown"]

        # Fail closed: any unclear statement blocks "not taking" unless another says current.
        if not current and unclear:
            where = sorted({"med list" if e.from_med_list else "notes" for e in unclear})
            checks.append(
                Check(
                    rule=rule,
                    category="MISSING_REQUIRED_DATA",
                    status="UNKNOWN",
                    evidence=f"Unclear whether {drug} is current ({' and '.join(where)}); "
                    "no statement confirms it is being taken or stopped.",
                    sources=sources,
                )
            )
            continue

        if not current:
            checks.append(
                Check(
                    rule=rule,
                    category=category,
                    status="PASS",
                    evidence=f"{drug} is not currently taken; no plan needed.",
                    sources=sources,
                )
            )
            continue

        plans = [p for p in facts.anticoag_plans if p.drug == drug]
        complete = [
            p
            for p in plans
            if p.has_pre_op_instruction
            and p.has_post_op_instruction
            and not p.deferred_to_other_team
        ]
        if complete:
            checks.append(
                Check(
                    rule=rule,
                    category=category,
                    status="PASS",
                    evidence=f"{drug} is current; plan has pre-op and post-op instructions.",
                    sources=sources + [complete[0].source],
                )
            )
            continue

        if plans:
            gaps = []
            for p in plans:
                missing = []
                if not p.has_pre_op_instruction:
                    missing.append("no pre-op instruction")
                if not p.has_post_op_instruction:
                    missing.append("no post-op instruction")
                if p.deferred_to_other_team:
                    missing.append("deferred to another team")
                gaps.append(f"{p.source.date or 'undated'}: {', '.join(missing)}")
            detail = "plan is incomplete (" + "; ".join(gaps) + ")"
        else:
            other = sorted({p.drug for p in facts.anticoag_plans})
            detail = "no plan on file" + (
                f" (only for {', '.join(other)})" if other else ""
            )
        checks.append(
            Check(
                rule=rule,
                category=category,
                status="FAIL",
                evidence=f"{drug} is current; {detail}.",
                sources=sources + [p.source for p in plans or facts.anticoag_plans],
            )
        )
    return checks


def _compare_readings(a: VitalReading, b: VitalReading) -> int:
    """Order two readings in time; positive means `a` is later."""

    if a.date != b.date:
        return 1 if a.date > b.date else -1
    same_note = a.from_note and b.from_note and a.source.id == b.source.id
    if same_note:
        return a.note_order - b.note_order
    if a.time is not None and b.time is not None:
        if a.time != b.time:
            return 1 if a.time > b.time else -1
        return int(a.from_note) - int(b.from_note)
    # Same day and at least one reading has no time: an untimed note reading wins.
    a_untimed_note = a.from_note and a.time is None
    b_untimed_note = b.from_note and b.time is None
    if a_untimed_note != b_untimed_note:
        return 1 if a_untimed_note else -1
    return int(a.from_note) - int(b.from_note) or a.note_order - b.note_order


def latest_reading(readings: list[VitalReading]) -> VitalReading | None:
    if not readings:
        return None
    return max(readings, key=cmp_to_key(_compare_readings))


def _reading_label(r: VitalReading) -> str:
    when = f"{r.date} {r.time.strftime('%H:%M')}" if r.time else str(r.date)
    return f"{when} ({r.source.type or r.source.kind})"


def check_vitals(facts: CaseFacts) -> list[Check]:
    checks: list[Check] = []

    bp = latest_reading(
        [v for v in facts.vitals if v.kind == "bp" and (v.systolic or v.diastolic)]
    )
    rule = "4.1 Blood pressure"
    if bp is None:
        checks.append(
            Check(
                rule=rule,
                category="MISSING_REQUIRED_DATA",
                status="UNKNOWN",
                evidence=f"No blood pressure found among {facts.structured_vitals_searched} "
                f"structured vitals and {facts.documents_searched} documents.",
                sources=[facts.procedure_source],
            )
        )
    else:
        sbp, dbp = bp.systolic, bp.diastolic
        reading = f"{sbp:g}/{dbp:g}" if sbp is not None and dbp is not None else (
            f"SBP {sbp:g}" if sbp is not None else f"DBP {dbp:g}"
        )
        breaches = []
        if sbp is not None and sbp >= SBP_LIMIT:
            breaches.append(f"systolic >= {SBP_LIMIT}")
        if dbp is not None and dbp >= DBP_LIMIT:
            breaches.append(f"diastolic >= {DBP_LIMIT}")
        checks.append(
            Check(
                rule=rule,
                category="ACUTE_SAFETY_EXCLUSION",
                status="FAIL" if breaches else "PASS",
                evidence=f"Most recent BP {reading} at {_reading_label(bp)}"
                + (f": {' and '.join(breaches)}." if breaches else "."),
                sources=[bp.source],
            )
        )

    temp = latest_reading([v for v in facts.vitals if v.kind == "temp" and v.temp_f is not None])
    rule = "4.2 Temperature"
    if temp is None:
        checks.append(
            Check(
                rule=rule,
                category="MISSING_REQUIRED_DATA",
                status="UNKNOWN",
                evidence=f"No temperature found among {facts.structured_vitals_searched} "
                f"structured vitals and {facts.documents_searched} documents.",
                sources=[facts.procedure_source],
            )
        )
    else:
        fever = temp.temp_f > TEMP_LIMIT_F
        checks.append(
            Check(
                rule=rule,
                category="ACUTE_SAFETY_EXCLUSION",
                status="FAIL" if fever else "PASS",
                evidence=f"Most recent temperature {temp.temp_f:.1f}°F at {_reading_label(temp)}"
                + (f": above {TEMP_LIMIT_F}°F." if fever else "."),
                sources=[temp.source],
            )
        )
    return checks


# -------------------------
# Decision
# -------------------------


def evaluate(facts: CaseFacts) -> TriageResult:
    """Apply Rules 1-4 and report every failing check as an issue."""

    checks = [
        *check_procedure_fields(facts),
        check_hp(facts),
        check_consent(facts),
        *check_labs(facts),
        *check_anticoag(facts),
        *check_vitals(facts),
    ]
    issues = [
        RuleIssue(category=c.category, rule=c.rule, evidence=c.evidence, sources=c.sources)
        for c in checks
        if c.status in ("FAIL", "UNKNOWN")
    ]

    decision: Decision
    if any(i.category == "ACUTE_SAFETY_EXCLUSION" for i in issues):
        decision = "NOT_CLEARED"
    elif issues:
        decision = "NEEDS_FOLLOW_UP"
    else:
        decision = "READY"

    return TriageResult(
        decision=decision,
        issues=issues,
        checks=checks,
        explanation=build_explanation(decision, issues),
    )


def build_explanation(decision: Decision, issues: list[RuleIssue]) -> str:
    if not issues:
        return "READY: all policy checks passed."
    categories = sorted({i.category for i in issues})
    noun = "issue" if len(issues) == 1 else "issues"
    lines = [f"{decision}: {len(issues)} {noun} ({', '.join(categories)})."]
    lines.extend(f"- [{i.rule}] {i.evidence}" for i in issues)
    return "\n".join(lines)
