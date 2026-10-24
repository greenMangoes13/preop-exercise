"""Structured submission fields -> rules.CaseFacts (no LLM).

Handles procedure fields, labs, structured vitals and the med list. Note-derived
facts (H&Ps, consents, note vitals, anticoag mentions and plans) come from
extract.py and are merged into the same CaseFacts.
"""

from __future__ import annotations

import datetime as dt
import re

from core import (
    BloodPressureVital,
    LabResult,
    Medication,
    PatientSubmission,
    TemperatureVital,
)
from rules import AnticoagEvidence, CaseFacts, LabRecord, Source, VitalReading

# -------------------------
# Vocabularies
# -------------------------

LAB_CODE_PREFIXES = ("LAB-",)
LAB_DISPLAY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("CBC", re.compile(r"complete blood count", re.I)),
    ("CMP", re.compile(r"comprehensive metabolic", re.I)),
    ("BMP", re.compile(r"basic metabolic", re.I)),
]

# Generic name -> brand names. Antiplatelets (aspirin, clopidogrel) are deliberately absent.
ANTICOAGULANTS: dict[str, tuple[str, ...]] = {
    "apixaban": ("eliquis",),
    "rivaroxaban": ("xarelto",),
    "dabigatran": ("pradaxa",),
    "edoxaban": ("savaysa", "lixiana"),
    "warfarin": ("coumadin", "jantoven"),
    "enoxaparin": ("lovenox",),
    "dalteparin": ("fragmin",),
    "fondaparinux": ("arixtra",),
    "heparin": (),
}
_ANTICOAG_PATTERNS = [
    (generic, re.compile(r"\b(" + "|".join((generic, *brands)) + r")\b", re.I))
    for generic, brands in ANTICOAGULANTS.items()
]
# Unnamed references ("her blood thinner") must not be dropped as "not an anticoagulant".
ANTICOAG_CLASS_PATTERN = re.compile(
    r"\b(blood[- ]thinners?|anticoagulants?|anticoagulation|anticoagulated)\b", re.I
)

# -------------------------
# Value helpers
# -------------------------


def normalize_lab_code(code: str | None, display: str | None = None) -> str | None:
    """`LAB-CBC` -> `CBC`. BMP stays BMP (it does not satisfy CMP)."""

    if code:
        normalized = code.strip().upper()
        for prefix in LAB_CODE_PREFIXES:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
        return normalized
    if display:
        for test, pattern in LAB_DISPLAY_PATTERNS:
            if pattern.search(display):
                return test
    return None


def anticoagulant_generic(name: str | None) -> str | None:
    """Generic name for an anticoagulant mention (brand or generic), else None."""

    if not name:
        return None
    for generic, pattern in _ANTICOAG_PATTERNS:
        if pattern.search(name):
            return generic
    return None


def names_anticoagulant(text: str | None) -> bool:
    """True if the text names an anticoagulant or uses a generic term for one."""

    return bool(text) and (
        anticoagulant_generic(text) is not None or bool(ANTICOAG_CLASS_PATTERN.search(text))
    )


def celsius_to_fahrenheit(value_c: float) -> float:
    return round(value_c * 9 / 5 + 32, 1)


def parse_date(value: str | None) -> dt.date | None:
    """Date part of an ISO date or timestamp; None if missing or unparseable."""

    if not value:
        return None
    try:
        return dt.date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


def parse_timestamp(value: str | None) -> tuple[dt.date, dt.time | None] | None:
    """ISO date or timestamp -> (date, time-or-None), taken as written (no tz shift)."""

    day = parse_date(value)
    if day is None:
        return None
    rest = value.strip()[10:]  # type: ignore[union-attr]
    if not rest.startswith("T"):
        return day, None
    try:
        parsed = dt.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))  # type: ignore[union-attr]
    except ValueError:
        return day, None
    return day, parsed.time().replace(tzinfo=None)


# -------------------------
# Record normalization
# -------------------------


def normalize_lab(lab: LabResult) -> LabRecord | None:
    test = normalize_lab_code(lab.code, lab.display)
    effective = parse_date(lab.effective_at)
    if test is None or effective is None:
        return None  # an unidentifiable or undated lab can't satisfy a window
    return LabRecord(
        source=Source(
            kind="lab",
            id=lab.id,
            type=test,
            date=lab.effective_at,
            value=f"{test} ({lab.status or 'no status'})",
        ),
        test=test,
        effective_at=effective,
        status=(lab.status or "").strip().lower() or None,
    )


def normalize_vital(vital: object, index: int) -> VitalReading | None:
    stamp = parse_timestamp(getattr(vital, "date", None))
    if stamp is None:
        return None
    day, time = stamp
    origin = getattr(vital, "source", None)
    source_id = f"vitals[{index}]"

    if isinstance(vital, BloodPressureVital) and (
        vital.systolic is not None or vital.diastolic is not None
    ):
        reading = f"BP {_fmt(vital.systolic)}/{_fmt(vital.diastolic)}"
        return VitalReading(
            kind="bp",
            source=Source(
                kind="vital",
                id=source_id,
                type="blood_pressure",
                date=vital.date,
                value=f"{reading} ({origin})" if origin else reading,
            ),
            date=day,
            time=time,
            systolic=vital.systolic,
            diastolic=vital.diastolic,
        )
    if isinstance(vital, TemperatureVital) and vital.value_f is not None:
        reading = f"Temp {float(vital.value_f):.1f}°F"
        return VitalReading(
            kind="temp",
            source=Source(
                kind="vital",
                id=source_id,
                type="temperature",
                date=vital.date,
                value=f"{reading} ({origin})" if origin else reading,
            ),
            date=day,
            time=time,
            temp_f=float(vital.value_f),
        )
    return None


def normalize_medication(med: Medication, index: int) -> AnticoagEvidence | None:
    generic = anticoagulant_generic(med.name)
    if generic is None:
        return None
    status = {True: "current", False: "discontinued", None: "unknown"}[med.active]
    return AnticoagEvidence(
        drug=generic,
        status=status,
        from_med_list=True,
        source=Source(
            kind="medication",
            id=f"medications[{index}]",
            type=med.name,
            value=f"{generic}, med list active={med.active}",
        ),
    )


def normalize_submission(submission: PatientSubmission) -> CaseFacts:
    """Structured parts of CaseFacts; note-derived lists are left empty."""

    procedure = submission.procedure
    risk = procedure.procedure_risk if procedure else None
    date_raw = procedure.procedure_date if procedure else None

    labs = [rec for lab in submission.labs if (rec := normalize_lab(lab)) is not None]
    vitals = [
        rec
        for i, vital in enumerate(submission.vitals)
        if (rec := normalize_vital(vital, i)) is not None
    ]
    anticoag = [
        rec
        for i, med in enumerate(submission.medications)
        if (rec := normalize_medication(med, i)) is not None
    ]

    return CaseFacts(
        procedure_source=Source(
            kind="procedure",
            id=(procedure.case_id if procedure else None) or "procedure",
            type=procedure.procedure_type if procedure else None,
            date=date_raw,
            value=f"risk {risk or 'blank'}, date {date_raw or 'blank'}",
        ),
        procedure_date=parse_date(date_raw),
        procedure_risk=risk,
        labs=labs,
        vitals=vitals,
        anticoag=anticoag,
        documents_searched=len(submission.documents),
    )


def _fmt(value: float | int | None) -> str:
    return "?" if value is None else f"{value:g}"
