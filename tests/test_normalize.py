from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from core import PatientSubmission
from normalize import (
    anticoagulant_generic,
    celsius_to_fahrenheit,
    names_anticoagulant,
    near_miss_anticoagulant,
    normalize_lab_code,
    normalize_submission,
    parse_timestamp,
)
from rules import evaluate

DATA = Path(__file__).resolve().parent.parent / "data" / "patients_sample_50.jsonl"


@pytest.mark.parametrize(
    ("code", "display", "expected"),
    [
        ("CBC", None, "CBC"),
        ("LAB-CBC", None, "CBC"),
        ("lab-cmp", None, "CMP"),
        ("LAB-BMP", None, "BMP"),
        ("BMP", None, "BMP"),
        ("PT-INR", None, "PT-INR"),
        (None, "Complete Blood Count w/ diff", "CBC"),
        (None, "Basic Metabolic Panel", "BMP"),
        (None, None, None),
    ],
)
def test_normalize_lab_code(code: str | None, display: str | None, expected: str | None) -> None:
    assert normalize_lab_code(code, display) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Eliquis 5 mg PO BID", "apixaban"),
        ("apixaban 5 mg PO BID", "apixaban"),
        ("Xarelto 20 mg PO daily with evening meal", "rivaroxaban"),
        ("Coumadin 5 mg PO daily (INR goal 2-3)", "warfarin"),
        ("warfarin (Coumadin) 5 mg PO daily", "warfarin"),
        ("dabigatran (Pradaxa) 150 mg PO BID", "dabigatran"),
        ("edoxaban (Savaysa) 60 mg PO daily", "edoxaban"),
        ("Lovenox 40 mg SC daily", "enoxaparin"),
        ("ASA 81 mg PO daily", None),
        ("Aspirin EC 81 mg PO daily", None),
        ("clopidogrel 75 mg PO daily", None),
        ("atorvastatin 40 mg PO nightly", None),
        (None, None),
    ],
)
def test_anticoagulant_generic(name: str | None, expected: str | None) -> None:
    assert anticoagulant_generic(name) == expected


@pytest.mark.parametrize(("c", "f"), [(37.0, 98.6), (37.9, 100.2), (38.0, 100.4), (38.6, 101.5)])
def test_celsius_to_fahrenheit(c: float, f: float) -> None:
    assert celsius_to_fahrenheit(c) == f


def test_parse_timestamp() -> None:
    assert parse_timestamp("2026-02-22T09:05:00Z") == (dt.date(2026, 2, 22), dt.time(9, 5))
    assert parse_timestamp("2026-02-22") == (dt.date(2026, 2, 22), None)
    assert parse_timestamp("not a date") is None
    assert parse_timestamp(None) is None


def submission(**overrides: object) -> PatientSubmission:
    payload: dict[str, object] = {
        "procedure": {
            "case_id": "case-1",
            "procedure_type": "Hernia repair",
            "procedure_risk": "LOW",
            "procedure_date": "2026-03-01",
        },
        "vitals": [
            {"type": "blood_pressure", "systolic": 122, "diastolic": 76,
             "date": "2026-02-26T09:05:00Z", "source": "Pre-op clinic"},
            {"type": "temperature", "value_f": 98.4,
             "date": "2026-02-26T09:10:00Z", "source": "Pre-op clinic"},
        ],
        "labs": [
            {"id": "lab-1", "code": "LAB-CBC", "effective_at": "2026-02-20T10:00:00Z", "status": "final"},
            {"id": "lab-2", "code": "CBC", "effective_at": "2026-02-25T10:00:00Z",
             "status": "entered-in-error"},
        ],
        "medications": [
            {"name": "Eliquis 5 mg PO BID", "active": True},
            {"name": "Coumadin 5 mg PO daily", "active": None},
            {"name": "aspirin 81 mg PO daily", "active": True},
        ],
        "documents": [{"doc_id": "d1", "text": "x"}, {"doc_id": "d2", "text": "y"}],
    }
    payload.update(overrides)
    return PatientSubmission.model_validate(payload)


def test_normalize_submission_structured_fields() -> None:
    facts = normalize_submission(submission())

    assert facts.procedure_date == dt.date(2026, 3, 1)
    assert facts.procedure_risk == "LOW"
    assert facts.procedure_source.id == "case-1"
    assert facts.documents_searched == 2

    assert [(lab.test, lab.status, lab.source.id) for lab in facts.labs] == [
        ("CBC", "final", "lab-1"),
        ("CBC", "entered-in-error", "lab-2"),
    ]

    bp, temp = facts.vitals
    assert (bp.kind, bp.systolic, bp.diastolic, bp.time) == ("bp", 122, 76, dt.time(9, 5))
    assert (temp.kind, temp.temp_f) == ("temp", 98.4)
    assert not bp.from_note
    assert bp.source.id == "vitals[0]"
    assert bp.source.value == "BP 122/76 (Pre-op clinic)"

    assert [(e.drug, e.status, e.source.id) for e in facts.anticoag] == [
        ("apixaban", "current", "medications[0]"),
        ("warfarin", "unknown", "medications[1]"),
    ]
    assert facts.hps == [] and facts.consents == [] and facts.anticoag_plans == []


def test_normalize_submission_blank_procedure_fields() -> None:
    facts = normalize_submission(
        submission(procedure={"case_id": "case-2", "procedure_risk": None, "procedure_date": None})
    )

    assert facts.procedure_date is None
    assert facts.procedure_risk is None
    assert facts.procedure_source.value == "risk blank, date blank"


def test_undated_or_unidentified_records_are_dropped() -> None:
    facts = normalize_submission(
        submission(
            vitals=[{"type": "blood_pressure", "systolic": 120, "diastolic": 80, "date": None}],
            labs=[{"id": "x", "code": "CBC", "effective_at": None}, {"id": "y"}],
        )
    )

    assert facts.vitals == []
    assert facts.labs == []


def test_all_sample_cases_normalize_and_evaluate() -> None:
    known_tests = {"CBC", "CMP", "BMP", "TSH", "UA", "T&S", "HBA1C", "PT-INR"}
    rows = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]

    for row in rows:
        sub = PatientSubmission.model_validate(row["submission"])
        facts = normalize_submission(sub)

        assert len(facts.labs) == len(sub.labs), row["case_id"]
        assert len(facts.vitals) == len(sub.vitals), row["case_id"]
        assert {lab.test for lab in facts.labs} <= known_tests, row["case_id"]
        evaluate(facts)  # structured-only facts must not crash the rules


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Eliquis 5 mg PO BID", True),
        ("takes her blood thinner nightly", True),
        ("Not currently anticoagulated.", True),
        ("perioperative management of anticoagulation", True),
        ("PMH: Hypertension, Atrial fibrillation.", False),
        ("ASA 81 mg PO daily", False),
        ("", False),
        # Near-miss spellings (case_00042) count as naming the drug.
        ("rivoraxaban 20 mg PO daily with evening meal", True),
        ("Elliquis 5 mg BID", True),
        ("Coumadine 5 mg daily", True),
        ("warfrin 5 mg", True),
        # Ordinary clinical words near a drug name must not.
        ("High-risk hepatic lobectomy", False),
        ("tissue fragment sent to pathology", False),
        ("Patient prefers Spanish, interpreter used", False),
    ],
)
def test_names_anticoagulant(text: str, expected: bool) -> None:
    assert names_anticoagulant(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("rivoraxaban", "rivaroxaban"), ("Elliquis", "eliquis"), ("hepatic", None), ("heparin", "heparin")],
)
def test_near_miss_anticoagulant(text: str, expected: str | None) -> None:
    assert near_miss_anticoagulant(text) == expected
