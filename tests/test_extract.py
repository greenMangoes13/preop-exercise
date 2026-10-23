from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import extract
from core import Document, PatientSubmission
from extract import (
    EXTRACTION_INSTRUCTIONS,
    AnticoagMention,
    AnticoagPlanExtract,
    CaseExtraction,
    ConsentExtract,
    HPCandidate,
    NoteVital,
    call_extraction_model,
    ground,
    normalize_text,
)

NURSING = Document(
    doc_id="nurse-1",
    type="Pre-op Nursing Intake",
    date="2026-02-28",
    text=(
        "VITAL SIGNS: On arrival at 10:12: NIBP 192/106, T 99.1 F, HR 76.\n"
        "After resting quietly for 24 minutes, BP was rechecked at 10:36: BP 183/104.\n"
        "  - Patient also reports taking Eliquis 5 mg (BID);  outside cardiologist."
    ),
)
CONSENT = Document(
    doc_id="consent-1",
    type="Surgical Consent",
    date="2026-02-20",
    text="Procedure: carpal tunnel release\nPatient signature: pending",
)
PLAN = Document(
    doc_id="plan-1",
    type="Cardiology Progress Note",
    date="2026-02-15",
    text=(
        "Stop warfarin 5 days before surgery (last dose 02/24/2026);\n"
        "Post-operative anticoagulation will be addressed after surgery."
    ),
)
DOCS = [NURSING, CONSENT, PLAN]


def empty_extraction(**overrides: object) -> CaseExtraction:
    base: dict[str, object] = {
        "hp_candidates": [],
        "consents": [],
        "vitals": [],
        "anticoagulant_mentions": [],
        "anticoagulant_plans": [],
    }
    base.update(overrides)
    return CaseExtraction.model_validate(base)


def bp_vital(quote: str, sbp: int, dbp: int, order: int, time: str | None) -> NoteVital:
    return NoteVital(
        doc_id="nurse-1", quote=quote, kind="bp", systolic=sbp, diastolic=dbp,
        temp_value=None, temp_unit=None, time=time, order_in_note=order,
    )


# -------------------------
# Grounding
# -------------------------


def test_normalize_text_collapses_whitespace_case_and_curly_quotes() -> None:
    assert normalize_text("  Patient\n  ALSO  reports ‘x’ ") == "patient also reports 'x'"


def test_grounded_quote_survives_whitespace_and_case_differences() -> None:
    mention = AnticoagMention(
        doc_id="nurse-1",
        quote="patient also reports taking ELIQUIS 5 mg (BID); outside cardiologist",
        drug="Eliquis",
        status="current",
    )
    grounded = ground(empty_extraction(anticoagulant_mentions=[mention]), DOCS)

    assert grounded.drops == []
    [evidence] = grounded.anticoag
    assert (evidence.drug, evidence.status, evidence.from_med_list) == ("apixaban", "current", False)
    assert evidence.source.id == "nurse-1"
    assert evidence.source.date == "2026-02-28"


@pytest.mark.parametrize(
    ("doc_id", "quote", "reason"),
    [
        ("nurse-1", "Patient reports taking Eliquis daily", "quote not in document"),
        ("consent-1", "Patient also reports taking Eliquis 5 mg", "quote not in document"),
        ("no-such-doc", "Patient also reports taking Eliquis 5 mg", "unknown doc_id"),
        ("nurse-1", "   ", "quote not in document"),
    ],
)
def test_ungrounded_anticoag_mention_is_dropped_and_kept_as_unverified(
    doc_id: str, quote: str, reason: str
) -> None:
    mention = AnticoagMention(doc_id=doc_id, quote=quote, drug="Eliquis", status="current")
    grounded = ground(empty_extraction(anticoagulant_mentions=[mention]), DOCS)

    assert grounded.anticoag == []
    assert [(d.kind, d.reason) for d in grounded.drops] == [("anticoag_mention", reason)]
    [unverified] = grounded.unverified
    assert (unverified.kind, unverified.reason) == ("anticoag", reason)
    assert unverified.source.id == doc_id


def test_ungrounded_consent_is_dropped_but_not_unverified() -> None:
    # Absence already fails closed for consents, so no safety-net entry is needed.
    consent = ConsentExtract(
        doc_id="consent-1", quote="Patient signature: /s/ signed", signed="true",
        procedure_named="carpal tunnel release", matches_scheduled_procedure="yes",
    )
    grounded = ground(empty_extraction(consents=[consent]), DOCS)

    assert grounded.consents == []
    assert grounded.drops[0].reason == "quote not in document"
    assert grounded.unverified == []


def test_vital_with_values_not_in_document_is_unverified() -> None:
    wrong = bp_vital("BP was rechecked at 10:36: BP 193/104", 193, 104, 2, "10:36")
    grounded = ground(empty_extraction(vitals=[wrong]), DOCS)

    assert grounded.vitals == []
    assert grounded.drops[0].reason == "values not found in document"
    [unverified] = grounded.unverified
    assert (unverified.kind, unverified.date) == ("bp", dt.date(2026, 2, 28))


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("BP was rechecked at 10:36: BP 183/104.", True),  # sentence-ending period
        ("Blood pressure 183/104 mmHg", True),
        ("BP 183 / 104", True),
        ("BP 183/1045", False),
        ("BP 183/104.5", False),
        ("BP 1183/104", False),
    ],
)
def test_bp_value_search(text: str, found: bool) -> None:
    doc = Document(doc_id="b", type="Note", date="2026-03-01", text=text)
    vital = NoteVital(doc_id="b", quote="x", kind="bp", systolic=183, diastolic=104,
                      temp_value=None, temp_unit=None, time=None, order_in_note=1)

    assert bool(ground(empty_extraction(vitals=[vital]), [doc]).vitals) is found


def test_garbled_quote_with_real_values_is_verified_by_value() -> None:
    # case_00047: the model emitted a control character where the note has "°C"
    doc = Document(
        doc_id="n47", type="Nursing Pre-procedure Assessment", date="2026-02-28",
        text="VITAL SIGNS:\n  09:05  BP: 161/117, 37.2 °C tympanic, HR 84, SpO2 95% RA",
    )
    garbled = "09:05  BP: 161/117, 37.2 \x176 tympanic"
    vitals = [
        NoteVital(doc_id="n47", quote=garbled, kind="bp", systolic=161, diastolic=117,
                  temp_value=None, temp_unit=None, time="09:05", order_in_note=1),
        NoteVital(doc_id="n47", quote=garbled, kind="temp", systolic=None, diastolic=None,
                  temp_value=37.2, temp_unit="C", time="09:05", order_in_note=1),
    ]
    grounded = ground(empty_extraction(vitals=vitals), [doc])

    assert grounded.drops == [] and grounded.unverified == []
    bp, temp = grounded.vitals
    assert (bp.systolic, bp.diastolic) == (161, 117)
    assert temp.temp_f == 99.0
    for evidence in grounded.vital_evidence:
        assert not evidence.model_quote_grounded
        assert evidence.document_line == "09:05  BP: 161/117, 37.2 °C tympanic, HR 84, SpO2 95% RA"


@pytest.mark.parametrize(
    ("text", "value", "unit", "found"),
    [
        ("T 99.1 F, HR 76", 99.1, "F", True),
        ("Recheck: Temp 98.6 °F 5 min later.", 98.6, "F", True),
        ("Temp: 37.2C. 9 min later", 37.2, "C", True),
        ("Temperature 98.8 °F (oral)", 98.8, "F", True),
        ("Temperature 98.8 °F (oral)", 98.8, "C", False),  # unit must match
        ("Temperature 98.8 °F (oral)", 98.0, "F", False),
        ("HR 99, SpO2 98% RA", 99.0, "F", False),  # a number without a unit is not a temp
    ],
)
def test_temperature_value_search(text: str, value: float, unit: str, found: bool) -> None:
    doc = Document(doc_id="t", type="Note", date="2026-03-01", text=text)
    vital = NoteVital(doc_id="t", quote="x", kind="temp", systolic=None, diastolic=None,
                      temp_value=value, temp_unit=unit, time=None, order_in_note=1)

    assert bool(ground(empty_extraction(vitals=[vital]), [doc]).vitals) is found


def test_note_vitals_use_document_date_time_and_order() -> None:
    vitals = [
        bp_vital("On arrival at 10:12: NIBP 192/106", 192, 106, 1, "10:12"),
        bp_vital("BP was rechecked at 10:36: BP 183/104", 183, 104, 2, "10:36"),
        NoteVital(
            doc_id="nurse-1", quote="T 99.1 F", kind="temp", systolic=None, diastolic=None,
            temp_value=99.1, temp_unit="F", time="10:12", order_in_note=1,
        ),
    ]
    grounded = ground(empty_extraction(vitals=vitals), DOCS)

    assert grounded.drops == []
    first, recheck, temp = grounded.vitals
    assert (first.date, first.time, first.note_order) == (dt.date(2026, 2, 28), dt.time(10, 12), 1)
    assert (recheck.systolic, recheck.time, recheck.from_note) == (183, dt.time(10, 36), True)
    assert recheck.source.value == "BP 183/104 at 10:36"
    assert temp.temp_f == 99.1


def test_celsius_note_temperature_is_converted() -> None:
    doc = Document(doc_id="n2", type="Nursing", date="2026-03-01", text="Temperature 38.6 °C (oral)")
    vital = NoteVital(
        doc_id="n2", quote="Temperature 38.6 °C", kind="temp", systolic=None, diastolic=None,
        temp_value=38.6, temp_unit="C", time=None, order_in_note=1,
    )
    [reading] = ground(empty_extraction(vitals=[vital]), [doc]).vitals

    assert reading.temp_f == 101.5
    assert reading.time is None
    assert reading.source.value == "Temp 38.6°C (101.5°F)"


def test_consent_enums_map_to_rules_values() -> None:
    consent = ConsentExtract(
        doc_id="consent-1", quote="Patient signature: pending", signed="false",
        procedure_named="carpal tunnel release", matches_scheduled_procedure="yes",
    )
    [fact] = ground(empty_extraction(consents=[consent]), DOCS).consents

    assert (fact.signed, fact.matches_scheduled_procedure) == (False, True)
    assert fact.date == dt.date(2026, 2, 20)


def test_hp_candidate_uses_document_date() -> None:
    doc = Document(doc_id="hp", type="Clinic H&P", date="2025-05-28",
                   text="Encounter: annual wellness / establish care")
    hp = HPCandidate(doc_id="hp", quote="Encounter: annual wellness / establish care",
                     is_preop_hp=False)
    [fact] = ground(empty_extraction(hp_candidates=[hp]), [doc]).hps

    assert (fact.date, fact.is_preop_hp) == (dt.date(2025, 5, 28), False)


def test_plan_components_count_only_when_their_quote_is_grounded() -> None:
    plan = AnticoagPlanExtract(
        doc_id="plan-1",
        drug="warfarin",
        pre_op_quote="Stop warfarin 5 days before surgery (last dose 02/24/2026);",
        post_op_quote="Resume warfarin the evening of surgery",  # hallucinated
        deferral_quote=None,
    )
    grounded = ground(empty_extraction(anticoagulant_plans=[plan]), DOCS)

    [fact] = grounded.anticoag_plans
    assert (fact.has_pre_op_instruction, fact.has_post_op_instruction) == (True, False)
    assert not fact.deferred_to_other_team
    assert [(d.kind, d.reason) for d in grounded.drops] == [("plan_post_op", "quote not in document")]


def test_antiplatelet_mention_is_dropped() -> None:
    doc = Document(doc_id="d", type="Note", date="2026-03-01", text="ASA 81 mg PO daily")
    mention = AnticoagMention(doc_id="d", quote="ASA 81 mg PO daily", drug="ASA", status="current")
    grounded = ground(empty_extraction(anticoagulant_mentions=[mention]), [doc])

    assert grounded.anticoag == []
    assert grounded.drops[0].reason == "not an anticoagulant: ASA"
    assert grounded.unverified == []  # classification miss, not a grounding failure


# -------------------------
# Model call
# -------------------------


@pytest.fixture
def parse_client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(
        output_parsed=empty_extraction(),
        usage=SimpleNamespace(input_tokens=1200, output_tokens=150),
    )
    monkeypatch.setattr(extract, "_client", client)
    return client


def test_call_uses_strict_parse_with_temperature_zero(parse_client: Mock) -> None:
    submission = PatientSubmission(
        procedure={"procedure_type": "Elective thyroidectomy"}, documents=[CONSENT]
    )
    extraction, meta = call_extraction_model(submission, model="test-model")

    assert extraction == empty_extraction()
    assert (meta["input_tokens"], meta["output_tokens"], meta["attempts"]) == (1200, 150, 1)
    kwargs = parse_client.responses.parse.call_args.kwargs
    assert kwargs["model"] == "test-model"
    assert kwargs["temperature"] == 0
    assert kwargs["text_format"] is CaseExtraction
    assert kwargs["instructions"] == EXTRACTION_INSTRUCTIONS
    assert "Scheduled procedure: Elective thyroidectomy" in kwargs["input"]
    assert '"doc_id": "consent-1"' in kwargs["input"]


def test_call_retries_once_when_output_is_not_parsed(parse_client: Mock) -> None:
    good = parse_client.responses.parse.return_value
    parse_client.responses.parse.side_effect = [SimpleNamespace(output_parsed=None), good]

    _, meta = call_extraction_model(PatientSubmission(), model="m")

    assert meta["attempts"] == 2
    assert parse_client.responses.parse.call_count == 2


def test_call_raises_after_parse_attempts_exhausted(parse_client: Mock) -> None:
    parse_client.responses.parse.return_value = SimpleNamespace(output_parsed=None)

    with pytest.raises(RuntimeError, match="Extraction failed"):
        call_extraction_model(PatientSubmission(), model="m")


def test_client_is_created_once_and_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[dict[str, object]] = []

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            created.append(kwargs)

    import sys

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setattr(extract, "_client", None)

    assert extract.get_client() is extract.get_client()
    assert created == [{"timeout": extract.REQUEST_TIMEOUT_SECONDS,
                        "max_retries": extract.TRANSPORT_RETRIES}]
