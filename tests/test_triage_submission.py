from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from core import (
    BASELINE_SYSTEM_PROMPT,
    PatientSubmission,
    TriageOutput,
    baseline_triage_submission,
    triage_output_json_schema,
    triage_submission,
)


@pytest.fixture
def openai_client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps(
            {
                "decision": "READY",
                "issues": [],
                "explanation": "All required criteria are satisfied.",
            }
        ),
    )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda: client))
    return client


@pytest.fixture
def submission_payload() -> dict[str, object]:
    return {
        "patient": {"id": "patient-1"},
        "procedure": {
            "case_id": "case-1",
            "procedure_risk": "LOW",
            "procedure_date": "2026-02-01",
        },
        "vitals": [
            {
                "type": "blood_pressure",
                "systolic": 120,
                "diastolic": 80,
                "date": "2026-01-25",
            }
        ],
        "labs": [
            {
                "code": "CBC",
                "display": "Complete blood count",
                "effective_at": "2026-01-20",
                "status": "final",
            }
        ],
        "medications": [],
        "conditions": [],
        "documents": [
            {
                "type": "history_and_physical",
                "date": "2026-01-20",
                "text": "History and physical completed.",
            },
            {
                "type": "surgical_consent",
                "date": "2026-01-22",
                "text": "Signed surgical consent.",
            },
        ],
    }


def test_triage_submission_returns_structured_output(
    openai_client: Mock,
    submission_payload: dict[str, object],
) -> None:
    output = baseline_triage_submission(submission_payload, model="test-model")

    assert isinstance(output, TriageOutput)
    assert output.decision == "READY"
    assert output.issues == []
    assert output.explanation == "All required criteria are satisfied."

    assert openai_client.responses.create.call_count == 1
    call = openai_client.responses.create.call_args.kwargs
    assert call["model"] == "test-model"
    assert call["instructions"] == BASELINE_SYSTEM_PROMPT

    text_config = call["text"]
    assert isinstance(text_config, dict)
    response_format = text_config["format"]
    assert isinstance(response_format, dict)
    assert response_format["type"] == "json_schema"
    assert response_format["name"] == "preop_triage_output"
    assert response_format["schema"] == triage_output_json_schema()
    assert response_format["strict"] is False


def test_triage_submission_builds_prompt_from_validated_submission(
    openai_client: Mock,
    submission_payload: dict[str, object],
) -> None:
    submission = PatientSubmission.model_validate(submission_payload)

    baseline_triage_submission(submission, model="test-model")

    assert openai_client.responses.create.call_count == 1
    call = openai_client.responses.create.call_args.kwargs
    message = call["input"][0]
    content = message["content"][0]
    prompt = content["text"]

    assert message["type"] == "message"
    assert message["role"] == "user"
    assert content["type"] == "input_text"
    assert json.loads(prompt.split("Submission JSON:\n", 1)[1]) == submission.model_dump()


def test_triage_submission_rejects_invalid_model_json(
    openai_client: Mock,
    submission_payload: dict[str, object],
) -> None:
    openai_client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps(
            {
                "decision": "MAYBE",
                "issues": [],
                "explanation": "Not a valid triage decision.",
            }
        )
    )

    with pytest.raises(ValidationError):
        baseline_triage_submission(submission_payload, model="test-model")

    assert openai_client.responses.create.call_count == 1


# -------------------------
# Grounded-extraction pipeline (mocked model)
# -------------------------


def _pipeline_payload() -> dict[str, object]:
    return {
        "procedure": {
            "case_id": "case-e2e",
            "procedure_type": "Carpal tunnel release",
            "procedure_risk": "LOW",
            "procedure_date": "2026-03-03",
        },
        "vitals": [
            {"type": "blood_pressure", "systolic": 124, "diastolic": 78,
             "date": "2026-02-20T09:00:00Z", "source": "Clinic"},
            {"type": "temperature", "value_f": 98.2, "date": "2026-02-20T09:00:00Z"},
        ],
        "labs": [{"id": "lab-cbc", "code": "CBC", "effective_at": "2026-02-25T08:00:00Z",
                  "status": "final"}],
        "documents": [
            {"doc_id": "hp", "type": "Pre-op H&P", "date": "2026-02-14",
             "text": "Planned procedure: carpal tunnel release (scheduled for 03/03/2026)"},
            {"doc_id": "consent", "type": "Surgical Consent", "date": "2026-02-26",
             "text": "Procedure: carpal tunnel release\nPatient signature: /s/ Ava Patel"},
            {"doc_id": "nursing", "type": "Nursing Assessment", "date": "2026-02-28",
             "text": "09:05  BP: 161/117, 37.2 °C tympanic, HR 84"},
        ],
    }


@pytest.fixture
def extraction_client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    import extract
    from extract import CaseExtraction

    extraction = CaseExtraction.model_validate(
        {
            "hp_candidates": [{"doc_id": "hp", "quote": "Planned procedure: carpal tunnel release",
                               "is_preop_hp": True}],
            "consents": [{"doc_id": "consent", "quote": "Patient signature: /s/ Ava Patel",
                          "signed": "true", "procedure_named": "carpal tunnel release",
                          "matches_scheduled_procedure": "yes"}],
            "vitals": [
                {"doc_id": "nursing", "quote": "BP: 161/117", "kind": "bp", "systolic": 161,
                 "diastolic": 117, "temp_value": None, "temp_unit": None, "time": "09:05",
                 "order_in_note": 1},
                {"doc_id": "nursing", "quote": "37.2 \x176 tympanic", "kind": "temp",
                 "systolic": None, "diastolic": None, "temp_value": 37.2, "temp_unit": "C",
                 "time": "09:05", "order_in_note": 1},
            ],
            "anticoagulant_mentions": [],
            "anticoagulant_plans": [],
        }
    )
    client = Mock()
    client.responses.parse.return_value = SimpleNamespace(
        output_parsed=extraction, usage=SimpleNamespace(input_tokens=10, output_tokens=5)
    )
    monkeypatch.setattr(extract, "_client", client)
    return client


def test_pipeline_decides_in_code_from_grounded_facts(extraction_client: Mock) -> None:
    output = triage_submission(_pipeline_payload(), model="test-model")

    assert extraction_client.responses.parse.call_count == 1
    assert output.decision == "NOT_CLEARED"
    assert [i.category for i in output.issues] == ["ACUTE_SAFETY_EXCLUSION"]
    [issue] = output.issues
    assert issue.rule == "4.1 Blood pressure"
    assert [(s.kind, s.id, s.date) for s in issue.sources] == [
        ("document", "nursing", "2026-02-28")
    ]
    assert {c.status for c in output.checks} >= {"PASS", "FAIL"}


def test_pipeline_output_is_scorable_and_quote_free(extraction_client: Mock) -> None:
    from run_evals import ScoredOutput

    output = triage_submission(_pipeline_payload(), model="test-model")
    dumped = output.model_dump()

    ScoredOutput.model_validate(dumped)
    assert "Patient signature" not in json.dumps(dumped)  # quotes stay in the sidecar


def test_trace_carries_quotes_and_usage(extraction_client: Mock) -> None:
    from core import triage_with_trace

    _, trace = triage_with_trace(_pipeline_payload(), model="test-model", case_id="case-e2e")

    assert trace["case_id"] == "case-e2e"
    assert (trace["input_tokens"], trace["output_tokens"]) == (10, 5)
    assert trace["extraction"]["consents"][0]["quote"] == "Patient signature: /s/ Ava Patel"
    assert trace["drop_count"] == 0
