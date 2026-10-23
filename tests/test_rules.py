"""Pure tests of rules.evaluate on hand-built facts. No API calls.

Each scenario mirrors a labeled case in data/patients_sample_50.jsonl where noted.
"""

from __future__ import annotations

import datetime as dt

import pytest

from rules import (
    AnticoagEvidence,
    AnticoagPlan,
    CaseFacts,
    ConsentFact,
    HPFact,
    LabRecord,
    Source,
    TriageResult,
    VitalReading,
    evaluate,
)

PROCEDURE_DATE = dt.date(2026, 3, 1)


def d(days_before: int) -> dt.date:
    return PROCEDURE_DATE - dt.timedelta(days=days_before)


def doc(doc_id: str, days_before: int, doc_type: str = "Note") -> Source:
    return Source(kind="document", id=doc_id, type=doc_type, date=d(days_before).isoformat())


def hp(days_before: int, *, preop: bool = True, doc_id: str = "hp-1") -> HPFact:
    return HPFact(source=doc(doc_id, days_before, "H&P"), date=d(days_before), is_preop_hp=preop)


def consent(
    days_before: int = 5,
    *,
    signed: bool | None = True,
    matches: bool | None = True,
    doc_id: str = "consent-1",
) -> ConsentFact:
    return ConsentFact(
        source=doc(doc_id, days_before, "Surgical Consent"),
        date=d(days_before),
        signed=signed,
        matches_scheduled_procedure=matches,
    )


def lab(
    test: str, days_before: int, *, status: str = "final", lab_id: str | None = None
) -> LabRecord:
    lab_id = lab_id or f"{test.lower()}-{days_before}"
    return LabRecord(
        source=Source(kind="lab", id=lab_id, type=test, date=d(days_before).isoformat()),
        test=test,
        effective_at=d(days_before),
        status=status,
    )


def bp(
    sbp: float,
    dbp: float,
    days_before: int = 3,
    *,
    time: dt.time | None = dt.time(9, 0),
    from_note: bool = False,
    doc_id: str = "vit-bp",
    order: int = 0,
) -> VitalReading:
    return VitalReading(
        kind="bp",
        source=Source(
            kind="document" if from_note else "vital",
            id=doc_id,
            type="blood_pressure",
            date=d(days_before).isoformat(),
        ),
        date=d(days_before),
        time=time,
        from_note=from_note,
        note_order=order,
        systolic=sbp,
        diastolic=dbp,
    )


def temp(
    temp_f: float,
    days_before: int = 3,
    *,
    time: dt.time | None = dt.time(9, 0),
    from_note: bool = False,
    doc_id: str = "vit-temp",
) -> VitalReading:
    return VitalReading(
        kind="temp",
        source=Source(
            kind="document" if from_note else "vital",
            id=doc_id,
            type="temperature",
            date=d(days_before).isoformat(),
        ),
        date=d(days_before),
        time=time,
        from_note=from_note,
        temp_f=temp_f,
    )


def ac(
    drug: str, status: str, *, from_med_list: bool = True, source_id: str | None = None
) -> AnticoagEvidence:
    return AnticoagEvidence(
        drug=drug,
        status=status,
        from_med_list=from_med_list,
        source=Source(
            kind="medication" if from_med_list else "document",
            id=source_id or f"{drug}-{'med' if from_med_list else 'note'}",
        ),
    )


def plan(
    drug: str,
    *,
    pre: bool = True,
    post: bool = True,
    deferred: bool = False,
    doc_id: str = "plan-1",
) -> AnticoagPlan:
    return AnticoagPlan(
        drug=drug,
        has_pre_op_instruction=pre,
        has_post_op_instruction=post,
        deferred_to_other_team=deferred,
        source=doc(doc_id, 7, "Perioperative Medication Plan"),
    )


def ready_facts(**overrides: object) -> CaseFacts:
    """A LOW-risk case that passes every rule."""

    base: dict[str, object] = {
        "procedure_source": Source(
            kind="procedure", id="case-x", type="procedure", date=PROCEDURE_DATE.isoformat()
        ),
        "procedure_date": PROCEDURE_DATE,
        "procedure_risk": "LOW",
        "hps": [hp(10)],
        "consents": [consent()],
        "labs": [lab("CBC", 10)],
        "vitals": [bp(122, 76), temp(98.6)],
        "documents_searched": 4,
    }
    base.update(overrides)
    return CaseFacts.model_validate(base)


def categories(result: TriageResult) -> list[str]:
    return sorted({i.category for i in result.issues})


def check(result: TriageResult, prefix: str):
    matches = [c for c in result.checks if c.rule.startswith(prefix)]
    assert len(matches) == 1, [c.rule for c in result.checks]
    return matches[0]


# -------------------------
# Baseline + precedence
# -------------------------


def test_ready_case_passes_every_check() -> None:
    result = evaluate(ready_facts())

    assert result.decision == "READY"
    assert result.issues == []
    assert {c.status for c in result.checks} == {"PASS"}
    assert result.explanation == "READY: all policy checks passed."


def test_fever_in_celsius_and_unsigned_consent_reports_both() -> None:
    # case_00002: 38.6 C (101.5 F) + portal e-signature NOT COMPLETED
    result = evaluate(
        ready_facts(
            consents=[consent(signed=False)],
            vitals=[bp(122, 76), temp(101.5, 0, time=None, from_note=True, doc_id="intake")],
        )
    )

    assert result.decision == "NOT_CLEARED"
    assert categories(result) == ["ACUTE_SAFETY_EXCLUSION", "REQUIRED_DOCUMENTATION"]


def test_issue_without_acute_exclusion_needs_follow_up() -> None:
    result = evaluate(ready_facts(labs=[]))

    assert result.decision == "NEEDS_FOLLOW_UP"
    assert categories(result) == ["REQUIRED_TESTING"]


def test_every_issue_points_to_a_record() -> None:
    result = evaluate(
        ready_facts(
            procedure_risk=None,
            hps=[],
            consents=[consent(signed=None)],
            vitals=[bp(185, 90)],
            anticoag=[ac("apixaban", "current")],
        )
    )

    assert result.issues
    for issue in result.issues:
        assert issue.sources, issue.rule
        assert all(s.id for s in issue.sources), issue.rule
        assert issue.evidence


# -------------------------
# Missing procedure fields
# -------------------------


def test_null_risk_only_missing_required_data() -> None:
    # case_00029: risk blank, everything else complete; CBC is not flagged
    result = evaluate(ready_facts(procedure_risk=None, labs=[lab("CBC", 5), lab("CMP", 2)]))

    assert result.decision == "NEEDS_FOLLOW_UP"
    assert categories(result) == ["MISSING_REQUIRED_DATA"]
    assert check(result, "2 ").status == "SKIPPED"


def test_null_date_skips_windows_but_still_checks_consent_and_vitals() -> None:
    # case_00044 (date blank) / case_00011 (date blank + no temperature)
    facts = ready_facts(procedure_date=None, hps=[hp(200)], labs=[lab("CBC", 300)])
    result = evaluate(facts)

    assert categories(result) == ["MISSING_REQUIRED_DATA"]
    assert check(result, "1.1").status == "SKIPPED"
    assert check(result, "2 ").status == "SKIPPED"

    no_temp = evaluate(facts.model_copy(update={"vitals": [bp(117, 71)]}))
    assert categories(no_temp) == ["MISSING_REQUIRED_DATA"]
    assert check(no_temp, "4.2").status == "UNKNOWN"

    unsigned = evaluate(facts.model_copy(update={"consents": [consent(signed=False)]}))
    assert categories(unsigned) == ["MISSING_REQUIRED_DATA", "REQUIRED_DOCUMENTATION"]


# -------------------------
# Rule 1: documentation
# -------------------------


@pytest.mark.parametrize(
    ("days_before", "status"), [(0, "PASS"), (30, "PASS"), (31, "FAIL"), (-1, "FAIL")]
)
def test_hp_window_boundaries(days_before: int, status: str) -> None:
    result = evaluate(ready_facts(hps=[hp(days_before)]))

    assert check(result, "1.1").status == status


def test_old_hp_and_wellness_visit_fail() -> None:
    # case_00021: pre-op H&P 44 days out; wellness visit doesn't count at any age
    result = evaluate(ready_facts(hps=[hp(44), hp(5, preop=False, doc_id="wellness")]))

    hp_check = check(result, "1.1")
    assert hp_check.status == "FAIL"
    assert "44 days" in hp_check.evidence
    assert [s.id for s in hp_check.sources] == ["hp-1"]


def test_no_hp_is_documentation_not_missing_data() -> None:
    # case_00013: no H&P anywhere
    result = evaluate(ready_facts(hps=[]))

    assert categories(result) == ["REQUIRED_DOCUMENTATION"]


def test_later_signed_consent_supersedes_pending() -> None:
    result = evaluate(
        ready_facts(
            consents=[
                consent(10, signed=None, doc_id="pending"),
                consent(3, signed=True, doc_id="signed"),
            ]
        )
    )

    assert check(result, "1.2").status == "PASS"
    assert check(result, "1.2").sources[0].id == "signed"


def test_signed_consent_for_other_procedure_fails() -> None:
    result = evaluate(ready_facts(consents=[consent(matches=False)]))

    assert check(result, "1.2").status == "FAIL"
    assert categories(result) == ["REQUIRED_DOCUMENTATION"]


# -------------------------
# Rule 2: testing
# -------------------------


@pytest.mark.parametrize(
    ("risk", "days_before", "status"),
    [
        ("LOW", 0, "PASS"),
        ("LOW", 30, "PASS"),
        ("LOW", 31, "FAIL"),
        ("MODERATE", 30, "PASS"),
        ("HIGH", 14, "PASS"),
        ("HIGH", 15, "FAIL"),
    ],
)
def test_cbc_window_boundaries(risk: str, days_before: int, status: str) -> None:
    result = evaluate(
        ready_facts(procedure_risk=risk, labs=[lab("CBC", days_before), lab("CMP", 3)])
    )

    assert check(result, "2 CBC").status == status


def test_high_risk_cbc_15_days_and_no_temperature() -> None:
    # case_00009
    result = evaluate(
        ready_facts(
            procedure_risk="HIGH",
            labs=[lab("CBC", 15), lab("CMP", 6)],
            vitals=[bp(130, 80)],
        )
    )

    assert categories(result) == ["MISSING_REQUIRED_DATA", "REQUIRED_TESTING"]
    assert "15 days" in check(result, "2 CBC").evidence
    assert check(result, "2 CMP").status == "PASS"


def test_bmp_does_not_count_as_cmp() -> None:
    result = evaluate(ready_facts(procedure_risk="HIGH", labs=[lab("CBC", 3), lab("BMP", 3)]))

    assert check(result, "2 CMP").status == "FAIL"
    assert check(result, "2 CMP").evidence == "No CMP found among 2 lab results."


def test_entered_in_error_falls_back_to_previous_result() -> None:
    # case_00004: newest CBC entered-in-error, prior CBC 24 days out
    result = evaluate(
        ready_facts(
            procedure_risk="MODERATE",
            labs=[lab("CBC", 23, status="entered-in-error"), lab("CBC", 24, lab_id="good")],
        )
    )

    cbc = check(result, "2 CBC")
    assert cbc.status == "PASS"
    assert cbc.sources[0].id == "good"
    assert "entered-in-error" in cbc.evidence


def test_most_recent_lab_dated_after_procedure_fails() -> None:
    result = evaluate(ready_facts(labs=[lab("CBC", 10), lab("CBC", -1)]))

    cbc = check(result, "2 CBC")
    assert cbc.status == "FAIL"
    assert "after the procedure" in cbc.evidence


def test_only_most_recent_result_counts() -> None:
    result = evaluate(ready_facts(labs=[lab("CBC", 31), lab("CBC", 40)]))

    assert check(result, "2 CBC").status == "FAIL"


# -------------------------
# Rule 3: anticoagulation
# -------------------------


def test_active_anticoag_with_complete_plan_passes() -> None:
    # case_00003: Eliquis on med list, hematology plan covers hold + resume
    result = evaluate(ready_facts(anticoag=[ac("apixaban", "current")], anticoag_plans=[plan("apixaban")]))

    assert result.decision == "READY"


def test_note_reported_anticoag_counts_as_active() -> None:
    # case_00027: not on med list, nursing note says patient takes Eliquis
    result = evaluate(ready_facts(anticoag=[ac("apixaban", "current", from_med_list=False)]))

    assert categories(result) == ["ANTICOAGULATION_MANAGEMENT"]
    assert "no plan on file" in result.issues[0].evidence


def test_plan_for_wrong_drug_fails() -> None:
    # case_00018: switched warfarin -> apixaban; plan is for warfarin
    result = evaluate(
        ready_facts(
            anticoag=[ac("apixaban", "current"), ac("warfarin", "discontinued")],
            anticoag_plans=[plan("warfarin")],
        )
    )

    assert categories(result) == ["ANTICOAGULATION_MANAGEMENT"]
    assert check(result, "3 Anticoagulation plan: warfarin").status == "PASS"
    apixaban = check(result, "3 Anticoagulation plan: apixaban")
    assert apixaban.status == "FAIL"
    assert "only for warfarin" in apixaban.evidence
    assert "plan-1" in [s.id for s in apixaban.sources]


@pytest.mark.parametrize(
    ("pre", "post", "deferred", "gap"),
    [
        (True, False, False, "no post-op instruction"),  # case_00020
        (False, True, False, "no pre-op instruction"),
        (False, False, True, "deferred to another team"),  # case_00023 / case_00036
    ],
)
def test_incomplete_plan_fails(pre: bool, post: bool, deferred: bool, gap: str) -> None:
    result = evaluate(
        ready_facts(
            anticoag=[ac("warfarin", "current")],
            anticoag_plans=[plan("warfarin", pre=pre, post=post, deferred=deferred)],
        )
    )

    assert categories(result) == ["ANTICOAGULATION_MANAGEMENT"]
    assert gap in result.issues[0].evidence


def test_inactive_anticoag_confirmed_by_note_is_ready() -> None:
    # case_00019 / case_00043: warfarin inactive on med list, H&P says stopped in 2020
    result = evaluate(
        ready_facts(
            anticoag=[
                ac("warfarin", "discontinued"),
                ac("warfarin", "discontinued", from_med_list=False),
            ]
        )
    )

    assert result.decision == "READY"


def test_unknown_med_list_status_is_missing_data() -> None:
    # case_00033: Coumadin active=null, patient couldn't confirm at pre-admission visit
    result = evaluate(
        ready_facts(
            procedure_risk="MODERATE",
            anticoag=[ac("warfarin", "unknown"), ac("warfarin", "unknown", from_med_list=False)],
        )
    )

    assert categories(result) == ["MISSING_REQUIRED_DATA"]


def test_current_statement_overrides_unclear_one() -> None:
    result = evaluate(
        ready_facts(
            anticoag=[ac("warfarin", "unknown"), ac("warfarin", "current", from_med_list=False)]
        )
    )

    assert categories(result) == ["ANTICOAGULATION_MANAGEMENT"]


@pytest.mark.parametrize(
    "evidence",
    [
        # Fail closed: an unclear statement is not cancelled by a "stopped" one.
        [ac("warfarin", "unknown"), ac("warfarin", "discontinued", from_med_list=False)],
        # Note-only unclear mention, nothing on the med list.
        [ac("apixaban", "unknown", from_med_list=False)],
    ],
)
def test_unclear_status_without_current_statement_is_missing_data(
    evidence: list[AnticoagEvidence],
) -> None:
    result = evaluate(ready_facts(anticoag=evidence))

    assert categories(result) == ["MISSING_REQUIRED_DATA"]


# -------------------------
# Rule 4: acute safety
# -------------------------


@pytest.mark.parametrize(
    ("sbp", "dbp", "decision"),
    [
        (179, 109, "READY"),
        (180, 80, "NOT_CLEARED"),
        (148, 110, "NOT_CLEARED"),  # case_00028: diastolic exactly 110
    ],
)
def test_bp_thresholds(sbp: float, dbp: float, decision: str) -> None:
    result = evaluate(ready_facts(vitals=[bp(sbp, dbp), temp(98.6)]))

    assert result.decision == decision


@pytest.mark.parametrize(("value", "decision"), [(100.4, "READY"), (100.5, "NOT_CLEARED")])
def test_temperature_threshold(value: float, decision: str) -> None:
    result = evaluate(ready_facts(vitals=[bp(120, 80), temp(value)]))

    assert result.decision == decision


def test_newer_note_vitals_override_structured() -> None:
    # case_00005: structured 119/78 on 03/01, nursing note 03/03 has 157/117
    result = evaluate(
        ready_facts(
            vitals=[
                bp(119, 78, 5),
                bp(157, 117, 3, time=None, from_note=True, doc_id="nursing"),
                temp(98.6),
            ]
        )
    )

    assert result.decision == "NOT_CLEARED"
    assert check(result, "4.1").sources[0].id == "nursing"


def test_recheck_later_in_same_note_wins() -> None:
    # case_00046: 192/106 on arrival, recheck 24 minutes later 183/104
    # case_00032 variant: high on arrival, recheck below the line -> recheck wins
    arrival = bp(196, 110, 1, time=None, from_note=True, doc_id="pat", order=0)
    recheck = bp(170, 98, 1, time=None, from_note=True, doc_id="pat", order=1)
    result = evaluate(ready_facts(vitals=[recheck, arrival, temp(98.6)]))

    assert result.decision == "READY"
    assert "170/98" in check(result, "4.1").evidence


def test_timed_structured_recheck_beats_earlier_timed_note_reading() -> None:
    note = bp(185, 100, 2, time=dt.time(9, 0), from_note=True, doc_id="clinic")
    structured = bp(130, 80, 2, time=dt.time(15, 0))
    result = evaluate(ready_facts(vitals=[note, structured, temp(98.6)]))

    assert result.decision == "READY"


def test_untimed_note_reading_beats_same_day_structured() -> None:
    note = bp(185, 100, 2, time=None, from_note=True, doc_id="clinic")
    structured = bp(130, 80, 2, time=dt.time(15, 0))
    result = evaluate(ready_facts(vitals=[structured, note, temp(98.6)]))

    assert result.decision == "NOT_CLEARED"


def test_latest_temperature_wins_regardless_of_list_order() -> None:
    # case_00031: vitals not in date order; latest temp 101.8 F
    result = evaluate(
        ready_facts(vitals=[bp(120, 80), temp(101.8, 5), temp(98.1, 20), temp(98.4, 10)])
    )

    assert result.decision == "NOT_CLEARED"


def test_no_vitals_at_all_is_missing_data() -> None:
    # case_00022: no BP and no temperature
    result = evaluate(ready_facts(vitals=[]))

    assert categories(result) == ["MISSING_REQUIRED_DATA"]
    assert check(result, "4.1").status == "UNKNOWN"
    assert check(result, "4.2").status == "UNKNOWN"


def test_absent_record_evidence_says_what_was_searched() -> None:
    result = evaluate(
        ready_facts(
            hps=[hp(5, preop=False, doc_id="wellness")],
            consents=[],
            labs=[lab("TSH", 5), lab("UA", 5), lab("CBC", 3, status="entered-in-error")],
            vitals=[bp(120, 80), bp(118, 76, 5)],
        )
    )

    assert check(result, "1.1").evidence == (
        "No dated pre-op H&P found among 4 documents (1 H&P document(s) not pre-op or undated)."
    )
    assert check(result, "1.2").evidence == "No surgical consent found among 4 documents."
    assert check(result, "2 CBC").evidence == (
        "No CBC found among 3 lab results (1 entered-in-error result ignored)."
    )
    assert check(result, "4.2").evidence == (
        "No temperature found among 2 structured vitals and 4 documents."
    )
    for prefix in ("1.2", "2 CBC", "4.2"):
        assert [s.kind for s in check(result, prefix).sources] == ["procedure"]
