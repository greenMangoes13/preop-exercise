"""One LLM call per case: documents -> typed facts, each with {doc_id, quote}.

The model only reads. Every fact must quote its document verbatim; the grounding
check drops facts whose quote isn't in that document's text, and the rules treat
the missing fact as absent (fail closed). Dates come from document metadata in
code, never from the model.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import re
import time
from typing import Any, Literal

from pydantic import BaseModel

from core import Document, PatientSubmission
from normalize import anticoagulant_generic, celsius_to_fahrenheit, parse_date
from rules import (
    AnticoagEvidence,
    AnticoagPlan,
    CaseFacts,
    ConsentFact,
    HPFact,
    Source,
    UnverifiedFact,
    VitalReading,
)

DEFAULT_MODEL = "gpt-4.1-mini"
REQUEST_TIMEOUT_SECONDS = 60.0
TRANSPORT_RETRIES = 3  # SDK retries connection errors, 429s and 5xx with backoff
PARSE_ATTEMPTS = 2  # extra attempt if the response has no parsed output

# -------------------------
# Extraction schema (what the model returns)
# -------------------------

TriState = Literal["true", "false", "unclear"]
Match = Literal["yes", "no", "unclear"]
AnticoagStatus = Literal["current", "discontinued", "unknown"]


class HPCandidate(BaseModel):
    doc_id: str
    quote: str
    is_preop_hp: bool


class ConsentExtract(BaseModel):
    doc_id: str
    quote: str
    signed: TriState
    procedure_named: str | None
    matches_scheduled_procedure: Match


class NoteVital(BaseModel):
    doc_id: str
    quote: str
    kind: Literal["bp", "temp"]
    systolic: int | None
    diastolic: int | None
    temp_value: float | None
    temp_unit: Literal["F", "C"] | None
    time: str | None  # "HH:MM", 24-hour
    order_in_note: int


class AnticoagMention(BaseModel):
    doc_id: str
    quote: str
    drug: str
    status: AnticoagStatus


class AnticoagPlanExtract(BaseModel):
    doc_id: str
    drug: str
    pre_op_quote: str | None
    post_op_quote: str | None
    deferral_quote: str | None


class CaseExtraction(BaseModel):
    hp_candidates: list[HPCandidate]
    consents: list[ConsentExtract]
    vitals: list[NoteVital]
    anticoagulant_mentions: list[AnticoagMention]
    anticoagulant_plans: list[AnticoagPlanExtract]


# -------------------------
# Prompt
# -------------------------

EXTRACTION_INSTRUCTIONS = """
You extract facts from pre-operative clinical documents. You do not decide whether the patient is ready for surgery; separate code applies the policy to what you extract.

Rules for every fact:
- doc_id: the id of the document the fact comes from, copied exactly.
- quote: a short excerpt (one line or sentence, at most 200 characters) copied character-for-character from that document's text. Do not paraphrase, abbreviate, fix typos, or join text from different places. Facts whose quote is not found in that document are discarded.
- Extract only what the documents state. Never fill gaps from general knowledge. When a document is ambiguous, use the "unclear"/"unknown"/null option.

What to extract:

1. hp_candidates: every document that is itself a history and physical (H&P), whatever its title or spelling ("H&P", "History and Physical", "Admission H&P", "Clinic H&P", "Pre-op H and P", ...).
   - is_preop_hp: true if it is an evaluation for a planned surgery; false if it is an annual wellness, establish-care, or other visit not done to evaluate for surgery.
   - quote: the line showing the encounter's purpose (e.g. "Encounter: annual wellness / establish care", or "Planned procedure: ...").
   Do not list documents that only refer to an H&P (e.g. "H&P reviewed", "H&P not on file").

2. consents: every surgical consent document.
   - signed: "true" only if the text shows the patient signed (patient signature, patient e-signature, "Signed by patient", "Patient signature on file - signed"). "false" if it shows the patient has not signed (blank signature line, "pending", "NOT COMPLETED", "not yet signed", printed for review). "unclear" otherwise. A physician attestation or physician signature is not a patient signature.
   - quote: the patient-signature line or signature-status line.
   - procedure_named: the procedure as written in the consent, or null.
   - matches_scheduled_procedure: "yes" if procedure_named is the same operation as the scheduled procedure given below (wording may differ), "no" if it is a different operation, "unclear" if no procedure is named.

3. vitals: every blood pressure and temperature reading recorded in a document's text. Skip placeholders such as "See flowsheet", "deferred", "not recorded".
   - kind: "bp" or "temp". One line with both a BP and a temperature yields two facts.
   - bp: systolic and diastolic as written; temp_value and temp_unit null.
   - temp: temp_value and temp_unit ("F" or "C") as written, without converting; systolic and diastolic null.
   - time: the reading's stated time as 24-hour "HH:MM" (e.g. "09:05", "rechecked at 10:36", "ADDENDUM 08:53"), else null.
   - order_in_note: 1 for the first reading of that kind in the document, 2 for the next (e.g. a recheck), and so on.
   - quote: text containing the reading's numbers.

4. anticoagulant_mentions: every statement about whether the patient takes an anticoagulant (warfarin/Coumadin, apixaban/Eliquis, rivaroxaban/Xarelto, dabigatran/Pradaxa, edoxaban/Savaysa, enoxaparin/Lovenox, heparin, or another anticoagulant). Exclude antiplatelets (aspirin/ASA, clopidogrel). Exclude a drug that appears only inside a perioperative instruction (e.g. "Bridge with enoxaparin").
   - drug: the name as written.
   - status: "current" if the patient is taking or maintained on it (including a medication-list line in the note); "discontinued" if it was stopped or the patient is "not currently anticoagulated"; "unknown" if the text says it is unclear or unconfirmed whether the patient takes it.
   - A planned or future hold/stop before surgery means the patient is current; record that instruction under anticoagulant_plans, not as discontinued. "discontinued" only means the patient has already stopped taking it.

5. anticoagulant_plans: each documented perioperative management plan for an anticoagulant.
   - drug: the drug the plan is written for, as written.
   - pre_op_quote: the instruction for holding/stopping before surgery, or null.
   - post_op_quote: the instruction for resuming/restarting after surgery, or null. A promise to address it later (e.g. "will be addressed after surgery") is not an instruction: use null.
   - deferral_quote: text that leaves the management to another team or to a later decision (e.g. "to be determined by the surgical team", "recommend the surgical team discuss"), or null. A concrete instruction that merely mentions a team (e.g. "Resume ... once hemostasis is secure, per surgical team") is not a deferral.

Return empty lists when nothing applies.
""".strip()


def build_extraction_input(submission: PatientSubmission) -> str:
    procedure = submission.procedure
    scheduled = (procedure.procedure_type if procedure else None) or "not stated"
    parts = [f"Scheduled procedure: {scheduled}", "", "Documents:"]
    for doc in submission.documents:
        parts.append(
            json.dumps(
                {"doc_id": doc.doc_id, "type": doc.type, "date": doc.date, "text": doc.text},
                ensure_ascii=False,
            )
        )
    return "\n".join(parts)


# -------------------------
# Client
# -------------------------

_client: Any = None


def get_client() -> Any:
    """Lazily create one OpenAI client and reuse it across cases."""

    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(timeout=REQUEST_TIMEOUT_SECONDS, max_retries=TRANSPORT_RETRIES)
    return _client


def reset_client() -> None:
    global _client
    _client = None


def call_extraction_model(
    submission: PatientSubmission, *, model: str = DEFAULT_MODEL
) -> tuple[CaseExtraction, dict[str, Any]]:
    """Return the parsed extraction plus call metadata (latency, tokens)."""

    client = get_client()
    started = time.monotonic()
    last_error: str | None = None
    for attempt in range(1, PARSE_ATTEMPTS + 1):
        response = client.responses.parse(
            model=model,
            instructions=EXTRACTION_INSTRUCTIONS,
            input=build_extraction_input(submission),
            text_format=CaseExtraction,
            temperature=0,
        )
        parsed = response.output_parsed
        if isinstance(parsed, CaseExtraction):
            usage = getattr(response, "usage", None)
            return parsed, {
                "model": model,
                "attempts": attempt,
                "latency_s": round(time.monotonic() - started, 3),
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
            }
        last_error = f"no parsed output (attempt {attempt})"
    raise RuntimeError(f"Extraction failed: {last_error}")


# -------------------------
# Grounding
# -------------------------

_QUOTE_CHARS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_QUOTE_CHARS)).strip().lower()


class Drop(BaseModel):
    kind: str
    doc_id: str
    quote: str | None
    reason: str


class Grounder:
    """Checks quotes against their documents and records every drop."""

    def __init__(self, documents: list[Document]) -> None:
        self.docs = {d.doc_id: d for d in documents if d.doc_id}
        self._text = {doc_id: normalize_text(d.text or "") for doc_id, d in self.docs.items()}
        self.drops: list[Drop] = []

    def check(self, kind: str, doc_id: str, quote: str | None) -> bool:
        if doc_id not in self.docs:
            reason = "unknown doc_id"
        elif not quote or not normalize_text(quote):
            reason = "empty quote"
        elif normalize_text(quote) not in self._text[doc_id]:
            reason = "quote not in document"
        else:
            return True
        self.drops.append(Drop(kind=kind, doc_id=doc_id, quote=quote, reason=reason))
        return False

    def drop(self, kind: str, doc_id: str, quote: str | None, reason: str) -> None:
        self.drops.append(Drop(kind=kind, doc_id=doc_id, quote=quote, reason=reason))

    def grounds(self, doc_id: str, quote: str | None) -> bool:
        """Quote check without recording a drop."""

        return bool(
            doc_id in self.docs
            and quote
            and normalize_text(quote)
            and normalize_text(quote) in self._text[doc_id]
        )

    def source(self, doc_id: str, value: str) -> Source:
        doc = self.docs.get(doc_id)
        if doc is None:
            return Source(kind="document", id=doc_id, value=value)
        return Source(kind="document", id=doc_id, type=doc.type, date=doc.date, value=value)

    def find_vital_line(self, v: NoteVital) -> str | None:
        """The document line that contains the extracted vital values, found by code."""

        doc = self.docs.get(v.doc_id)
        if doc is None:
            return None
        for line in (doc.text or "").splitlines():
            if v.kind == "bp" and v.systolic is not None and v.diastolic is not None:
                if re.search(rf"(?<![\d.]){v.systolic}\s*/\s*{v.diastolic}(?!\.?\d)", line):
                    return line.strip()
            elif v.kind == "temp" and v.temp_value is not None and v.temp_unit is not None:
                for number, unit in _TEMP_PATTERN.findall(line):
                    if float(number) == float(v.temp_value) and unit.upper() == v.temp_unit:
                        return line.strip()
        return None


_TEMP_PATTERN = re.compile(r"(?<![\d.])(\d{2,3}(?:\.\d+)?)\s*°?\s*([FfCc])\b")


class VitalEvidence(BaseModel):
    """Sidecar record of how a note vital was verified."""

    doc_id: str
    kind: str
    model_quote: str
    model_quote_grounded: bool
    document_line: str


def _parse_hhmm(value: str | None) -> dt.time | None:
    if not value:
        return None
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", value.strip())
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    return dt.time(hour, minute) if hour < 24 and minute < 60 else None


# -------------------------
# Extraction -> rules types
# -------------------------


class GroundedFacts(BaseModel):
    hps: list[HPFact]
    consents: list[ConsentFact]
    vitals: list[VitalReading]
    anticoag: list[AnticoagEvidence]
    anticoag_plans: list[AnticoagPlan]
    unverified: list[UnverifiedFact]
    vital_evidence: list[VitalEvidence]
    drops: list[Drop]


def ground(extraction: CaseExtraction, documents: list[Document]) -> GroundedFacts:
    g = Grounder(documents)
    hps: list[HPFact] = []
    consents: list[ConsentFact] = []
    vitals: list[VitalReading] = []
    anticoag: list[AnticoagEvidence] = []
    plans: list[AnticoagPlan] = []
    unverified: list[UnverifiedFact] = []
    vital_evidence: list[VitalEvidence] = []

    def unverify(kind: Literal["bp", "temp", "anticoag"], doc_id: str, quote: str | None,
                 reason: str, value: str) -> None:
        g.drop("vital" if kind != "anticoag" else "anticoag_mention", doc_id, quote, reason)
        doc = g.docs.get(doc_id)
        unverified.append(
            UnverifiedFact(
                kind=kind,
                date=parse_date(doc.date) if doc else None,
                reason=reason,
                source=g.source(doc_id, f"unverified: {value}"),
            )
        )

    for hp in extraction.hp_candidates:
        if g.check("hp", hp.doc_id, hp.quote):
            label = "pre-op H&P" if hp.is_preop_hp else "H&P, not pre-op"
            hps.append(
                HPFact(
                    source=g.source(hp.doc_id, label),
                    date=parse_date(g.docs[hp.doc_id].date),
                    is_preop_hp=hp.is_preop_hp,
                )
            )

    for c in extraction.consents:
        if g.check("consent", c.doc_id, c.quote):
            consents.append(
                ConsentFact(
                    source=g.source(
                        c.doc_id,
                        f"consent for {c.procedure_named or 'unnamed procedure'}, signed={c.signed}",
                    ),
                    date=parse_date(g.docs[c.doc_id].date),
                    signed={"true": True, "false": False, "unclear": None}[c.signed],
                    matches_scheduled_procedure={"yes": True, "no": False, "unclear": None}[
                        c.matches_scheduled_procedure
                    ],
                )
            )

    # Vitals are verified by their values: code finds the document line containing the
    # extracted numbers and unit, so a garbled model quote can't lose a real reading.
    for v in extraction.vitals:
        reading_time = _parse_hhmm(v.time)
        at = f" at {reading_time.strftime('%H:%M')}" if reading_time else ""
        if v.kind == "bp":
            value = f"BP {v.systolic}/{v.diastolic}{at}"
        else:
            value = f"Temp {v.temp_value}{v.temp_unit or ''}{at}"

        if v.doc_id not in g.docs:
            unverify(v.kind, v.doc_id, v.quote, "unknown doc_id", value)
            continue
        line = g.find_vital_line(v)
        if line is None:
            unverify(v.kind, v.doc_id, v.quote, "values not found in document", value)
            continue
        day = parse_date(g.docs[v.doc_id].date)
        if day is None:
            unverify(v.kind, v.doc_id, v.quote, "undated document", value)
            continue
        vital_evidence.append(
            VitalEvidence(
                doc_id=v.doc_id,
                kind=v.kind,
                model_quote=v.quote,
                model_quote_grounded=g.grounds(v.doc_id, v.quote),
                document_line=line,
            )
        )

        if v.kind == "bp":
            reading = VitalReading(
                kind="bp",
                source=g.source(v.doc_id, value),
                date=day,
                time=reading_time,
                from_note=True,
                note_order=v.order_in_note,
                systolic=v.systolic,
                diastolic=v.diastolic,
            )
        else:
            assert v.temp_value is not None  # find_vital_line matched a value and unit
            if v.temp_unit == "C":
                temp_f = celsius_to_fahrenheit(v.temp_value)
                value = f"Temp {v.temp_value:g}°C ({temp_f:.1f}°F){at}"
            else:
                temp_f = float(v.temp_value)
                value = f"Temp {temp_f:.1f}°F{at}"
            reading = VitalReading(
                kind="temp",
                source=g.source(v.doc_id, value),
                date=day,
                time=reading_time,
                from_note=True,
                note_order=v.order_in_note,
                temp_f=temp_f,
            )
        vitals.append(reading)

    for m in extraction.anticoagulant_mentions:
        generic = anticoagulant_generic(m.drug)
        if generic is None:
            # A classification miss (e.g. aspirin), not a grounding failure: doesn't block READY.
            g.drop("anticoag_mention", m.doc_id, m.quote, f"not an anticoagulant: {m.drug}")
            continue
        if not g.grounds(m.doc_id, m.quote):
            reason = "unknown doc_id" if m.doc_id not in g.docs else "quote not in document"
            unverify("anticoag", m.doc_id, m.quote, reason, f"{generic} {m.status}")
            continue
        anticoag.append(
            AnticoagEvidence(
                drug=generic,
                status=m.status,
                from_med_list=False,
                source=g.source(m.doc_id, f"{generic} {m.status} (note)"),
            )
        )

    for p in extraction.anticoagulant_plans:
        generic = anticoagulant_generic(p.drug)
        if p.doc_id not in g.docs:
            g.drop("anticoag_plan", p.doc_id, None, "unknown doc_id")
            continue
        if generic is None:
            g.drop("anticoag_plan", p.doc_id, None, f"not an anticoagulant: {p.drug}")
            continue
        # Each plan component counts only if its own quote is grounded.
        has_pre = p.pre_op_quote is not None and g.check("plan_pre_op", p.doc_id, p.pre_op_quote)
        has_post = p.post_op_quote is not None and g.check(
            "plan_post_op", p.doc_id, p.post_op_quote
        )
        deferred = p.deferral_quote is not None and g.check(
            "plan_deferral", p.doc_id, p.deferral_quote
        )
        parts = [
            "pre-op" if has_pre else "no pre-op",
            "post-op" if has_post else "no post-op",
        ] + (["deferred"] if deferred else [])
        plans.append(
            AnticoagPlan(
                drug=generic,
                has_pre_op_instruction=has_pre,
                has_post_op_instruction=has_post,
                deferred_to_other_team=deferred,
                source=g.source(p.doc_id, f"{generic} plan: {', '.join(parts)}"),
            )
        )

    return GroundedFacts(
        hps=hps,
        consents=consents,
        vitals=vitals,
        anticoag=anticoag,
        anticoag_plans=plans,
        unverified=unverified,
        vital_evidence=vital_evidence,
        drops=g.drops,
    )


def sidecar_row(
    case_id: str, extraction: CaseExtraction, grounded: GroundedFacts, meta: dict[str, Any]
) -> dict[str, Any]:
    """Quotes and call metadata for data/extractions.jsonl (not part of the scored output)."""

    return {
        "case_id": case_id,
        **meta,
        "extraction": extraction.model_dump(),
        "vital_evidence": [e.model_dump() for e in grounded.vital_evidence],
        "drops": [d.model_dump() for d in grounded.drops],
        "drop_count": len(grounded.drops),
        "unverified_count": len(grounded.unverified),
    }


def merge_facts(structured: CaseFacts, grounded: GroundedFacts) -> CaseFacts:
    """Add grounded note facts to the structured facts from normalize.py."""

    return structured.model_copy(
        update={
            "hps": grounded.hps,
            "consents": grounded.consents,
            "vitals": [*structured.vitals, *grounded.vitals],
            "anticoag": [*structured.anticoag, *grounded.anticoag],
            "anticoag_plans": grounded.anticoag_plans,
            "unverified": grounded.unverified,
        }
    )


# -------------------------
# Sample run (inspection only; the pipeline entry point is core.triage_submission)
# -------------------------


def _main() -> None:
    from normalize import normalize_submission
    from rules import evaluate

    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run extraction on selected cases and print facts.")
    parser.add_argument("--input", default=str(root / "data" / "patients_sample_50.jsonl"))
    parser.add_argument("--sidecar", default=str(root / "data" / "extractions.jsonl"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("case_ids", nargs="+")
    args = parser.parse_args()

    rows = {
        row["case_id"]: row
        for line in Path(args.input).read_text().splitlines()
        if line.strip() and (row := json.loads(line))
    }
    with Path(args.sidecar).open("a", encoding="utf-8") as sidecar:
        for case_id in args.case_ids:
            row = rows[case_id]
            submission = PatientSubmission.model_validate(row["submission"])
            extraction, meta = call_extraction_model(submission, model=args.model)
            grounded = ground(extraction, submission.documents)
            facts = merge_facts(normalize_submission(submission), grounded)
            result = evaluate(facts)
            sidecar.write(json.dumps(sidecar_row(case_id, extraction, grounded, meta)) + "\n")

            label = row["label"]
            print(f"\n=== {case_id}  {meta['latency_s']}s  "
                  f"{meta['input_tokens']} in / {meta['output_tokens']} out")
            for name in ("hps", "consents", "anticoag_plans"):
                for fact in getattr(grounded, name):
                    print(f"  {name:15} {fact.source.id[:8]} {fact.source.date}  {fact.source.value}")
            for fact in grounded.vitals:
                print(f"  {'note vital':15} {fact.source.id[:8]} {fact.source.date}  "
                      f"{fact.source.value} (order {fact.note_order})")
            for fact in grounded.anticoag:
                print(f"  {'anticoag':15} {fact.source.id[:8]} {fact.source.date}  {fact.source.value}")
            for ev in grounded.vital_evidence:
                if not ev.model_quote_grounded:
                    print(f"  VALUE-VERIFIED {ev.kind} {ev.doc_id[:8]}: model quote "
                          f"{ev.model_quote!r} -> document line {ev.document_line!r}")
            for drop in grounded.drops:
                print(f"  DROP {drop.kind} {drop.doc_id[:8]}: {drop.reason}: {drop.quote!r}")
            for u in grounded.unverified:
                print(f"  UNVERIFIED (blocks READY) {u.kind} {u.source.id[:8]} {u.date}: {u.reason}")
            got = sorted({i.category for i in result.issues})
            ok = result.decision == label["decision"] and got == sorted(set(label["categories"]))
            print(f"  -> {result.decision} {got}   label {label['decision']} "
                  f"{sorted(label['categories'])}   {'MATCH' if ok else 'MISMATCH'}")


if __name__ == "__main__":
    _main()
