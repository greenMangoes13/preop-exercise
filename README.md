# Pre-Op Triage

## How to run

If you use Anaconda, run `conda deactivate` first. Anaconda's Brotli 1.0.9 leaks into the uv environment and breaks response decompression in the OpenAI client (`TypeError: process() takes no keyword arguments`). The scripts and test target also pin `brotli>=1.2` for the same reason.

With `OPENAI_API_KEY` set:

```bash
make test          # unit tests, no network
make baseline      # runs triage_submission on all 50 cases -> data/baseline_outputs.jsonl
make evals         # scores outputs against the labels -> data/eval_report.json
make score         # prints the aggregate score
make determinism   # 10 repeated runs of one case -> data/determinism_report.json
make report        # interactive per-case viewer
```

`make baseline` also writes `data/extractions.jsonl`, which keeps each case's raw extraction, quotes, grounding drops, tokens and latency. The original single-call implementation is kept as `baseline_triage_submission`, and its outputs are saved in `data/naive_*` so the comparison below is reproducible. The Claude Code session export is in `claude-session.txt`.

## Approach

The core question in this task isn't whether an LLM can read a chart. It's whether you can tell when it read the chart wrong. A single LLM call that reads and decides at once can't answer that: when it's wrong, you can't see which step failed.

So the LLM only reads, and code decides. One `gpt-4.1-mini` call per case extracts typed facts from the free-text documents (H&Ps, consents, note vitals, anticoagulant mentions and plans). Every fact has to carry a document id and a verbatim quote. Code then checks each quote against its document. The source text is the reference, so verifying an extraction doesn't need a gold label, just the document. Plain Python applies the four policy rules and builds the output.

The scored output is entirely code-derived: the decision, each issue with sources pointing to the exact record, and a list of every check including passes, so a nurse can verify a sampled READY case quickly. Quotes stay in the sidecar log, so wording drift in the model can't change the scored output.

I took this approach from my summer work evaluating a literature-extraction agent, where grounding outputs in the source text was the most reliable way to catch hallucinations, and where we found LLM judges are especially weak at verifying evidence. That's why no LLM makes a policy decision here. More detail is in [DESIGN.md](DESIGN.md).

## Results

| | Naive (single LLM call) | Final |
|---|---|---|
| Aggregate score | 70.0% | 99.33% |
| Decision matches label | 62% | 100% |
| Issue categories match label | 48% | 98% |
| Determinism, identical output over 10 runs | 10% | 100% |
| Determinism, same decision over 10 runs | 100% | 100% |
| Tokens per case (in / out) | not recorded | 3,611 / 530 |
| Latency per case (mean / max) | not recorded | 3.8 s / 6.1 s |

To be honest about these numbers: the 50 labeled cases were also my development set. I used them for error analysis and every fix below, so the score is optimistic for unseen data. What I'd expect to carry over is the direction of errors. The design fails closed, so mistakes on new data should lean toward NEEDS_FOLLOW_UP, not a false READY. Determinism was measured on one case (`case_00000`), the `make determinism` default.

## Key decisions

### Fail closed

Nurses review every non-READY case but only sample READY ones, so a false READY is the expensive mistake. Anything unknown, missing or unverifiable never produces READY. Absence fails the rule it belongs to: no signed consent is REQUIRED_DOCUMENTATION, and no CBC is REQUIRED_TESTING.

### Let the labels settle ambiguous rules

The anticoagulation rules were the most ambiguous part of the policy, so instead of guessing I checked every case that mentions an anticoagulant. The unclear-status case (`case_00033`) is labeled MISSING_REQUIRED_DATA, which confirmed my fail-closed default. The cases where the patient reports taking one with no plan (`case_00027`, `case_00038`) are labeled ANTICOAGULATION_MANAGEMENT, which corrected an assumption we had made.

### A small spot check before the full run caught a fail-open bug

Before running all 50 cases, I ran 5 diverse ones and read the extracted facts. In `case_00047`, the model intermittently emitted a control character where the note said "°C", so every note-vital quote failed grounding. With those readings dropped, the rules fell back to older, normal structured vitals and returned READY for a patient with BP 161/117.

Grounding was working as designed. The problem was that dropping a safety fact made the patient look safer. I fixed it in two ways. First, vitals are now verified by their values: code finds the document line containing the extracted numbers and unit, instead of trusting the model's quote. The corruption hit about a quarter of vital quotes across the 50 cases, and all were recovered this way. Second, an unverifiable vital that could be the most recent reading, or any unverifiable anticoagulant mention, now blocks READY instead of disappearing. A unit test locks that property in.

### No fuzzy grounding

I considered tolerant text matching for quotes, but the 50-case data argued against it. In `case_00038`, the model fabricated "Coumadin, status unknown" from a paraphrased past-history line that never mentions Coumadin. Fuzzy matching could have accepted that as grounded. So matching stays exact, and the fixes are targeted:

- A miscopied document id is repaired only when the quote appears verbatim in exactly one document.
- An anticoagulant quote must name the drug or a generic term like "blood thinner", so an unnamed mention is never silently lost.
- Placeholder vitals ("See flowsheet") are ignored.

### Near-miss drug names

The name check opened a new fail-open path: a real mention with a misspelled drug ("rivoraxaban") failed the check and was dropped without blocking. Spellings within one or two letters of a known anticoagulant now count as naming it, so the mention stays unverified and blocks READY. I scanned every word in all 50 cases' documents to make sure no ordinary word ("hepatic", for example) triggers it. Grounding itself is still exact.

### Post-procedure labs fail

If the most recent required lab is dated after the procedure, the check fails even when an earlier result was in window. A future-dated result is a data anomaly a human should see, and the policy says to use only the most recent result.

## Remaining mismatch

`case_00042` gets the right decision (NEEDS_FOLLOW_UP for a stale CBC) plus an extra MISSING_REQUIRED_DATA. The model quoted a real rivaroxaban mention with a misspelling. In earlier runs it shortened the quote with "..." instead. Either way the quote can't be verified, and unverified anticoagulant evidence blocks READY by design. I'm accepting this mismatch as the cost of the anticoagulation safety net.

## Next steps

- **Held-out evaluation.** Score on labeled cases that weren't used during development. This is the most important missing piece.
- **Error-injection suite.** Mutate READY cases with known answers (a CBC one day past its window, diastolic exactly 110, consent "signature pending", latest lab entered-in-error) to measure per-rule recall without new labels, and run it in CI.
- **False-READY rate as the main metric.** Aggregate accuracy weights every error equally and hides where the errors are. For sampled review, the number that matters is how often a case that should be held comes out READY. Track it per model and per prompt change.