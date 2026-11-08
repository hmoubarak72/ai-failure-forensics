"""Versioned prompt templates.

Prompt versions are how the pipeline improves over time. Regression tracking re-runs
the accumulated eval dataset against each version, which is what produces the
"known issues resolved" trend line.

  v1  baseline prompts
  v2  + strict grounding rule (fixes extraction hallucinations)
      + explicit per-amount currency rule (fixes ignored currency instruction)
  v3  + primary-purpose classification rule (fixes ambiguous misclassification)
      + copy-amounts-verbatim summary rule (fixes propagation of amount values)
"""
from __future__ import annotations

VERSIONS = ("v1", "v2", "v3")

# Rule snippets. The mock LLM reacts to their presence the way a real model reacts
# to a clearer instruction, so prompt changes have observable, testable effects.
RULE_GROUNDING = "GROUNDING RULE: Only extract entities that appear verbatim in the document. If a field has no evidence, return an empty list. Never invent names or dates."
RULE_CURRENCY = "CURRENCY RULE: Documents may mix currencies. Determine the currency of EACH amount from its own symbol or ISO code."
RULE_PURPOSE = "PURPOSE RULE: Classify by the document's primary communicative purpose. A letter that merely encloses or mentions an invoice is correspondence; an analysis about a contract is a report."
RULE_COPY_AMOUNTS = "AMOUNT RULE: Copy every amount exactly from the entities JSON as '<CURRENCY> <value with thousands separators and 2 decimals>'. Do not rescale."

CONFIDENCE_INSTRUCTION = (
    'Also include an integer field "confidence" from 1 (guessing) to 5 (certain) '
    "rating how confident you are in your own answer."
)


def _rules(version: str, *candidates: tuple[str, str]) -> str:
    order = {v: i for i, v in enumerate(VERSIONS)}
    active = [rule for min_v, rule in candidates if order[version] >= order[min_v]]
    return ("\n" + "\n".join(active)) if active else ""


def extraction_prompt(text: str, version: str) -> str:
    return f"""You are an information-extraction engine.
Extract structured entities from the document below.
Return ONLY JSON with this schema:
{{"names": [str], "dates": [str], "amounts": [{{"value": float, "currency": "ISO-4217", "raw": str}}], "key_terms": [str], "confidence": int}}
- names: people named in the document (signatories, addressees, authors)
- dates: dates exactly as written
- amounts: monetary amounts; preserve the original currency of each amount
- key_terms: important legal/financial/business terms
{CONFIDENCE_INSTRUCTION}{_rules(version, ("v2", RULE_GROUNDING), ("v2", RULE_CURRENCY))}

<document>
{text}
</document>"""


def classification_prompt(text: str, version: str) -> str:
    return f"""You are a document classifier.
Classify the document as exactly one of: contract, invoice, report, correspondence.
Return ONLY JSON: {{"doc_type": str, "rationale": str, "confidence": int}}
{CONFIDENCE_INSTRUCTION}{_rules(version, ("v3", RULE_PURPOSE))}

<document>
{text}
</document>"""


def summarization_prompt(text: str, doc_type: str, entities_json: str, version: str) -> str:
    return f"""You write structured summaries tailored to the document type.
Document type: <doc_type>{doc_type}</doc_type>
Use the extracted entities as the authoritative facts:
<entities>
{entities_json}
</entities>
Return ONLY JSON:
{{"headline": str, "parties": [str], "key_dates": [str], "key_amounts": [str], "bullets": [str], "confidence": int}}
- For a contract emphasise parties, term and obligations; for an invoice emphasise amounts and due dates;
  for a report emphasise findings; for correspondence emphasise the request or action item.
- key_amounts must be formatted as '<CURRENCY> <value>'.
- Include every date and amount from the entities.
{CONFIDENCE_INSTRUCTION}{_rules(version, ("v3", RULE_COPY_AMOUNTS))}

<document>
{text}
</document>"""


JUDGE_PROMPT = """You are an impartial reviewer auditing ONE step of a multi-step document pipeline.
Judge whether the step's OUTPUT is a faithful, instruction-following transformation of its INPUT.
Pipeline steps: intake -> extraction -> classification -> summarization.

Step: {step_name}
Step instructions (prompt sent to the model, may be empty):
<prompt>
{prompt}
</prompt>
Original source document:
<source>
{source}
</source>
Step INPUT:
<input>
{step_input}
</input>
Step OUTPUT:
<output>
{step_output}
</output>
{reference}
Failure taxonomy:
- extraction_hallucination: entities in the output that do not exist in the source
- misclassification: wrong document type
- propagation_error: the step's input was correct but this step misinterpreted it
- prompt_failure: the model ignored an explicit instruction
- context_loss: important information present in the input was dropped
- none: the step is fine

Judge ONLY this step's own transformation. If the output repeats an error that was already in its input,
the step is not at fault for that error.
Return ONLY JSON: {{"score": 1-5, "issues": [str], "category": str}}
(5 = faithful, 3 = minor problems, 1-2 = significant quality drop)."""
