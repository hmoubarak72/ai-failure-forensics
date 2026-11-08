"""Synthetic demo corpus: 50 documents with golden labels.

38 clean documents across the four types plus 12 deliberately adversarial ones that
trigger every failure category in the taxonomy (and one hard pipeline error).
Generation is seeded so the demo is fully reproducible.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .llm import fmt_amount
from .models import RawDocument

FIRST = ["Amira", "Daniel", "Sofia", "Omar", "Grace", "Liam", "Nadia", "Victor", "Hannah", "Karim", "Elena",
         "Marcus", "Leila", "Peter", "Yusuf", "Clara", "Tariq", "Maya", "Samuel", "Rania", "Julian", "Fatima"]
LAST = ["Haddad", "Okafor", "Lindqvist", "Mansour", "Whitfield", "Moreau", "Castillo", "Nakamura", "Brennan",
        "Farouk", "Petrov", "Duarte", "Adeyemi", "Kowalski", "Sinclair", "Rahman", "Ferreira", "Hughes"]
COMPANIES = ["Northwind Analytics", "Bluefin Logistics", "Cedar Ridge Energy", "Orion Health Partners",
             "Atlas Freight Co", "Helix Data Systems", "Sable Creek Mining", "Meridian Advisory"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


class _Gen:
    def __init__(self, seed: int = 42):
        self.r = random.Random(seed)
        self.used: set[str] = set()

    def name(self) -> str:
        while True:
            n = f"{self.r.choice(FIRST)} {self.r.choice(LAST)}"
            if n not in self.used and n != "John Smith":
                self.used.add(n)
                return n

    def company(self) -> str:
        return self.r.choice(COMPANIES)

    def date(self, year: int = 2025) -> str:
        return f"{self.r.choice(MONTHS)} {self.r.randint(1, 28)}, {year}"

    def money(self, lo: int, hi: int) -> float:
        return float(self.r.randint(lo, hi)) + self.r.choice([0, 0.5, 0.25, 0.99])


def _usd(v: float) -> str:
    return f"${v:,.2f}"


def _golden(doc_type: str, names: list[str], dates: list[str], amounts: list[tuple[float, str]],
            failure: str = "none") -> dict[str, Any]:
    return {
        "doc_type": doc_type,
        "entities": {"names": names, "dates": dates,
                     "amounts": [{"value": v, "currency": c} for v, c in amounts]},
        "summary": {"key_dates": dates, "key_amounts": [fmt_amount(v, c) for v, c in amounts]},
        "expected_failure": failure,
    }


# --------------------------------------------------------------------------- #
# Clean templates
# --------------------------------------------------------------------------- #
def contract(g: _Gen, i: int) -> tuple[RawDocument, dict]:
    a, b = g.name(), g.name()
    d1, d2 = g.date(2025), g.date(2026)
    fee, late = g.money(2000, 9000), g.money(50, 300)
    text = f"""SERVICE AGREEMENT
This Service Agreement is entered into on {d1} between {a}, of {g.company()}, and {b}, of {g.company()}.
1. Term. The term shall begin on {d1} and terminate on {d2}, subject to renewal by written notice.
2. Fees. The client shall pay {_usd(fee)} per month. A late fee of {_usd(late)} applies to overdue balances.
3. Confidentiality. Each party shall keep the other party's information confidential.
4. Governing law. This agreement is governed by the laws of the State of Texas.
Signed: {a}
Signed: {b}"""
    return RawDocument(doc_id=f"doc-{i:03d}", title="Service Agreement", text=text), \
        _golden("contract", [a, b], [d1, d2], [(fee, "USD"), (late, "USD")])


def invoice(g: _Gen, i: int) -> tuple[RawDocument, dict]:
    a = g.name()
    d1, d2 = g.date(2025), g.date(2025)
    sub = g.money(800, 20000)
    tax = round(sub * 0.0825, 2)
    total = round(sub + tax, 2)
    fmt = "pdf_text" if i % 3 == 0 else "text"
    page = "\nPage 1 of 1\n" if fmt == "pdf_text" else "\n"
    text = f"""INVOICE #INV-{1000 + i}
Invoice date: {d1}
Bill To: {a}, {g.company()}
Description: Professional services and deliver-
ables for the period.{page}Subtotal: {_usd(sub)}
Tax: {_usd(tax)}
Total amount due: {_usd(total)}
Due date: {d2}
Payment terms: net 30. Remit payment to {g.company()}."""
    return RawDocument(doc_id=f"doc-{i:03d}", title=f"Invoice INV-{1000 + i}", text=text, source_format=fmt), \
        _golden("invoice", [a], [d1, d2], [(sub, "USD"), (tax, "USD"), (total, "USD")])


def report(g: _Gen, i: int) -> tuple[RawDocument, dict]:
    a, b = g.name(), g.name()
    d = g.date(2025)
    rev = g.money(100000, 900000)
    text = f"""QUARTERLY OPERATIONS REPORT
Prepared by: {a}
Report date: {d}
Summary of findings: quarterly revenue reached {_usd(rev)}, and the operating margin improved by two points.
Analysis: results were driven by lower logistics costs; the main risk is supplier concentration.
Metrics are tracked in the appendix.
Recommendation: expand the supplier base before the next quarter.
Reviewed by: {b}"""
    return RawDocument(doc_id=f"doc-{i:03d}", title="Quarterly Operations Report", text=text, source_format="markdown"), \
        _golden("report", [a, b], [d], [(rev, "USD")])


def letter(g: _Gen, i: int) -> tuple[RawDocument, dict]:
    a, b = g.name(), g.name()
    d = g.date(2025)
    text = f"""Dear {a},
I am writing to follow up on our meeting last week about the site visit.
Could you confirm the agenda and the attendee list by {d}? Thank you for your help with the arrangements.
Please let me know if anything changes on your side.
Sincerely,
{b}"""
    return RawDocument(doc_id=f"doc-{i:03d}", title="Follow-up letter", text=text), \
        _golden("correspondence", [a, b], [d], [])


# --------------------------------------------------------------------------- #
# Adversarial templates (deliberate failure modes)
# --------------------------------------------------------------------------- #
def contract_no_dates(g: _Gen, i: int) -> tuple[RawDocument, dict]:
    a, b = g.name(), g.name()
    fee = g.money(3000, 9000)
    text = f"""CONSULTING AGREEMENT
This Consulting Agreement is made between {a}, of {g.company()}, and {b}, of {g.company()}.
1. Term. The engagement shall begin upon signature and continue for twelve months.
2. Fees. The client shall pay {_usd(fee)} per month for the deliverables.
3. Termination. Either party may terminate with thirty days notice.
Signed: {a}
Signed: {b}"""
    return RawDocument(doc_id=f"doc-{i:03d}", title="Consulting Agreement", text=text), \
        _golden("contract", [a, b], [], [(fee, "USD")], "extraction_hallucination")


def contract_unsigned(g: _Gen, i: int) -> tuple[RawDocument, dict]:
    a, b = g.name(), g.name()
    d = g.date(2025)
    fee = g.money(1000, 5000)
    text = f"""SUPPLY AGREEMENT
This Supply Agreement is entered into on {d} between {a}, of {g.company()}, and {b}, of {g.company()}.
1. Supply. The supplier shall deliver goods against each purchase order.
2. Price. Each shipment is billed at {_usd(fee)}; warranty and liability terms are in Schedule A.
Signed: {a}
Signed: ______________"""
    return RawDocument(doc_id=f"doc-{i:03d}", title="Supply Agreement", text=text), \
        _golden("contract", [a, b], [d], [(fee, "USD")], "extraction_hallucination")


def invoice_mixed_currency(g: _Gen, i: int) -> tuple[RawDocument, dict]:
    a = g.name()
    d1, d2 = g.date(2025), g.date(2025)
    eur, gbp = g.money(2000, 6000), g.money(300, 1200)
    text = f"""INVOICE #INV-{2000 + i}
Invoice date: {d1}
Bill To: {a}, {g.company()}
Consulting services (Berlin office): €{eur:,.2f}
Travel expenses (London): £{gbp:,.2f}
Each line is payable in its own currency. Due date: {d2}
Payment terms: net 60."""
    return RawDocument(doc_id=f"doc-{i:03d}", title=f"Invoice INV-{2000 + i}", text=text), \
        _golden("invoice", [a], [d1, d2], [(eur, "EUR"), (gbp, "GBP")], "prompt_failure")


def ambiguous_letter_invoice(g: _Gen, i: int) -> tuple[RawDocument, dict]:
    a, b = g.name(), g.name()
    d = g.date(2025)
    total = g.money(1000, 8000)
    text = f"""Dear {a},
Please find enclosed invoice INV-{3000 + i} for the payment discussed. The invoice total amount due is {_usd(total)},
and the due date for payment is {d}. The invoice lists the bill to address and the payment remit details;
if the invoice payment total needs adjusting, please reply.
Kind regards,
{b}"""
    return RawDocument(doc_id=f"doc-{i:03d}", title="Cover letter", text=text), \
        _golden("correspondence", [a, b], [d], [(total, "USD")], "misclassification")


def ambiguous_contract_report(g: _Gen, i: int) -> tuple[RawDocument, dict]:
    a = g.name()
    d = g.date(2025)
    text = f"""CONTRACT REVIEW REPORT
Prepared by: {a}
Report date: {d}
This report reviews the master agreement. The agreement says each party shall notify the other party
before it may terminate; the term clause and the renewal clause shall be read together. The parties
hereby keep the indemnification clause. Under the agreement a party shall not assign. Our finding is that
the termination clause is unclear and should be renegotiated."""
    return RawDocument(doc_id=f"doc-{i:03d}", title="Contract review", text=text, source_format="markdown"), \
        _golden("report", [a], [d], [], "misclassification")


def big_amount_contract(g: _Gen, i: int) -> tuple[RawDocument, dict]:
    a, b = g.name(), g.name()
    d1, d2 = g.date(2025), g.date(2027)
    value = float(g.r.choice([1_250_000, 1_500_000, 2_750_000, 4_200_000]))
    text = f"""MASTER SERVICES AGREEMENT
This Master Services Agreement is entered into on {d1} between {a}, of {g.company()}, and {b}, of {g.company()}.
1. Term. The term shall run until {d2}.
2. Contract value. The total contract value is {_usd(value)}, invoiced quarterly.
3. Indemnification. Each party shall indemnify the other party.
Signed: {a}
Signed: {b}"""
    return RawDocument(doc_id=f"doc-{i:03d}", title="Master Services Agreement", text=text), \
        _golden("contract", [a, b], [d1, d2], [(value, "USD")], "propagation_error")


def long_invoice(g: _Gen, i: int) -> tuple[RawDocument, dict]:
    a = g.name()
    dates = [g.date(2025) for _ in range(4)]
    lines = [g.money(200, 3000) for _ in range(4)]
    total = round(sum(lines), 2)
    text = f"""INVOICE #INV-{4000 + i}
Invoice date: {dates[0]}
Bill To: {a}, {g.company()}
Milestone 1 delivered {dates[1]}: {_usd(lines[0])}
Milestone 2 delivered {dates[2]}: {_usd(lines[1])}
Milestone 3: {_usd(lines[2])}
Milestone 4: {_usd(lines[3])}
Total amount due: {_usd(total)}
Due date: {dates[3]}
Payment terms: net 30. A late fee applies after the due date."""
    amounts = [(v, "USD") for v in lines] + [(total, "USD")]
    return RawDocument(doc_id=f"doc-{i:03d}", title=f"Invoice INV-{4000 + i}", text=text), \
        _golden("invoice", [a], dates, amounts, "context_loss")


def empty_doc(g: _Gen, i: int) -> tuple[RawDocument, dict]:
    return RawDocument(doc_id=f"doc-{i:03d}", title="Scanned page (OCR failed)", text="   \n\n  ", source_format="pdf_text"), \
        {"doc_type": None, "entities": {}, "summary": {}, "expected_failure": "pipeline_error"}


def build_corpus(seed: int = 42) -> tuple[list[RawDocument], dict[str, dict]]:
    g = _Gen(seed)
    plan = (
        [contract] * 10 + [invoice] * 10 + [report] * 9 + [letter] * 9
        + [contract_no_dates] * 2 + [contract_unsigned] + [invoice_mixed_currency] * 2
        + [ambiguous_letter_invoice, ambiguous_contract_report] + [big_amount_contract] * 2
        + [long_invoice] * 2 + [empty_doc]
    )
    random.Random(seed).shuffle(plan)
    docs, golden = [], {}
    for i, maker in enumerate(plan, start=1):
        doc, gold = maker(g, i)
        docs.append(doc)
        golden[doc.doc_id] = gold
    return docs, golden


def write_corpus(out_dir: Path, seed: int = 42) -> tuple[Path, Path]:
    docs, golden = build_corpus(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    dp, gp = out_dir / "documents.jsonl", out_dir / "golden.json"
    dp.write_text("\n".join(d.model_dump_json() for d in docs) + "\n", encoding="utf-8")
    gp.write_text(json.dumps(golden, indent=2), encoding="utf-8")
    return dp, gp


def load_documents(path: Path) -> list[RawDocument]:
    return [RawDocument.model_validate_json(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def load_golden(path: Path) -> dict[str, dict]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
