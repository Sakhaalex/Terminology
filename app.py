from __future__ import annotations

import base64
import io
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from typing import Any

import matplotlib
from flask import Flask, render_template, request, redirect, url_for, session

matplotlib.use("Agg")
import matplotlib.pyplot as plt

APP_TITLE = "AX_PCore"
USERNAME = "Alex"
PASSWORD = "1"
USD_RATE = 91.69

CATEGORIES = [
    "General",
    "Food",
    "Transport",
    "Housing",
    "Utilities",
    "Shopping",
    "Health",
    "Education",
    "Entertainment",
    "Savings",
]

app = Flask(__name__)
app.secret_key = "ax_pcore_phase1"


@dataclass
class Transaction:
    id: int
    date: str
    subject: str
    category: str
    amount: float
    type: str
    note: str = ""
    transaction_id: str = ""
    utr: str = ""
    source: str = ""
    counterparty: str = ""
    account_type: str = "normal"
    created_at: str = ""
    edited_at: str = ""


TRANSACTIONS: list[Transaction] = []
NEXT_ID = 1


NOISE_PATTERNS = [
    re.compile(r"page\s+\d+\s+of\s+\d+", re.IGNORECASE),
    re.compile(r"disclaimer", re.IGNORECASE),
    re.compile(r"https?://"),
    re.compile(r"phonepe", re.IGNORECASE),
    re.compile(r"transaction\s+history", re.IGNORECASE),
]

NUMERIC_DATE_PATTERN = re.compile(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})")
TEXT_DATE_PATTERN = re.compile(r"([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})")
AMOUNT_PATTERN = re.compile(r"₹\s*([0-9,]+(?:\.\d+)?)")
TXN_ID_PATTERN = re.compile(r"(?:txn|transaction)\s*id[:\s]*([A-Za-z0-9-]+)", re.IGNORECASE)
UTR_PATTERN = re.compile(r"utr[:\s]*([A-Za-z0-9]+)", re.IGNORECASE)


def normalize_date(value: str) -> str:
    value = value.strip()
    for fmt in (
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d/%m/%y",
        "%d-%m-%y",
        "%b %d, %Y",
        "%B %d, %Y",
    ):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return date.today().isoformat()


def extract_date(text: str) -> str | None:
    numeric_match = NUMERIC_DATE_PATTERN.search(text)
    if numeric_match:
        return numeric_match.group(1)
    text_match = TEXT_DATE_PATTERN.search(text)
    if text_match:
        return text_match.group(1)
    return None


def parse_upi_text(text: str) -> list[dict[str, Any]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    filtered: list[str] = []
    for line in lines:
        if any(pattern.search(line) for pattern in NOISE_PATTERNS):
            continue
        filtered.append(line)

    blocks: list[list[str]] = []
    current: list[str] = []
    for line in filtered:
        if extract_date(line) and current:
            blocks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(current)

    transactions: list[dict[str, Any]] = []
    for block in blocks:
        block_text = " ".join(block)
        date_value = extract_date(block_text)
        amount_match = AMOUNT_PATTERN.search(block_text)
        if not date_value or not amount_match:
            continue
        subject = ""
        tx_type = ""
        if re.search(r"paid\s+to", block_text, re.IGNORECASE):
            tx_type = "Debit"
            subject = re.split(r"paid\s+to", block_text, flags=re.IGNORECASE)[-1]
        elif re.search(r"received\s+from", block_text, re.IGNORECASE):
            tx_type = "Credit"
            subject = re.split(r"received\s+from", block_text, flags=re.IGNORECASE)[-1]
        else:
            tx_type = "Debit"
            subject = block_text
        subject = subject.split("₹")[0].strip(" -")
        subject = subject.split("Transaction")[0].strip()

        amount = float(amount_match.group(1).replace(",", ""))
        txn_id = ""
        utr = ""
        txn_match = TXN_ID_PATTERN.search(block_text)
        utr_match = UTR_PATTERN.search(block_text)
        if txn_match:
            txn_id = txn_match.group(1)
        if utr_match:
            utr = utr_match.group(1)

        note = block_text
        for token in [date_value, amount_match.group(0)]:
            note = note.replace(token, "")
        note = re.sub(r"\s+", " ", note).strip()

        transactions.append(
            {
                "date": normalize_date(date_value),
                "subject": subject or "Unknown",
                "category": "General",
                "amount": amount,
                "type": tx_type,
                "note": note,
                "transaction_id": txn_id,
                "utr": utr,
                "source": "UPI_IMPORT",
                "counterparty": "",
                "account_type": "normal",
            }
        )
    return transactions


def login_required() -> bool:
    return bool(session.get("logged_in"))


def add_transaction(payload: dict[str, Any]) -> None:
    global NEXT_ID
    now = datetime.now().isoformat(timespec="seconds")
    transaction = Transaction(
        id=NEXT_ID,
        date=payload["date"],
        subject=payload["subject"],
        category=payload["category"],
        amount=float(payload["amount"]),
        type=payload["type"],
        note=payload.get("note", ""),
        transaction_id=payload.get("transaction_id", ""),
        utr=payload.get("utr", ""),
        source=payload.get("source", ""),
        counterparty=payload.get("counterparty", ""),
        account_type=payload.get("account_type", "normal"),
        created_at=now,
    )
    NEXT_ID += 1
    TRANSACTIONS.append(transaction)


def can_edit(transaction: Transaction) -> bool:
    created = datetime.fromisoformat(transaction.created_at)
    return datetime.now() - created <= timedelta(hours=48)


def usd(value: float) -> float:
    return value / USD_RATE


def totals(transactions: list[Transaction]) -> dict[str, float]:
    spend = sum(t.amount for t in transactions if t.type == "Debit")
    income = sum(t.amount for t in transactions if t.type == "Credit")
    return {"spend": spend, "income": income}


def chart_image(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def analytics_images(transactions: list[Transaction]) -> dict[str, str]:
    if not transactions:
        return {}
    totals_data = totals(transactions)
    fig1, ax1 = plt.subplots()
    ax1.bar(
        ["Spend", "Income"],
        [totals_data["spend"], totals_data["income"]],
        color=["#d9534f", "#5cb85c"],
    )
    ax1.set_title("Total Spend vs Total Income")
    ax1.set_ylabel("Amount (INR)")
    images = {"totals": chart_image(fig1)}

    category_totals: dict[str, float] = {}
    for t in transactions:
        if t.type == "Debit":
            category_totals[t.category] = category_totals.get(t.category, 0) + t.amount
    if category_totals:
        fig2, ax2 = plt.subplots()
        ax2.pie(category_totals.values(), labels=category_totals.keys(), autopct="%1.0f%%")
        ax2.set_title("Category Spending Distribution")
        images["categories"] = chart_image(fig2)

    subject_totals: dict[str, float] = {}
    subject_frequency: dict[str, int] = {}
    for t in transactions:
        if t.type == "Debit":
            subject_totals[t.subject] = subject_totals.get(t.subject, 0) + t.amount
        subject_frequency[t.subject] = subject_frequency.get(t.subject, 0) + 1
    if subject_totals:
        top_subjects = sorted(subject_totals.items(), key=lambda x: x[1], reverse=True)[:10]
        fig3, ax3 = plt.subplots()
        ax3.barh(
            [s for s, _ in top_subjects][::-1],
            [v for _, v in top_subjects][::-1],
            color="#d9534f",
        )
        ax3.set_title("Top Subjects by Spend")
        ax3.set_xlabel("Amount (INR)")
        images["top_subjects"] = chart_image(fig3)

    if subject_frequency:
        freq_sorted = sorted(subject_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        fig4, ax4 = plt.subplots()
        ax4.bar([s for s, _ in freq_sorted], [v for _, v in freq_sorted], color="#5bc0de")
        ax4.set_title("Transaction Frequency by Subject")
        ax4.set_ylabel("Count")
        ax4.tick_params(axis="x", rotation=45)
        images["frequency"] = chart_image(fig4)

    daily_totals: dict[str, float] = {}
    for t in transactions:
        if t.type == "Debit":
            daily_totals[t.date] = daily_totals.get(t.date, 0) + t.amount
    if daily_totals:
        dates_sorted = sorted(daily_totals.items())
        fig5, ax5 = plt.subplots()
        ax5.plot(
            [d for d, _ in dates_sorted],
            [v for _, v in dates_sorted],
            marker="o",
            color="#d9534f",
        )
        ax5.set_title("Daily Spend Trend")
        ax5.set_ylabel("Amount (INR)")
        ax5.tick_params(axis="x", rotation=45)
        images["trend"] = chart_image(fig5)

    if daily_totals:
        total_spend = sum(daily_totals.values())
        days = len(daily_totals)
        burn_rate = total_spend / days if days else 0
        fig6, ax6 = plt.subplots()
        ax6.bar(["Burn Rate (Avg/Day)"], [burn_rate], color="#d9534f")
        ax6.set_title("Burn Rate")
        ax6.set_ylabel("Amount (INR)")
        images["burn_rate"] = chart_image(fig6)

    return images


def loan_balances(transactions: list[Transaction]) -> dict[str, float]:
    balances: dict[str, float] = {}
    for transaction in transactions:
        if transaction.account_type != "loan":
            continue
        key = transaction.counterparty or "Unspecified"
        balances.setdefault(key, 0.0)
        if transaction.type == "Credit":
            balances[key] += transaction.amount
        else:
            balances[key] -= transaction.amount
    return balances


@app.route("/", methods=["GET", "POST"])
def login():
    if login_required():
        return redirect(url_for("dashboard"))
    error = ""
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == USERNAME and password == PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        error = "Invalid credentials."
    return render_template("login.html", title=APP_TITLE, error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if not login_required():
        return redirect(url_for("login"))

    message = ""
    edit_message = ""
    if request.method == "POST" and request.form.get("action") == "add_or_edit":
        form = request.form
        payload = {
            "date": form.get("date") or date.today().isoformat(),
            "subject": form.get("subject", "").strip(),
            "category": form.get("category", "General"),
            "amount": form.get("amount", "0"),
            "type": form.get("type", "Debit"),
            "note": form.get("note", "").strip(),
            "counterparty": form.get("counterparty", "").strip(),
            "account_type": form.get("account_type", "normal").strip() or "normal",
        }
        edit_id = form.get("edit_id")
        if not payload["subject"] or float(payload["amount"]) <= 0:
            message = "Subject and amount are required (amount must be positive)."
        elif edit_id:
            transaction = next((t for t in TRANSACTIONS if str(t.id) == edit_id), None)
            if transaction and can_edit(transaction):
                transaction.date = payload["date"]
                transaction.subject = payload["subject"]
                transaction.category = payload["category"]
                transaction.amount = float(payload["amount"])
                transaction.type = payload["type"]
                transaction.note = payload["note"]
                transaction.counterparty = payload.get("counterparty", "")
                transaction.account_type = payload.get("account_type", "normal")
                transaction.edited_at = datetime.now().isoformat(timespec="seconds")
                edit_message = f"Record edited at {transaction.edited_at}"
            else:
                message = "Record is read-only after 48 hours."
        else:
            add_transaction(payload)

    if request.method == "POST" and request.form.get("action") == "import_parse":
        raw_text = request.form.get("upi_text", "")
        preview = parse_upi_text(raw_text)
        session["import_preview"] = preview

    if request.method == "POST" and request.form.get("action") == "import_commit":
        preview = session.get("import_preview", [])
        for entry in preview:
            add_transaction(entry)
        session["import_preview"] = []

    q = request.args.get("q", "").strip()
    category_filter = request.args.get("category", "")
    type_filter = request.args.get("type", "")

    filtered = TRANSACTIONS
    if q:
        filtered = [
            t
            for t in filtered
            if q.lower() in t.subject.lower() or q.lower() in t.note.lower()
        ]
    if category_filter:
        filtered = [t for t in filtered if t.category == category_filter]
    if type_filter:
        filtered = [t for t in filtered if t.type == type_filter]

    filtered_sorted = sorted(filtered, key=lambda t: (t.date, t.created_at))

    edit_id = request.args.get("edit_id")
    edit_transaction = None
    if edit_id:
        edit_transaction = next((t for t in TRANSACTIONS if str(t.id) == edit_id), None)

    preview = session.get("import_preview", [])
    images = analytics_images(TRANSACTIONS)
    totals_data = totals(TRANSACTIONS)

    return render_template(
        "index.html",
        title=APP_TITLE,
        categories=CATEGORIES,
        transactions=filtered_sorted,
        totals=totals_data,
        usd_rate=USD_RATE,
        usd=usd,
        query=q,
        category_filter=category_filter,
        type_filter=type_filter,
        message=message,
        edit_message=edit_message,
        edit_transaction=edit_transaction,
        can_edit=can_edit,
        import_preview=preview,
        images=images,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
