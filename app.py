from __future__ import annotations

import base64
import io
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any

import matplotlib
from flask import Flask, render_template, request, redirect, url_for, session, jsonify

matplotlib.use("Agg")
import matplotlib.pyplot as plt

APP_TITLE = "AX_PCore"
USERNAME = "Alex"
PASSWORD = "1"
USD_RATE = 91.69

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMPORTS_DIR = DATA_DIR / "imports"
LEDGER_PATH = DATA_DIR / "ledger.json"
CONFIG_PATH = BASE_DIR / "config.json"
PROFILE_PATH = BASE_DIR / "profile.json"

DEFAULT_CONFIG = {
    "categories": [
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
    ],
    "subcategories": [
        "Essentials",
        "Lifestyle",
        "Bills",
        "Travel",
        "Subscriptions",
    ],
    "payment_modes": ["Cash", "UPI", "Bank", "Card", "Loan", "Other"],
}

app = Flask(__name__)
app.secret_key = "ax_pcore_phase1"


@dataclass
class Transaction:
    id: int
    date: str
    subject: str
    category: str
    subcategory: str
    mode: str
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


def transaction_from_dict(data: dict[str, Any]) -> Transaction:
    return Transaction(
        id=int(data.get("id", 0)),
        date=data.get("date", date.today().isoformat()),
        subject=data.get("subject", ""),
        category=data.get("category", "General"),
        subcategory=data.get("subcategory", ""),
        mode=data.get("mode", "Other"),
        amount=float(data.get("amount", 0)),
        type=data.get("type", "Debit"),
        note=data.get("note", ""),
        transaction_id=data.get("transaction_id", ""),
        utr=data.get("utr", ""),
        source=data.get("source", ""),
        counterparty=data.get("counterparty", ""),
        account_type=data.get("account_type", "normal"),
        created_at=data.get("created_at", ""),
        edited_at=data.get("edited_at", ""),
    )


TRANSACTIONS: list[Transaction] = []
CONFIG: dict[str, list[str]] = {}
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


def ensure_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
    if not PROFILE_PATH.exists():
        PROFILE_PATH.write_text(
            json.dumps({"username": USERNAME, "password": PASSWORD}, indent=2),
            encoding="utf-8",
        )
    if not LEDGER_PATH.exists():
        LEDGER_PATH.write_text("[]", encoding="utf-8")


def load_config() -> dict[str, list[str]]:
    ensure_storage()
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for key, defaults in DEFAULT_CONFIG.items():
        values = data.get(key, defaults)
        data[key] = sorted({item.strip() for item in values if item.strip()})
    return data


def save_config(config: dict[str, list[str]]) -> None:
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_profile() -> dict[str, str]:
    ensure_storage()
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def load_ledger() -> list["Transaction"]:
    ensure_storage()
    raw = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    return [transaction_from_dict(item) for item in raw]


def save_ledger(transactions: list["Transaction"]) -> None:
    LEDGER_PATH.write_text(
        json.dumps([asdict(transaction) for transaction in transactions], indent=2),
        encoding="utf-8",
    )


def add_config_value(config: dict[str, list[str]], key: str, value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return ""
    if cleaned not in config[key]:
        config[key].append(cleaned)
        config[key] = sorted(set(config[key]))
        save_config(config)
    return cleaned


CONFIG = load_config()
TRANSACTIONS = load_ledger()
NEXT_ID = max((t.id for t in TRANSACTIONS), default=0) + 1


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
        for token in [
            date_value,
            amount_match.group(0),
            subject,
            "Paid to",
            "paid to",
            "Received from",
            "received from",
        ]:
            if token:
                note = note.replace(token, "")
        note = re.sub(r"\s+", " ", note).strip()

        transactions.append(
            {
                "date": normalize_date(date_value),
                "subject": subject or "Unknown",
                "category": CONFIG["categories"][0] if CONFIG["categories"] else "General",
                "subcategory": "",
                "mode": "UPI",
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


def transaction_key(data: dict[str, Any] | Transaction) -> tuple[str, str, float, str]:
    if isinstance(data, Transaction):
        date_value = data.date
        subject = data.subject
        amount = data.amount
        utr = data.utr
    else:
        date_value = data.get("date", "")
        subject = data.get("subject", "")
        amount = float(data.get("amount", 0))
        utr = data.get("utr", "")
    return (
        date_value,
        subject.strip().lower(),
        float(amount),
        utr.strip().lower() if utr else "",
    )


def save_import_text(raw_text: str) -> str:
    ensure_storage()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = IMPORTS_DIR / f"import_{timestamp}.txt"
    path.write_text(raw_text, encoding="utf-8")
    return str(path)


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
        subcategory=payload.get("subcategory", ""),
        mode=payload.get("mode", "Other"),
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
    save_ledger(TRANSACTIONS)


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


def analytics_lists(transactions: list[Transaction]) -> dict[str, list[dict[str, Any]]]:
    debit_transactions = [t for t in transactions if t.type == "Debit"]
    amounts = [t.amount for t in debit_transactions]
    average = sum(amounts) / len(amounts) if amounts else 0
    variance = (
        sum((amount - average) ** 2 for amount in amounts) / len(amounts)
        if amounts
        else 0
    )
    std_dev = math.sqrt(variance) if variance else 0
    outlier_threshold = average + (2 * std_dev)

    above_average = [
        {"date": t.date, "subject": t.subject, "amount": t.amount}
        for t in sorted(debit_transactions, key=lambda t: t.amount, reverse=True)
        if t.amount > average
    ][:5]

    subject_totals: dict[str, float] = {}
    for t in debit_transactions:
        subject_totals[t.subject] = subject_totals.get(t.subject, 0) + t.amount
    top_subjects = [
        {"subject": subject, "amount": amount}
        for subject, amount in sorted(subject_totals.items(), key=lambda x: x[1], reverse=True)
    ][:5]

    category_totals: dict[str, float] = {}
    for t in debit_transactions:
        category_totals[t.category] = category_totals.get(t.category, 0) + t.amount
    total_spend = sum(category_totals.values()) or 1
    category_dominance = [
        {
            "category": category,
            "amount": amount,
            "share": (amount / total_spend) * 100,
        }
        for category, amount in sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
    ][:5]

    outliers = [
        {"date": t.date, "subject": t.subject, "amount": t.amount}
        for t in sorted(debit_transactions, key=lambda t: t.amount, reverse=True)
        if t.amount >= outlier_threshold and outlier_threshold > 0
    ][:5]

    return {
        "above_average": above_average,
        "top_subjects": top_subjects,
        "category_dominance": category_dominance,
        "outliers": outliers,
    }


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
        profile = load_profile()
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == profile.get("username") and password == profile.get("password"):
            session["logged_in"] = True
            return redirect(url_for("dashboard"))
        error = "Invalid credentials."
    return render_template("login.html", title=APP_TITLE, error=error)


@app.route("/auth_check", methods=["POST"])
def auth_check():
    profile = load_profile()
    username = request.json.get("username", "")
    password = request.json.get("password", "")
    if username == profile.get("username") and password == profile.get("password"):
        session["logged_in"] = True
        return jsonify({"ok": True})
    return jsonify({"ok": False})


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
        category_choice = form.get("category", "General")
        subcategory_choice = form.get("subcategory", "")
        mode_choice = form.get("mode", "Other")
        category_new = form.get("category_new", "")
        subcategory_new = form.get("subcategory_new", "")
        mode_new = form.get("mode_new", "")
        if category_new:
            category_choice = add_config_value(CONFIG, "categories", category_new)
        if subcategory_new:
            subcategory_choice = add_config_value(CONFIG, "subcategories", subcategory_new)
        if mode_new:
            mode_choice = add_config_value(CONFIG, "payment_modes", mode_new)
        payload = {
            "date": form.get("date") or date.today().isoformat(),
            "subject": form.get("subject", "").strip(),
            "category": category_choice,
            "subcategory": subcategory_choice,
            "amount": form.get("amount", "0"),
            "type": form.get("type", "Debit"),
            "mode": mode_choice,
            "note": form.get("note", "").strip(),
            "counterparty": form.get("counterparty", "").strip(),
            "account_type": form.get("account_type", "normal").strip() or "normal",
        }
        edit_id = form.get("edit_id")
        if not payload["date"] or not payload["subject"] or float(payload["amount"]) <= 0:
            message = "Date, subject, and amount are required (amount must be positive)."
        elif edit_id:
            transaction = next((t for t in TRANSACTIONS if str(t.id) == edit_id), None)
            if transaction and can_edit(transaction):
                transaction.date = payload["date"]
                transaction.subject = payload["subject"]
                transaction.category = payload["category"]
                transaction.subcategory = payload.get("subcategory", "")
                transaction.mode = payload.get("mode", "Other")
                transaction.amount = float(payload["amount"])
                transaction.type = payload["type"]
                transaction.note = payload["note"]
                transaction.counterparty = payload.get("counterparty", "")
                transaction.account_type = payload.get("account_type", "normal")
                transaction.edited_at = datetime.now().isoformat(timespec="seconds")
                edit_message = f"Record edited at {transaction.edited_at}"
                save_ledger(TRANSACTIONS)
            else:
                message = "Record is read-only after 48 hours."
        else:
            add_transaction(payload)

    if request.method == "POST" and request.form.get("action") == "import_parse":
        raw_text = request.form.get("upi_text", "")
        preview = parse_upi_text(raw_text)
        existing_keys = {transaction_key(t) for t in TRANSACTIONS}
        new_items: list[dict[str, Any]] = []
        for item in preview:
            key = transaction_key(item)
            if key in existing_keys:
                item["is_duplicate"] = True
            else:
                item["is_duplicate"] = False
                existing_keys.add(key)
                new_items.append(item)
        if raw_text.strip():
            save_import_text(raw_text)
        session["import_preview"] = preview
        session["import_meta"] = {
            "detected": len(preview),
            "new": len(new_items),
        }

    if request.method == "POST" and request.form.get("action") == "import_commit":
        preview = session.get("import_preview", [])
        for entry in preview:
            if not entry.get("is_duplicate"):
                add_transaction(entry)
        session["import_preview"] = []
        session["import_meta"] = {}

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
    if not q and not category_filter and not type_filter:
        cutoff = date.today() - timedelta(days=2)
        filtered = [
            t for t in filtered if date.fromisoformat(t.date) >= cutoff
        ]

    filtered_sorted = sorted(filtered, key=lambda t: (t.date, t.created_at))

    edit_id = request.args.get("edit_id")
    edit_transaction = None
    if edit_id:
        edit_transaction = next((t for t in TRANSACTIONS if str(t.id) == edit_id), None)

    preview = session.get("import_preview", [])
    import_meta = session.get("import_meta", {})
    images = analytics_images(TRANSACTIONS)
    analytics = analytics_lists(TRANSACTIONS)
    totals_data = totals(TRANSACTIONS)

    return render_template(
        "index.html",
        title=APP_TITLE,
        categories=CONFIG["categories"],
        subcategories=CONFIG["subcategories"],
        modes=CONFIG["payment_modes"],
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
        import_meta=import_meta,
        images=images,
        analytics=analytics,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
