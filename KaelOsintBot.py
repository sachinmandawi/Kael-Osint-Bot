# -*- coding: utf-8 -*-
# Minimal Number Lookup Bot
# Python 3.10+  |  Requires: python-telegram-bot==22.5, requests

import re
import io
import csv
import json
import time
import html
import logging
from typing import Dict, List, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters
)

# ====== LOGGING ======
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("osint-bot")

# ====== CONFIG ======
BOT_TOKEN = "8492547058:AAFcTuoVvPeeVlVCN4ik5QXPUQU0837rMN0"  # your BotFather token

API_ENDPOINT = {
    "url": "https://chut.voidnetwork.in/api",
    "headers": {
        'accept': '*/*',
        'content-type': 'application/json',
        'origin': 'https://chut.voidnetwork.in',
        'referer': 'https://chut.voidnetwork.in/',
        'user-agent': 'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Mobile Safari/537.36',
    }
}

RESULTS_PER_PAGE = 10
TELEGRAM_MSG_LIMIT = 4000
MERGE_VALUE_SEP = " • "

FIELD_NAMES = {
    "mobile": "Mobile Number",
    "alt": "Alternate Number",
    "name": "Name",
    "fname": "Father Name",
    "address": "Address",
    "circle": "Region",
    "id": "Aadhaar Number",
    "email": "Email",
}
LOGICAL_FIELDS = list(FIELD_NAMES.keys())

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

# ====== HTTP HELPERS ======
def build_session():
    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"])
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

# ====== UI ======
def make_start_menu() -> InlineKeyboardMarkup:
    # One row with Commands and Developer buttons side-by-side
    row1 = [
        InlineKeyboardButton("📜 Commands", callback_data="action:commands"),
        InlineKeyboardButton("👨‍💻 Developer", callback_data="action:dev"),
    ]
    return InlineKeyboardMarkup([row1])

def make_final_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back", callback_data="action:back")]])

def make_pagination_kb(idx: int, total: int) -> InlineKeyboardMarkup:
    nav_row = []
    if idx > 0:
        nav_row.append(InlineKeyboardButton("⬅️ Previous", callback_data=f"page:{idx-1}"))
    if idx < total - 1:
        nav_row.append(InlineKeyboardButton("Next ➡️", callback_data=f"page:{idx+1}"))
    rows = []
    if nav_row:
        rows.append(nav_row)
    rows.append([InlineKeyboardButton("🧩 Final Result", callback_data="action:final")])
    return InlineKeyboardMarkup(rows)

START_WELCOME = "⚡ Send a mobile number to search"

# ====== COMMANDS TEXT ======
def build_commands_text() -> str:
    # Minimal command list: only optional /num
    return (
        "📜 <b>Available Commands</b>\n\n"
        "🔎 <b>Lookups</b>\n"
        "• <code>/num &lt;number&gt;</code> — Mobile lookup (auto +91)\n\n"
        "📝 <b>Example</b>\n"
        "• <code>/num 9876543210</code>\n"
        "<i>Tip: You can also just send a number without /num.</i>"
    )

# ====== UTILS ======
def pretty_key(k: str) -> str:
    return FIELD_NAMES.get(k.lower(), k.capitalize())

def prettify_value(val: str) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if s.isdigit() or "@" in s:
        return s
    return s.title()

def last10(s: str) -> str:
    d = re.sub(r"\D", "", s or "")
    return d[-10:] if len(d) >= 10 else d

def last12(s: str) -> str:
    d = re.sub(r"\D", "", s or "")
    return d[-12:] if len(d) >= 12 else d

def normalize_number(raw: str) -> str:
    digits = re.sub(r"[^\d+]", "", raw or "").strip()
    if not digits:
        raise ValueError("Empty number")
    if digits.startswith("+"):
        return digits
    if re.fullmatch(r"\d{10}", digits):
        return "+91" + digits
    if re.fullmatch(r"91\d{10}", digits):
        return "+" + digits
    return "+" + digits

def clean_address(raw_address: str) -> str:
    if not raw_address:
        return ""
    parts = [p.strip() for p in str(raw_address).split("!") if p.strip()]
    seen, unique = set(), []
    for p in parts:
        pl = p.lower()
        if pl not in seen:
            pretty = p[:1].upper() + p[1:]
            unique.append(pretty)
            seen.add(pl)
    return ", ".join(unique)

def _add_unique_combined(base_val: str | None, new_val: str) -> str:
    if not new_val:
        return base_val or ""
    if not base_val:
        return new_val
    parts = [x.strip() for x in str(base_val).split(MERGE_VALUE_SEP) if x.strip()]
    if new_val.strip() not in parts:
        parts.append(new_val.strip())
    return MERGE_VALUE_SEP.join(parts)

def order_values_for_target(values: set, searched_query: str) -> List[str]:
    tgt = last10(searched_query)
    sorted_vals = sorted(v for v in values if v)
    def score(v: str):
        return (0 if last10(v) == tgt else 1, v)
    return [v for v in sorted(sorted_vals, key=score)]

def chunk_list(lst: list, n: int) -> list:
    return [lst[i:i+n] for i in range(0, len(lst), n)]

# ====== API CALL ======
def fetch_api(term_type: str, term: str) -> dict:
    """
    POST JSON: {"type": "mobile", "term": "<value>"}
    Normalizes to {"data": [...]} or {"error": {...}}.
    """
    url = API_ENDPOINT["url"].rstrip("/")
    headers = {"Accept": "application/json"}
    headers.update(API_ENDPOINT.get("headers", {}))
    payload = {"type": term_type, "term": term}
    s = build_session()

    def _normalize(resp_obj):
        if isinstance(resp_obj, list):
            return {"data": resp_obj}
        if isinstance(resp_obj, dict):
            if "data" in resp_obj and isinstance(resp_obj["data"], list):
                return {"data": resp_obj["data"]}
            if "result" in resp_obj and isinstance(resp_obj["result"], list):
                return {"data": resp_obj["result"]}
            if "records" in resp_obj and isinstance(resp_obj["records"], list):
                return {"data": resp_obj["records"]}
            if any(k in resp_obj for k in ("mobile", "email", "name", "id", "aadhaar", "aadhar")):
                return {"data": [resp_obj]}
            return {"error": {"message": resp_obj.get("message", "Unknown response shape"), "raw": resp_obj}}
        return {"error": {"message": "Non-JSON response"}}

    try:
        r = s.post(url, json=payload, headers=headers, timeout=20)
        if r.ok:
            try:
                return _normalize(r.json())
            except Exception as e:
                return {"error": {"message": f"Invalid JSON: {e}"}}

        # fallback GET
        r2 = s.get(url, params=payload, headers=headers, timeout=20)
        if r2.ok:
            try:
                return _normalize(r2.json())
            except Exception as e:
                return {"error": {"message": f"Invalid JSON: {e}"}}

        return {"error": {"status": r.status_code, "message": "HTTP error"}}
    except Exception as e:
        logger.exception("fetch_api failed")
        return {"error": {"message": str(e)}}

def _safe_list_from_any(obj):
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        v = obj.get("data")
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
        return [obj]
    return []

# ====== MERGE & ORDER ======
def merge_records(records: List[dict]) -> List[dict]:
    def signature(d: dict) -> Tuple[str, str]:
        m = last10(d.get("mobile", ""))
        a = last10(d.get("alt", ""))
        e = (d.get("email") or "").strip().lower()
        id12 = re.sub(r"\D", "", str(d.get("id", "")))[-12:]
        key = m or a or e or id12
        tag = "mob" if (m or a) else ("mail" if e else ("aad" if id12 else "x"))
        return (tag, key)

    buckets: Dict[Tuple[str, str], dict] = {}
    for rec in records:
        if rec.get("address"):
            rec["address"] = clean_address(rec["address"])
        sig = signature(rec)
        if not sig[1]:
            buckets[(id(rec), "x")] = rec
            continue
        if sig not in buckets:
            buckets[sig] = dict(rec)
        else:
            base = buckets[sig]
            for k in LOGICAL_FIELDS:
                v = rec.get(k)
                if not v:
                    continue
                if k not in base or not base[k]:
                    base[k] = v
                else:
                    base[k] = _add_unique_combined(str(base[k]), str(v))
    return list(buckets.values())

def order_for_query(data_list: list, searched_number: str) -> list:
    tgt = last10(searched_number)
    def score(idx_entry):
        idx, e = idx_entry
        mob = last10(str(e.get("mobile", "")))
        alt = last10(str(e.get("alt", "")))
        if mob == tgt:
            return (0, idx)
        if alt == tgt:
            return (1, idx)
        return (2, idx)
    return [e for _, e in sorted(list(enumerate(data_list)), key=score)]

# ====== RENDERING ======
def format_single_result_block(entry: dict, esc) -> list:
    lines = []
    for logical in ["mobile","alt","name","fname","address","circle","id","email"]:
        v = entry.get(logical)
        if not v:
            continue
        if logical == "address":
            v = clean_address(v)
        parts = [p.strip() for p in str(v).split(MERGE_VALUE_SEP) if p.strip()]
        if len(parts) <= 1:
            lines.append(f"➖ {pretty_key(logical)}: <code>{esc(prettify_value(v))}</code>")
        else:
            lines.append(f"➖ {pretty_key(logical)} ({len(parts)}):")
            for pv in parts:
                lines.append(f"   • <code>{esc(prettify_value(pv))}</code>")
    return lines

def format_page_text(title: str, query_line: str, items: list, page_index: int,
                     total_pages: int, start_result_num: int,
                     elapsed_first_page: float | None = None) -> str:
    esc = html.escape
    out = [f"✨ <b>{esc(title)}</b>", ""]
    out.append(f"🔹 <b>Query:</b> <code>{esc(query_line)}</code>")
    out.append(f"📑 <b>Page:</b> {page_index+1}/{total_pages}")
    out.append("\n🔹 <b>Results:</b>")
    num = start_result_num
    for entry in items:
        if not isinstance(entry, dict):
            continue
        out.append(f"\n📄 <b>Result {num}</b>")
        out.append("—" * 18)
        out.extend(format_single_result_block(entry, esc))
        num += 1
    if page_index == 0 and elapsed_first_page is not None:
        out.append(f"\n⏱️ Done in {elapsed_first_page:.2f}s")
    return "\n".join(out)

def build_final_boxes(data_list: list, searched_query: str) -> str:
    esc = html.escape
    uniq = {k: set() for k in ["mobile","alt","email","name","fname","address","circle","id"]}

    for e in (data_list or []):
        if not isinstance(e, dict):
            continue
        for k in uniq.keys():
            v = e.get(k)
            if not v:
                continue
            if k == "address":
                v = clean_address(v)
            [uniq[k].add(s.strip()) for s in str(v).split(MERGE_VALUE_SEP) if s.strip()]

    def box(title: str, values_list: List[str]) -> str:
        if not values_list:
            return f"📦 <b>{esc(title)}</b> — <i>None</i>\n"
        items = "\n".join(f"• <code>{esc(v)}</code>" for v in values_list)
        return f"📦 <b>{esc(title)} ({len(values_list)})</b>\n{items}\n"

    out = []
    out.append(f"🧩 <b>Final Result</b>\n")
    out.append(f"🔎 <b>Query:</b> <code>{esc(searched_query)}</code>\n")
    mobiles_ordered = order_values_for_target(uniq["mobile"], searched_query)
    alts_ordered    = order_values_for_target(uniq["alt"], searched_query)
    out.append(box("Mobile Numbers", mobiles_ordered))
    out.append(box("Alternate Numbers", alts_ordered))
    out.append(box("Emails", sorted(v for v in uniq["email"] if v)))
    out.append(box("Names", sorted(v for v in uniq["name"] if v)))
    out.append(box("Father Names", sorted(v for v in uniq["fname"] if v)))
    out.append(box("Addresses", sorted(v for v in uniq["address"] if v)))
    out.append(box("Regions", sorted(v for v in uniq["circle"] if v)))
    out.append(box("Aadhaar Numbers", sorted(v for v in uniq["id"] if v)))
    return "\n".join(out).strip()

# ====== PAGINATION & SEND ======
async def paginate_and_send(update: Update, context: ContextTypes.DEFAULT_TYPE,
                            title: str, query: str, items_list: list, elapsed: float):
    pages = []
    context.user_data["last_title"] = title
    context.user_data["last_query"] = query

    if items_list:
        data = order_for_query(list(items_list), query)
        context.user_data["last_raw_list"] = list(data)

        chunks = chunk_list(data, RESULTS_PER_PAGE)
        total = len(chunks)
        counter = 1
        for i, items in enumerate(chunks):
            page_text = format_page_text(
                title, query, items, i, total, counter,
                elapsed if i == 0 else None
            )
            pages.append(page_text)
            counter += len(items)
    else:
        pages.append(format_page_text(title, query, [], 0, 1, 1, elapsed))

    context.user_data["pages_text"] = pages
    context.user_data["pages_total"] = len(pages)
    kb = make_pagination_kb(0, len(pages))
    await update.message.reply_text(pages[0], parse_mode="HTML", reply_markup=kb)

# ====== CALLBACKS ======
async def on_page_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    idx = int(q.data.split(":")[1])
    pages = context.user_data.get("pages_text", [])
    total = context.user_data.get("pages_total", 0)
    if not pages or idx < 0 or idx >= total:
        return
    kb = make_pagination_kb(idx, total)
    await q.edit_message_text(pages[idx], parse_mode="HTML", reply_markup=kb)

async def on_action_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    action = q.data.split(":")[1]

    if action == "commands":
        text = build_commands_text()
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Back", callback_data="action:back")]
        ])
        context.user_data["back_target"] = "welcome"   # Back → Welcome
        await q.edit_message_text(text, parse_mode="HTML", reply_markup=kb)
        return

    if action == "dev":
        dev_text = (
            "👨‍💻 <b>Developer Info</b>\n\n"
            "• <b>Name:</b> Kabir Kael\n"
            "• <b>Telegram:</b> <a href=\"https://t.me/KabirKael\">@KabirKael</a>\n\n"
            "For custom features, support, or to purchase source code."
        )
        kb = InlineKeyboardMarkup([
            [InlineKeyboardButton("💬 Message @KabirKael", url="https://t.me/KabirKael")],
            [InlineKeyboardButton("🔙 Back", callback_data="action:back")]
        ])
        context.user_data["back_target"] = "welcome"   # Back → Welcome
        await q.edit_message_text(dev_text, parse_mode="HTML", reply_markup=kb)
        return

    if action == "final":
        raw_list = context.user_data.get("last_raw_list", [])
        query = context.user_data.get("last_query", "")
        if not raw_list:
            await q.edit_message_text("ℹ️ No data to merge yet. Send a number to search.", parse_mode="HTML")
            return
        merged_text = build_final_boxes(raw_list, query)
        context.user_data["back_target"] = "results"   # Back → Results pages
        await send_long_text(q, merged_text, reply_markup=make_final_kb())
        return

    if action == "back":
        target = context.user_data.get("back_target", "results")
        if target == "welcome":
            await q.edit_message_text(START_WELCOME, parse_mode="HTML", reply_markup=make_start_menu())
        else:
            pages = context.user_data.get("pages_text", [])
            if not pages:
                await q.edit_message_text("ℹ️ No previous results.", parse_mode="HTML")
                return
            kb = make_pagination_kb(0, len(pages))
            await q.edit_message_text(pages[0], parse_mode="HTML", reply_markup=kb)
        return

# ====== SEND LONG TEXT ======
async def send_long_text(update_or_query, text: str, reply_markup=None):
    parts = []
    while len(text) > TELEGRAM_MSG_LIMIT:
        cut = text.rfind("\n", 0, TELEGRAM_MSG_LIMIT)
        if cut == -1:
            cut = TELEGRAM_MSG_LIMIT
        parts.append(text[:cut])
        text = text[cut:]
    parts.append(text)
    first = True
    target_msg = getattr(update_or_query, "message", None)
    for p in parts:
        if first:
            try:
                await update_or_query.edit_message_text(p, parse_mode="HTML", reply_markup=reply_markup)
            except Exception:
                if target_msg:
                    await target_msg.reply_text(p, parse_mode="HTML", reply_markup=reply_markup)
            first = False
        else:
            if target_msg:
                await target_msg.reply_text(p, parse_mode="HTML")

# ====== ERROR HANDLER ======
async def error_handler(update, context):
    logger.exception("Unhandled exception while handling update", exc_info=context.error)
    try:
        if update and update.effective_message:
            await update.effective_message.reply_text("⚠️ An unexpected error occurred. Please try again.")
    except Exception:
        pass

# ====== LOOKUP ======
MOBILE_FINDER = re.compile(r"(?:\+?\d[\d\-\s]{8,}\d)")

def extract_number_like(text: str) -> str | None:
    if not text:
        return None
    m = MOBILE_FINDER.search(text)
    if not m:
        return None
    candidate = m.group(0)
    try:
        return normalize_number(candidate)
    except Exception:
        return None

async def perform_number_lookup(update: Update, context: ContextTypes.DEFAULT_TYPE, raw: str):
    try:
        number = normalize_number(raw)
    except Exception as e:
        await update.message.reply_text(
            f"❌ <b>Error</b>\n➖ <b>Message:</b> <code>{html.escape(str(e))}</code>",
            parse_mode="HTML"
        ); return

    t0 = time.time()
    api_resp = fetch_api("mobile", number)
    api_list = api_resp.get("data", []) if isinstance(api_resp, dict) else []
    api_clean = _safe_list_from_any(api_list)
    merged = merge_records(api_clean)
    elapsed = time.time() - t0

    if isinstance(api_resp, dict) and "error" in api_resp and not merged:
        err = api_resp["error"]
        msg = err.get("message", "Something went wrong")
        await update.message.reply_text(
            f"❌ <b>Error</b>\n➖ <b>Message:</b> <code>{html.escape(msg)}</code>\n⏱️ Failed in {elapsed:.2f}s",
            parse_mode="HTML"
        ); return

    # From results state → back should return to results, not welcome
    context.user_data["back_target"] = "results"
    await paginate_and_send(update, context, "📱 Number Lookup Result", number, merged, elapsed)

# ====== COMMAND & TEXT HANDLERS ======
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["back_target"] = "welcome"
    await update.message.reply_text(START_WELCOME, parse_mode="HTML", reply_markup=make_start_menu())

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["back_target"] = "welcome"
    await update.message.reply_text(build_commands_text(), parse_mode="HTML")

async def num_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("⚠️ Usage: <code>/num 9876543210</code>", parse_mode="HTML"); return
    raw = " ".join(context.args).strip()
    await perform_number_lookup(update, context, raw)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    number = extract_number_like(text)
    if number:
        await perform_number_lookup(update, context, number)
        return
    context.user_data["back_target"] = "welcome"
    hint = "🔎 Send a 10-digit mobile number (or with +91) to search.\nExample: <code>9876543210</code>"
    await update.message.reply_text(hint, parse_mode="HTML", reply_markup=make_start_menu())

# ====== MAIN ======
def main():
    app = Application.builder().token(BOT_TOKEN).build()

    # Core
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("num", num_cmd))  # optional

    # Text
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Callbacks
    app.add_handler(CallbackQueryHandler(on_page_callback, pattern=r"^page:\d+$"))
    app.add_handler(CallbackQueryHandler(on_action_callback, pattern=r"^action:(commands|final|back|dev)$"))

    app.add_error_handler(error_handler)

    print("🤖 Bot is running (Minimal number search)...")
    app.run_polling()

if __name__ == "__main__":
    main()
