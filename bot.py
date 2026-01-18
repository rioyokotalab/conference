#!/usr/bin/env python3
"""
Slack deadlines bot that posts TWO MARKDOWN-TABLES in ONE message:

1) AI conference *full paper* deadlines (from huggingface/ai-deadlines on GitHub)
2) HPC conference *full paper* deadlines (from NOWLAB CFP table)

Requested behavior (rolled up from recent changes):
- ONLY show full paper deadlines
- Show deadlines that have NOT passed, up to 1 year from now (configurable via LOOKAHEAD_DAYS)
- Output TWO separate tables (AI + HPC) with columns:
    {Conference name, Full paper deadline (days until deadline), Conference dates, Location}
- If no in-window deadline exists for a conference (or the current year's entry isn't available yet),
  fall back to the most recent known schedule, SHIFT deadline + conference dates forward by +1 year
  repeatedly until the deadline is in the future, and mark affected cells with "(from {source_year})".

Required env vars:
  - SLACK_BOT_TOKEN
  - SLACK_CHANNEL_ID

Optional env vars:
  - LOOKAHEAD_DAYS: look ahead window (default: 365)
  - DRY_RUN: "1" prints message instead of posting to Slack
  - DEBUG: "1" prints debug logs
  - GITHUB_TOKEN (or GH_TOKEN): raises GitHub API rate limit for listing AI YAML files

AI options:
  - AI_WATCH_TITLES: comma-separated titles (default: a top-tier-ish list)
  - AI_DEADLINES_OWNER / AI_DEADLINES_REPO / AI_DEADLINES_REF / AI_DEADLINES_DIR: override dataset source
  - AI_MAX_ROWS: max rows in AI table (default: 20)

HPC options:
  - HPC_CFP_URL: override NOWLAB CFP URL (default: https://nowlab.cse.ohio-state.edu/cfp/)
  - HPC_WATCH_TITLES: comma-separated base titles to include (optional override)
      example: "SC,HPDC,ICPP,IPDPS,ICS,PPoPP,Euro-Par,ISC,HiPC,CLUSTER,CCGrid"
  - HPC_MAX_ROWS: max rows in HPC table (default: 20)
  - HPC_ASSUME_TZ: timezone used for interpreting date-only deadlines (default: AoE)
      values: "AoE" (UTC-12), "UTC", or an IANA timezone like "America/New_York"

Dependencies (requirements.txt):
  - slack_sdk
  - requests
  - PyYAML
  - python-dateutil
"""

from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import urljoin

import requests
import yaml
from dateutil import parser as dateparser
from dateutil import tz as dateutil_tz
from dateutil.relativedelta import relativedelta
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# ----------------------------
# Global config
# ----------------------------

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_CHANNEL_ID = os.environ["SLACK_CHANNEL_ID"]

DEBUG = os.getenv("DEBUG", "").strip().lower() in {"1", "true", "yes", "y"}
DRY_RUN = os.getenv("DRY_RUN", "").strip().lower() in {"1", "true", "yes", "y"}

LOOKAHEAD_DAYS = int(os.getenv("LOOKAHEAD_DAYS", "365"))

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")

# ----------------------------
# AI source (huggingface/ai-deadlines on GitHub)
# ----------------------------

AI_OWNER = os.getenv("AI_DEADLINES_OWNER", "huggingface")
AI_REPO = os.getenv("AI_DEADLINES_REPO", "ai-deadlines")
AI_REF = os.getenv("AI_DEADLINES_REF", "main")
AI_DIR = os.getenv("AI_DEADLINES_DIR", "src/data/conferences")

AI_MAX_ROWS = int(os.getenv("AI_MAX_ROWS", "20"))

AI_DEFAULT_WATCH = {
    "NeurIPS",
    "ICML",
    "ICLR",
    "AAAI",
    "IJCAI",
    "ACL",
    "EMNLP",
    "NAACL",
    "COLM",
    "CVPR",
    "ICCV",
    "ECCV",
    "KDD",
    "AISTATS",
    "ICRA",
    "IROS",
}

AI_WATCH_TITLES_RAW = os.getenv("AI_WATCH_TITLES", "").strip()
AI_WATCH_TITLES: List[str] = (
    [t.strip() for t in AI_WATCH_TITLES_RAW.split(",") if t.strip()]
    if AI_WATCH_TITLES_RAW
    else sorted(AI_DEFAULT_WATCH)
)

# We interpret these as "full paper deadline" in ai-deadlines.
AI_PAPER_TYPES = {"paper", "submission"}


# ----------------------------
# HPC source (NOWLAB CFP table)
# ----------------------------

HPC_CFP_URL = os.getenv("HPC_CFP_URL", "https://nowlab.cse.ohio-state.edu/cfp/")
HPC_MAX_ROWS = int(os.getenv("HPC_MAX_ROWS", "20"))

HPC_DEFAULT_WATCH = {
    "SC",
    "HPDC",
    "IPDPS",
    "ICPP",
    "ICS",
    "PPoPP",
    "Euro-Par",
    "ISC",
    "IEEE Cluster",
    "CCGrid",
}

HPC_WATCH_TITLES_RAW = os.getenv("HPC_WATCH_TITLES", "").strip()
HPC_WATCH_TITLES: Set[str] = (
    {t.strip().lower() for t in HPC_WATCH_TITLES_RAW.split(",") if t.strip()}
    if HPC_WATCH_TITLES_RAW
    else {t.strip().lower() for t in HPC_DEFAULT_WATCH}
)

HPC_ASSUME_TZ = os.getenv("HPC_ASSUME_TZ", "AoE").strip()  # AoE (UTC-12), UTC, or IANA TZ


# ----------------------------
# Data model
# ----------------------------

@dataclass(frozen=True)
class TableRow:
    conference_name: str
    paper_deadline_cell: str
    conf_dates_cell: str
    location: str
    sort_deadline_utc: Optional[datetime]


# ----------------------------
# Utilities
# ----------------------------

_TZ_ABBREV_MAP = {
    # US
    "PST": "America/Los_Angeles",
    "PDT": "America/Los_Angeles",
    "MST": "America/Denver",
    "MDT": "America/Denver",
    "CST": "America/Chicago",
    "CDT": "America/Chicago",
    "EST": "America/New_York",
    "EDT": "America/New_York",
    # Europe
    "CET": "Europe/Paris",
    "CEST": "Europe/Paris",
    "BST": "Europe/London",
}

AOE_TZ = timezone(timedelta(hours=-12))  # Anywhere on Earth (AoE) ~ UTC-12


def log(msg: str) -> None:
    if DEBUG:
        print(msg)


def require_env(name: str, value: str) -> None:
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")


def norm_title(s: str) -> str:
    return " ".join((s or "").split()).strip().lower()


def parse_timezone(tz_str: Optional[str]):
    """Parse timezone strings used in ai-deadlines YAML files."""
    if not tz_str:
        return timezone.utc

    s = str(tz_str).strip()
    upper = s.upper()

    if upper in {"AOE", "ANYWHERE ON EARTH"}:
        return AOE_TZ

    if upper in {"UTC", "GMT"}:
        return timezone.utc

    m = re.fullmatch(r"(UTC|GMT)\s*([+-])\s*(\d{1,2})(?::?(\d{2}))?", upper)
    if m:
        sign = 1 if m.group(2) == "+" else -1
        hours = int(m.group(3))
        minutes = int(m.group(4) or "0")
        return timezone(sign * timedelta(hours=hours, minutes=minutes))

    if upper in _TZ_ABBREV_MAP:
        tzinfo = dateutil_tz.gettz(_TZ_ABBREV_MAP[upper])
        if tzinfo:
            return tzinfo

    tzinfo = dateutil_tz.gettz(s)
    return tzinfo if tzinfo else timezone.utc


def parse_dt(date_str: str, tzinfo) -> datetime:
    dt = dateparser.parse(str(date_str))
    if dt is None:
        raise ValueError(f"Could not parse datetime: {date_str}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tzinfo)
    return dt


def safe_shift_datetime(dt: datetime, years: int) -> datetime:
    """Shift dt by a number of years, preserving tzinfo."""
    return dt + relativedelta(years=years)


def safe_shift_date(d: date, years: int) -> date:
    return (datetime(d.year, d.month, d.day) + relativedelta(years=years)).date()


def days_until(dt_utc: datetime) -> int:
    """Days until deadline (ceil)."""
    now = datetime.now(timezone.utc)
    sec = (dt_utc - now).total_seconds()
    return int(math.ceil(sec / 86400.0))


def fmt_iso_date_range(start: Optional[date], end: Optional[date]) -> str:
    if start and end:
        return f"{start.isoformat()} to {end.isoformat()}"
    if start:
        return start.isoformat()
    return "—"


def escape_pipes(s: str) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ")


def render_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    hdr = "| " + " | ".join(escape_pipes(h) for h in headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(escape_pipes(c) for c in r) + " |" for r in rows)
    return "\n".join([hdr, sep, body]) if body else "\n".join([hdr, sep])


def normalize_url(href: str, base_url: str) -> str:
    if not href:
        return ""
    return urljoin(base_url, href)


# ----------------------------
# GitHub helpers (AI listing)
# ----------------------------

def github_headers() -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "deadlines-slack-bot",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def github_list_download_urls(owner: str, repo: str, path: str, ref: str) -> List[str]:
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    r = requests.get(api_url, headers=github_headers(), timeout=30)

    if r.status_code == 403 and "rate limit" in r.text.lower():
        raise SystemExit(
            "GitHub API rate limit hit while listing AI conference files. "
            "Set GITHUB_TOKEN (or GH_TOKEN) in your workflow env to raise the limit."
        )

    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise SystemExit(f"Unexpected GitHub contents response type: {type(data)}")

    urls: List[str] = []
    for item in data:
        if item.get("type") != "file":
            continue
        name = str(item.get("name") or "").lower()
        if not name.endswith((".yml", ".yaml")):
            continue
        dl = item.get("download_url")
        if dl:
            urls.append(dl)

    if not urls:
        raise SystemExit(f"No YAML files found at {owner}/{repo}:{path}@{ref}")

    return sorted(urls)


# ----------------------------
# AI: load + extract
# ----------------------------

def load_ai_entries() -> List[Dict[str, Any]]:
    urls = github_list_download_urls(AI_OWNER, AI_REPO, AI_DIR, AI_REF)
    log(f"[AI] Found {len(urls)} YAML files in {AI_OWNER}/{AI_REPO}:{AI_DIR}")

    entries: List[Dict[str, Any]] = []
    for url in urls:
        log(url)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        parsed = yaml.safe_load(resp.text)
        if parsed is None:
            continue
        if isinstance(parsed, list):
            entries.extend([x for x in parsed if isinstance(x, dict)])
        elif isinstance(parsed, dict):
            entries.append(parsed)

    log(f"[AI] Loaded {len(entries)} conference entries across years.")
    return entries


def group_by_title(entries: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        title = str(e.get("title") or "").strip()
        if not title:
            continue
        by.setdefault(norm_title(title), []).append(e)

    def year_key(x: Dict[str, Any]) -> int:
        try:
            return int(str(x.get("year") or "0"))
        except Exception:
            return 0

    for k in by:
        by[k].sort(key=year_key)
    return by


def extract_ai_paper_deadline(entry: Dict[str, Any]) -> Optional[datetime]:
    """Return full-paper deadline as UTC datetime, if available."""
    deadlines = entry.get("deadlines")
    if isinstance(deadlines, list):
        # Prefer 'paper' then 'submission'
        for t in ("paper", "submission"):
            candidates: List[datetime] = []
            for d in deadlines:
                if not isinstance(d, dict):
                    continue
                dtype = str(d.get("type") or "").strip().lower()
                if dtype != t:
                    continue
                date_val = d.get("date")
                if not date_val:
                    continue
                tz_label = str(d.get("timezone") or entry.get("timezone") or "UTC").strip()
                tzinfo = parse_timezone(tz_label)
                try:
                    dt_local = parse_dt(str(date_val), tzinfo)
                except Exception:
                    continue
                candidates.append(dt_local.astimezone(timezone.utc))
            if candidates:
                return min(candidates)

    # Legacy fallbacks
    if entry.get("submission_deadline"):
        tz_label = str(entry.get("timezone_submission") or entry.get("timezone") or "UTC").strip()
        tzinfo = parse_timezone(tz_label)
        dt_local = parse_dt(str(entry["submission_deadline"]), tzinfo)
        return dt_local.astimezone(timezone.utc)

    if entry.get("deadline"):
        tz_label = str(entry.get("timezone") or "UTC").strip()
        tzinfo = parse_timezone(tz_label)
        dt_local = parse_dt(str(entry["deadline"]), tzinfo)
        return dt_local.astimezone(timezone.utc)

    return None


def extract_ai_conf_dates(entry: Dict[str, Any]) -> Tuple[Optional[date], Optional[date], str]:
    """Return (start_date, end_date, raw_date_string)."""
    start_s = entry.get("start")
    end_s = entry.get("end")
    raw = str(entry.get("date") or "").strip()

    start_d: Optional[date] = None
    end_d: Optional[date] = None

    try:
        if start_s:
            start_d = dateparser.parse(str(start_s)).date()
        if end_s:
            end_d = dateparser.parse(str(end_s)).date()
    except Exception:
        start_d = end_d = None

    return start_d, end_d, raw


def extract_ai_location(entry: Dict[str, Any]) -> str:
    city = str(entry.get("city") or "").strip()
    country = str(entry.get("country") or "").strip()
    if city and country:
        if city.lower() == country.lower():
            return city
        return f"{city}, {country}"
    return city or country or "—"


def parse_entry_year(entry: Dict[str, Any]) -> Optional[int]:
    try:
        return int(str(entry.get("year") or ""))
    except Exception:
        return None


def build_ai_rows(all_entries: Sequence[Dict[str, Any]]) -> List[TableRow]:
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(days=LOOKAHEAD_DAYS)

    by_title = group_by_title(all_entries)
    rows: List[TableRow] = []

    for watch_title in AI_WATCH_TITLES:
        series = by_title.get(norm_title(watch_title), [])
        if not series:
            continue

        # 1) Prefer a real entry whose paper deadline is within [now, horizon]
        best_entry: Optional[Dict[str, Any]] = None
        best_deadline: Optional[datetime] = None

        for e in series:
            dl = extract_ai_paper_deadline(e)
            if not dl:
                continue
            if now <= dl <= horizon:
                if best_deadline is None or dl < best_deadline:
                    best_deadline = dl
                    best_entry = e

        if best_entry and best_deadline:
            year_str = str(best_entry.get("year") or "").strip()
            conf_name = f"{watch_title} {year_str}".strip()

            start_d, end_d, raw_date = extract_ai_conf_dates(best_entry)
            conf_dates_cell = fmt_iso_date_range(start_d, end_d)
            if conf_dates_cell == "—" and raw_date:
                conf_dates_cell = raw_date

            location = extract_ai_location(best_entry)
            deadline_cell = f"{best_deadline.date().isoformat()} ({days_until(best_deadline)}d)"

            rows.append(
                TableRow(
                    conference_name=conf_name,
                    paper_deadline_cell=deadline_cell,
                    conf_dates_cell=conf_dates_cell,
                    location=location,
                    sort_deadline_utc=best_deadline,
                )
            )
            continue

        # 2) If no real entry is in-window, estimate from the latest entry that has a deadline
        template: Optional[Dict[str, Any]] = None
        template_year: Optional[int] = None
        template_deadline: Optional[datetime] = None

        for e in reversed(series):
            y = parse_entry_year(e)
            dl = extract_ai_paper_deadline(e)
            if y and dl:
                template = e
                template_year = y
                template_deadline = dl
                break

        if not (template and template_year and template_deadline):
            continue

        est_year = template_year
        est_deadline = template_deadline

        while est_deadline < now:
            est_deadline = safe_shift_datetime(est_deadline, 1)
            est_year += 1

        if est_deadline > horizon:
            continue

        # Dates: shift start/end if available; otherwise adjust raw date year string if present.
        ps, pe, praw = extract_ai_conf_dates(template)
        diff = est_year - template_year
        start_d = safe_shift_date(ps, diff) if ps else None
        end_d = safe_shift_date(pe, diff) if pe else None

        conf_dates_cell = fmt_iso_date_range(start_d, end_d)
        if conf_dates_cell == "—" and praw:
            # Replace any 4-digit year in the string.
            updated = re.sub(r"\b(20\d{2})\b", str(est_year), praw)
            # If no year was present, append it.
            if updated == praw and str(est_year) not in updated:
                updated = f"{updated} {est_year}".strip()
            conf_dates_cell = f"{updated} (from {template_year})"
        elif conf_dates_cell != "—":
            conf_dates_cell = f"{conf_dates_cell} (from {template_year})"

        location = extract_ai_location(template)
        if location != "—":
            location = f"{location} (from {template_year})"

        deadline_cell = f"{est_deadline.date().isoformat()} ({days_until(est_deadline)}d) (from {template_year})"
        conf_name = f"{watch_title} {est_year}"

        rows.append(
            TableRow(
                conference_name=conf_name,
                paper_deadline_cell=deadline_cell,
                conf_dates_cell=conf_dates_cell,
                location=location,
                sort_deadline_utc=est_deadline,
            )
        )

    rows.sort(key=lambda r: r.sort_deadline_utc or datetime.max.replace(tzinfo=timezone.utc))
    return rows[:AI_MAX_ROWS]


# ----------------------------
# HPC: parse NOWLAB CFP table
# ----------------------------

@dataclass
class HPCEntry:
    base_title: str
    year: int
    location: str
    conf_start: Optional[date]
    conf_end: Optional[date]
    paper_deadline: Optional[date]  # date-only (full paper)


@dataclass
class _Cell:
    text: str
    href: Optional[str]


class _TableRowParser(HTMLParser):
    """Extract all HTML table rows into a list of cell objects (text + first link)."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: List[List[_Cell]] = []
        self._in_tr = False
        self._in_cell = False
        self._cell_text: List[str] = []
        self._cell_href: Optional[str] = None
        self._current_row: List[_Cell] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag == "tr":
            self._in_tr = True
            self._current_row = []
            return

        if self._in_tr and tag in {"td", "th"}:
            self._in_cell = True
            self._cell_text = []
            self._cell_href = None
            return

        if self._in_cell and tag == "a":
            href = dict(attrs).get("href")
            if href and self._cell_href is None:
                self._cell_href = href

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._in_cell:
            text = " ".join("".join(self._cell_text).split())
            self._current_row.append(_Cell(text=text, href=self._cell_href))
            self._in_cell = False
            self._cell_text = []
            self._cell_href = None
            return

        if tag == "tr" and self._in_tr:
            if any(c.text for c in self._current_row):
                self.rows.append(self._current_row)
            self._in_tr = False
            self._current_row = []


def split_hpc_conf_name(name: str) -> Tuple[str, int]:
    """Split NOWLAB conference name cell into (base_title, year)."""
    s = " ".join((name or "").split()).strip()
    if not s:
        return "", 0

    # 4-digit year
    m = re.search(r"\b(20\d{2})\b", s)
    if m:
        year = int(m.group(1))
        base = (s[: m.start()] + s[m.end() :]).strip()
        base = re.sub(r"\s+", " ", base).strip(" -")
        return (base or s), year

    # Apostrophe year ('26 or ’26)
    m = re.search(r"[\u2019'](\d{2})\b", s)
    if m:
        year = int("20" + m.group(1))
        base = (s[: m.start()] + s[m.end() :]).strip()
        base = re.sub(r"\s+", " ", base).strip(" -")
        return (base or s), year

    # No year in name (e.g., CCGrid)
    return s, 0


def hpc_assumed_tzinfo():
    if not HPC_ASSUME_TZ:
        return AOE_TZ
    upper = HPC_ASSUME_TZ.strip().upper()
    if upper in {"AOE", "ANYWHERE ON EARTH"}:
        return AOE_TZ
    if upper in {"UTC", "GMT"}:
        return timezone.utc
    tzinfo = dateutil_tz.gettz(HPC_ASSUME_TZ.strip())
    return tzinfo if tzinfo else AOE_TZ


def parse_deadline_date_only(date_str: str) -> Optional[date]:
    s = (date_str or "").strip()
    if not s or s.lower() in {"tba", "n/a", "na", "-"}:
        return None
    try:
        dt = dateparser.parse(s)
        if dt is None:
            return None
        return dt.date()
    except Exception:
        return None


def parse_conference_date_range(date_str: str) -> Tuple[Optional[date], Optional[date]]:
    """Parse strings like 'Jul 13 - 16, 2026' or 'Sep 28 - Oct 01, 2026'."""
    s = " ".join((date_str or "").split()).strip()
    if not s:
        return None, None

    m = re.fullmatch(r"([A-Za-z]{3,})\s+(\d{1,2})\s*-\s*(\d{1,2}),\s*(20\d{2})", s)
    if m:
        mon, d1, d2, y = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        try:
            start = dateparser.parse(f"{mon} {d1} {y}").date()
            end = dateparser.parse(f"{mon} {d2} {y}").date()
            return start, end
        except Exception:
            return None, None

    m = re.fullmatch(r"([A-Za-z]{3,})\s+(\d{1,2})\s*-\s*([A-Za-z]{3,})\s+(\d{1,2}),\s*(20\d{2})", s)
    if m:
        mon1, d1, mon2, d2, y = m.group(1), int(m.group(2)), m.group(3), int(m.group(4)), int(m.group(5))
        try:
            start = dateparser.parse(f"{mon1} {d1} {y}").date()
            end = dateparser.parse(f"{mon2} {d2} {y}").date()
            return start, end
        except Exception:
            return None, None

    # Best-effort fallback
    try:
        dt = dateparser.parse(s)
        if dt:
            d = dt.date()
            return d, d
    except Exception:
        pass

    return None, None


def infer_year_from_text(*texts: str) -> int:
    for t in texts:
        m = re.search(r"\b(20\d{2})\b", t or "")
        if m:
            return int(m.group(1))
    return 0


def load_hpc_entries_from_nowlab() -> List[HPCEntry]:
    r = requests.get(HPC_CFP_URL, timeout=30)
    r.raise_for_status()

    parser = _TableRowParser()
    parser.feed(r.text)

    rows = parser.rows
    log(f"[HPC] Parsed {len(rows)} HTML table rows from NOWLAB")

    out: List[HPCEntry] = []
    for row in rows:
        if len(row) < 5:
            continue

        name = row[0].text
        if not name or "conference name" in name.lower():
            continue

        conf_dates_str = row[2].text.strip() if row[2].text else ""
        paper_deadline_str = row[4].text.strip() if row[4].text else ""

        base, year = split_hpc_conf_name(name)

        # CCGrid-style: year missing from name, but present in date column.
        if year == 0:
            year = infer_year_from_text(conf_dates_str, paper_deadline_str)

        if not base or not year:
            continue

        # Apply default/override watchlist
        if HPC_WATCH_TITLES and base.strip().lower() not in HPC_WATCH_TITLES:
            continue

        location = row[1].text.strip() if row[1].text else "—"
        conf_start, conf_end = parse_conference_date_range(conf_dates_str)
        paper_deadline = parse_deadline_date_only(paper_deadline_str)

        out.append(
            HPCEntry(
                base_title=base.strip(),
                year=year,
                location=location or "—",
                conf_start=conf_start,
                conf_end=conf_end,
                paper_deadline=paper_deadline,
            )
        )

    log(f"[HPC] Loaded {len(out)} HPC entries after filtering.")
    return out


def build_hpc_rows(hpc_entries: Sequence[HPCEntry]) -> List[TableRow]:
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(days=LOOKAHEAD_DAYS)
    tzinfo = hpc_assumed_tzinfo()

    # Group by base title
    by: Dict[str, List[HPCEntry]] = {}
    for e in hpc_entries:
        by.setdefault(e.base_title.lower(), []).append(e)
    for k in by:
        by[k].sort(key=lambda x: x.year)

    rows: List[TableRow] = []

    for _, series in by.items():
        base_title = series[0].base_title

        # 1) Prefer real entry with soonest in-window paper deadline
        best: Optional[HPCEntry] = None
        best_deadline_utc: Optional[datetime] = None

        for e in series:
            if not e.paper_deadline:
                continue
            dt_local = datetime(e.paper_deadline.year, e.paper_deadline.month, e.paper_deadline.day, 23, 59, tzinfo=tzinfo)
            dt_utc = dt_local.astimezone(timezone.utc)

            if now <= dt_utc <= horizon:
                if best_deadline_utc is None or dt_utc < best_deadline_utc:
                    best_deadline_utc = dt_utc
                    best = e

        if best and best_deadline_utc:
            deadline_cell = f"{best_deadline_utc.date().isoformat()} ({days_until(best_deadline_utc)}d)"
            conf_dates_cell = fmt_iso_date_range(best.conf_start, best.conf_end)
            location = best.location
            conf_name = f"{base_title} {best.year}".strip()

            rows.append(
                TableRow(
                    conference_name=conf_name,
                    paper_deadline_cell=deadline_cell,
                    conf_dates_cell=conf_dates_cell,
                    location=location,
                    sort_deadline_utc=best_deadline_utc,
                )
            )
            continue

        # 2) Estimate from latest entry that has a deadline
        template: Optional[HPCEntry] = next((x for x in reversed(series) if x.paper_deadline), None)
        if not template or not template.paper_deadline:
            continue

        est_year = template.year
        dt_local = datetime(template.paper_deadline.year, template.paper_deadline.month, template.paper_deadline.day, 23, 59, tzinfo=tzinfo)
        est_deadline_utc = dt_local.astimezone(timezone.utc)

        while est_deadline_utc < now:
            est_deadline_utc = safe_shift_datetime(est_deadline_utc, 1)
            est_year += 1

        if est_deadline_utc > horizon:
            continue

        # Shift conf dates too
        diff = est_year - template.year
        s = safe_shift_date(template.conf_start, diff) if template.conf_start else None
        e = safe_shift_date(template.conf_end, diff) if template.conf_end else None
        conf_dates_cell = fmt_iso_date_range(s, e)
        if conf_dates_cell != "—":
            conf_dates_cell = f"{conf_dates_cell} (from {template.year})"

        location = template.location
        if location != "—":
            location = f"{location} (from {template.year})"

        deadline_cell = f"{est_deadline_utc.date().isoformat()} ({days_until(est_deadline_utc)}d) (from {template.year})"
        conf_name = f"{base_title} {est_year}".strip()

        rows.append(
            TableRow(
                conference_name=conf_name,
                paper_deadline_cell=deadline_cell,
                conf_dates_cell=conf_dates_cell,
                location=location,
                sort_deadline_utc=est_deadline_utc,
            )
        )

    rows.sort(key=lambda r: r.sort_deadline_utc or datetime.max.replace(tzinfo=timezone.utc))
    return rows[:HPC_MAX_ROWS]


# ----------------------------
# Message building + Slack post
# ----------------------------

TABLE_HEADERS = [
    "Conference name",
    "Full paper deadline (days until deadline)",
    "Conference dates",
    "Location",
]


def build_message(ai_rows: Sequence[TableRow], hpc_rows: Sequence[TableRow]) -> str:
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(days=LOOKAHEAD_DAYS)

    lines: List[str] = []
    lines.append(f"*Upcoming full-paper deadlines (next {LOOKAHEAD_DAYS} days)*")
    lines.append(f"_Window: {now.date().isoformat()} to {horizon.date().isoformat()}_")
    lines.append("")

    lines.append("*AI conferences — full paper deadlines*")
    ai_table_rows = [[r.conference_name, r.paper_deadline_cell, r.conf_dates_cell, r.location] for r in ai_rows]
    lines.append(render_markdown_table(TABLE_HEADERS, ai_table_rows))
    lines.append("")

    lines.append("*HPC conferences — full paper deadlines*")
    hpc_table_rows = [[r.conference_name, r.paper_deadline_cell, r.conf_dates_cell, r.location] for r in hpc_rows]
    lines.append(render_markdown_table(TABLE_HEADERS, hpc_table_rows))
    lines.append("")

    lines.append(
        "_Rows marked “(from YYYY)” are estimated by shifting the most recent known schedule forward by +1 year. "
        "Always confirm on the official conference site._"
    )

    return "\n".join(lines)


def post_to_slack(text: str) -> None:
    require_env("SLACK_BOT_TOKEN", SLACK_BOT_TOKEN)
    require_env("SLACK_CHANNEL_ID", SLACK_CHANNEL_ID)

    if DRY_RUN:
        print(text)
        return

    client = WebClient(token=SLACK_BOT_TOKEN)
    try:
        client.chat_postMessage(channel=SLACK_CHANNEL_ID, text=text)
    except SlackApiError as e:
        err = e.response.get("error") if hasattr(e, "response") else str(e)
        raise SystemExit(f"Slack API error: {err}") from e


def main() -> None:
    ai_entries = load_ai_entries()
    ai_rows = build_ai_rows(ai_entries)

    hpc_entries = load_hpc_entries_from_nowlab()
    hpc_rows = build_hpc_rows(hpc_entries)

    msg = build_message(ai_rows, hpc_rows)
    if DEBUG:
        print(msg)
    else:
        post_to_slack(msg)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        raise
