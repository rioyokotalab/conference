import os
import re
from datetime import datetime, timedelta
import requests
import yaml
from dateutil import parser, tz
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

DATA_URL = os.getenv(
    "DEADLINES_YAML_URL",
    "https://raw.githubusercontent.com/abhshkdz/ai-deadlines/gh-pages/_data/conferences.yml",
)

#SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
#SLACK_CHANNEL_ID = os.environ["SLACK_CHANNEL_ID"]

DEFAULT_WATCH = {
    "NeurIPS", "ICML", "ICLR",
    "AAAI", "IJCAI",
    "ACL", "EMNLP", "NAACL",
    "CVPR", "ICCV", "ECCV",
    "KDD", "SIGIR", "AISTATS",
    "ICRA", "IROS",
}

WATCH_TITLES = set(
    t.strip() for t in os.getenv("WATCH_TITLES", "").split(",") if t.strip()
) or DEFAULT_WATCH

DAYS_AHEAD = int(os.getenv("DAYS_AHEAD", "60"))
MAX_ITEMS = int(os.getenv("MAX_ITEMS", "15"))

def parse_timezone(tz_str: str):
    if not tz_str:
        return tz.UTC
    tz_str = str(tz_str).strip()
    if tz_str in {"UTC", "GMT"}:
        return tz.UTC

    m = re.match(r"^(UTC|GMT)([+-]\d{1,2})(?::?(\d{2}))?$", tz_str)
    if m:
        hours = int(m.group(2))
        mins = int(m.group(3) or "0")
        sign = 1 if hours >= 0 else -1
        offset_seconds = hours * 3600 + sign * mins * 60
        return tz.tzoffset(None, offset_seconds)

    tzinfo = tz.gettz(tz_str)
    return tzinfo or tz.UTC

def parse_deadline(deadline_value, tzinfo):
    if deadline_value is None:
        return None
    dt = parser.parse(str(deadline_value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tzinfo)
    return dt

def slack_date(dt_utc: datetime, fallback: str):
    ts = int(dt_utc.timestamp())
    token_string = "{date_short_pretty} at {time}"
    return f"<!date^{ts}^{token_string}|{fallback}>"

def load_conferences():
    r = requests.get(DATA_URL, timeout=30)
    r.raise_for_status()
    return yaml.safe_load(r.text)

def choose_next_deadlines(confs):
    now_utc = datetime.now(tz=tz.UTC)
    horizon_utc = now_utc + timedelta(days=DAYS_AHEAD)
    next_by_title = {}

    for c in confs:
        title = (c.get("title") or "").strip()
        if title not in WATCH_TITLES:
            continue

        tzinfo = parse_timezone(c.get("timezone"))
        deadline_dt = parse_deadline(c.get("deadline"), tzinfo)
        if not deadline_dt:
            continue

        deadline_utc = deadline_dt.astimezone(tz.UTC)
        if not (now_utc <= deadline_utc <= horizon_utc):
            continue

        if title not in next_by_title or deadline_utc < next_by_title[title][0]:
            next_by_title[title] = (deadline_utc, deadline_dt, c)

    items = sorted(next_by_title.items(), key=lambda kv: kv[1][0])
    return items[:MAX_ITEMS]

def format_message(items):
    if not items:
        return f"*AI conference deadlines:* nothing due in the next {DAYS_AHEAD} days for your watchlist."

    lines = [f"*AI conference deadlines (next {DAYS_AHEAD} days):*", ""]
    for title, (deadline_utc, deadline_local, c) in items:
        year = c.get("year", "")
        link = c.get("link", "")
        tz_str = c.get("timezone", "UTC")
        fallback = f"{deadline_local.strftime('%Y-%m-%d %H:%M')} ({tz_str})"
        when = slack_date(deadline_utc, fallback=fallback)
        link_txt = f"<{link}|CFP>" if link else ""
        note = c.get("note")
        note_txt = f" — {note}" if note else ""
        lines.append(f"• *{title} {year}* — {when} {link_txt}{note_txt}")

    lines += ["", "_Always confirm on the official CFP page; deadlines move._"]
    return "\n".join(lines)

#def post_to_slack(text):
#    client = WebClient(token=SLACK_BOT_TOKEN)
#    try:
#        client.chat_postMessage(channel=SLACK_CHANNEL_ID, text=text)
#    except SlackApiError as e:
#        raise SystemExit(f"Slack API error: {e.response['error']}") from e

def main():
    confs = load_conferences()
    items = choose_next_deadlines(confs)
    msg = format_message(items)
    print(msg)
#    post_to_slack(msg)

if __name__ == "__main__":
    main()
