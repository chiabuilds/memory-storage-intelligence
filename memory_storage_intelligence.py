import feedparser
import pandas as pd
import anthropic
import smtplib
import os
import json
import re
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import parsedate_to_datetime

FEEDS_FILE = "feeds.txt"
RECIPIENT_EMAIL = "chiabuilds@gmail.com"
SENDER_EMAIL = "austinkhchia@gmail.com"

NET_COLORS = {
    "bullish": "#c0392b",
    "bearish": "#1a7a4a",
    "balanced": "#e67e22",
}

NET_LABELS = {
    "bullish": "▲ BULLISH",
    "bearish": "▼ BEARISH",
    "balanced": "◆ BALANCED",
}

CURRENT_DAYS = 3   # current window: last N days
PRIOR_DAYS = 7     # prior window: N days before current window


def load_feeds(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def extract_outlet(title: str) -> str:
    parts = title.rsplit(" - ", 1)
    return parts[1].strip() if len(parts) == 2 else "Unknown"


def parse_pub_date(pub: str) -> datetime | None:
    try:
        return parsedate_to_datetime(pub)
    except Exception:
        return None


def fetch_articles(feed_urls: list[str]) -> pd.DataFrame:
    articles = []
    for url in feed_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = entry.get("title", "")
            articles.append({
                "title": title,
                "link": entry.get("link", ""),
                "published": entry.get("published", ""),
                "summary": entry.get("summary", ""),
                "source": feed.feed.get("title", url),
                "outlet": extract_outlet(title),
            })
    df = pd.DataFrame(articles)
    df["pub_dt"] = df["published"].apply(parse_pub_date)
    return df


def split_windows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, tuple, tuple]:
    now = datetime.now(timezone.utc)
    current_start = now - timedelta(days=CURRENT_DAYS)
    prior_end = current_start
    prior_start = current_start - timedelta(days=PRIOR_DAYS)

    current_df = df[df["pub_dt"] >= current_start].copy().reset_index(drop=True)
    prior_df = df[(df["pub_dt"] >= prior_start) & (df["pub_dt"] < prior_end)].copy().reset_index(drop=True)

    def fmt_range(d: pd.DataFrame) -> tuple[str, str]:
        dates = d["pub_dt"].dropna()
        if dates.empty:
            return "N/A", "N/A"
        return dates.min().strftime("%b %d"), dates.max().strftime("%b %d")

    return current_df, prior_df, fmt_range(current_df), fmt_range(prior_df)


def analyze_with_claude(df: pd.DataFrame) -> dict:
    client = anthropic.Anthropic()
    articles = df.head(50).reset_index(drop=True)
    numbered_headlines = "\n".join(
        f"[{i+1}] {row['title']} | {row['outlet']} | {row['published']} | {row['link']}"
        for i, (_, row) in enumerate(articles.iterrows())
    )

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": (
                "You are a semiconductor market analyst evaluating conditions from a MANUFACTURER/INVESTOR perspective.\n\n"
                "SCORING LOGIC — follow this exactly:\n"
                "- Supply WEAKER (manufacturers cutting/constraining output) + Demand STRONGER = prices RISE = BULLISH (score 60-100)\n"
                "- Supply STRONGER (oversupply) + Demand WEAKER = prices FALL = BEARISH (score 1-40)\n"
                "- Mixed signals = BALANCED (score 40-60)\n"
                "- 'supply_signal' refers to SUPPLY AVAILABILITY: 'weaker' = less supply available, 'stronger' = more supply available\n"
                "- Rising prices, tight supply, strong demand = BULLISH for manufacturers\n"
                "- Falling prices, oversupply, weak demand = BEARISH for manufacturers\n\n"
                "Return this exact JSON structure:\n"
                "{\n"
                '  "supply_signal": "stronger" | "weaker" | "same",\n'
                '  "supply_detail": "one sentence explaining ROOT CAUSE of supply signal (which manufacturers, capacity, inventory)",\n'
                '  "demand_signal": "stronger" | "weaker" | "same",\n'
                '  "demand_detail": "one sentence explaining ROOT CAUSE of demand signal (which sectors, applications)",\n'
                '  "net": "bullish" | "bearish" | "balanced",\n'
                '  "score": <integer 1-100, MUST follow scoring logic above>,\n'
                '  "key_takeaway": "2-3 sentence executive summary that MUST align with net and score, use [N] citations",\n'
                '  "sections": [\n'
                '    {"title": "Key Price Trends (DRAM, NAND)", "points": ["point with **bold key terms** and citation [1]", ...]},\n'
                '    {"title": "Supply Chain Signals", "points": ["point with **bold key terms**"]},\n'
                '    {"title": "Demand Drivers", "points": ["point with **bold key terms**"]},\n'
                '    {"title": "Notable Vendor Developments", "points": ["point with **bold key terms**"]}\n'
                '  ],\n'
                '  "cited_indices": [1, 3, 5]\n'
                "}\n\n"
                f"Articles:\n{numbered_headlines}"
            ),
        }],
    )

    raw = message.content[0].text
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        raise ValueError("Claude did not return valid JSON")

    result = json.loads(json_match.group())
    result["articles"] = articles
    return result


def analyze_score_only(df: pd.DataFrame) -> dict:
    """Lightweight Claude call for prior window — score + net only."""
    client = anthropic.Anthropic()
    articles = df.head(50).reset_index(drop=True)
    headlines = "\n".join(
        f"[{i+1}] {row['title']} | {row['outlet']} | {row['published']}"
        for i, (_, row) in enumerate(articles.iterrows())
    )

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": (
                "You are a semiconductor market analyst. Score these headlines from a MANUFACTURER/INVESTOR perspective.\n"
                "SCORING: tight supply + strong demand = BULLISH (60-100). Oversupply + weak demand = BEARISH (1-40). Mixed = BALANCED (40-60).\n"
                "Return ONLY: {\"net\": \"bullish\"|\"bearish\"|\"balanced\", \"score\": <1-100>}\n\n"
                f"Headlines:\n{headlines}"
            ),
        }],
    )

    raw = message.content[0].text
    json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"net": "balanced", "score": 50}


def build_footnotes(articles: pd.DataFrame, cited_indices: list[int]) -> dict:
    footnotes_by_outlet = {}
    for idx in cited_indices:
        if idx < 1 or idx > len(articles):
            continue
        row = articles.iloc[idx - 1]
        outlet = row["outlet"]
        if outlet not in footnotes_by_outlet:
            footnotes_by_outlet[outlet] = []
        footnotes_by_outlet[outlet].append({
            "num": idx,
            "title": row["title"].rsplit(" - ", 1)[0].strip(),
            "link": row["link"],
            "published": row["published"],
        })
    return dict(sorted(footnotes_by_outlet.items()))


def strip_bold_after_colon(text: str) -> str:
    """Keep bold before first colon, remove bold markers after it."""
    if ": " in text:
        before, after = text.split(": ", 1)
        after = re.sub(r"\*\*(.+?)\*\*", r"\1", after)
        return f"{before}: {after}"
    return text


def md_to_html(text: str, articles: pd.DataFrame) -> str:
    """Convert markdown bold and citations to HTML."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    def replace(match):
        idx = int(match.group(1))
        if 1 <= idx <= len(articles):
            link = articles.iloc[idx - 1]["link"]
            return f'<a href="{link}" style="color:#2980b9;">[{idx}]</a>'
        return match.group(0)
    return re.sub(r"\[(\d+)\]", replace, text)


def fmt_score_line(curr_score: int, curr_range: tuple, prior_score: int | None, prior_range: tuple | None) -> tuple[str, str]:
    curr_label = f"{curr_range[0]} to {curr_range[1]}"
    if prior_score is not None and prior_range and prior_range[0] != "N/A":
        prior_label = f"{prior_range[0]} to {prior_range[1]}"
        delta = curr_score - prior_score
        arrow = "▲" if delta > 0 else "▼" if delta < 0 else "→"
        md = f"**Prior ({prior_label}): {prior_score} → Current ({curr_label}): {curr_score}** _{arrow} {delta:+d}_"
        html = (
            f'<span style="font-size:16px;font-weight:bold;">'
            f'Prior ({prior_label}): {prior_score} &nbsp;→&nbsp; Current ({curr_label}): {curr_score}'
            f'</span>'
            f'&nbsp;<span style="font-size:13px;">{arrow} <sup>{delta:+d}</sup></span>'
        )
    else:
        md = f"**Current ({curr_label}): {curr_score}/100** _(no prior data)_"
        html = f'<span style="font-size:16px;font-weight:bold;">Current ({curr_label}): {curr_score}/100</span> <span style="font-size:12px;color:#ccc;">(no prior data)</span>'
    return md, html


def build_markdown(result: dict, date_str: str, curr_range: tuple, prior_score: int | None, prior_range: tuple | None) -> str:
    net = result["net"]
    supply = result.get("supply_signal", "same").upper()
    demand = result.get("demand_signal", "same").upper()
    net_label = NET_LABELS[net]
    articles = result["articles"]
    footnotes_by_outlet = build_footnotes(articles, result.get("cited_indices", []))
    score = result.get("score", 50)
    score_md, _ = fmt_score_line(score, curr_range, prior_score, prior_range)

    lines = [
        f"# Memory & Storage Market Update — {date_str}",
        f"\n## Supply: {supply} | Demand: {demand} → Net: {net_label}",
        score_md,
        f"- **Supply:** {result.get('supply_detail', '')}",
        f"- **Demand:** {result.get('demand_detail', '')}\n",
        f"### Key Takeaway\n{result['key_takeaway']}\n",
        "---\n",
    ]

    for section in result["sections"]:
        lines.append(f"## {section['title']}\n")
        for point in section["points"]:
            lines.append(f"- {strip_bold_after_colon(point)}")
        lines.append("")

    lines.append("---\n")
    lines.append("## Sources\n")
    lines.append("| # | Outlet | Date | Article |")
    lines.append("|---|--------|------|---------|")
    for outlet, entries in footnotes_by_outlet.items():
        for i, e in enumerate(entries):
            try:
                pub_date = parsedate_to_datetime(e["published"]).strftime("%b %d, %Y")
            except Exception:
                pub_date = e["published"]
            outlet_col = outlet if i == 0 else ""
            lines.append(f"| [{e['num']}] | {outlet_col} | {pub_date} | [{e['title']}]({e['link']}) |")

    return "\n".join(lines)


def build_html(result: dict, date_str: str, curr_range: tuple, prior_score: int | None, prior_range: tuple | None) -> str:
    net = result["net"]
    color = NET_COLORS[net]
    net_label = NET_LABELS[net]
    supply = result.get("supply_signal", "same").upper()
    demand = result.get("demand_signal", "same").upper()
    articles = result["articles"]
    footnotes_by_outlet = build_footnotes(articles, result.get("cited_indices", []))
    score = result.get("score", 50)
    _, score_html = fmt_score_line(score, curr_range, prior_score, prior_range)

    sections_html = ""
    for section in result["sections"]:
        points = "".join(f"<li>{md_to_html(strip_bold_after_colon(p), articles)}</li>" for p in section["points"])
        sections_html += f"<h2 style='color:#1a1a2e;'>{section['title']}</h2><ul>{points}</ul>"

    rows = ""
    for outlet, entries in footnotes_by_outlet.items():
        for i, e in enumerate(entries):
            try:
                pub_date = parsedate_to_datetime(e["published"]).strftime("%b %d, %Y")
            except Exception:
                pub_date = e["published"]
            outlet_cell = (
                f'<td style="padding:4px 8px;font-weight:bold;vertical-align:top;" rowspan="{len(entries)}">{outlet}</td>'
                if i == 0 else ""
            )
            rows += (
                f'<tr>'
                f'<td style="padding:4px 8px;color:#555;vertical-align:top;">[{e["num"]}]</td>'
                f'{outlet_cell}'
                f'<td style="padding:4px 8px;color:#888;white-space:nowrap;font-size:12px;vertical-align:top;">{pub_date}</td>'
                f'<td style="padding:4px 8px;vertical-align:top;"><a href="{e["link"]}" style="color:#2980b9;">{e["title"]}</a></td>'
                f'</tr>'
            )
    footnotes_html = (
        f'<table style="width:100%;border-collapse:collapse;font-size:13px;">'
        f'<thead><tr style="background:#f0f0f0;">'
        f'<th style="padding:6px 8px;text-align:left;">#</th>'
        f'<th style="padding:6px 8px;text-align:left;">Outlet</th>'
        f'<th style="padding:6px 8px;text-align:left;">Date</th>'
        f'<th style="padding:6px 8px;text-align:left;">Article</th>'
        f'</tr></thead><tbody>{rows}</tbody></table>'
    )

    key_takeaway_html = md_to_html(result["key_takeaway"], articles)

    return f"""
    <html><body style="font-family:Arial,sans-serif;max-width:700px;margin:auto;padding:20px;">
      <h1 style="color:#1a1a2e;">Memory & Storage Market Update</h1>
      <p style="color:#888;">{date_str}</p>

      <div style="background:{color};color:white;padding:16px 20px;border-radius:8px;">
        <div style="font-size:13px;letter-spacing:1px;opacity:0.85;">SUPPLY: {supply} &nbsp;|&nbsp; DEMAND: {demand} &nbsp;→&nbsp; NET</div>
        <div style="font-size:24px;font-weight:bold;margin-top:4px;">{net_label}</div>
        <div style="margin-top:8px;">{score_html}</div>
        <div style="font-size:13px;margin-top:10px;opacity:0.9;">&#8226; Supply: {result.get('supply_detail', '')}</div>
        <div style="font-size:13px;opacity:0.9;">&#8226; Demand: {result.get('demand_detail', '')}</div>
      </div>

      <div style="background:#f5f5f5;border-left:4px solid {color};padding:12px 16px;margin:16px 0;">
        <strong>Key Takeaway:</strong> {key_takeaway_html}
      </div>

      <hr>
      {sections_html}
      <hr>

      <h2>Sources</h2>
      {footnotes_html}

      <hr>
      <p style="color:#aaa;font-size:12px;">Generated by sourcing-intelligence script</p>
    </body></html>
    """


def send_email(subject: str, markdown_body: str, html_body: str):
    app_password = os.environ.get("GMAIL_APP_PASSWORD")
    if not app_password:
        print("Warning: GMAIL_APP_PASSWORD not set, skipping email.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    msg.attach(MIMEText(markdown_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SENDER_EMAIL, app_password)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
    print(f"Email sent to {RECIPIENT_EMAIL}")


def main():
    date_str = datetime.now().strftime("%Y-%m-%d")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Fetching feeds...")

    feed_urls = load_feeds(FEEDS_FILE)
    df = fetch_articles(feed_urls)
    print(f"Fetched {len(df)} articles from {len(feed_urls)} feeds.")

    if df.empty:
        print("No articles found.")
        return

    current_df, prior_df, curr_range, prior_range = split_windows(df)
    print(f"Current window ({curr_range[0]} to {curr_range[1]}): {len(current_df)} articles")
    print(f"Prior window ({prior_range[0]} to {prior_range[1]}): {len(prior_df)} articles\n")

    if current_df.empty:
        print("No articles in current window.")
        return

    print("Analyzing current window with Claude...")
    result = analyze_with_claude(current_df)

    prior_score = None
    if not prior_df.empty:
        print("Analyzing prior window with Claude...")
        prior_result = analyze_score_only(prior_df)
        prior_score = prior_result.get("score")
        print(f"Prior score: {prior_score}")
    else:
        print("No prior window articles found — skipping prior score.")
        prior_range = None

    markdown = build_markdown(result, date_str, curr_range, prior_score, prior_range)
    html = build_html(result, date_str, curr_range, prior_score, prior_range)

    report_file = f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    with open(report_file, "w") as f:
        f.write(markdown)
    print(f"Report saved to {report_file}")

    send_email(f"Memory & Storage Market Update — {date_str}", markdown, html)

    print("\n" + "=" * 60)
    print(markdown)
    print("=" * 60)


if __name__ == "__main__":
    main()
