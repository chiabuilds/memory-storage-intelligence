import feedparser
import pandas as pd
import anthropic
import smtplib
import os
import re
import json
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import parsedate_to_datetime

FEEDS_FILE = "feeds.txt"
RECIPIENT_EMAIL = "austin.chia@hpe.com"
SENDER_EMAIL = "chiabuilds@gmail.com"
WINDOW_HOURS = 24

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
                "outlet": extract_outlet(title),
            })
    df = pd.DataFrame(articles)
    df["pub_dt"] = df["published"].apply(parse_pub_date)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=WINDOW_HOURS)
    return df[df["pub_dt"] >= cutoff].reset_index(drop=True)


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
                "You are a semiconductor market analyst evaluating conditions from a MANUFACTURER/INVESTOR perspective.\n"
                "Focus ONLY on NAND flash, DRAM/memory, and AI chips. Ignore unrelated semiconductor topics.\n\n"
                "SCORING LOGIC — follow this exactly:\n"
                "- Supply WEAKER (manufacturers cutting/constraining output) + Demand STRONGER = prices RISE = BULLISH (score 60-100)\n"
                "- Supply STRONGER (oversupply) + Demand WEAKER = prices FALL = BEARISH (score 1-40)\n"
                "- Mixed signals = BALANCED (score 40-60)\n"
                "- Rising prices, tight supply, strong demand = BULLISH for manufacturers\n"
                "- Falling prices, oversupply, weak demand = BEARISH for manufacturers\n\n"
                "Return this exact JSON structure:\n"
                "{\n"
                '  "supply_signal": "stronger" | "weaker" | "same",\n'
                '  "supply_detail": "one sentence on ROOT CAUSE for NAND/DRAM/AI chip supply, bold 1-2 key terms with **bold**",\n'
                '  "demand_signal": "stronger" | "weaker" | "same",\n'
                '  "demand_detail": "one sentence on ROOT CAUSE for NAND/DRAM/AI chip demand, bold 1-2 key terms with **bold**",\n'
                '  "net": "bullish" | "bearish" | "balanced",\n'
                '  "score": <integer 1-100, MUST follow scoring logic above>,\n'
                '  "sections": [\n'
                '    {"title": "Price Trends", "points": ["**Bold label**: description with **1-2 key highlights bolded** and citation [N]", ...]},\n'
                '    {"title": "Demand", "points": ["**Bold label**: description with **1-2 key highlights bolded** and citation [N]"]},\n'
                '    {"title": "Supply & Inventory", "points": ["**Bold label**: description focused on top fab makers (Samsung, SK Hynix, Micron, TSMC, Kioxia, WD) and largest module houses (Kingston, Crucial, Corsair, ADATA) with **1-2 key highlights bolded** and citation [N]"]},\n'
                '    {"title": "Other", "points": ["**Bold label**: description with **1-2 key highlights bolded**"]}\n'
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
    if ": " in text:
        before, after = text.split(": ", 1)
        after = re.sub(r"\*\*(.+?)\*\*", r"\1", after)
        return f"{before}: {after}"
    return text


def expand_citations(text: str) -> str:
    def expand(match):
        nums = [n.strip() for n in match.group(1).split(",")]
        return "".join(f"[{n}]" for n in nums if n.isdigit())
    return re.sub(r"\[(\d+(?:,\s*\d+)+)\]", expand, text)


def bold_to_html(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)


def md_to_html(text: str, articles: pd.DataFrame) -> str:
    text = expand_citations(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    def replace(match):
        idx = int(match.group(1))
        if 1 <= idx <= len(articles):
            link = articles.iloc[idx - 1]["link"]
            return f'<a href="{link}" style="color:#2980b9;">[{idx}]</a>'
        return match.group(0)
    return re.sub(r"\[(\d+)\]", replace, text)


def bullet_to_html(text: str, articles: pd.DataFrame) -> str:
    def linkify(t):
        t = expand_citations(t)
        def replace(match):
            idx = int(match.group(1))
            if 1 <= idx <= len(articles):
                link = articles.iloc[idx - 1]["link"]
                return f'<a href="{link}" style="color:#2980b9;">[{idx}]</a>'
            return match.group(0)
        return re.sub(r"\[(\d+)\]", replace, t)
    if ": " in text:
        before, after = text.split(": ", 1)
        before_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", before)
        after_html = re.sub(r"\*\*(.+?)\*\*", r'<strong style="color:#2980b9;">\1</strong>', after)
        return f"{before_html}: {linkify(after_html)}"
    return linkify(re.sub(r"\*\*(.+?)\*\*", r'<strong style="color:#2980b9;">\1</strong>', text))


def build_markdown(result: dict, date_str: str) -> str:
    net = result["net"]
    supply = result.get("supply_signal", "same").upper()
    demand = result.get("demand_signal", "same").upper()
    net_label = NET_LABELS[net]
    score = result.get("score", 50)
    articles = result["articles"]
    footnotes_by_outlet = build_footnotes(articles, result.get("cited_indices", []))

    lines = [
        f"# Memory & Storage Market: Daily Update — {date_str}",
        f"\n## Supply: {supply} | Demand: {demand} → Net: {net_label}",
        f"**Score: {score}/100**",
        f"- **Supply:** {result.get('supply_detail', '')}",
        f"- **Demand:** {result.get('demand_detail', '')}\n",
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


def build_html(result: dict, date_str: str) -> str:
    net = result["net"]
    color = NET_COLORS[net]
    net_label = NET_LABELS[net]
    supply = result.get("supply_signal", "same").upper()
    demand = result.get("demand_signal", "same").upper()
    score = result.get("score", 50)
    articles = result["articles"]
    footnotes_by_outlet = build_footnotes(articles, result.get("cited_indices", []))

    sections_html = ""
    for section in result["sections"]:
        points = "".join(
            f"<li>{bullet_to_html(p, articles)}</li>"
            for p in section["points"]
        )
        sections_html += f"<h2 style='color:#1a1a2e;font-size:13px;font-family:Aptos,Arial,sans-serif;margin-bottom:4px;'>{section['title']}</h2><ul style='font-size:13px;font-family:Aptos,Arial,sans-serif;margin-top:4px;'>{points}</ul>"

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

    return f"""
    <html><body style="font-family:Aptos,Arial,sans-serif;max-width:700px;margin:auto;padding:20px;">
      <h1 style="color:#1a1a2e;font-family:Aptos,Arial,sans-serif;font-size:18px;">Memory & Storage Market: Daily Update</h1>
      <p style="color:#888;font-family:Aptos,Arial,sans-serif;font-size:13px;">{date_str} — Last 24 Hours</p>

      <div style="padding:12px 0 8px 0;">
        <div style="font-size:13px;font-family:Aptos,Arial,sans-serif;color:#555;letter-spacing:1px;">SUPPLY: {supply} &nbsp;|&nbsp; DEMAND: {demand} &nbsp;→&nbsp; NET</div>
        <div style="font-size:22px;font-weight:bold;margin-top:4px;font-family:Aptos,Arial,sans-serif;color:{color};">{net_label} &nbsp; <span style="font-size:18px;">{score}/100</span></div>
        <div style="font-size:13px;font-family:Aptos,Arial,sans-serif;margin-top:8px;color:#333;">&#8226; Supply: {md_to_html(result.get('supply_detail', ''), articles)}</div>
        <div style="font-size:13px;font-family:Aptos,Arial,sans-serif;color:#333;">&#8226; Demand: {md_to_html(result.get('demand_detail', ''), articles)}</div>
      </div>

      <hr>
      {sections_html}
      <hr>

      <h2 style="font-size:13px;font-family:Aptos,Arial,sans-serif;">Sources</h2>
      {footnotes_html}

      <hr>
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
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Fetching last 24hrs of articles...")

    feed_urls = load_feeds(FEEDS_FILE)
    df = fetch_articles(feed_urls)
    print(f"Found {len(df)} articles in last 24 hours.\n")

    if df.empty:
        print("No articles found in last 24 hours.")
        return

    print("Analyzing with Claude...")
    result = analyze_with_claude(df)

    markdown = build_markdown(result, date_str)
    html = build_html(result, date_str)

    report_file = f"daily_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    with open(report_file, "w") as f:
        f.write(markdown)
    print(f"Report saved to {report_file}")

    send_email(f"Memory & Storage Market: Daily Update — {date_str}", markdown, html)

    print("\n" + "=" * 60)
    print(markdown)
    print("=" * 60)


if __name__ == "__main__":
    main()
