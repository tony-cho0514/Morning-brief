#!/usr/bin/env python3
"""Send an HTML email newsletter to a list of recipients via SMTP."""

import argparse
import csv
import logging
import os
import smtplib
import ssl
import sys
import time
from email.message import EmailMessage
from email.utils import formataddr, make_msgid
from pathlib import Path
from string import Template

sys.path.insert(0, str(Path(__file__).resolve().parent))
from market_data import build_market_context  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("newsletter")


def load_recipients(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "email" not in (reader.fieldnames or []):
            raise ValueError("Recipients CSV must contain an 'email' column.")
        recipients = [row for row in reader if row.get("email", "").strip()]
    log.info("Loaded %d recipient(s) from %s", len(recipients), csv_path)
    return recipients


def render_template(template_str: str, context: dict) -> str:
    safe_context = {k: (v if v is not None else "") for k, v in context.items()}
    return Template(template_str).safe_substitute(safe_context)


def build_message(
    *,
    sender_name: str,
    sender_email: str,
    to_email: str,
    subject: str,
    html_body: str,
    text_body: str,
    reply_to: str | None,
    list_unsubscribe: str | None,
) -> EmailMessage:
    msg = EmailMessage()
    msg["From"] = formataddr((sender_name, sender_email))
    msg["To"] = to_email
    msg["Subject"] = subject
    msg["Message-ID"] = make_msgid()
    if reply_to:
        msg["Reply-To"] = reply_to
    if list_unsubscribe:
        msg["List-Unsubscribe"] = f"<{list_unsubscribe}>"
        msg["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")
    return msg


def html_to_text(html: str) -> str:
    import re

    text = re.sub(r"<\s*br\s*/?\s*>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"</\s*p\s*>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def open_smtp(host: str, port: int, username: str, password: str, use_ssl: bool, use_starttls: bool):
    context = ssl.create_default_context()
    if use_ssl:
        smtp = smtplib.SMTP_SSL(host, port, context=context, timeout=30)
    else:
        smtp = smtplib.SMTP(host, port, timeout=30)
        smtp.ehlo()
        if use_starttls:
            smtp.starttls(context=context)
            smtp.ehlo()
    if username:
        smtp.login(username, password)
    return smtp


def main() -> int:
    parser = argparse.ArgumentParser(description="Send an HTML email newsletter.")
    parser.add_argument("--recipients", type=Path, default=Path("newsletter/recipients.csv"))
    parser.add_argument("--template", type=Path, default=Path("newsletter/newsletter.html"))
    parser.add_argument(
        "--subject",
        default=os.environ.get("NEWSLETTER_SUBJECT", "오늘의 시장 브리핑 — $market_date"),
    )
    parser.add_argument(
        "--tickers",
        default=os.environ.get("NEWSLETTER_TICKERS", "SPY,QQQ,AAPL,NVDA,TSLA"),
        help="Comma-separated tickers to include.",
    )
    parser.add_argument(
        "--news-feed",
        default=os.environ.get(
            "NEWSLETTER_NEWS_FEED",
            "https://news.google.com/rss/search?q=US+stock+market+OR+SPY+OR+QQQ+OR+AAPL+OR+NVDA+OR+TSLA&hl=en-US&gl=US&ceid=US:en",
        ),
    )
    parser.add_argument("--news-limit", type=int, default=int(os.environ.get("NEWSLETTER_NEWS_LIMIT", "3")))
    parser.add_argument(
        "--schedule",
        default=os.environ.get("NEWSLETTER_SCHEDULE"),
        help="Run as a daemon and send daily at HH:MM (24h, local time). E.g. 07:30",
    )
    parser.add_argument("--sender-name", default=os.environ.get("SENDER_NAME", "Newsletter"))
    parser.add_argument("--sender-email", default=os.environ.get("SENDER_EMAIL"))
    parser.add_argument("--reply-to", default=os.environ.get("REPLY_TO"))
    parser.add_argument("--list-unsubscribe", default=os.environ.get("LIST_UNSUBSCRIBE"))
    parser.add_argument("--smtp-host", default=os.environ.get("SMTP_HOST"))
    parser.add_argument("--smtp-port", type=int, default=int(os.environ.get("SMTP_PORT", "587")))
    parser.add_argument("--smtp-username", default=os.environ.get("SMTP_USERNAME", ""))
    parser.add_argument("--smtp-password", default=os.environ.get("SMTP_PASSWORD", ""))
    parser.add_argument("--ssl", action="store_true", help="Use SMTPS (implicit TLS).")
    parser.add_argument("--no-starttls", action="store_true", help="Disable STARTTLS upgrade.")
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds to wait between sends.")
    parser.add_argument("--dry-run", action="store_true", help="Render and print without sending.")
    args = parser.parse_args()

    if not args.sender_email:
        log.error("Missing sender email. Set SENDER_EMAIL or pass --sender-email.")
        return 2
    if not args.dry_run and not args.smtp_host:
        log.error("Missing SMTP host. Set SMTP_HOST or pass --smtp-host.")
        return 2
    if not args.recipients.exists():
        log.error("Recipients file not found: %s", args.recipients)
        return 2
    if not args.template.exists():
        log.error("Template file not found: %s", args.template)
        return 2

    if args.schedule:
        return run_scheduled(args)
    return run_once(args)


def run_once(args) -> int:
    template_str = args.template.read_text(encoding="utf-8")
    recipients = load_recipients(args.recipients)
    if not recipients:
        log.warning("No recipients to send to. Exiting.")
        return 0

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    market_context = build_market_context(
        tickers=tickers,
        feed_url=args.news_feed,
        headline_limit=args.news_limit,
    )

    smtp = None
    if not args.dry_run:
        log.info("Connecting to SMTP %s:%d", args.smtp_host, args.smtp_port)
        smtp = open_smtp(
            host=args.smtp_host,
            port=args.smtp_port,
            username=args.smtp_username,
            password=args.smtp_password,
            use_ssl=args.ssl,
            use_starttls=not args.no_starttls and not args.ssl,
        )

    sent = 0
    failed: list[tuple[str, str]] = []
    try:
        for row in recipients:
            to_email = row["email"].strip()
            context = {**market_context, **row, "subject": args.subject}
            html_body = render_template(template_str, context)
            subject = render_template(args.subject, context)
            text_body = html_to_text(html_body)

            msg = build_message(
                sender_name=args.sender_name,
                sender_email=args.sender_email,
                to_email=to_email,
                subject=subject,
                html_body=html_body,
                text_body=text_body,
                reply_to=args.reply_to,
                list_unsubscribe=args.list_unsubscribe,
            )

            if args.dry_run:
                log.info("[dry-run] Would send to %s | subject=%r", to_email, subject)
                continue

            try:
                smtp.send_message(msg)
                sent += 1
                log.info("Sent to %s (%d/%d)", to_email, sent, len(recipients))
            except Exception as exc:
                failed.append((to_email, str(exc)))
                log.error("Failed to send to %s: %s", to_email, exc)

            if args.delay > 0:
                time.sleep(args.delay)
    finally:
        if smtp is not None:
            try:
                smtp.quit()
            except Exception:
                pass

    log.info("Done. Sent: %d, Failed: %d, Total: %d", sent, len(failed), len(recipients))
    if failed:
        log.warning("Failures:")
        for email, err in failed:
            log.warning("  %s -> %s", email, err)
        return 1
    return 0


def run_scheduled(args) -> int:
    try:
        import schedule
    except ImportError:
        log.error("The 'schedule' package is required for --schedule. Install it with: pip install schedule")
        return 2

    try:
        hh, mm = args.schedule.split(":")
        send_time = f"{int(hh):02d}:{int(mm):02d}"
    except Exception:
        log.error("Invalid --schedule value %r. Use HH:MM (24h).", args.schedule)
        return 2

    def job():
        log.info("Scheduled send triggered at %s", time.strftime("%Y-%m-%d %H:%M:%S"))
        try:
            run_once(args)
        except Exception as exc:
            log.exception("Scheduled send failed: %s", exc)

    schedule.every().day.at(send_time).do(job)
    log.info("Scheduler started. Sending daily at %s (local time). Press Ctrl+C to stop.", send_time)
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    sys.exit(main())
