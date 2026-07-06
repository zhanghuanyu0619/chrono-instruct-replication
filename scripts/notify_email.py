#!/usr/bin/env python3
"""Send a short notification email (e.g. "vintage 1999 finished"). Best-effort.

Used by scripts/train_all_vintages.sh to ping you as each vintage completes, so
you don't have to babysit a multi-hour sweep. Credentials are read from the
ENVIRONMENT (never committed):

  NOTIFY_SMTP_USER   sender address, e.g. zhanghuanyu0619@gmail.com
  NOTIFY_SMTP_PASS   an APP PASSWORD for that account (NOT your login password;
                     Gmail: account needs 2FA, then create one under
                     Google Account -> Security -> App passwords)
  NOTIFY_TO          recipient (optional; defaults to NOTIFY_SMTP_USER)
  NOTIFY_SMTP_HOST   default smtp.gmail.com
  NOTIFY_SMTP_PORT   default 587 (STARTTLS)

If the creds are absent it prints a hint and exits 0. It ALWAYS exits 0 — a
notification failure must never abort a training sweep.

  python scripts/notify_email.py --subject "[chrono] 1999 done" --body "$(cat summary.json)"
"""
import argparse
import os
import smtplib
import ssl
import sys
from email.message import EmailMessage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--body", default="")
    args = ap.parse_args()

    user = os.environ.get("NOTIFY_SMTP_USER")
    pw = os.environ.get("NOTIFY_SMTP_PASS")
    if not (user and pw):
        print("[notify] NOTIFY_SMTP_USER / NOTIFY_SMTP_PASS not set -> skipping email "
              "(export them to enable). Continuing.")
        return
    to = os.environ.get("NOTIFY_TO", user)
    host = os.environ.get("NOTIFY_SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("NOTIFY_SMTP_PORT", "587"))

    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = to
    msg["Subject"] = args.subject
    msg.set_content(args.body or args.subject)

    try:
        with smtplib.SMTP(host, port, timeout=30) as s:
            s.starttls(context=ssl.create_default_context())
            s.login(user, pw)
            s.send_message(msg)
        print(f"[notify] emailed '{args.subject}' -> {to}")
    except Exception as e:  # never propagate; the sweep must keep going
        print(f"[notify] email failed ({type(e).__name__}: {e}); continuing.", file=sys.stderr)


if __name__ == "__main__":
    main()
