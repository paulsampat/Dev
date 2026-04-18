"""
Email skill — sends an email from the command line or Claude CLI.

Usage:
    python email_skill.py "Subject" "Body text"

Reads config from .env:
    EMAIL_FROM         sender address (e.g. you@hotmail.com)
    EMAIL_APP_PASSWORD SMTP password (or app password)
    EMAIL_TO           recipient work email
    EMAIL_SMTP         SMTP server (default: smtp-mail.outlook.com)
    EMAIL_PORT         SMTP port   (default: 587)
"""

import os
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()


def send_response(subject: str, body: str) -> bool:
    """Send agent response to work email. Returns True on success."""
    smtp_server = os.environ.get("EMAIL_SMTP", "smtp.gmail.com")
    smtp_port   = int(os.environ.get("EMAIL_PORT", "587"))
    from_addr   = os.environ["EMAIL_FROM"]
    password    = os.environ["EMAIL_APP_PASSWORD"]
    to_addr     = os.environ["EMAIL_TO"]

    msg = MIMEMultipart()
    msg["From"]    = from_addr
    msg["To"]      = to_addr
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(from_addr, password)
        server.sendmail(from_addr, to_addr, msg.as_string())

    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python email_skill.py \"Subject\" \"Body\"")
        sys.exit(1)
    subject = sys.argv[1]
    body = sys.argv[2]
    send_response(subject, body)
    print(f"Email sent to {os.environ['EMAIL_TO']}")
