import smtplib
from email.message import EmailMessage

def send_email(to_email, subject, body):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = "autoheal@yourcompany.com"
    msg["To"] = to_email
    msg.set_content(body)

    with smtplib.SMTP("smtp.yourprovider.com", 587) as smtp:
        smtp.starttls()
        smtp.login("youruser", "yourpass")
        smtp.send_message(msg)