import imaplib
import sys
import email
from email.header import decode_header
from email.utils import parsedate_to_datetime
import os
import requests
import json
from dotenv import load_dotenv
from abc import ABC, abstractmethod
import argparse
import glob
import re
import markdown
from xhtml2pdf import pisa
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables
# Global Settings (initialized in load_settings)
IMAP_SERVER = None
IMAP_PORT = 993
SMTP_SERVER = None
SMTP_PORT = 587
EMAIL_USER = None
EMAIL_PASSWORD = None
LLM_SERVICE = "ollama"
LLM_MODEL = None
DEFAULT_PROMPT = (
    "Please provide a comprehensive consolidated summary for the following group of emails, targeting a length of approximately 3 pages. "
    "Highlight key information from each if relevant, but provide a coherent overview.\n\n"
    "CRITICAL STRICTNESS INSTRUCTIONS:\n"
    "1. Do NOT make up any information or fill in blanks. All content must be explicitly derived from the emails.\n"
    "2. If context is missing, verify only what is present.\n\n"
    "CONTENT INSTRUCTIONS:\n"
    "1. Include any images that seem highly important to the context of the message (embedded as Markdown images).\n"
    "2. Preserve important links from the original emails.\n"
    "3. If there are any upcoming events mentioned, include a consolidated 'Upcoming Events' calendar section at the end.\n"
    "4. Include a 'Mailing Lists' section listing the sources of these emails and any sign-up/subscription links found."
)
LLM_PROMPT = DEFAULT_PROMPT
EMAIL_ALLOWLIST = []
EMAIL_BLOCKLIST = []

def load_settings(env_file=None):
    global IMAP_SERVER, IMAP_PORT, SMTP_SERVER, SMTP_PORT, EMAIL_USER, EMAIL_PASSWORD, LLM_SERVICE, LLM_MODEL, LLM_PROMPT, EMAIL_ALLOWLIST, EMAIL_BLOCKLIST
    
    load_dotenv(dotenv_path=env_file)
    
    IMAP_SERVER = os.getenv("IMAP_SERVER")
    IMAP_PORT = int(os.getenv("IMAP_PORT", 993))
    
    SMTP_SERVER = os.getenv("SMTP_SERVER")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 587))

    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
    LLM_SERVICE = os.getenv("LLM_SERVICE", "ollama").lower()
    LLM_MODEL = os.getenv("LLM_MODEL")
    LLM_PROMPT = os.getenv("LLM_PROMPT", DEFAULT_PROMPT)

    # Filtering Config
    EMAIL_ALLOWLIST = [d.strip().lower() for d in os.getenv("EMAIL_ALLOWLIST", "").split(",") if d.strip()]
    EMAIL_BLOCKLIST = [d.strip().lower() for d in os.getenv("EMAIL_BLOCKLIST", "").split(",") if d.strip()]

class SummarizerService(ABC):
    @abstractmethod
    def summarize(self, text):
        pass

class OllamaSummarizer(SummarizerService):
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = LLM_MODEL or "llama3"

    def summarize(self, text):
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": f"{LLM_PROMPT}\n\n{text}",
            "stream": False
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "Error: No response from Ollama")
        except Exception as e:
            return f"Error summarizing with Ollama: {e}"

def generate_pdf(text, filename):
    """
    Generates a PDF file from Markdown text.
    """
    try:
        # Convert Markdown to HTML
        html_content = markdown.markdown(text)
        
        # Simple CSS for better PDF rendering
        css = """
        <style>
            body { font-family: Helvetica, sans-serif; font-size: 12px; }
            h1 { color: #333; font-size: 18px; border-bottom: 1px solid #ccc; padding-bottom: 5px; }
            h2 { color: #444; font-size: 16px; margin-top: 20px; }
            p { line-height: 1.5; }
            pre { background-color: #f5f5f5; padding: 10px; border: 1px solid #ddd; }
        </style>
        """
        
        full_html = f"<html><head>{css}</head><body>{html_content}</body></html>"

        with open(filename, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(full_html, dest=pdf_file)
            
        if pisa_status.err:
            print(f"Error generating PDF: {pisa_status.err}")
        else:
            print(f"PDF summary saved to {filename}")
            
    except Exception as e:
        print(f"Error generating PDF: {e}")

def send_email_summary(to_address, subject, body_text, as_html=False):
    """
    Sends the summary via email using SMTP.
    If as_html is True, sends as multipart/alternative (Text + HTML).
    """
    if not SMTP_SERVER or not EMAIL_USER or not EMAIL_PASSWORD:
        raise ValueError("SMTP_SERVER, EMAIL_USER, and EMAIL_PASSWORD must be set to send emails via SMTP.")

    try:
        if as_html:
            msg = MIMEMultipart("alternative")
            
            # Record the MIME types of both parts - text/plain and text/html.
            part1 = MIMEText(body_text, 'plain')
            
            # Convert Markdown to HTML for the second part
            html_content = markdown.markdown(body_text)
            # Add some basic styling
            css = """
            <style>
                body { font-family: Helvetica, sans-serif; font-size: 14px; line-height: 1.6; color: #333; }
                h1 { color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }
                h2 { color: #34495e; margin_top: 20px; }
                a { color: #3498db; text-decoration: none; }
                a:hover { text-decoration: underline; }
                code { background-color: #f8f9fa; padding: 2px 4px; border-radius: 4px; font-family: monospace; }
                pre { background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
                blockquote { border-left: 4px solid #ddd; margin-left: 0; padding-left: 15px; color: #7f8c8d; }
                img { max-width: 100%; height: auto; }
            </style>
            """
            full_html = f"<html><head>{css}</head><body>{html_content}</body></html>"
            part2 = MIMEText(full_html, 'html')

            # Attach parts into message container.
            # According to RFC 2046, the last part of a multipart message, in this case
            # the HTML message, is best and preferred.
            msg.attach(part1)
            msg.attach(part2)
            
        else:
            msg = MIMEText(body_text)

        msg['Subject'] = subject
        msg['From'] = EMAIL_USER
        msg['To'] = to_address

        print(f"Connecting to SMTP server {SMTP_SERVER}:{SMTP_PORT}...")
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.send_message(msg)
        
        print(f"Email sent successfully to {to_address}")
            
    except Exception as e:
        print(f"Error sending email: {e}")

class OpenAISummarizer(SummarizerService):
    def __init__(self):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)
        self.model = LLM_MODEL or "gpt-3.5-turbo"

    def summarize(self, text):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates a consolidated summary from multiple emails."},
                    {"role": "user", "content": f"{LLM_PROMPT}\n\n{text}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error summarizing with OpenAI: {e}"

class AnthropicSummarizer(SummarizerService):
    def __init__(self):
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = LLM_MODEL or "claude-3-haiku-20240307"

    def summarize(self, text):
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": f"{LLM_PROMPT}\n\n{text}"}
                ]
            )
            return message.content[0].text
        except Exception as e:
            return f"Error summarizing with Anthropic: {e}"

class GeminiSummarizer(SummarizerService):
    def __init__(self):
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        genai.configure(api_key=api_key)
        # Using the model name as updated by the user previously
        self.model_name = LLM_MODEL or "gemini-3-pro-preview"
        self.model = genai.GenerativeModel(self.model_name)

    def summarize(self, text):
        try:
            response = self.model.generate_content(f"{LLM_PROMPT}\n\n{text}")
            return response.text
        except Exception as e:
            return f"Error summarizing with Gemini: {e}"


def get_summarizer():
    if LLM_SERVICE == "openai":
        return OpenAISummarizer()
    elif LLM_SERVICE == "anthropic":
        return AnthropicSummarizer()
    elif LLM_SERVICE == "gemini":
        return GeminiSummarizer()
    else:
        return OllamaSummarizer()


def connect_imap():
    """Connect to the IMAP server."""
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        mail.login(EMAIL_USER, EMAIL_PASSWORD)
        return mail
    except Exception as e:
        print(f"Error connecting to IMAP server: {e}")
        return None

def get_email_body(msg):
    """Extracts the body of the email."""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))

            if "attachment" not in content_disposition:
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        encoding = part.get_content_charset()
                        if encoding:
                            return payload.decode(encoding, errors="replace")
                        else:
                            return payload.decode(errors="replace")
                except Exception:
                    pass
    else:
        try:
           payload = msg.get_payload(decode=True)
           if payload:
               encoding = msg.get_content_charset()
               if encoding:
                    return payload.decode(encoding, errors="replace")
               else:
                    return payload.decode(errors="replace")
        except Exception:
            pass
    return ""

def format_email_for_summary(sender, subject, body):
    return f"--- EMAIL START ---\nFrom: {sender}\nSubject: {subject}\nBody:\n{body}\n--- EMAIL END ---\n"

def extract_original_sender(header_sender, body):
    """
    Extracts the 'innermost' sender from a forwarded email chain.
    It looks for 'From: <email>' patterns in the body.
    Returns the last found sender if present, otherwise the header sender.
    """
    # Regex to find email addresses in lines starting with "From:"
    # This handles typical forwarding headers like "From: John Doe <john@example.com>"
    # or "From: john@example.com"
    matches = re.findall(r'From:.*[\<]([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})[\>]|From:\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', body, re.IGNORECASE)
    
    if matches:
        # matches is a list of tuples due to the OR in regex groups.
        # We take the last match.
        last_match = matches[-1]
        # Valid email is in either group 1 or group 2
        return last_match[0] if last_match[0] else last_match[1]
    
    # Fallback to header sender if no forwarding detected
    if "<" in header_sender:
         # simple extraction from "Name <email>" format if needed for consistency, 
         # though simple string matching works for the check usually.
         # Let's clean it up to be safe.
         m = re.search(r'<(.+?)>', header_sender)
         return m.group(1) if m else header_sender
    return header_sender

def is_email_allowed(sender, allowlist, blocklist):
    """
    Checks if an email address is allowed based on the lists.
    Blocklist takes precedence.
    """
    sender = sender.lower().strip()
    domain = sender.split('@')[-1]
    
    # Check Blocklist first
    for blocked in blocklist:
        if blocked == sender or blocked == domain:
            return False
            
    # Check Allowlist
    if allowlist:
        start_allowed = False
        for allowed in allowlist:
            if allowed == sender or allowed == domain:
                start_allowed = True
                break
        if not start_allowed:
            return False
            
    return True

def process_local_emails(directory):
    """Process all email files in a directory and return formatted strings and dates."""
    print(f"Reading local emails from {directory}...")
    
    files = glob.glob(os.path.join(directory, "*"))
    email_files = [f for f in files if f.endswith('.eml') or f.endswith('.txt')]
    
    if not email_files:
        print("No .eml or .txt files found in the directory.")
        return [], []

    email_contents = []
    email_dates = []

    for file_path in email_files:
        try:
            print(f"Reading {file_path}...")
            with open(file_path, 'rb') as f:
                msg = email.message_from_binary_file(f)
                
                subject, encoding = decode_header(msg.get("Subject", "No Subject"))[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding if encoding else "utf-8", errors="replace")
                
                sender = msg.get("From", "Unknown Sender")
                
                # Extract date
                date_header = msg.get("Date")
                if date_header:
                    try:
                        dt = parsedate_to_datetime(date_header)
                        email_dates.append(dt)
                    except Exception:
                        pass
                
                body = get_email_body(msg)
                if body:
                    effective_sender = extract_original_sender(sender, body)
                    if is_email_allowed(effective_sender, EMAIL_ALLOWLIST, EMAIL_BLOCKLIST):
                        email_contents.append(format_email_for_summary(sender, subject, body))
                    else:
                        print(f"Skipping email from {effective_sender} (blocked or not in allowlist)")
                else:
                    print(f"Could not extract body from {file_path}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            
    return email_contents, email_dates

def main():
    parser = argparse.ArgumentParser(description="Email Summarizer")
    parser.add_argument("--local-dir", type=str, help="Path to directory containing email files to process locally")
    parser.add_argument("--output-file", "-o", type=str, help="[Deprecated] Path to save the summary output (alias for --output-file-md)")
    parser.add_argument("--output-file-md", type=str, help="Path to save the summary as a Markdown file")
    parser.add_argument("--output-file-pdf", type=str, help="Path to save the summary as a PDF file")
    parser.add_argument("--output-email", type=str, help="Email address to send the summary to")
    parser.add_argument("--html-email", action="store_true", help="Send the email as HTML (with plaintext fallback)")
    parser.add_argument("--keep-unread", "-k", action="store_true", help="Keep emails as unread in the inbox after processing")
    parser.add_argument("--env-file", type=str, help="Path to a custom .env file")
    args = parser.parse_args()

    # Create output directory if it doesn't exist (basic check for MD/PDF paths)
    for out_path in [args.output_file, args.output_file_md, args.output_file_pdf]:
        if out_path:
             out_dir = os.path.dirname(out_path)
             if out_dir and not os.path.exists(out_dir):
                 os.makedirs(out_dir, exist_ok=True)

    # Load settings after parsing arguments
    load_settings(args.env_file)

    try:
        summarizer = get_summarizer()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        return
    except ImportError as e:
        print(f"Dependency Error: {e}. Please install requirements.")
        return

    email_contents = []
    email_dates = []

    if args.local_dir:
        if not os.path.exists(args.local_dir):
            print(f"Error: Directory {args.local_dir} does not exist.")
            return
        email_contents, email_dates = process_local_emails(args.local_dir)
    
    else:
        # IMAP Logic
        if not all([IMAP_SERVER, EMAIL_USER, EMAIL_PASSWORD]):
            print("Please set IMAP_SERVER, EMAIL_USER, and EMAIL_PASSWORD in .env file, or use --local-dir")
            return

        mail = connect_imap()
        if not mail:
            return

        mail.select("inbox")

        status, messages = mail.search(None, "UNSEEN")
        if status != "OK":
            print("No unread emails found or error searching.")
            return

        email_ids = messages[0].split()
        if not email_ids:
            print("No unread emails.")
            return

        print(f"Found {len(email_ids)} unread emails. Processing...")

        for e_id in email_ids:
            try:
                # Fetch the email
                res, msg_data = mail.fetch(e_id, "(RFC822)")
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        
                        subject, encoding = decode_header(msg["Subject"])[0]
                        if isinstance(subject, bytes):
                            subject = subject.decode(encoding if encoding else "utf-8", errors="replace")
                        
                        sender = msg.get("From")
                        print(f"Reading email from {sender}: {subject}")

                        # Extract date
                        date_header = msg.get("Date")
                        if date_header:
                            try:
                                dt = parsedate_to_datetime(date_header)
                                email_dates.append(dt)
                            except Exception:
                                pass

                        body = get_email_body(msg)
                        if body:
                             effective_sender = extract_original_sender(sender, body)
                             if is_email_allowed(effective_sender, EMAIL_ALLOWLIST, EMAIL_BLOCKLIST):
                                 email_contents.append(format_email_for_summary(sender, subject, body))
                             else:
                                 print(f"Skipping email from {effective_sender} (blocked or not in allowlist)")
                        else:
                            print("Could not extract body.")
                
                if args.keep_unread:
                    # Mark as unseen again if we want to keep it unread
                    mail.store(e_id, '-FLAGS', '\\Seen')
                
            except Exception as e:
                print(f"Error processing email ID {e_id}: {e}")

        mail.close()
        mail.logout()

    if email_contents:
        print(f"\nGenerating aggregated summary for {len(email_contents)} emails using {LLM_SERVICE}...")
        combined_text = "\n".join(email_contents)
        summary = summarizer.summarize(combined_text)

        output_text = "\n" + "="*30 + "\n"
        output_text += "       AGGREGATED SUMMARY       \n"
        output_text += "="*30 + "\n\n"
        output_text += summary + "\n"
        
        # Handle generic/legacy --output-file as MD
        md_file = args.output_file_md or args.output_file
        
        if md_file:
            try:
                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(summary)
                print(f"\nMarkdown summary saved to {md_file}")
            except Exception as e:
                print(f"Error writing to file {md_file}: {e}")
        
        if args.output_file_pdf:
            print(f"\nGenerating PDF summary...")
            generate_pdf(summary, args.output_file_pdf)

        if args.output_email:
            print(f"\nSending summary via email...")
            
            email_subject = "Email Summarizer Report"
            if email_dates:
                try:
                    oldest_date = min(email_dates)
                    email_subject = f"{oldest_date.strftime('%m/%d/%Y')} Summary"
                except Exception as e:
                    print(f"Error calculating oldest date for subject: {e}")

            try:
                send_email_summary(args.output_email, email_subject, summary, as_html=args.html_email)
            except ValueError as e:
                print(f"Error: {e}")
                sys.exit(1)

        if not md_file and not args.output_file_pdf:
            print(output_text)

    else:
        print("No email content to summarize.")

if __name__ == "__main__":
    main()
