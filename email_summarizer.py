import imaplib
import email
from email.header import decode_header
import os
import requests
import json
from dotenv import load_dotenv
from abc import ABC, abstractmethod
import argparse
import glob

# Load environment variables
load_dotenv()

IMAP_SERVER = os.getenv("IMAP_SERVER")
IMAP_PORT = int(os.getenv("IMAP_PORT", 993))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
LLM_SERVICE = os.getenv("LLM_SERVICE", "ollama").lower()
LLM_MODEL = os.getenv("LLM_MODEL") # Default will be handled by service
DEFAULT_PROMPT = "Please provide a single consolidated summary for the following group of emails. Highlight key information from each if relevant, but provide a coherent overview. Preserve important links from the original emails. If there are any upcoming events mentioned, please include a consolidated 'Upcoming Events' calendar section at the end. Finally, include a 'Mailing Lists' section listing the sources of these emails and any sign-up/subscription links found."
LLM_PROMPT = os.getenv("LLM_PROMPT", DEFAULT_PROMPT)

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

def process_local_emails(directory):
    """Process all email files in a directory and return formatted strings."""
    print(f"Reading local emails from {directory}...")
    
    files = glob.glob(os.path.join(directory, "*"))
    email_files = [f for f in files if f.endswith('.eml') or f.endswith('.txt')]
    
    if not email_files:
        print("No .eml or .txt files found in the directory.")
        return []

    email_contents = []
    for file_path in email_files:
        try:
            print(f"Reading {file_path}...")
            with open(file_path, 'rb') as f:
                msg = email.message_from_binary_file(f)
                
                subject, encoding = decode_header(msg.get("Subject", "No Subject"))[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding if encoding else "utf-8", errors="replace")
                
                sender = msg.get("From", "Unknown Sender")
                
                body = get_email_body(msg)
                if body:
                    email_contents.append(format_email_for_summary(sender, subject, body))
                else:
                    print(f"Could not extract body from {file_path}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            
    return email_contents

def main():
    parser = argparse.ArgumentParser(description="Email Summarizer")
    parser.add_argument("--local-dir", type=str, help="Path to directory containing email files to process locally")
    parser.add_argument("--output-file", "-o", type=str, help="Path to save the summary output (e.g., summary.md)")
    parser.add_argument("--keep-unread", "-k", action="store_true", help="Keep emails as unread in the inbox after processing")
    args = parser.parse_args()

    try:
        summarizer = get_summarizer()
    except ValueError as e:
        print(f"Configuration Error: {e}")
        return
    except ImportError as e:
        print(f"Dependency Error: {e}. Please install requirements.")
        return

    email_contents = []

    if args.local_dir:
        if not os.path.exists(args.local_dir):
            print(f"Error: Directory {args.local_dir} does not exist.")
            return
        email_contents = process_local_emails(args.local_dir)
    
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

                        body = get_email_body(msg)
                        if body:
                             email_contents.append(format_email_for_summary(sender, subject, body))
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
        print(f"\nGeneratig aggregated summary for {len(email_contents)} emails using {LLM_SERVICE}...")
        combined_text = "\n".join(email_contents)
        summary = summarizer.summarize(combined_text)

        output_text = "\n" + "="*30 + "\n"
        output_text += "       AGGREGATED SUMMARY       \n"
        output_text += "="*30 + "\n\n"
        output_text += summary + "\n"

        if args.output_file:
            try:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    f.write(summary) # Write just the summary content, arguably cleaner for markdown
                print(f"\nSummary saved to {args.output_file}")
            except Exception as e:
                print(f"Error writing to file {args.output_file}: {e}")
                print(output_text) # Fallback output
        else:
            print(output_text)

    else:
        print("No email content to summarize.")

if __name__ == "__main__":
    main()
