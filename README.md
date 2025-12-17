# Email Summarizer

A Python tool that connects to an email account via IMAP or reads local email files, and generates concise summaries using Large Language Models. It supports **Ollama** (local), **OpenAI**, **Gemini**, and **Anthropic**.

## Setup

1.  **Clone the repository** (if you haven't already).
2.  **Set up a virtual environment** (Recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The script uses a `.env` file to manage secrets and settings. It automatically loads this file when it runs, so you don't need to manually set environment variables in your shell.

1.  **Create the configuration file**:
    ```bash
    cp .env.example .env
    ```
2.  **Edit `.env`**:
    Open the `.env` file in your text editor and fill in your details (API keys, email credentials, etc.).


```env
# --- Email Server Settings (Required for IMAP) ---
# --- Email Server Settings (Required for IMAP) ---
IMAP_SERVER=imap.gmail.com   # e.g., imap.mail.yahoo.com
IMAP_PORT=993                # Default is 993 (SSL)

# --- SMTP Settings (Required for sending emails) ---
SMTP_SERVER=smtp.gmail.com   # e.g., smtp.gmail.com
SMTP_PORT=587                # Default is 587 (TLS)

EMAIL_USER=your_email@example.com
EMAIL_PASSWORD=your_app_password

### Gmail Setup Guide
To use a Gmail account, you must use an **App Password** instead of your regular login password:
1.  Enable **2-Step Verification** on your Google Account.
2.  Go to [Google Account Security](https://myaccount.google.com/security).
3.  Under "How you sign in to Google", search for or select **App passwords**.
4.  Create a new App Password (e.g., name it "Email Summarizer").
5.  Use this 16-character code as your `EMAIL_PASSWORD` in the `.env` file.
6.  Set `IMAP_SERVER=imap.gmail.com` and `IMAP_PORT=993`.


# --- General Settings ---
LLM_SERVICE=ollama           # Options: ollama, openai, gemini, anthropic
LLM_MODEL=                   # Optional: Override default model for the chosen service
LLM_PROMPT=                  # Optional: Override the default summarization prompt

# --- Filtering Settings ---
# Comma-separated list of domains (e.g., example.com) or specific emails.
# Blocklist takes precedence over Allowlist.
EMAIL_ALLOWLIST=             # e.g., trusted.com, friend@example.com
EMAIL_BLOCKLIST=             # e.g., spam.com, annoyance@test.com


# --- Service Specific Settings ---

# Ollama (Local)
OLLAMA_BASE_URL=http://localhost:11434
# Default Model: llama3

# OpenAI
OPENAI_API_KEY=sk-proj-...
# Default Model: gpt-3.5-turbo

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...
# Default Model: claude-3-haiku-20240307

# Google Gemini
GEMINI_API_KEY=AIzA...
# Default Model: gemini-1.5-flash
```

## Usage

### Command Line Options

| Option | Description |
| :--- | :--- |
| `--keep-unread`, `-k` | Keep emails as unread in your inbox after processing (default: processed emails are marked as read). |
| `--local-dir DIR` | Process `.eml` or `.txt` files from a local directory instead of connecting to IMAP. |
| `--output-file FILE`, `-o` | Save the generated summary to a specific file (e.g., `summary.md`). |
| `--env-file FILE` | Path to a custom configuration file (default: `.env`). |
| `--output-file-md FILE` | Save the generated summary as a Markdown file. |
| `--output-file-pdf FILE` | Save the generated summary as a PDF file. |
| `--output-email EMAIL` | Email address to send the summary to. |
| `--html-email` | Send the email as HTML (multipart/alternative) with plaintext fallback. |
| `--template-file FILE` | Path to a Markdown file containing the summary prompt template. |

### Examples

**1. Process unread emails and mark them as read (Default):**
```bash
python email_summarizer.py
```

**2. Process unread emails but keep them marked as unread (Safe Mode):**
```bash
python email_summarizer.py --keep-unread
```

**3. Process local test files and save output:**
```bash
python email_summarizer.py --local-dir ./test_emails --output-file-md summary.md
```

**4. Generate a PDF summary:**
```bash
python email_summarizer.py --output-file-pdf summary.pdf
```

**5. Send summary via email (Plaintext):**
```bash
python email_summarizer.py --output-email myemail@example.com
```

**6. Send summary via email (HTML with Plaintext fallback):**
```bash
python email_summarizer.py --output-email myemail@example.com --html-email
```

**7. Use a custom summary template:**
```bash
python email_summarizer.py --local-dir ./test_emails --template-file summary_template.md
```

## Examples

**Running with local test files:**

```bash
$ python email_summarizer.py --local-dir ./test_emails

Processing local emails from ./test_emails...
Reading ./test_emails/sample1.eml...
Reading ./test_emails/sample2.eml...

==============================
       EMAIL SUMMARIES       
==============================

**File:** sample1.eml
**From:** The Boss <boss@example.com>
**Subject:** Project Deadline Extension
**Summary:** The deadline for the "Antigravity" project has been extended by two weeks to December 23rd. The team is asked to use this time to fix unit tests and update documentation.

**File:** sample2.eml
**From:** Shop Marketing <marketing@shop.com>
**Subject:** Huge Sale! 50% Off Everything!
**Summary:** The store is having a 50% off sale on everything for the next 24 hours. A link is provided to shop.
```

## Testing
To run the included unit tests:

```bash
python -m unittest test_summarizer.py
```
