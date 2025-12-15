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

Control the behavior of the script using environment variables. Create a `.env` file in the root directory:

```env
# --- Email Server Settings (Required for IMAP) ---
IMAP_SERVER=imap.gmail.com   # e.g., imap.mail.yahoo.com
IMAP_PORT=993                # Default is 993 (SSL)
EMAIL_USER=your_email@example.com
EMAIL_PASSWORD=your_app_password

# --- General Settings ---
LLM_SERVICE=ollama           # Options: ollama, openai, gemini, anthropic
LLM_MODEL=                   # Optional: Override default model for the chosen service
LLM_PROMPT=                  # Optional: Override the default summarization prompt

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

### 1. Process Emails via IMAP
To connect to your email server and summarize unread messages. **Note**: This will mark processed emails as read by default.

```bash
python email_summarizer.py
```

To keep emails as unread (e.g., for testing):
```bash
python email_summarizer.py --keep-unread
```

### 2. Process Local Email Files
To process a folder containing `.eml` or `.txt` email files (useful for testing):

```bash
python email_summarizer.py --local-dir ./test_emails
```

### 3. Save Summary to File
To save the summary to a file (e.g., Markdown):

```bash
python email_summarizer.py --output-file summary.md
```

You can combine this with `--local-dir`:
```bash
python email_summarizer.py --local-dir ./test_emails --output-file summary.md
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
