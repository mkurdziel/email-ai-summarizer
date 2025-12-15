import unittest
from unittest.mock import MagicMock, patch, mock_open
import email_summarizer
import email
import sys
import os

# Mock modules
sys.modules['openai'] = MagicMock()
sys.modules['anthropic'] = MagicMock()
sys.modules['google.generativeai'] = MagicMock()

import email_summarizer

class TestEmailSummarizer(unittest.TestCase):

    @patch('email_summarizer.imaplib.IMAP4_SSL')
    def test_connect_imap_success(self, mock_imap):
        email_summarizer.IMAP_SERVER = "imap.test.com"
        email_summarizer.EMAIL_USER = "user"
        email_summarizer.EMAIL_PASSWORD = "password"
        
        mail = email_summarizer.connect_imap()
        
        mock_imap.assert_called_with("imap.test.com", 993)
        self.assertIsNotNone(mail)

    @patch('email_summarizer.requests.post')
    def test_ollama_summarizer(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Ollama summary."}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        summarizer = email_summarizer.OllamaSummarizer()
        summary = summarizer.summarize("Long email body text.")
        
        self.assertEqual(summary, "Ollama summary.")

    @patch('email_summarizer.glob.glob')
    @patch('builtins.open', new_callable=mock_open)
    def test_process_local_emails_returns_content(self, mock_file, mock_glob):
        mock_glob.return_value = ['/path/to/test.eml']
        
        # We don't pass summarizer anymore
        
        with patch('email.message_from_binary_file') as mock_email_parser:
            mock_msg = MagicMock()
            mock_msg.get.side_effect = lambda k, d=None: "Test Subject" if k == "Subject" else "sender" if k == "From" else d
            mock_msg.is_multipart.return_value = False
            mock_msg.get_payload.return_value = b"Body content"
            mock_msg.get_content_charset.return_value = "utf-8"
            mock_email_parser.return_value = mock_msg
            
            email_contents = email_summarizer.process_local_emails('/path/to')
        
        self.assertEqual(len(email_contents), 1)
        # Check that it returns the formatted string, not the summary
        self.assertIn("--- EMAIL START ---", email_contents[0])
        self.assertIn("Body content", email_contents[0])

if __name__ == '__main__':
    unittest.main()
