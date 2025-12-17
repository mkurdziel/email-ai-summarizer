import unittest
from email_summarizer import extract_original_sender, is_email_allowed

class TestEmailFiltering(unittest.TestCase):

    def test_extract_original_sender_no_forward(self):
        header_sender = "User <user@example.com>"
        body = "Here is some text."
        self.assertEqual(extract_original_sender(header_sender, body), "user@example.com")

    def test_extract_original_sender_simple_forward(self):
        header_sender = "Forwarder <me@me.com>"
        body = """
        ---------- Forwarded message ---------
        From: Original Sender <original@sender.com>
        Date: Fri, Dec 15, 2023 at 10:00 AM
        Subject: Test
        
        Some content.
        """
        self.assertEqual(extract_original_sender(header_sender, body), "original@sender.com")

    def test_extract_original_sender_nested_forward(self):
        header_sender = "Me <me@me.com>"
        body = """
        Begin forwarded message:
        
        From: Middle Man <middle@man.com>
        
        Begin forwarded message:
        
        From: Source <source@origin.com>
        
        Real content.
        """
        self.assertEqual(extract_original_sender(header_sender, body), "source@origin.com")

    def test_extract_original_sender_plain_email_in_body(self):
        header_sender = "Forwarder <me@me.com>"
        body = """
        From: plain@email.com
        Subject: FW: test
        """
        self.assertEqual(extract_original_sender(header_sender, body), "plain@email.com")
        
    def test_is_email_allowed_no_lists(self):
        self.assertTrue(is_email_allowed("test@example.com", [], []))

    def test_is_email_allowed_blocklist(self):
        blocklist = ["blocked.com", "bad@actor.com"]
        self.assertFalse(is_email_allowed("user@blocked.com", [], blocklist))
        self.assertFalse(is_email_allowed("bad@actor.com", [], blocklist))
        self.assertTrue(is_email_allowed("good@user.com", [], blocklist))

    def test_is_email_allowed_allowlist(self):
        allowlist = ["allowed.com", "friend@vip.com"]
        self.assertTrue(is_email_allowed("user@allowed.com", allowlist, []))
        self.assertTrue(is_email_allowed("friend@vip.com", allowlist, []))
        self.assertFalse(is_email_allowed("random@stranger.com", allowlist, []))

    def test_is_email_allowed_block_precedence(self):
        allowlist = ["example.com"]
        blocklist = ["bad@example.com"]
        # Explicit block should override domain allow
        self.assertFalse(is_email_allowed("bad@example.com", allowlist, blocklist))
        # Other email in allowed domain should pass
        self.assertTrue(is_email_allowed("good@example.com", allowlist, blocklist))

if __name__ == '__main__':
    unittest.main()
