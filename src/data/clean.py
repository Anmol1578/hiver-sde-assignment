"""
Text cleaning, mention sanitization, and language filtering module.
"""

import re
import html

# Regex patterns for cleaning tweets
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
WHITESPACE_PATTERN = re.compile(r"\s+")
AGENT_SIGNATURE_PATTERN = re.compile(r"\s*[\^/][A-Z]{2,3}$")

def is_predominantly_english(text: str, threshold: float = 0.82) -> bool:
    """
    Determines if text is predominantly English by checking ASCII alphanumeric ratio.
    Filters out Japanese, Chinese, Arabic, Cyrillic, etc.
    """
    if not text or not text.strip():
        return False
    # Count printable ASCII characters
    ascii_chars = sum(1 for c in text if 32 <= ord(c) <= 126)
    ratio = ascii_chars / len(text)
    return ratio >= threshold

def clean_tweet_text(text: str, remove_mentions: bool = True, remove_urls: bool = False, strip_signature: bool = True) -> str:
    """
    Normalizes a tweet's text by unescaping HTML entities, removing agent signatures,
    normalizing whitespace, and optionally sanitizing handles and URLs.
    """
    if not text:
        return ""
    
    # Unescape HTML (&amp; -> &, &gt; -> >, etc.)
    text = html.unescape(text)
    
    # Strip brand agent signature like ^TN, ^AG, /KM
    if strip_signature:
        text = AGENT_SIGNATURE_PATTERN.sub("", text)
    
    # Remove or normalize user mentions (@115820 -> "")
    if remove_mentions:
        text = MENTION_PATTERN.sub("", text)
        
    # Replace URLs with token or strip if requested
    if remove_urls:
        text = URL_PATTERN.sub("", text)
    else:
        # Standardize URLs
        text = URL_PATTERN.sub(lambda m: "[LINK]", text)
        
    # Normalize whitespaces
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text

def clean_reference_reply(text: str) -> str:
    """
    Prepares a historical brand response to serve as high-quality ground-truth evidence.
    Keeps customer service links and instructions, removes agent tags and customer usernames.
    """
    if not text:
        return ""
    text = html.unescape(text)
    # Remove leading customer mentions e.g. "@115820 "
    text = re.sub(r"^(@\w+\s*)+", "", text)
    # Strip agent signature at the end
    text = AGENT_SIGNATURE_PATTERN.sub("", text)
    return text.strip()
