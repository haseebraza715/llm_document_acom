from __future__ import annotations

import re


URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_PATTERN = re.compile(r"\b[\w.\-+%]+@[\w.\-]+\.[A-Za-z]{2,}\b")
REPEATED_PUNCT_PATTERN = re.compile(r"([!?.,;:])\1{2,}")
EXCESSIVE_BLANK_LINES_PATTERN = re.compile(r"\n{3,}")
EMAIL_ARTIFACT_LINE_PATTERN = re.compile(
    r"^\s*(from|subject|organization|reply-to|nntp-posting-host|distribution|lines|keywords)\s*:\s*.*$",
    re.IGNORECASE | re.MULTILINE,
)


def normalize_line_breaks(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def trim_whitespace(text: str) -> str:
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def remove_urls(text: str) -> str:
    return URL_PATTERN.sub(" ", text)


def remove_email_artifacts(text: str) -> str:
    text = EMAIL_ARTIFACT_LINE_PATTERN.sub("", text)
    return EMAIL_PATTERN.sub(" ", text)


def reduce_repeated_punctuation(text: str) -> str:
    return REPEATED_PUNCT_PATTERN.sub(r"\1\1", text)


def remove_excessive_blank_lines(text: str) -> str:
    return EXCESSIVE_BLANK_LINES_PATTERN.sub("\n\n", text)


def light_clean_text(text: str) -> str:
    cleaned = normalize_line_breaks(text)
    cleaned = remove_urls(cleaned)
    cleaned = remove_email_artifacts(cleaned)
    cleaned = reduce_repeated_punctuation(cleaned)
    cleaned = trim_whitespace(cleaned)
    cleaned = remove_excessive_blank_lines(cleaned)
    return cleaned.strip()


def is_valid_document(text: str, min_words: int = 20, min_chars: int = 120) -> bool:
    if not text:
        return False
    if len(text) < min_chars:
        return False
    return len(text.split()) >= min_words
