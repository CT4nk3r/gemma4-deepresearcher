from __future__ import annotations

import re
from collections import Counter

from .models import Note, SourceDocument

_STOPWORDS = {
    "about",
    "after",
    "also",
    "analysis",
    "because",
    "between",
    "from",
    "have",
    "into",
    "that",
    "their",
    "there",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}


def extract_keywords(question: str, *, limit: int = 10) -> list[str]:
    tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", question)
        if token.lower() not in _STOPWORDS
    ]
    counts = Counter(tokens)
    return [token for token, _count in counts.most_common(limit)]


def extract_notes(
    question: str, documents: list[SourceDocument], *, max_notes: int
) -> list[Note]:
    keywords = extract_keywords(question)
    notes: list[Note] = []
    for document in documents:
        for sentence in _sentences(document.text):
            if not _matches(sentence, keywords):
                continue
            quote = sentence[:600].strip()
            notes.append(
                Note(
                    source_id=document.id,
                    title=document.title,
                    url=document.url,
                    claim=_claim_from_sentence(sentence),
                    quote=quote,
                    relevance=_relevance(sentence, keywords),
                )
            )
            if len(notes) >= max_notes:
                return notes
    return notes


def _sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []
    chunks = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", compact)
    return [chunk.strip() for chunk in chunks if len(chunk.strip()) > 40]


def _matches(sentence: str, keywords: list[str]) -> bool:
    lowered = sentence.lower()
    if not keywords:
        return True
    return any(keyword in lowered for keyword in keywords)


def _claim_from_sentence(sentence: str) -> str:
    sentence = sentence.strip()
    if len(sentence) <= 220:
        return sentence
    return f"{sentence[:217].rstrip()}..."


def _relevance(sentence: str, keywords: list[str]) -> str:
    lowered = sentence.lower()
    matched = [keyword for keyword in keywords if keyword in lowered]
    if matched:
        return f"Matches query terms: {', '.join(matched[:5])}"
    return "General contextual evidence"
