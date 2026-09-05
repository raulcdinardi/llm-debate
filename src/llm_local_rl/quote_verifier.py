from __future__ import annotations

from dataclasses import dataclass
import re


_QUOTE_RE = re.compile(r"<quote>(.*?)</quote>", re.IGNORECASE | re.DOTALL)
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class QuoteVerificationResult:
    text: str
    metrics: dict[str, float | int]


def normalize_for_quote_match(text: str) -> str:
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    return _SPACE_RE.sub(" ", text).strip().casefold()


def verify_quotes(text: str, *, source_text: str) -> QuoteVerificationResult:
    normalized_source = normalize_for_quote_match(source_text)
    verified = 0
    unverified = 0
    quoted_words = 0
    seen_quotes: set[str] = set()
    duplicate_quotes = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal verified, unverified, quoted_words, duplicate_quotes
        quote = match.group(1).strip()
        normalized_quote = normalize_for_quote_match(quote)
        quoted_words += len(quote.split())
        if normalized_quote in seen_quotes:
            duplicate_quotes += 1
        seen_quotes.add(normalized_quote)
        if normalized_quote and normalized_quote in normalized_source:
            verified += 1
            return f"<v_quote>{quote}</v_quote>"
        unverified += 1
        return f"<u_quote>{quote}</u_quote>"

    rewritten = _QUOTE_RE.sub(replace, text)
    total = verified + unverified
    precision = float(verified / total) if total else 0.0
    return QuoteVerificationResult(
        text=rewritten,
        metrics={
            "verified_quote_count": verified,
            "unverified_quote_count": unverified,
            "quote_count": total,
            "quoted_words": quoted_words,
            "quote_precision": precision,
            "duplicate_quote_count": duplicate_quotes,
        },
    )
