"""Honor the existing corpus hash schema WITHOUT normalizing training inputs."""
from hashlib import sha256
import unicodedata


def validate_source_text(row, arm):
    text = row["text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Missing/empty training source text")
    if arm == "external_wikipedia":
        # Exact producer formula: NFKC + casefold + whitespace collapse, for
        # deduplication identity ONLY. Whole-file byte checksums pin raw text.
        normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
        if sha256(normalized.encode()).hexdigest() != row.get("text_sha256"):
            raise ValueError("Normalized source identity hash mismatch")
    elif arm == "approved_articles":
        if "text_sha256" in row and sha256(text.encode()).hexdigest() != row["text_sha256"]:
            raise ValueError("Raw source text checksum mismatch")
    else:
        raise ValueError("Unknown training arm")
    return text
