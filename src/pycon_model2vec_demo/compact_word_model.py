"""Tokenization primitives for a genuinely compact bilingual static model.

Model2Vec's public ``vocabulary=`` argument retains the teacher tokenizer's
internal vocabulary and adds custom words.  That is useful for vocabulary
expansion but cannot satisfy this project's strict package-size requirement
with SEA-LION's 249,999-token tokenizer.  This module builds a replacement
WordLevel tokenizer from the already approved balanced vocabulary instead.
The word splitting exactly matches the frozen vocabulary-counting protocol.
"""

from __future__ import annotations

from typing import Sequence

from tokenizers import Regex, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase, NFKC, Sequence as NormalizerSequence
from tokenizers.pre_tokenizers import Split


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
QUERY_TOKEN = "[QRY]"
SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN, QUERY_TOKEN)
TOKEN_PATTERN = r"\w+|[^\w\s]+"


def build_compact_word_tokenizer(vocabulary: Sequence[str]) -> Tokenizer:
    """Build a serializable replacement tokenizer without hidden base tokens."""

    cleaned: list[str] = []
    seen = set(SPECIAL_TOKENS)
    for raw in vocabulary:
        token = str(raw).strip()
        if not token or token in seen:
            continue
        # The vocabulary builder applies Unicode-aware casefolding. NFKC plus
        # lowercase is the closest serializable tokenizers normalization and
        # retains the exact approved vocabulary spelling.
        seen.add(token)
        cleaned.append(token)
    if not cleaned:
        raise ValueError("compact vocabulary is empty")
    entries = [*SPECIAL_TOKENS, *cleaned]
    mapping = {token: index for index, token in enumerate(entries)}
    tokenizer = Tokenizer(WordLevel(mapping, unk_token=UNK_TOKEN))
    tokenizer.normalizer = NormalizerSequence([NFKC(), Lowercase()])
    tokenizer.pre_tokenizer = Split(Regex(TOKEN_PATTERN), behavior="removed", invert=True)
    if tokenizer.add_special_tokens([QUERY_TOKEN]) != 1:
        raise ValueError("query token was not added exactly once")
    if tokenizer.get_vocab_size(with_added_tokens=True) != len(entries):
        raise ValueError("replacement tokenizer contains unexpected hidden tokens")
    for token, expected in zip(SPECIAL_TOKENS, range(3), strict=True):
        if tokenizer.token_to_id(token) != expected:
            raise ValueError(f"special token id changed: {token}")
    if tokenizer.encode(QUERY_TOKEN, add_special_tokens=False).ids != [2]:
        raise ValueError("query token is not atomic")
    return tokenizer


def unknown_fraction(tokenizer: Tokenizer, texts: Sequence[str]) -> tuple[int, int, float]:
    unknown = tokenizer.token_to_id(UNK_TOKEN)
    if unknown is None:
        raise ValueError("tokenizer has no unknown token")
    total = missing = 0
    for encoding in tokenizer.encode_batch_fast(list(texts), add_special_tokens=False):
        total += len(encoding.ids)
        missing += sum(token_id == unknown for token_id in encoding.ids)
    return missing, total, missing / total if total else 0.0
