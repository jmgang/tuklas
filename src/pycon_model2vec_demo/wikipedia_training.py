"""Deterministic, unlabeled Wikipedia preparation; no model or API execution."""
from collections import Counter, defaultdict
from hashlib import sha256
import heapq
import re
import unicodedata
from urllib.parse import unquote, urlsplit


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def title_key(text):
    return normalize(unquote(text).replace("_", " "))


def article_key(language, title):
    return language, title_key(title)


def url_key(url):
    parsed = urlsplit(url)
    if not parsed.hostname or not parsed.hostname.endswith(".wikipedia.org"):
        raise ValueError("Expected a Wikipedia article URL")
    if not parsed.path.startswith("/wiki/"):
        raise ValueError("Expected a /wiki/ source URL")
    return article_key(parsed.hostname.split(".")[0], parsed.path[6:])


def text_hash(text):
    return sha256(normalize(text).encode()).hexdigest()


class ContentGuard:
    """Exact lexical overlap guard, not a claim of semantic decontamination.

    Index 32-word snippets at a fixed stride, anchored on a rare benchmark word.
    Search every word position in training text, not just the benchmark stride.
    Whole-article identity exclusion is handled separately by the caller.
    """
    def __init__(self, parents, policy):
        self.policy = policy
        self.full = {}
        self.paragraphs = {}
        self.anchors = defaultdict(lambda: defaultdict(dict))
        texts = [(p["parent_id"], normalize(p["text"])) for p in parents]
        frequencies = Counter(word for _, text in texts for word in text.split())
        width = policy["snippet_words"]
        stride = policy["snippet_stride_words"]
        for parent, text in texts:
            self.full[text_hash(text)] = parent
        for parent in parents:
            parent_id = parent["parent_id"]
            for paragraph in re.split(r"\n\s*\n", parent["text"]):
                if len(normalize(paragraph)) >= policy["paragraph_minimum_characters"]:
                    self.paragraphs[text_hash(paragraph)] = parent_id
            words = normalize(parent["text"]).split()
            starts = list(range(0, max(0, len(words) - width + 1), stride))
            if len(words) >= width:
                starts.append(len(words) - width)
            for start in starts:
                snippet = tuple(words[start:start + width])
                if len(" ".join(snippet)) < policy["snippet_minimum_characters"]:
                    continue
                offset = min(range(width), key=lambda i: (frequencies[snippet[i]], i))
                self.anchors[snippet[offset]][offset][snippet] = parent_id

    def find(self, text):
        parent = self.full.get(text_hash(text))
        if parent:
            return "normalized-full-context", parent
        for paragraph in re.split(r"\n\s*\n", text):
            if len(normalize(paragraph)) >= self.policy["paragraph_minimum_characters"]:
                parent = self.paragraphs.get(text_hash(paragraph))
                if parent:
                    return "exact-long-paragraph", parent
        words = normalize(text).split()
        width = self.policy["snippet_words"]
        for position, word in enumerate(words):
            offsets = self.anchors.get(word)
            if not offsets:
                continue
            for offset, snippets in offsets.items():
                start = position - offset
                if start >= 0 and start + width <= len(words):
                    parent = snippets.get(tuple(words[start:start + width]))
                    if parent:
                        return "sampled-exact-32-word-snippet", parent
        return None


def passage_windows(text, maximum, minimum):
    """Nonoverlapping source slices; prefer a whitespace boundary, retain offsets.

    Short fragments are explicitly not selected, not silently claimed as covered.
    This is a training selection, not the lossless evaluation layout.
    """
    if minimum <= 0 or maximum < minimum:
        raise ValueError("Invalid passage bounds")
    start = 0
    while start < len(text):
        end = min(start + maximum, len(text))
        if end < len(text):
            boundary = max(text.rfind(" ", start + minimum, end),
                           text.rfind("\n", start + minimum, end))
            if boundary > start:
                end = boundary + 1
        value = text[start:end]
        if len(value.strip()) >= minimum:
            yield {"start": start, "end": end, "text": value}
        start = end


def stable_priority(seed, identity):
    return int(sha256((seed + "\0" + identity).encode()).hexdigest(), 16)


class HashSample:
    """Streaming bottom-k sample, independent of parquet row order and model scores."""
    def __init__(self, maximum):
        if maximum < 1:
            raise ValueError("Positive sample size required")
        self.maximum = maximum
        self.heap = []

    def add(self, record, priority):
        item = (-priority, record["passage_id"], record)
        if len(self.heap) < self.maximum:
            heapq.heappush(self.heap, item)
        elif item > self.heap[0]:
            heapq.heapreplace(self.heap, item)

    def records(self):
        return [item[2] for item in sorted(self.heap, key=lambda item: (-item[0], item[1]))]


def article_passages(language, row, source_path, row_index, sampling):
    identity = language + ":" + str(row["id"])
    candidates = []
    for window in passage_windows(row["text"], sampling["maximum_passage_characters"],
                                  sampling["minimum_passage_characters"]):
        passage_id = identity + f':{window["start"]}:{window["end"]}'
        record = {**window, "passage_id": passage_id, "language": language,
                  "source_article_id": identity, "source_url": row["url"],
                  "source_title": row["title"], "source_file": source_path,
                  "source_row_index": row_index, "text_sha256": text_hash(window["text"])}
        candidates.append((stable_priority(sampling["seed"], passage_id), record))
    candidates.sort(key=lambda item: (item[0], item[1]["passage_id"]))
    return candidates[:sampling["maximum_passages_per_article"]], len(candidates)
