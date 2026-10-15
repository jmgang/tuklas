"""Data-only safety adapters for the official TokenLearn 2.0 recipe.

No PCA, custom loss, retrieval labels or alternative training algorithm here.
Inputs are existing document text; article identities control early-stop splits.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import math


def article_partition(article_id: str, seed: int = 42) -> str:
    value = int(sha256(f"{seed}:{article_id}".encode()).hexdigest()[:16], 16)
    return "validation" if value % 10 == 0 else "train"


def safe_windows(text: str, tokenizer, max_tokens: int = 512, max_chars: int = 1024):
    """Lossless original-character windows, checked with special tokens included."""
    if not text or not text.strip():
        raise ValueError("Empty/whitespace-only source text")
    windows = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if len(tokenizer.encode(text[start:end], add_special_tokens=True)) > max_tokens:
            lo, hi = start + 1, end
            best = None
            while lo <= hi:
                middle = (lo + hi) // 2
                if len(tokenizer.encode(text[start:middle], add_special_tokens=True)) <= max_tokens:
                    best = middle
                    lo = middle + 1
                else:
                    hi = middle - 1
            if best is None:
                raise ValueError("Single character exceeds the teacher budget")
            end = best
        fragment = text[start:end]
        if not tokenizer.encode(fragment, add_special_tokens=False):
            if windows:
                a, _, previous = windows[-1]
                merged = previous + fragment
                if len(tokenizer.encode(merged, add_special_tokens=True)) <= max_tokens:
                    windows[-1] = (a, end, merged)
                    start = end
                    continue
            raise ValueError("Token-empty window cannot be preserved safely")
        windows.append((start, end, fragment))
        start = end
    if "".join(fragment for _, _, fragment in windows) != text:
        raise AssertionError("Source character coverage changed")
    for _, _, fragment in windows:
        if len(fragment) > 5120:
            raise ValueError("Model2Vec training character guard would truncate text")
        if len(tokenizer.encode(fragment, add_special_tokens=True)) > max_tokens:
            raise AssertionError("Preflight produced an over-budget window")
    return windows


def pad_collate(batch, pad_id: int):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    ids, targets = zip(*batch)
    return (
        pad_sequence([torch.tensor(x, dtype=torch.long) for x in ids],
                     batch_first=True, padding_value=pad_id),
        torch.stack(targets),
    )


@contextmanager
def official_training_adapter(validation_texts, validation_vectors, pad_token: str,
                              on_validation=None, checkpoint_selector=None):
    """Use TokenLearn's CLI unchanged except explicit validation/padding/telemetry.

TokenLearn 2.0 imports Model2Vec's trainable similarity class. This scoped
subclass calls the real class's fit; its cosine objective, Adam optimizer,
plateau scheduler, token-vector training and export stay unchanged.
"""
    import torch
    import model2vec.train as public_train
    import model2vec.train.dataset as dataset_module
    import model2vec.train.trainer as trainer
    original_class = public_train.StaticModelForSimilarity
    original_collate = dataset_module.TextDataset.collate_fn
    original_validation = trainer._run_validation
    state = {}

    class CorpusValidatedSimilarity(original_class):
        @classmethod
        def from_static_model(cls, *, model, **kwargs):
            if not torch.isfinite(torch.from_numpy(model.embedding)).all():
                raise ValueError("Nonfinite initialization; refusing implicit repair")
            kwargs["pad_token"] = pad_token
            result = super().from_static_model(model=model, **kwargs)
            state["pad_id"] = result.pad_id
            return result

        def fit(self, X, y, **kwargs):
            # Reject inputs BEFORE the library's [:512]/[:5120] slices.
            for texts in (X, validation_texts):
                for index in range(0, len(texts), 1024):
                    batch = texts[index:index + 1024]
                    if any(len(x) > 5120 for x in batch):
                        raise ValueError("Training character truncation would occur")
                    encodings = self.tokenizer.encode_batch_fast(batch, add_special_tokens=False)
                    if any(not e.ids or len(e.ids) > 512 for e in encodings):
                        raise ValueError("Training token truncation/empty input would occur")
            kwargs["X_val"] = validation_texts
            kwargs["y_val"] = torch.from_numpy(validation_vectors)
            kwargs["random_seed"] = 42
            # Same cadence: once per full epoch for BOTH corpus arms.
            kwargs["validation_steps"] = math.ceil(len(X) / kwargs["batch_size"])
            result = super().fit(X, y, **kwargs)
            # TokenLearn's installed Model2Vec trainer documents that the best
            # validation checkpoint is restored, but the current loop returns
            # the most recent validation checkpoint.  Preserve the documented
            # behaviour without changing its cosine loss, optimizer or
            # scheduler: restore the lowest validation-loss state observed by
            # the real validation function.
            best_state = state.get("best_state_dict")
            if best_state is None:
                raise RuntimeError("TokenLearn completed without a validation checkpoint")
            self.load_state_dict(best_state)
            state["selected_validation_index"] = state["best_validation_index"]
            return result

    def collate(batch):
        return pad_collate(batch, state["pad_id"])

    def validation(*args, **kwargs):
        metrics = original_validation(*args, **kwargs)
        event_index = int(state.get("validation_count", 0))
        state["validation_count"] = event_index + 1
        if checkpoint_selector is None:
            value = float(metrics["val_loss"])
            selection_event = {
                "selection_value": -value,
                "selection_metric": "negative_validation_loss",
                "event_index": event_index,
                "val_loss": value,
            }
            state["selection_mode"] = "validation_loss"
            improved = value < float(state.get("best_validation_value", float("inf")))
            if improved:
                state["best_validation_value"] = value
        else:
            selection_event = checkpoint_selector(args[0], event_index, metrics)
            value = float(selection_event["selection_value"])
            if not math.isfinite(value):
                raise ValueError("checkpoint selector returned a nonfinite score")
            state["selection_mode"] = "development_retrieval"
            improved = value > float(state.get("best_selection_value", float("-inf")))
            if improved:
                state["best_selection_value"] = value
                state["best_validation_value"] = float(metrics["val_loss"])
        state.setdefault("selection_events", []).append(selection_event)
        if improved:
            state["best_validation_index"] = event_index
            state["best_state_dict"] = {
                key: tensor.detach().cpu().clone()
                for key, tensor in args[0].state_dict().items()
            }
        if on_validation is not None:
            on_validation(metrics)
        return metrics

    public_train.StaticModelForSimilarity = CorpusValidatedSimilarity
    dataset_module.TextDataset.collate_fn = staticmethod(collate)
    trainer._run_validation = validation
    try:
        yield state
    finally:
        public_train.StaticModelForSimilarity = original_class
        dataset_module.TextDataset.collate_fn = staticmethod(original_collate)
        trainer._run_validation = original_validation
