from pycon_model2vec_demo.compact_word_model import build_compact_word_tokenizer, unknown_fraction


def test_word_tokenizer_has_no_hidden_teacher_vocabulary_and_atomic_query_role():
    tokenizer = build_compact_word_tokenizer(["hello", ",", "world", "kamusta"])
    assert tokenizer.get_vocab_size(with_added_tokens=True) == 7
    assert tokenizer.encode("[QRY] Hello, world", add_special_tokens=False).tokens == ["[QRY]", "hello", ",", "world"]


def test_word_tokenizer_unknown_audit_uses_the_real_serializable_tokenization():
    tokenizer = build_compact_word_tokenizer(["hello", "world"])
    missing, total, fraction = unknown_fraction(tokenizer, ["Hello world", "Hello moon"])
    assert (missing, total, fraction) == (1, 4, 0.25)
