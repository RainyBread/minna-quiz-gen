from core.locks import LockTable
from core.quizgen import _source_slots, _vocab_stem_conflicts, style_key
from core.similarity import overlap


def test_source_slots_preserve_ratio():
    slots = _source_slots(10, {"original": 0.5, "variant": 0.3, "new": 0.2})
    assert slots.count("original") == 5
    assert slots.count("variant") == 3
    assert slots.count("new") == 2


def test_style_key_distinguishes_polite_plain_and_te():
    assert style_key("読みます") == "polite"
    assert style_key("読めませんでした") == "polite"
    assert style_key("並べて") == "te"
    assert style_key("食べた") == "ta"
    assert style_key("から") == "particle"


def test_vocab_stem_conflict_detects_non_polite_context():
    assert _vocab_stem_conflicts("足が[[blank]]て、転んだ", "滑ります") is True
    assert _vocab_stem_conflicts("足が[[blank]]から、気をつけて", "滑ります") is False


def test_overlap_is_low_for_different_sentences():
    assert overlap("今日は学校へ行きます", "明日は図書館で勉強します") < 0.3


def test_lock_table_rejects_bare_framework():
    locks = LockTable(
        {
            "clue_markers": ["から", "ので"],
            "fields": {"taste": ["からい", "あまい"]},
            "antonyms": [["かたい", "やわらかい"]],
        }
    )
    assert locks.is_clue_present("みんなで[[blank]]。", "") is False
    assert locks.is_clue_present("味が[[blank]]ので、塩を足す。", "") is True
    assert locks.field_of("からい") == "taste"
    assert locks.antonym_of("かたい") == "やわらかい"

