from core.tagformat import parse_question_tags, parse_tags


def test_parse_full_with_answer_marker_and_blank():
    raw = (
        "[FULL]赤ちゃんが [[ans]]泣きます[[/ans]]ので、母は慌てた。 [[blank]][/FULL]\n"
        "[D]笑います[/D]\n[D]歌います[/D]\n[D]食べます[/D]\n"
        "[EXP]解析[/EXP]\n[ZH]翻译[/ZH]"
    )
    parsed = parse_question_tags(raw)
    assert "[[ans]]泣きます[[/ans]]" in parsed["full"]
    assert parsed["distractors"] == ["笑います", "歌います", "食べます"]
    assert parsed["explanation"] == "解析"
    assert parsed["translation"] == "翻译"


def test_missing_closing_tag_and_full_width_brackets():
    raw = "［STEM］题干 [[blank]] ここ\n[D]一[/D]\n[D]二[/D]"
    tags = parse_tags(raw)
    assert tags["STEM"] == ["题干 [[blank]] ここ"]
    assert tags["D"] == ["一", "二"]


def test_trailing_html_tag_is_cleaned():
    parsed = parse_question_tags(
        "[STEM]あ[[blank]]い[/STEM]\n[D]一[/D]\n[D]二[/D]\n[D]三[/D]\n[EXP]解析</EXP>"
    )
    assert parsed["explanation"] == "解析"
