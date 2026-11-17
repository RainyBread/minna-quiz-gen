import pytest

from config import Semester, SemesterViolation, normalize_base_url


def test_semester_parse_allowed():
    assert Semester.parse("big2_autumn") is Semester.BIG2_AUTUMN
    assert Semester.parse("big1_spring") is Semester.BIG1_SPRING


@pytest.mark.parametrize("value", ["big2_spring", "big2_summer", "sophomore_spring"])
def test_semester_forbidden(value):
    with pytest.raises(SemesterViolation):
        Semester.parse(value)


def test_base_url_normalize():
    assert (
        normalize_base_url("https://api.deepseek.com/v1/chat/completions")
        == "https://api.deepseek.com/v1"
    )
    assert normalize_base_url("https://api.example.com/v1/") == "https://api.example.com/v1"
