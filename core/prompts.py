"""Short, construction-first prompts.

The model writes only a clue and a complete sentence containing the pinned
answer.  It never writes options, an answer letter, or an explanation; those are
assembled locally from lock tables / vocabulary data.  This avoids asking the
model to argue why a choice is unique on a bare framework.
"""

from __future__ import annotations

from typing import Any

from core.models import QuestionType

TAG_TEMPLATE = (
    "[CLUE]决定性线索（类别、用途、反义、搭配、原因等）[/CLUE]\n"
    "[SENTENCE]完整句子或短对话，正确项用 [[ans]]...[[/ans]] 标出[/SENTENCE]\n"
    "[ZH]中文翻译[/ZH]"
)


def build_question_prompt(spec: Any) -> dict[str, str]:
    qtype: QuestionType = spec.qtype
    system = (
        "你是日语语境作者。请输出三样东西："
        "一个能唯一支持固定项的日语线索、一个让固定项原样出现的完整句子或短对话、"
        "以及中文翻译。固定项用 [[ans]]...[[/ans]] 标出。"
        "线索写成一个短短语（不超过 30 字），只陈述决定答案的事实；"
        "句子要自然，并让线索真正决定固定项。"
    )
    if qtype == QuestionType.VOCAB_CHOICE:
        target = (
            f"固定项：{spec.written}（{spec.reading}）\n"
            f"中文：{spec.meaning or '（无）'}\n"
            f"课次：{spec.lesson or '跨课'}\n"
        )
        if spec.scope == "extend":
            target += f"这是拓展学习词，主题：{spec.topic or '自由'}。\n"
        instruction = (
            "请写一句自然的日语，让固定项原样出现；"
            "句子里必须有一个只能支持它的线索（类别、用途、反义对比、固定搭配或原因）。"
            "把线索单独写进 [CLUE]，把完整句写进 [SENTENCE]。"
        )
    elif qtype == QuestionType.EXPRESSION_CHOICE:
        target = (
            f"语法/表达点：{spec.pattern or spec.written}\n"
            f"意思：{spec.meaning or '（无）'}\n"
            f"固定表达：{spec.correct_text}\n"
            f"参考例句：{spec.example or '（无）'}\n"
        )
        instruction = (
            "请写一段很短的日语对话（2-3 句），让固定表达原样出现在应答中，"
            "并用上下文线索说明为什么必须用这个表达。"
            "把线索写进 [CLUE]，把对话写进 [SENTENCE]。"
        )
    else:
        target = (
            f"语法点：{spec.pattern or spec.written}\n"
            f"意思：{spec.meaning or '（无）'}\n"
            f"固定形式：{spec.correct_text}\n"
            f"参考例句：{spec.example or '（无）'}\n"
        )
        instruction = (
            "请写一个自然的日语单句，让固定形式原样出现；"
            "句子里必须有一个只能支持这个形式的线索（接续、搭配、时态、语态或原因）。"
            "把线索写进 [CLUE]，把完整句写进 [SENTENCE]。"
        )
    hint = getattr(spec, "clue_hint", "")
    anchor = ""
    if hint:
        anchor += f"本地线索提示：{hint}\n"
    if getattr(spec, "anchor_example", ""):
        anchor += (
            "课内参考例句（只作难度/风格锚，请另起一个不同的语境）：\n"
            f"{spec.anchor_example}\n"
        )
    user = (
        f"题型：{qtype.value}\n"
        f"{target}"
        f"{anchor}"
        f"{instruction}\n"
        "请按以下标签输出：\n"
        f"{TAG_TEMPLATE}"
    )
    return {"system": system, "user": user}


def build_intent_prompt(text: str) -> dict[str, str]:
    system = (
        "你是 JLPTCraft 的意图解析器。请分析用户需求，并用标签输出可执行计划。"
        "题型从 kanji_to_kana、kana_to_kanji、vocab_choice、expression_choice、grammar_fill 中选择；"
        "范围从 in_syllabus、mdict_distractors、extend 中选择。"
    )
    user = (
        f"用户需求：{text}\n"
        "请输出：\n"
        "[MODE]practice 或 mock[/MODE]\n"
        "[COUNT]题量整数[/COUNT]\n"
        "[LESSONS]教材课次，如 37-50 或 3 或 37,39[/LESSONS]\n"
        "[TYPES]题型=数量，如 kanji_to_kana=5,grammar_fill=5[/TYPES]\n"
        "[SCOPE]in_syllabus、mdict_distractors 或 extend[/SCOPE]\n"
        "[TOPIC]主题，可留空[/TOPIC]"
    )
    return {"system": system, "user": user}
