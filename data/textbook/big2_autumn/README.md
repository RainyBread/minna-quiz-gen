# 教材数据结构与覆盖说明

本目录只保留**结构化结果**，不保存教材 PDF/图片、docx、OCR 原始页面或词典原文。

## 当前覆盖

| 数据 | 文件 | 状态 |
|---|---|---|
| 37–43 课词表 | `vocab.json` | 369 条，已审核；构建期 UniDic 复核 |
| 44–50 课词表 | `vocab.json` | 134 条，视觉 OCR + UnitDic 复核后合并 |
| 语法参考 | `grammar.json` | 72 点；52 approved；52 个点有 answer forms（21 跨课 + 31 个 44–50 点已审核合并） |
| 2024 主 blueprint | `exam_blueprint.json` | 官方 100 题；第13题 OCR 不可用 |
| 课程背景/配比 | `course_profile.json` | 课程信息、80 题五类模拟卷、来源配比预设 |
| 拓展词种子 | `extension_vocab.json` | 日文词/读音/关系；无中文释义（运行时查本地 MDict） |
| 唯一性锁表 | `locks.json` | 反义对/语义场/搭配/线索标记；构造干扰项用 |
| PDF 页面映射 | `pdf_page_map.json`（本地，不提交） | 开发期 OCR 映射 |

当前 `vocab.json` 共 **14 课 / 503 条**（37–50）。UniDic 复核结果：**471 条一致、32 条不一致**（含 3 条教材特殊读音）。不一致条目不用于读音/汉字题；词义题保留。

## 数据来源

- 37–43：旧项目清洗后的教材词条。
- 44–50：扫描版辅导用书单词页 + 主教材语法页，视觉 OCR 后人工审核合并。
- 2024 真题：结构化导入；OCR 不清/重复选项/答案冲突题隔离。
- 课程说明/考试通知：只提取结构化课程背景，原 docx 不进仓库。

## 44–50 语法 forms

`drafts/grammar_forms_44_50.draft.json` 保存 LLM 从例句提取、再人工整理到“例句中实际出现”的候选形式。31 个点已审核合并；draft 保留为备份。重新生成/审核：

```bash
python tools/review_grammar_forms.py
python tools/review_grammar_forms.py --merge
```

只有 `grammar.json` 的 `approved_answer_forms`/`form_tags` 会用于语法/表达题；当前 52 个点可用。

## 质量说明

- 读音题/汉字题只使用 `unidic_verified=true` 且词形干净的词条；`その後` 等多读音词条跳过。
- 3 条教材特殊读音保留词义，不用作读音/汉字题。
- 词汇题使用词性与活用形一致性过滤；与正确项语义过近的干扰项会用本地关联词过滤。
- 原题库 `data/exams/item_bank.local.json` 只在本机构建，不提交；没有它时自动纯新题生成。
- 扩展词中文释义运行时从本地 MDict 获取，不写入本目录。

## 版权

教材 PDF、OCR 页面图像、UniDic/MDict、原题库均不进入仓库，也不随项目分发；本目录仅保留复习所需的结构化字段和来源引用。
