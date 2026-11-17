# 数据分层

`data/` 只保留结构化结果；原始教材、真题、词典、数据库和缓存都在本地并被 gitignore。

## 运行时数据（提交）

```text
textbook/big2_autumn/
  vocab.json                 教材词表 + UniDic 复核标记
  grammar.json               语法点 + answer forms
  exam_blueprint.json        2024 真题配比
  course_profile.json        课程背景 + 来源配比预设 + mock blueprint
  extension_vocab.json       拓展词种子（日文词/读音/关系；无中文释义）
  locks.json                 唯一性锁表（反义对/语义场/搭配/线索标记）
  README.md
rules/
  question_types.json        题型与权重
  common_mistakes.json       设误规则（展示/本地参考）
exams/
  existing_question_index.json  真题去重索引（哈希/目标词频；无原文）
  big1_spring/.gitkeep
  big2_autumn/.gitkeep
```

## 本地构建/缓存（不提交）

```text
exams/
  raw_*.json                 原始结构化语料
  item_bank.local.json       本地原题库（原题/变体来源）
  big2_autumn/               生成批次 + rejected
textbook/big2_autumn/
  drafts/                    OCR/语法 forms 草稿
  pdf_page_map.json          开发期 OCR 页面映射
db/                          SQLite
cache/                       MDict/UniDic/备份缓存
logs/                        日志
```

## 原则

- 仓库只 `add` 运行必需的结构化 JSON 与代码。
- 原题、词典释义、OCR 图片、docx、PDF 不提交。
- 没有 `item_bank.local.json` 时自动纯新题生成；有原题库时启用原题/本地变体配比。
- 曝光记录写入本地 SQLite，不提交。
