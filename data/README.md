# 数据分层

`data/` 只保留运行必需的结构化结果；原始教材、PDF、词典、数据库、缓存和生成批次都在本地并被 gitignore。

- 数据契约：[docs/10-契约/17-数据分层与隐私.md](../docs/10-契约/17-数据分层与隐私.md)
- 数据制备：[docs/30-操作/32-数据制备.md](../docs/30-操作/32-数据制备.md)

## 提交

```text
textbook/big2_autumn/{vocab,grammar,exam_blueprint,course_profile,extension_vocab,locks}.json
textbook/big2_autumn/README.md
rules/{question_types,common_mistakes}.json
exams/existing_question_index.json
```

## 本地生成、不提交

```text
exams/item_bank.local.json
exams/raw_*.json
exams/items/
exams/quarantine/
exams/big2_autumn/
db/ cache/ logs/
mdicts/ unidic-cwj-*/
textbook/*/drafts/
textbook/*/ocr/
```

真实词典名、中文释义、原题库、DB、日志和 API Key 不进入 Git。
