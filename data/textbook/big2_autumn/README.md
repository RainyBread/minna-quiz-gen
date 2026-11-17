# big2_autumn 运行数据

本目录只放运行必需的结构化 JSON；字段契约见 [12-Item-schema](../../../docs/10-契约/12-Item-schema.md) 与 [14-任务单与配比](../../../docs/10-契约/14-任务单与配比.md)。

| 文件 | 用途 |
|---|---|
| `vocab.json` | 教材词表：词形、读音、中文、课次、UniDic 复核标记。 |
| `grammar.json` | 语法点、批准 answer forms、例句、审核状态。 |
| `exam_blueprint.json` | 2024 真题配比（构建期/参考）。 |
| `course_profile.json` | 课程背景、来源配比、模拟卷 blueprint。 |
| `extension_vocab.json` | 拓展词种子；只存日文词/读音/关系，不存中文释义。 |
| `locks.json` | 唯一性锁表：反义对/语义场/搭配/线索标记。 |
| `drafts/` | OCR/语法 forms 草稿（本地，不提交）。 |

覆盖检查：

```bash
python init.py --check
```

数据分层与提交边界见 [docs/10-契约/17-数据分层与隐私.md](../../../docs/10-契约/17-数据分层与隐私.md)。
