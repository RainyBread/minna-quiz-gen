---
档次: L1
说明: canonical Item v1 的字段、slot 不变量、hash/ID、legacy 投影与 items 表
代码: core/items.py#Item, core/items.py#validate_item, core/models.py#GeneratedQuestion, core/db.py#save_items
验收: python main.py --self-test
校验: 2026-11-17
红线: 7
---

# Item schema

<a id="redline-7"></a>
## 红线 7：Item 是唯一事实源，答案由 key 派生

`core.items.Item` 是题目的唯一事实源。`GeneratedQuestion`、Markdown、控制台、答案册都是它的投影/视图；新逻辑不得从题干或选项顺序反推答案。

## 1. 顶层字段（schema_version = "1"）

| 字段 | 类型 | 含义 |
|---|---|---|
| `schema_version` | str | 固定 `"1"`。 |
| `id` | str | `stable_item_id` 生成的 UUID5；同 seed/题型/考点/carrier/key 稳定。 |
| `type` | str | `QuestionType` 值，如 `vocab_choice`。 |
| `carrier` | str | 去标签的完整句子/对话。 |
| `slots` | list[ItemSlot] | carrier 中的目标区间；至少一个。 |
| `key` | ItemKey | 正确项本体与证据。 |
| `distractors` | list[ItemDistractor] | 3 个唯一干扰项。 |
| `explanation` | str | 本地拼装的解析；不得为空。 |
| `meta` | ItemMeta | 课次、考点、域、来源、hash、缺陷、provenance。 |

## 2. 子结构

| 结构 | 字段 |
|---|---|
| `ItemSlot` | `start`, `end`, `span`, `slot_type`, `left_boundary`, `right_boundary` |
| `ItemKey` | `value`, `lemma`, `reading`, `pos`, `sense_id`, `attestation` |
| `ItemDistractor` | `value`, `origin`, `slot_type` |
| `ItemMeta` | `lessons`, `point`, `domain`, `source`, `level`, `seed`, `hashes`, `defects`, `repairs`, `clue`, `prompt_visible`, `provenance` |

`domain` 与 `source` 不得混用：

- `domain`: `in`（课内）/ `out`（拓展）。
- `source`: `original`（原题）/ `variant`（变体）/ `ai`（新造）。

## 3. slot 不变量

1. `0 <= slot.start < slot.end <= len(carrier)`。
2. `carrier[slot.start:slot.end] == slot.span`。
3. 填空槽（`vocab`/`grammar_form`/`expression`）必须满足 `slot.span == key.value`。
4. 读音/汉字使用 focus slot：slot 标出被问的词，`key.value` 是读音/汉字答案，不要求等于 span。
5. `ItemSlot.slot_type` 取值：`reading`、`kanji`、`vocab`、`expression`、`grammar_form`、`other`。

违反 1–3 产生 `E_ANCHOR`；空 carrier/无 slot/空解析产生 `E_PARSE`；key 为空/干扰项不足/重复产生 `E_ZERO`。

## 4. hash 与 ID

`compute_hashes` 产出三个 SHA-256：

- `carrier`: `sha256(carrier)`。
- `key`: canonical JSON（排序键、无空格）的 `key`。
- `item`: canonical JSON 去掉 `meta.hashes` 后的整条 Item。

`stable_item_id` 使用 `uuid5(namespace, f"{seed}|{type}|{point}|{carrier_hash}|{key_hash}")`。同一条目重复生成应得到同一 ID；不同 seed 会改变 ID。

`meta.hashes` 只由 `refresh_hashes` 写入；手工改字段后必须重算，否则 `validate_item` 报 `E_PARSE: hash does not match canonical item`。

## 5. legacy 投影

`project_generated_question(item)` 生成旧消费者所需的 `GeneratedQuestion`：

- 正确项固定放在 `A`，`answer_key = "A"`。
- 干扰项按 `item.distractors[:3]` 顺序放 `B/C/D`。
- `stem` 由填 slot 替换为 `[[blank]]`；focus slot 保留 carrier。
- `source_kind`：`item.meta.source == "ai"` → `new`，否则原样。
- `scope`：`domain=out` → `extend`；`provenance.extension_distractors` → `mdict_distractors`；否则 `in_syllabus`。

投影只用于兼容；答案仍来自 `item.key.value`，不从 stem 反推。

## 6. 持久化

- `GeneratedBatch.items` 保存 canonical Item JSON；`GeneratedBatch.questions` 保存 legacy 投影。
- SQLite `items` 表（schema v3）列：`item_id`, `batch_id`, `schema_version`, `carrier`, `item_json`, `carrier_hash`, `key_hash`, `item_hash`, `created_at`；`item_hash` 唯一。
- `core.db.insert_batch` 同时写 `items` 与 `questions`；`save_items` 供迁移工具写 Items。
- 旧批次没有 Items 时，`tools/migrate_items_v1.py` 从 `questions`/归档 JSON 重建；失败写 `data/exams/quarantine/`。

## 7. 验收

```bash
python main.py --self-test
python tools/migrate_items_v1.py --dry-run
python main.py --audit
```

`--self-test` 会构造 Item、校验 slot、归档并写库；`--audit` 对最近一批做回环恒等检查。

## 8. 已知缺陷

- DRIFT-03：`core.exporter` 在批次没有 Items 时仍回退 legacy Markdown；回退路径只兼容旧批次，新逻辑不得依赖。
- 没有 Item 级已提交 quarantine 夹具；迁移工具写本地 `data/exams/quarantine/`。
- `ItemMeta.repairs` 只是保留字段，当前没有修复算子。
- 相关文档：[15-不变量与缺陷码](15-不变量与缺陷码.md)、[24-装配审计与降级](../20-模块/24-装配审计与降级.md)、[27-导出与视图](../20-模块/27-导出与视图.md)。
