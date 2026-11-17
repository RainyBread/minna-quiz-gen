---
模块: 生成与 LLM 适配
档次: L2
说明: 载体句提示词、标签解析、LLM provider/重试与本地选项装配
代码: core/quizgen.py#generate_llm, core/prompts.py#build_carrier_prompt, core/llm.py#LLMClient, core/tagformat.py#parse_carrier_tags, core/request_log.py#log_llm_request
契约: 10-契约/13-标签传输格式.md
验收: python -m pytest -q tests/test_llm_generation.py tests/test_tagformat.py tests/test_llm.py
校验: 2026-11-17
---

# 生成与 LLM 适配

## 1. 职责与边界

调用 LLM 生成载体句，解析标签，校验 span，并本地构造选项/解析/答案字母。LLM 不写选项、答案、解析、答案字母。

- `core.prompts.build_carrier_prompt`：新协议 `[s]/[t]`。
- `core.tagformat.parse_carrier_tags`：宽容解析与 offsets。
- `core.quizgen.generate_llm`：提示词 → LLM → 解析 → span 校验 → 本地干扰项/解析 → `GeneratedQuestion`。
- `core.llm.LLMClient`：OpenAI-compatible/Anthropic 文本与视觉适配。
- `core.request_log`：只记 prompt 的审计日志。

## 2. 输入输出契约

输入：`QuestionSpec`、`LLMClient`、`Settings`、`TextbookPools`、`LockTable`、`MDictIndex`、`ExtensionPool`。

输出：`GeneratedQuestion`，随后由 runner `_finalize` 转成 canonical `Item`。

提示词契约：

```text
[s]完整句子或短对话，把要考的词或形式用 [t]...[/t] 标出[/s]
```

旧 `[SENTENCE]/[[ans]]`、全角括号、缺闭合、`[[word]]` 由解析器兼容，但新提示词只承诺 `[s]/[t]`。

## 3. 数据与依赖

- `spec.correct_text` / `approved_forms` / `written`：本地事实，不能由 LLM 改写。
- `locks.clue_hint` / `is_clue_present`：本地线索。
- `mdict.chinese_gloss`：拓展词/干扰项释义。
- `settings.llm_*`：provider、base URL、model、timeout、retries、temperature。
- `request_log` 写入 `data/logs/llm_requests/*.jsonl`（gitignored）。

## 4. 本地不变量

- 词汇题会显示目标词（它就是正确项）；语法/表达题隐藏固定形式。线索提示经 `mask_secrets` 清洗，不把批准形式/正确项拼进提示。
- `[t]` span 必须等于 `correct_text`，或属于批准形式/目标词形；否则 `E_ZERO`/`E_PARSE`。
- 缺完整句 → `E_PARSE`；缺决定性线索 → `E_MULTI`。
- LLM 输出的选项/解析不进入活跃路径；活跃路径只取 carrier/clue/translation。
- 错误信息经 `sanitize_secret_text` 清洗；请求日志不含 Key/headers/response。

## 5. 失败与降级

- LLM 调用重试 `llm_retries`；401/403/认证类错误立即熔断。
- 单题失败：记录 `RunFailure`，继续；由 runner 走 original→variant→local-grammar。
- LLM 不可用：读音/汉字题仍可离线；模拟卷报 `LLMUnavailable`。
- 标签缺失/span 不合法：拒题，不尝试语义修复。

## 6. 扩展点

- 新 provider：在 `core.llm` 增加分支，保持 `available`/`complete_text` 接口与密钥清洗。
- 新提示词：改 `build_carrier_prompt`，同步 `tagformat` 协议与测试；不要恢复“让模型写选项”的旧模式。
- 新标签：在 `parse_carrier_tags` 增加兼容，并在 L1-13 更新契约。
- 新审计字段：在 `request_log` 增加元数据，禁止写入 Key/response。

## 7. 验收

```bash
python -m pytest -q tests/test_llm_generation.py tests/test_tagformat.py tests/test_llm.py
python main.py "生成2道第44课词汇题" --dry-run
```

`test_llm_generation` 用 FakeLLM 验证 `[s]/[t]` 正确项钉死与本地选项装配；`test_tagformat` 覆盖全角/缺闭合/`[[ans]]`。

## 8. 已知缺陷

- DRIFT-04：`E_MULTI` 语义是“缺线索”；多个 `[t]` 只取第一个。
- DRIFT-05：`build_question_prompt`/`TAG_TEMPLATE` 是未使用 legacy；新协议以 `build_carrier_prompt` 为准。
- 没有运行时 LLM mock；测试使用 FakeLLM，`MockRetriever` 不是 LLM mock。
- 相关文档：[13-标签传输格式](../10-契约/13-标签传输格式.md)、[24-装配审计与降级](24-装配审计与降级.md)、[28-CLI与配置](28-CLI与配置.md)。
