---
模块: CLI 与配置
档次: L2
说明: 命令行参数、REPL、init 启动器、Settings/环境变量与配置错误
代码: main.py#parse_args, init.py#main, quiz.py#_run, config.py#load_settings
契约: 10-契约/17-数据分层与隐私.md
验收: python main.py --help
校验: 2026-11-17
---

# CLI 与配置

## 1. 职责与边界

解析 CLI 参数、进入 REPL、加载 `Settings`、处理配置错误。业务逻辑在 `core`；CLI 只做编排与展示。

## 2. 输入输出契约

`main.py`：

| 参数 | 作用 |
|---|---|
| `request`（位置） | 自然语言需求；省略则 REPL。 |
| `--mock` | 生成 80 题五类模拟卷；无 request 时注入“生成一套模拟卷”。 |
| `--scope` / `--topic` / `--use-case` | 范围/主题/来源配比。 |
| `--workers` / `--seed` | LLM 并发/随机种子。 |
| `--answers` | `auto/separate/inline/only/hidden`。 |
| `--reveal` | hidden 模式逐题揭示答案。 |
| `--hide-answers` | 流式不打印答案/解析。 |
| `--no-stream` / `--json` | 全部完成后渲染/输出批次 JSON。 |
| `--dry-run` | 不写 DB/Markdown/曝光（仍可能调用 LLM）。 |
| `--offline` | 禁用 LLM，仅本地读音/汉字题。 |
| `--health` / `--list` / `--export` / `--no-answers` / `--self-test` | 环境/批次/导出/自检。 |
| `--audit [BATCH_ID]` | 审计 Item/视图/请求日志；省略批次审计最近一批。 |

`quiz.py`：`python quiz.py audit [BATCH_ID] [--json]`；其余参数转交 `main.main`。

`init.py`：`--check`、`--check-dev`、`--remine`、`--offline`、`--yes`；已废弃的 `--skip-mining`/`--no-import`/`--no-ocr` 是 no-op。

REPL：`/help /list /show /export /rules /approve /reject /remine /stats /env /dict /extend /mode /seed /workers /quit`。

## 3. 数据与依赖

`config.load_settings` 读取 `.env`（shell 环境变量优先）：

- `ACTIVE_SEMESTER`、`DATA_DIR`/`DB_DIR`/`CACHE_DIR`/`LOG_DIR`、`UNIDIC_PATH`/`UNIDIC_ROOT`、`MDICT_DIR`、`LEGACY_QUIZ_ROOT`、`TEXTBOOK_PDF_DIR`。
- `LLM_PROVIDER`（`openai`/`anthropic`）、`LLM_BASE_URL`、`LLM_MODEL`、`LLM_API_KEY`、`LLM_CONCURRENCY`、`LLM_TIMEOUT`、`LLM_MAX_TOKENS`、`LLM_TEMPERATURE`、`LLM_RETRIES`。
- `VISION_*`：OCR/视觉可选。

默认：DeepSeek-compatible `https://api.deepseek.com/v1` / `deepseek-chat`，并发 6，超时 60s，max_tokens 1200，temperature 0.7，retries 2。

## 4. 本地不变量

- `Semester.parse` 只接受 `big1_spring`/`big2_autumn`；拒绝大二下/未知学期。
- 非本地 `http://` base URL 拒绝；`LLM_API_KEY` 是显式覆盖，否则按 host 选 provider Key。
- `.env` 创建尝试 0600；`.env`/`.env.*` 不提交。
- `--dry-run` 与 `--offline` 不是一回事：前者不写库，后者禁用 LLM。
- 配置错误（`ConfigurationError`/`SemesterViolation`）返回 exit 2，不打印 Key。

## 5. 失败与降级

- 配置错误 → `main`/`init` 捕获，打印“配置错误”，exit 2。
- 无 LLM 且请求模拟卷 → `LLMUnavailable`；非模拟卷裁剪为读音/汉字。
- 无 UniDic/MDict → 自动降级；见 [22-备料与词典](22-备料与词典.md)。
- `--audit` 失败 → exit 1；成功 → 0。
- 未知 REPL 命令 → 提示 `/help`，不退出。

## 6. 扩展点

- 新 CLI 参数：`parse_args` + `_generate` 传参 + L2-28/L3-33 文档 + 测试。
- 新 REPL 命令：`run_repl` 分支；保持 `QuizPlan` 入口。
- 新 env：`Settings` 字段 + `load_settings` + `env.template` + L1-17。
- 新 provider：`core.llm` + `_select_llm_api_key` + 安全测试。

## 7. 验收

```bash
python main.py --help
python main.py --health
python init.py --check
python -m pytest -q tests/test_config_semester.py
```

`--help` 应列出 `--answers/--reveal/--audit`；`init.py --check` 显示环境/覆盖；配置测试覆盖学期与 base URL 校验。

## 8. 已知缺陷

- DRIFT-01：模拟卷配比键名漂移，当前硬编码回退。
- DRIFT-02：`init.py --check` 的 72/72 forms 与运行时 `approved_points` 口径不同。
- DRIFT-08：`is_active_target` 未用于生成守卫；禁用学期更多靠数据缺失。
- 相关文档：[17-数据分层与隐私](../10-契约/17-数据分层与隐私.md)、[33-命令手册](../30-操作/33-命令手册.md)、[35-排障与降级](../30-操作/35-排障与降级.md)。
