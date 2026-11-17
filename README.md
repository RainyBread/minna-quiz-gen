# JLPTCraft

大二上学期《综合日语3》日语题库生成与教研系统。纯 CLI，无前端，通过 `init.py` 启动。

## 定位

- **课内是锚**：教材词表/语法点提供考点、难度与答案权威。
- **AI 是面**：LLM 只负责语境和干扰项，正确答案由本地注入。
- **本地优先**：本地随机抽样 → 原题/本地变体 → MDict 补充 → 每题一次 LLM → 本地清洗/组卷。
- **配比归编排，不归模型**：来源配比在扇出前确定；同一考点第二次出现必须换新表面。
- **不超纲是默认**：用户可以自愿选择“课内 + 拓展干扰项”或“拓展学习”。

## 快速开始

```bash
cd E:/MyData/大工/2026/minna-quiz-gen
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash
pip install -r requirements.txt

cp env.template .env
# 只填 LLM_API_KEY；默认 DeepSeek-compatible / deepseek-chat

python init.py --check
python init.py
```

`init.py` 默认只做快速检查并进入 REPL；**不会**每次导入语料、跑 OCR 或重新挖掘规则。clone 后只需要配置 LLM Key，运行数据已随仓库提供。

### 自然语言示例

```text
jp> 生成一套模拟卷
jp> 生成3道第44课读音题
jp> 生成2道第44课语法题
jp> 围绕旅行生成10道拓展题
jp> 只出课内题
jp> 生成20题单元测
jp> 生成一份100题第44-50课考卷
```

如果输入缺少关键信息（范围/题量/题型），CLI 会**追问缺失项**，而不是直接抛错。例如：

```text
未指定题型，按真题配比生成？[Y/n]
检测到考卷请求：按 80 题五类（2024 配比）生成？[Y/n]
本期不生成：阅读。改用读音/汉字/词汇/表达/语法五类配比？[Y/n]
```

## 出题流程

```text
需求解析（本地优先 + 轻量标签兜底）
  -> 考点抽样（课次分层、避开已用/重复）
  -> 每题来源编排：原题 / 本地变体 / 新生成
  -> 本地确定正确答案
  -> 每题一次 LLM（标签输出）
  -> 本地清洗选项、注入答案、打乱 A-D
  -> 并行执行，完成一题显示一题
  -> 归档 JSON/Markdown + DB + 曝光记录
```

关键点：

- LLM 只输出 `[CLUE]决定性线索[/CLUE]` + `[SENTENCE]...[[ans]]正确项[[/ans]]...[/SENTENCE]` + 翻译；**不写选项、不写答案字母、不写解析**，不要求严格 JSON。
- **正确答案由本地钉死**；完整句中必须原样出现固定正确项，系统再把该项替换成空格。这样不会出现“正确项是ます形、干扰项变形更对”的漂移。
- 干扰项由本地锁词表/词表构造：优先选与正确项不同语义场、同活用形/同词性的词；不够时本地补位。解析由本地模板拼装“线索 + 正确项释义 + 干扰项释义”，不做排除式论证。
- 裸框架题（无线索槽）直接弃掉补位；同场聚堆（四选项同语义场且无线索）也弃掉。
- 单题失败只记录并继续；LLM 连续认证/连接失败才熔断。
- 不做语义校验/LLM-as-judge；正确性来自本地答案、锁词表、来源锚点和本地清洗。

## 来源配比

用途 → 预设档位，写死在 `data/textbook/big2_autumn/course_profile.json`，扇出前每题来源已确定：

| 用途 | 课内原题 | 本地变体 | 新生成 |
|---|---:|---:|---:|
| 首学 / 巩固 `first_learn` | 50% | 30% | 20% |
| 单元测 `unit_test` | 20% | 30% | 50% |
| 模拟卷 `mock` | 5% | 20% | 75% |
| 错题复现 `review` | 0% | 40% | 60% |

- 原题/本地变体需要本地原题库（`data/exams/item_bank.local.json`，不提交）。
- 没有本地原题库时自动降级为纯新题，配额由新题补位；不会因缺原题而中断。
- `--use-case first_learn|unit_test|mock|review` 可显式指定；自然语言中的“单元测/错题复习/模拟卷/考卷”会自动识别。
- 曝光记账：同一点第二次出现时，原题不再使用，必须走变体或新题。

## MDict 扩展

把任意 `.mdx` 放进本地 MDict 文件夹（默认 `data/mdicts/`）。系统会**递归扫描全部 `.mdx`**，运行时按内容特征自动分类（中文/双语、关联词、发音、单语等），不依赖具体词典名称。

三种范围：

```bash
python main.py "生成3道第44课读音题"                 # 默认课内
python main.py "课内题，加拓展干扰项"                 # 目标课内，干扰项可拓展
python main.py "围绕旅行生成10道拓展题"               # 拓展学习（目标可教材外）
```

- `in_syllabus`：目标、正确项、干扰项全部来自教材；模拟卷强制此模式。
- `mdict_distractors`：目标/正确项课内，少量干扰项/解析来自拓展词。
- `extend`：目标可为教材外词，题目和导出标注“拓展学习”。

扩展词的中文释义在运行时从本地 MDict 获取，**不写进提交的 JSON**；仓库只保存日文词/读音/关系/通用来源类型。

命令：

```text
/dict <word>                 查询全部本地词典
/extend <topic>              查看主题拓展词
/extend build <topic>        构建/合并拓展词
```

## 数据分层

```text
data/
  textbook/big2_autumn/
    vocab.json               运行数据：教材词表（已 UniDic 复核）
    grammar.json             运行数据：语法点/答案形式
    exam_blueprint.json      2024 真题配比
    course_profile.json      课程背景 + 来源配比
    extension_vocab.json     拓展词种子（无中文释义）
    locks.json               唯一性锁表（反义对/语义场/搭配）
    README.md
  rules/
    question_types.json      运行数据：题型/权重
    common_mistakes.json     运行数据：设误规则（展示/本地参考）
  exams/
    existing_question_index.json  去重索引（哈希/目标词频）
    item_bank.local.json     本地原题库（gitignored）
    raw_*.json               本地原始语料（gitignored）
    big1_spring/             只读归档
    big2_autumn/             生成批次（gitignored）
  db/ cache/ logs/           运行时数据（gitignored）
```

仓库只提交代码 + 运行必需结构化数据；教材 PDF、docx、OCR 图片、UniDic、MDict、DB、缓存、日志、原始题库都不提交。

## 开发期数据工具

```bash
# 1) 用 UniDic 复核全部教材读音（构建期跑一次）
python tools/verify_vocab.py

# 2) 构建本地原题库（可选用；没有则纯新题）
python tools/build_item_bank.py

# 3) 44-50 语法 answer forms：LLM 提取候选 -> 人工审核 -> 合并
python tools/fill_grammar_forms.py
python tools/review_grammar_forms.py
python tools/review_grammar_forms.py --merge

# 4) 真题去重索引
python tools/build_novelty_index.py

# 5) 拓展词种子（读取本地 MDict）
python tools/build_extension_vocab.py

# 6) 开发语料重建（会保留 44-50 已审核数据和语法 forms）
python tools/rebuild_data.py

# 7) OCR / 旧数据（可选，仅开发期）
python tools/extract_textbook.py --force
python tools/review_textbook.py
python tools/check_drafts.py
```

`.env` 只保留 LLM Key；现有复杂 `.env` 可用：

```bash
python tools/simplify_env.py
```

## 测试与验收

```bash
python -m pytest -q
python main.py --self-test
python init.py --check
python main.py "生成3道第44课读音题" --dry-run       # 不需要 UniDic/LLM
python main.py "生成一套模拟卷" --dry-run              # 需要 LLM
python main.py "围绕旅行生成10道拓展题" --dry-run      # 需要本地 MDict
python tools/check_repo_hygiene.py
```

`--self-test` 走通本地读音题 → 归档 → DB → Markdown，不需要 LLM/UniDic。

## CLI 参数

```text
--mock             生成一套 80 题五类模拟卷
--scope             in_syllabus | mdict_distractors | extend
--topic            拓展主题
--use-case         first_learn | unit_test | mock | review
--workers          LLM 并发数（默认 6）
--seed             随机种子
--hide-answers     流式显示时不打印答案/解析
--no-stream        全部完成后再输出
--dry-run          不写 DB/Markdown
--json             打印批次 JSON
--offline          仅本地读音/汉字题
--health / --list / --export / --self-test
```

REPL 命令：`/help /list /show /export /rules /approve /reject /remine /stats /env /dict /extend /mode /seed /workers /quit`。

## 支持范围

- 生成：读音、假名选汉字、词汇、表达、语法。
- 模拟卷：80 题五类（30/10/5/10/25），不含阅读/排序/完形 20 题。
- 读音：优先使用构建期 UniDic 复核为 `true` 的教材 `kana`；没有 UniDic 也能运行。3 条特殊读音与 32 条 UniDic 不匹配条目跳过读音/汉字题，词义题保留。
- 语法：44-50 的 31 个点已审核合并，共 52 个点有 answer forms。
- 表达：固定表达由本地语法点钉死；LLM 只写含线索的对话语境，选项/解析本地构造。

## 版权与隐私

- 不复制、不分发教材/真题 PDF、图片、docx。
- UniDic/MDict 路径与 API Key 只在本地 `.env`，不随仓库分发。
- 仓库可见文件不出现具体 MDict 词典名称；代码/README/committed JSON 只使用通用类型。
- 原题库、词典释义、生成批次只在本地；请勿分发。
- 旧项目 `dlut-nihongo-quiz/` 只读，不修改、不提交。
