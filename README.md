# RAG / Agent 业务大模型评测

基于大模型裁判（LLM-as-a-Judge）的**业务场景定制化**自动化评测工具，针对两类典型业务场景提供独立的评分维度与打分标准：

- **知识库问答（RAG）**：评估基于检索知识的回答质量，重点防幻觉、查引用
- **智能体任务（Agent）**：评估智能体执行任务的过程与结果，重点考察工具调用与规划

## 特性

- **场景化打分 Prompt**：RAG 与 Agent 各一套系统提示词，维度、权重、总分公式独立定制
- **OpenAI 兼容协议**：支持 DeepSeek / 通义千问 / 智谱 GLM / OpenAI / 火山方舟 / Ollama 本地等
- **环境变量与 OpenCompass 评测环境一致**：`LLM_API_KEY` / `LLM_API_BASE` / `LLM_MODEL_NAME`
- **输入灵活**：JSONL（推荐）或 CSV，`type` 字段自动识别场景，可混排
- **输出完整**：逐条结果 CSV + 分组汇总 JSON + 汇总 Markdown
- **工程化**：并发打分、失败重试、JSON 解析容错、Decimal 精确四舍五入

## 目录结构

| 文件 | 作用 |
|---|---|
| [rag_agent_judge_eval.py](rag_agent_judge_eval.py) | 可执行评测脚本，内嵌 RAG / Agent 两套打分 Prompt |
| [samples_demo.jsonl](samples_demo.jsonl) | 示例评测样本（5 条，含 RAG 与 Agent 两类） |
| [dryrun_test.py](dryrun_test.py) | 不调用 API 的逻辑验证脚本（41 项自检） |
| [RAG-Agent业务评测使用说明书.docx](RAG-Agent业务评测使用说明书.docx) | 标题分级的详细使用说明书（含完整源码附录） |
| [README.md](README.md) | 本说明文档 |

## 快速开始

### 1. 安装依赖

需要 Python 3.9 及以上版本，仅依赖 openai 官方 SDK：

```bash
pip install openai
```

### 2. 配置环境变量

**Linux / macOS：**

```bash
export LLM_API_KEY=sk-xxxx
export LLM_API_BASE=https://api.deepseek.com
export LLM_MODEL_NAME=deepseek-chat
```

**Windows PowerShell：**

```powershell
$env:LLM_API_KEY="sk-xxxx"
$env:LLM_API_BASE="https://api.deepseek.com"
$env:LLM_MODEL_NAME="deepseek-chat"
```

### 3. 运行评测

```bash
python rag_agent_judge_eval.py --input samples_demo.jsonl --output results/
```

只评测某一场景（如仅 RAG）：

```bash
python rag_agent_judge_eval.py --input samples.jsonl --output results/ --type rag
```

### 4. 快速验证（dry-run）

不调用任何 API，仅验证脚本自身的加载、解析与汇总逻辑：

```bash
python dryrun_test.py
# 全部输出 [OK] 即表示脚本逻辑正常
```

## 样本格式

JSONL 每行一个 JSON 对象，`type` 必填（`rag` / `agent`），`question` 必填：

```json
{"type": "rag", "question": "根据知识库，我们公司报销差旅费的审批流程是什么？", "reference_answer": "出差结束后 5 个工作日内提交报销单，部门负责人审批后财务复核。", "answer": "出差回来后 5 个工作日内提交报销单即可。", "context": "[片段1] 员工应在出差结束之日起 5 个工作日内完成差旅费报销申请。\n[片段2] 审批流程：部门负责人审批 → 财务部复核。"}
{"type": "agent", "question": "请帮我查一下上周五的销售数据，并生成一份环比对比报告。", "reference_answer": "查询销售数据后生成环比报告并输出关键结论。", "answer": "已生成环比对比报告：销售额环比 +12.3%。", "tool_calls": "1. query_sales_data(date='2026-10-16') -> {ok: true}\n2. generate_report(metric='mom') -> {ok: true}"}
```

CSV 表头必须包含 `type` 与 `question` 列。

| 字段 | 是否必填 | 说明 |
|---|---|---|
| `type` | 必填 | 场景类型：`rag`（知识库问答）或 `agent`（智能体任务） |
| `question` | 必填 | 待评测的用户问题或任务指令 |
| `reference_answer` | 建议 | 参考答案，供裁判评分参照 |
| `answer` | 建议 | 被测模型的回答或智能体最终输出 |
| `context` | 条件建议 | RAG 场景：检索到的知识库片段 |
| `tool_calls` | 条件建议 | Agent 场景：工具调用轨迹 |

## 打分维度与权重

### RAG（知识库问答）

| 维度 | 权重 | 考察内容 |
|---|---|---|
| faithfulness（忠实度） | 0.30 | 是否忠实于知识库内容，是否存在幻觉/编造 |
| relevance（相关性） | 0.25 | 是否直接回应问题，是否正确使用知识片段 |
| completeness（完整性） | 0.20 | 是否覆盖问题全部关键点 |
| citation（引用准确性） | 0.15 | 引用/溯源是否准确对应知识库片段 |
| clarity（清晰性） | 0.10 | 结构是否清晰、表达是否通顺 |

总分公式：`total = round(0.30*faithfulness + 0.25*relevance + 0.20*completeness + 0.15*citation + 0.10*clarity, 1)`

### Agent（智能体任务）

| 维度 | 权重 | 考察内容 |
|---|---|---|
| task_completion（任务完成度） | 0.30 | 是否完整达成用户任务目标 |
| tool_accuracy（工具调用正确性） | 0.25 | 工具选择、参数、调用时机是否正确 |
| planning（规划合理性） | 0.20 | 任务拆解与步骤规划是否合理 |
| recovery（错误处理） | 0.15 | 遇到报错/异常时能否正确恢复 |
| efficiency（效率） | 0.10 | 是否用最少必要步骤达成目标 |

总分公式：`total = round(0.30*task_completion + 0.25*tool_accuracy + 0.20*planning + 0.15*recovery + 0.10*efficiency, 1)`

两套系统提示词全文内嵌于 [rag_agent_judge_eval.py](rag_agent_judge_eval.py)，详见使用说明书第 4 章。

## 命令行参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input` | 必填 | 评测样本文件路径（.jsonl 或 .csv） |
| `--output` | `results` | 输出目录 |
| `--type` | （全部） | 只评测指定场景：`rag` / `agent` |
| `--api-key` | 环境变量 | API Key（`LLM_API_KEY`） |
| `--base-url` | 环境变量 | API Base URL（`LLM_API_BASE`） |
| `--model` | 环境变量 | 裁判模型名（`LLM_MODEL_NAME`） |
| `--workers` | `4` | 并发线程数 |
| `--max-samples` | `0` | 最多评测样本数（0 表示全部） |
| `--timeout` | `60.0` | 单次请求超时（秒） |
| `--max-retries` | `3` | 失败重试次数 |

## 输出结果

运行完成后，输出目录下生成三个文件：

- **judge_results.csv**：逐条结果，含 type、question、total、全部 10 个维度分、reason、ok、error
- **summary.json**：结构化汇总数据（总体 + 按场景分组）
- **summary.md**：人类可读的分组汇总报告

汇总报告按场景分组，包含：样本数/成功/失败、平均总分（最低/最高）、各维度均值、总分五档分布（0-2 / 2-4 / 4-6 / 6-8 / 8-10）、每场景最佳与最差样本。

## 常见问题

| 问题 / 注意点 | 处理方式 |
|---|---|
| RAG 样本缺少 context | 不影响运行，但会降低忠实度与引用准确性评分的可信度，建议尽量提供 |
| Agent 样本缺少 tool_calls | 会影响工具调用正确性与规划合理性评分，建议记录真实工具调用轨迹 |
| 场景混排 | 同一文件可混合两类样本，脚本按 type 自动分流并分组汇总 |
| API 限流 | 触发 429 时调低 `--workers`、调大 `--timeout` |
| 密钥安全 | 不要把真实 API Key 写入样本文件或提交到代码仓库 |
| 输出目录 | 重复运行会覆盖同名输出文件，建议按批次使用不同 `--output` 目录 |

## License

本项目仅用于评测学习与研究用途，使用前请遵守所调用模型服务商的条款与当地法律法规。
