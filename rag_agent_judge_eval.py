#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG / Agent 业务大模型评测脚本
==============================

用途：针对两类业务场景，使用一个"裁判模型"(Judge Model) 按场景化多维评分标准
自动打分，并输出逐条结果与分组汇总统计：

- RAG（知识库问答）：评估基于检索知识的回答质量
  - faithfulness（忠实度）：回答是否忠实于检索到的知识库内容，是否存在幻觉/编造
  - relevance（相关性）：是否直接回应问题、是否正确使用了知识片段
  - completeness（完整性）：是否覆盖问题全部关键点
  - citation（引用准确性）：引用/溯源是否准确对应知识库片段
  - clarity（清晰性）：表达是否清晰、结构是否合理

- Agent（智能体任务）：评估智能体执行用户任务的过程与结果质量
  - task_completion（任务完成度）：是否完成用户任务目标
  - tool_accuracy（工具调用正确性）：工具选择、参数与调用时机是否正确
  - planning（规划合理性）：步骤规划是否合理、顺序是否正确
  - recovery（错误处理）：遇到错误/异常时能否正确恢复
  - efficiency（效率）：步骤是否冗余、是否高效达成目标

特性：
- OpenAI 兼容协议：支持 DeepSeek / 通义千问 / 智谱 GLM / OpenAI / 火山方舟 / Ollama 本地等
- 环境变量与 OpenCompass 评测环境保持一致：LLM_API_KEY / LLM_API_BASE / LLM_MODEL_NAME
- 输入：JSONL（推荐）或 CSV，字段为 type / question / reference_answer / answer，
  RAG 场景可含 context（检索片段），Agent 场景可含 tool_calls（工具调用轨迹）
- 输出：逐条结果 CSV + 分组汇总报告 JSON + 汇总 Markdown
- 并发打分、失败重试、JSON 解析容错

依赖：
    pip install openai

用法示例：
    # 方式一：环境变量驱动
    export LLM_API_KEY=sk-xxx
    export LLM_API_BASE=https://api.deepseek.com
    export LLM_MODEL_NAME=deepseek-chat
    python rag_agent_judge_eval.py --input samples.jsonl --output results/

    # 方式二：命令行参数优先（可选 --type 只评测指定场景）
    python rag_agent_judge_eval.py --input samples.csv --output results/ \
        --api-key sk-xxx --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
        --model qwen-plus --workers 4 --max-samples 100 --type rag

Windows PowerShell 下设置环境变量的等价写法：
    $env:LLM_API_KEY='sk-xxx'
    $env:LLM_API_BASE='https://api.deepseek.com'
    $env:LLM_MODEL_NAME='deepseek-chat'
    python rag_agent_judge_eval.py --input samples.jsonl --output results/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    print(
        "[错误] 缺少 openai 依赖，请先执行：pip install openai",
        file=sys.stderr,
    )
    sys.exit(1)


# ============================================================
# 一、场景化打分 Prompt 设计
# ============================================================

# 每个场景：维度列表（顺序）、权重、系统提示词
SCENE_CONFIG: Dict[str, Dict[str, Any]] = {
    "rag": {
        "label": "知识库问答（RAG）",
        "dims": [
            ("faithfulness", "忠实度"),
            ("relevance", "相关性"),
            ("completeness", "完整性"),
            ("citation", "引用准确性"),
            ("clarity", "清晰性"),
        ],
        "weights": {
            "faithfulness": 0.30,
            "relevance": 0.25,
            "completeness": 0.20,
            "citation": 0.15,
            "clarity": 0.10,
        },
        "system_prompt": """你是一位严谨、公正、经验丰富的知识库问答（RAG）系统评测专家，负责对基于检索增强生成（Retrieval-Augmented Generation）的回答进行打分。你的评分必须客观、可复现，严格依据评分标准执行，不受模型名称、回答风格或长度影响。

# 评分维度（每项 0-10 分）

1. faithfulness（忠实度）：回答是否忠实于检索到的知识库内容，是否出现知识库中不存在的幻觉、编造或篡改信息。
2. relevance（相关性）：回答是否直接回应问题，是否正确使用了与问题相关的知识片段，是否答非所问。
3. completeness（完整性）：回答是否覆盖问题的全部关键点，是否有重大遗漏。
4. citation（引用准确性）：回答中的引用、溯源、编号是否准确对应所提供的知识库片段，是否张冠李戴。
5. clarity（清晰性）：回答结构是否清晰、表达是否通顺、是否易于理解。

# 评分标准（各维度通用）

- 9-10 分：完全符合该维度要求，无明显瑕疵。
- 7-8 分：基本符合，仅有轻微不足。
- 5-6 分：部分符合，存在明显不足或部分错误。
- 3-4 分：较大程度不符合，存在严重错误或重大遗漏。
- 0-2 分：基本不符合，答非所问、严重幻觉或完全错误。

# 总分计算

- total = round(0.30*faithfulness + 0.25*relevance + 0.20*completeness + 0.15*citation + 0.10*clarity, 1)
- 总分范围 0-10，保留 1 位小数。

# 输出要求（必须严格遵守）

- 只输出一个 JSON 对象，不要输出任何其他文字、解释或 Markdown 代码块标记。
- JSON 结构如下：
{"scores": {"faithfulness": 8, "relevance": 7, "completeness": 6, "citation": 9, "clarity": 8}, "total": 7.6, "reason": "简要说明扣分原因与优点，不超过 150 字。"}
""",
        "user_template": """请对以下知识库问答场景的模型回答进行评分。

## 问题
{question}

## 检索到的知识库片段
{context}

## 参考答案（评分参考，非唯一标准）
{reference_answer}

## 待评模型回答
{answer}

请严格按照系统提示词中的评分维度、评分标准和输出格式给出评分。""",
    },
    "agent": {
        "label": "智能体任务（Agent）",
        "dims": [
            ("task_completion", "任务完成度"),
            ("tool_accuracy", "工具调用正确性"),
            ("planning", "规划合理性"),
            ("recovery", "错误处理"),
            ("efficiency", "效率"),
        ],
        "weights": {
            "task_completion": 0.30,
            "tool_accuracy": 0.25,
            "planning": 0.20,
            "recovery": 0.15,
            "efficiency": 0.10,
        },
        "system_prompt": """你是一位严谨、公正、经验丰富的智能体（Agent）评测专家，负责对智能体执行用户任务的过程与结果进行打分。你的评分必须客观、可复现，严格依据评分标准执行，不受模型名称、回答风格或长度影响。

# 评分维度（每项 0-10 分）

1. task_completion（任务完成度）：智能体是否完整达成用户的任务目标，输出结果是否满足需求。
2. tool_accuracy（工具调用正确性）：工具选择是否恰当、参数是否正确、调用时机是否合理，是否误用或漏用工具。
3. planning（规划合理性）：任务拆解与步骤规划是否合理、顺序是否正确、是否有必要的前置操作。
4. recovery（错误处理）：遇到工具报错、数据缺失或意外情况时，是否能正确识别、调整并继续完成。
5. efficiency（效率）：是否用最少的必要步骤达成目标，是否存在明显冗余、空转或重复操作。

# 评分标准（各维度通用）

- 9-10 分：完全符合该维度要求，无明显瑕疵。
- 7-8 分：基本符合，仅有轻微不足。
- 5-6 分：部分符合，存在明显不足或部分错误。
- 3-4 分：较大程度不符合，存在严重错误或重大遗漏。
- 0-2 分：基本不符合，任务未完成或完全错误。

# 总分计算

- total = round(0.30*task_completion + 0.25*tool_accuracy + 0.20*planning + 0.15*recovery + 0.10*efficiency, 1)
- 总分范围 0-10，保留 1 位小数。

# 输出要求（必须严格遵守）

- 只输出一个 JSON 对象，不要输出任何其他文字、解释或 Markdown 代码块标记。
- JSON 结构如下：
{"scores": {"task_completion": 8, "tool_accuracy": 7, "planning": 6, "recovery": 9, "efficiency": 8}, "total": 7.5, "reason": "简要说明扣分原因与优点，不超过 150 字。"}
""",
        "user_template": """请对以下智能体任务的执行过程与结果进行评分。

## 用户任务
{question}

## 工具调用轨迹
{tool_calls}

## 参考答案（评分参考，非唯一标准）
{reference_answer}

## 智能体最终输出
{answer}

请严格按照系统提示词中的评分维度、评分标准和输出格式给出评分。""",
    },
}


def get_scene(scene_type: str) -> Dict[str, Any]:
    """获取场景配置，校验场景名"""
    if scene_type not in SCENE_CONFIG:
        raise ValueError(
            f"不支持的场景类型：{scene_type}，可选：{' / '.join(SCENE_CONFIG.keys())}"
        )
    return SCENE_CONFIG[scene_type]


def build_messages(sample: "Sample") -> List[Dict[str, str]]:
    """按样本场景组装裁判调用消息"""
    cfg = get_scene(sample.type)
    if sample.type == "rag":
        user_prompt = cfg["user_template"].format(
            question=sample.question,
            context=sample.context or "（未提供检索片段）",
            reference_answer=sample.reference_answer or "（未提供）",
            answer=sample.answer or "（模型未输出）",
        )
    else:  # agent
        user_prompt = cfg["user_template"].format(
            question=sample.question,
            tool_calls=sample.tool_calls or "（未提供工具调用轨迹）",
            reference_answer=sample.reference_answer or "（未提供）",
            answer=sample.answer or "（智能体未输出）",
        )
    return [
        {"role": "system", "content": cfg["system_prompt"]},
        {"role": "user", "content": user_prompt},
    ]


# ============================================================
# 二、数据结构
# ============================================================


@dataclass
class Sample:
    """单个评测样本，type 区分 rag / agent 场景"""

    type: str
    question: str
    reference_answer: str = ""
    answer: str = ""
    context: str = ""  # RAG：检索到的知识库片段
    tool_calls: str = ""  # Agent：工具调用轨迹
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeResult:
    """单个样本的裁判打分结果"""

    sample: Sample
    scores: Dict[str, float] = field(default_factory=dict)
    total: Optional[float] = None
    reason: str = ""
    ok: bool = False
    error: str = ""


# ============================================================
# 三、样本加载
# ============================================================


def load_samples(input_path: str) -> List[Sample]:
    """加载 JSONL 或 CSV 格式的评测样本。

    JSONL：每行一个 JSON 对象，支持字段 type / question / reference_answer /
    answer / context / tool_calls / meta
    CSV：表头必须包含 type 与 question，建议包含 reference_answer / answer
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    suffix = path.suffix.lower()
    samples: List[Sample] = []

    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[警告] 第 {line_no} 行 JSON 解析失败，已跳过：{e}")
                    continue
                samples.append(_item_to_sample(item))
    elif suffix == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "question" not in reader.fieldnames:
                raise ValueError("CSV 表头必须包含 question 列")
            for row in reader:
                samples.append(_item_to_sample(row))
    else:
        raise ValueError("仅支持 .jsonl 或 .csv 格式输入")

    if not samples:
        raise ValueError("输入文件中没有有效样本")
    return samples


def _item_to_sample(item: Dict[str, Any]) -> Sample:
    """将一条输入记录转换为 Sample"""
    scene_type = str(item.get("type", "") or "rag").strip().lower()
    if scene_type not in SCENE_CONFIG:
        raise ValueError(
            f"样本 type 字段非法：{scene_type}，可选：{' / '.join(SCENE_CONFIG.keys())}"
        )
    question = str(item.get("question", "")).strip()
    if not question:
        raise ValueError("样本缺少 question 字段")
    extra_keys = ("type", "question", "reference_answer", "answer", "context", "tool_calls", "meta")
    meta = {k: v for k, v in item.items() if k not in extra_keys}
    if isinstance(item.get("meta"), dict):
        meta = {**item["meta"], **meta}
    return Sample(
        type=scene_type,
        question=question,
        reference_answer=str(item.get("reference_answer", "") or "").strip(),
        answer=str(item.get("answer", "") or "").strip(),
        context=str(item.get("context", "") or "").strip(),
        tool_calls=str(item.get("tool_calls", "") or "").strip(),
        meta=meta,
    )


# ============================================================
# 四、裁判调用与结果解析
# ============================================================

# 兜底：当模型不返回 JSON 时，用正则提取分数
_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")
_DIM_ALIAS = {
    "faithfulness": ("faithfulness", "忠实度", "忠实"),
    "relevance": ("relevance", "相关性", "相关"),
    "completeness": ("completeness", "完整性", "完整"),
    "citation": ("citation", "引用准确性", "引用"),
    "clarity": ("clarity", "清晰性", "清晰"),
    "task_completion": ("task_completion", "任务完成度", "完成度", "任务"),
    "tool_accuracy": ("tool_accuracy", "工具调用正确性", "工具调用", "工具"),
    "planning": ("planning", "规划合理性", "规划"),
    "recovery": ("recovery", "错误处理", "恢复"),
    "efficiency": ("efficiency", "效率"),
}


def parse_score(text: str, scene_type: str) -> Dict[str, Any]:
    """解析裁判输出，优先 JSON，失败时正则兜底。

    返回结构：{"scores": {...}, "total": float|None, "reason": str}
    """
    cfg = get_scene(scene_type)
    dim_keys = [d[0] for d in cfg["dims"]]
    text = (text or "").strip()
    # 去掉可能的 ```json ... ``` 包裹
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    if m:
        text = m.group(1).strip()

    result: Dict[str, Any] = {"scores": {}, "total": None, "reason": ""}

    # 尝试 JSON 解析
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            scores = data.get("scores") or data
            if isinstance(scores, dict):
                for dim in dim_keys:
                    val = scores.get(dim)
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        result["scores"][dim] = float(val)
            total = data.get("total")
            if isinstance(total, (int, float)) and not isinstance(total, bool):
                result["total"] = float(total)
            reason = data.get("reason", "")
            if isinstance(reason, str):
                result["reason"] = reason.strip()
            return result
    except json.JSONDecodeError:
        pass

    # 正则兜底：寻找 "维度名": 数字 或 "维度名：数字"
    for dim in dim_keys:
        for alias in _DIM_ALIAS.get(dim, (dim,)):
            pat = re.compile(
                rf'["\']?{re.escape(alias)}["\']?\s*[:：]\s*(\d+(?:\.\d+)?)'
            )
            mm = pat.search(text)
            if mm:
                val = float(mm.group(1))
                result["scores"][dim] = min(10.0, val)
                break

    # 总分兜底
    m_total = re.search(r'["\']?total["\']?\s*[:：]\s*(\d+(?:\.\d+)?)', text)
    if m_total:
        result["total"] = min(10.0, float(m_total.group(1)))
    # 理由提取：取末尾一段文字
    reason_m = re.search(r'["\']?reason["\']?\s*[:：]\s*["\']?(.*?)["\']?\s*$', text, re.S)
    if reason_m:
        result["reason"] = reason_m.group(1).strip()

    return result


def compute_total(scores: Dict[str, float], scene_type: str) -> float:
    """按场景加权公式计算总分（与 Prompt 中一致，保留 1 位小数）"""
    cfg = get_scene(scene_type)
    if not scores:
        return 0.0
    dim_keys = [d[0] for d in cfg["dims"]]
    weights = cfg["weights"]
    total = sum(scores.get(k, 0.0) * weights.get(k, 0.0) for k in dim_keys)
    # 用 Decimal 做四舍五入，避免二进制浮点误差导致 round 不稳定
    from decimal import Decimal, ROUND_HALF_UP
    return float(Decimal(str(total)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def call_judge(
    client: OpenAI,
    model: str,
    sample: Sample,
    timeout: float = 60.0,
    max_retries: int = 3,
    temperature: float = 0.0,
) -> JudgeResult:
    """调用裁判模型打分，失败自动重试"""
    result = JudgeResult(sample=sample)
    messages = build_messages(sample)

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("裁判模型返回空内容")
            parsed = parse_score(text, sample.type)

            if parsed["scores"]:
                result.scores = parsed["scores"]
            if parsed["total"] is None and result.scores:
                result.total = compute_total(result.scores, sample.type)
            else:
                result.total = parsed["total"]
            result.reason = parsed["reason"]
            result.ok = bool(result.scores or result.total is not None)
            if result.ok:
                return result
            raise ValueError("无法从裁判输出中解析出分数")
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** (attempt - 1)  # 1s, 2s, 4s 退避
                print(f"[重试] 第 {attempt} 次失败：{e}，{wait}s 后重试...")
                time.sleep(wait)
            else:
                result.ok = False
                result.error = str(e)
    return result


# ============================================================
# 五、并发评测与汇总
# ============================================================


def judge_all(
    client: OpenAI,
    model: str,
    samples: List[Sample],
    workers: int = 4,
    timeout: float = 60.0,
    max_retries: int = 3,
) -> List[JudgeResult]:
    """并发评测所有样本，按输入顺序返回结果"""
    results: List[Optional[JudgeResult]] = [None] * len(samples)

    def _judge_one(i: int, s: Sample) -> tuple[int, JudgeResult]:
        return i, call_judge(client, model, s, timeout, max_retries)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_judge_one, i, s) for i, s in enumerate(samples)]
        done = 0
        for fut in as_completed(futures):
            i, res = fut.result()
            results[i] = res
            done += 1
            status = "OK" if res.ok else f"FAIL({res.error[:40]})"
            print(f"[进度] {done}/{len(samples)}  {status}")
    return [r for r in results if r is not None]


def _scene_dim_keys(scene_type: str) -> List[str]:
    return [d[0] for d in get_scene(scene_type)["dims"]]


def summarize(results: List[JudgeResult]) -> Dict[str, Any]:
    """汇总统计，按场景分组"""
    ok_results = [r for r in results if r.ok]
    scene_types = sorted({r.sample.type for r in results})

    summary: Dict[str, Any] = {
        "total_samples": len(results),
        "judged_ok": len(ok_results),
        "judged_fail": len(results) - len(ok_results),
        "score": None,
        "scenes": {},
    }

    for st in scene_types:
        scene_ok = [r for r in ok_results if r.sample.type == st]
        scene_all = [r for r in results if r.sample.type == st]
        dim_keys = _scene_dim_keys(st)
        dims = {}
        for dim in dim_keys:
            vals = [r.scores.get(dim) for r in scene_ok if dim in r.scores]
            dims[dim] = {
                "mean": round(statistics.mean(vals), 2) if vals else None,
                "min": min(vals) if vals else None,
                "max": max(vals) if vals else None,
            }
        totals = [r.total for r in scene_ok if r.total is not None]
        buckets = {"0-2": 0, "2-4": 0, "4-6": 0, "6-8": 0, "8-10": 0}
        for t in totals:
            if t < 2:
                buckets["0-2"] += 1
            elif t < 4:
                buckets["2-4"] += 1
            elif t < 6:
                buckets["4-6"] += 1
            elif t < 8:
                buckets["6-8"] += 1
            else:
                buckets["8-10"] += 1
        scene_entry = {
            "label": get_scene(st)["label"],
            "total": len(scene_all),
            "judged_ok": len(scene_ok),
            "judged_fail": len(scene_all) - len(scene_ok),
            "score": round(statistics.mean(totals), 2) if totals else None,
            "score_min": min(totals) if totals else None,
            "score_max": max(totals) if totals else None,
            "dims": dims,
            "distribution": buckets,
        }
        if scene_ok:
            valid = [(r, r.total) for r in scene_ok if r.total is not None]
            if valid:
                valid.sort(key=lambda x: x[1])
                worst = valid[0][0]
                best = valid[-1][0]
                scene_entry["best_sample"] = {
                    "question": best.sample.question[:100],
                    "total": best.total,
                }
                scene_entry["worst_sample"] = {
                    "question": worst.sample.question[:100],
                    "total": worst.total,
                }
        summary["scenes"][st] = scene_entry

    all_totals = [r.total for r in ok_results if r.total is not None]
    if all_totals:
        summary["score"] = round(statistics.mean(all_totals), 2)
    return summary


# ============================================================
# 六、输出
# ============================================================


def write_results(results: List[JudgeResult], out_dir: Path, summary: Dict[str, Any]) -> Path:
    """写入逐条结果 CSV，返回文件路径"""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "judge_results.csv"
    all_dim_keys: List[str] = []
    for st in SCENE_CONFIG:
        for dim in _scene_dim_keys(st):
            if dim not in all_dim_keys:
                all_dim_keys.append(dim)
    header = ["type", "question", "reference_answer", "answer", "total"] + all_dim_keys + ["reason", "ok", "error"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in results:
            row = [
                r.sample.type,
                r.sample.question,
                r.sample.reference_answer,
                r.sample.answer,
                r.total if r.total is not None else "",
            ]
            for dim in all_dim_keys:
                row.append(r.scores.get(dim, ""))
            row += [r.reason, "1" if r.ok else "0", r.error]
            writer.writerow(row)
    return csv_path


def write_summary(summary: Dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """写入汇总 JSON 与 Markdown，返回 (json_path, md_path)"""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md_path = out_dir / "summary.md"
    lines = [
        "# RAG / Agent 业务评测汇总",
        "",
        f"- 评测时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 样本总数：{summary.get('total_samples')}",
        f"- 评测成功：{summary.get('judged_ok')}",
        f"- 评测失败：{summary.get('judged_fail')}",
        f"- 平均总分：{summary.get('score')}",
        "",
    ]
    for st, scene in summary.get("scenes", {}).items():
        lines += [
            f"## {scene['label']}（{st}）",
            "",
            f"- 样本数：{scene['total']}，成功：{scene['judged_ok']}，失败：{scene['judged_fail']}",
            f"- 平均总分：{scene['score']}（最低 {scene.get('score_min')}，最高 {scene.get('score_max')}）",
            "",
            "| 维度 | 均值 | 最低 | 最高 |",
            "|---|---|---|---|",
        ]
        for dim, v in scene.get("dims", {}).items():
            lines.append(f"| {dim} | {v.get('mean')} | {v.get('min')} | {v.get('max')} |")
        if scene.get("distribution"):
            lines += ["", "| 总分区间 | 数量 |", "|---|---|"]
            for bucket, cnt in scene["distribution"].items():
                lines.append(f"| {bucket} | {cnt} |")
        lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return json_path, md_path


# ============================================================
# 七、主入口
# ============================================================


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG / Agent 业务大模型评测脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="评测样本文件路径（.jsonl 或 .csv）")
    parser.add_argument("--output", default="results", help="输出目录")
    parser.add_argument("--type", default="", help="只评测指定场景：rag / agent（默认全部）")
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", ""), help="API Key（默认取环境变量 LLM_API_KEY）")
    parser.add_argument("--base-url", default=os.environ.get("LLM_API_BASE", ""), help="API Base URL（默认取环境变量 LLM_API_BASE）")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL_NAME", ""), help="裁判模型名（默认取环境变量 LLM_MODEL_NAME）")
    parser.add_argument("--workers", type=int, default=4, help="并发数")
    parser.add_argument("--max-samples", type=int, default=0, help="最多评测样本数（0=全部）")
    parser.add_argument("--timeout", type=float, default=60.0, help="单次请求超时（秒）")
    parser.add_argument("--max-retries", type=int, default=3, help="失败重试次数")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # 校验配置
    missing = []
    if not args.api_key:
        missing.append("API Key（--api-key 或环境变量 LLM_API_KEY）")
    if not args.base_url:
        missing.append("Base URL（--base-url 或环境变量 LLM_API_BASE）")
    if not args.model:
        missing.append("模型名（--model 或环境变量 LLM_MODEL_NAME）")
    if missing:
        print("[错误] 缺少必要配置：" + "；".join(missing), file=sys.stderr)
        print("示例：python rag_agent_judge_eval.py --input samples.jsonl --output results/ --api-key sk-xxx --base-url https://api.deepseek.com --model deepseek-chat")
        return 1

    try:
        samples = load_samples(args.input)
    except (FileNotFoundError, ValueError) as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1

    if args.type:
        try:
            get_scene(args.type)
        except ValueError as e:
            print(f"[错误] {e}", file=sys.stderr)
            return 1
        samples = [s for s in samples if s.type == args.type]
        if not samples:
            print(f"[错误] 没有类型为 {args.type} 的样本", file=sys.stderr)
            return 1

    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]
    scene_counts = {}
    for s in samples:
        scene_counts[s.type] = scene_counts.get(s.type, 0) + 1
    print(f"[信息] 加载 {len(samples)} 个样本（{scene_counts}），开始评测（模型：{args.model}）")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    t0 = time.time()
    results = judge_all(
        client=client,
        model=args.model,
        samples=samples,
        workers=args.workers,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    elapsed = time.time() - t0
    print(f"[信息] 评测完成，耗时 {elapsed:.1f}s")

    summary = summarize(results)
    out_dir = Path(args.output)
    csv_path = write_results(results, out_dir, summary)
    json_path, md_path = write_summary(summary, out_dir)

    print("\n========== 评测汇总 ==========")
    print(f"样本总数：{summary['total_samples']}，成功：{summary['judged_ok']}，失败：{summary['judged_fail']}")
    print(f"平均总分：{summary.get('score')}")
    for st, scene in summary.get("scenes", {}).items():
        print(f"[{st}] {scene['label']}：score={scene.get('score')} n={scene['total']}")
        for dim, v in scene.get("dims", {}).items():
            print(f"    {dim}: mean={v.get('mean')} min={v.get('min')} max={v.get('max')}")
    print("==============================")
    print(f"逐条结果：{csv_path}")
    print(f"汇总 JSON：{json_path}")
    print(f"汇总 Markdown：{md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
