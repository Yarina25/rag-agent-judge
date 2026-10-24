#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dry-run 测试：不调用任何 API，仅验证 rag_agent_judge_eval.py 的
场景配置、样本加载、消息组装、分数解析、加权总分与分组汇总逻辑。
运行：python dryrun_test.py
"""

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).parent / "rag_agent_judge_eval.py"
spec = importlib.util.spec_from_file_location("judge", SCRIPT)
judge = importlib.util.module_from_spec(spec)
sys.modules["judge"] = judge  # dataclass 需要模块注册，否则报 AttributeError
spec.loader.exec_module(judge)

PASS = 0
FAIL = 0


def check(name: str, cond: bool, extra: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"[OK] {name}")
    else:
        FAIL += 1
        print(f"[FAIL] {name} {extra}")


def make_sample(stype, question="q", **kw):
    return judge.Sample(
        type=stype,
        question=question,
        reference_answer=kw.get("reference_answer", "ref"),
        answer=kw.get("answer", "ans"),
        context=kw.get("context", "ctx"),
        tool_calls=kw.get("tool_calls", "tool trace"),
    )


# ---------- 1. 场景配置 ----------
check("场景类型包含 rag 与 agent", {"rag", "agent"} <= set(judge.SCENE_CONFIG.keys()))
check("RAG 5 维度", len(judge.get_scene("rag")["dims"]) == 5)
check("Agent 5 维度", len(judge.get_scene("agent")["dims"]) == 5)
check(
    "RAG 权重合计=1",
    abs(sum(judge.get_scene("rag")["weights"].values()) - 1.0) < 1e-9,
    str(judge.get_scene("rag")["weights"]),
)
check(
    "Agent 权重合计=1",
    abs(sum(judge.get_scene("agent")["weights"].values()) - 1.0) < 1e-9,
    str(judge.get_scene("agent")["weights"]),
)
try:
    judge.get_scene("xxx")
    check("非法场景应抛错", False)
except ValueError:
    check("非法场景应抛错", True)

# ---------- 2. 样本加载（JSONL） ----------
sample_lines = [
    {
        "type": "rag",
        "question": "质保期多久？",
        "reference_answer": "整机 2 年",
        "answer": "整机 2 年",
        "context": "产品手册：整机质保 2 年。",
    },
    {
        "type": "agent",
        "question": "更新合同状态",
        "reference_answer": "先查后改",
        "answer": "已更新",
        "tool_calls": "find_contract -> update",
        "meta": {"case_id": "A-1"},
    },
]
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "s.jsonl"
    p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in sample_lines), encoding="utf-8")
    samples = judge.load_samples(str(p))
    check("JSONL 加载 2 条", len(samples) == 2)
    check("rag 样本 context 保留", samples[0].context == "产品手册：整机质保 2 年。")
    check("agent 样本 tool_calls 保留", samples[1].tool_calls == "find_contract -> update")
    check("meta 保留", samples[1].meta.get("case_id") == "A-1")

    # CSV
    p2 = Path(td) / "s.csv"
    p2.write_text("type,question,reference_answer,answer\nrag,问题1,参考答案1,回答1\nagent,问题2,参考答案2,回答2\n", encoding="utf-8")
    s2 = judge.load_samples(str(p2))
    check("CSV 加载 2 条", len(s2) == 2 and s2[0].type == "rag" and s2[1].type == "agent")

    # 非法 type
    p3 = Path(td) / "bad.jsonl"
    p3.write_text(json.dumps({"type": "xx", "question": "q"}, ensure_ascii=False), encoding="utf-8")
    try:
        judge.load_samples(str(p3))
        check("非法 type 应抛错", False)
    except ValueError:
        check("非法 type 应抛错", True)

    # 空文件
    p4 = Path(td) / "empty.jsonl"
    p4.write_text("", encoding="utf-8")
    try:
        judge.load_samples(str(p4))
        check("空文件应抛错", False)
    except ValueError:
        check("空文件应抛错", True)

# ---------- 3. 消息组装 ----------
rag_s = make_sample("rag", question="问题", context="片段A", answer="回答", reference_answer="参考")
msgs = judge.build_messages(rag_s)
check("RAG 消息 2 条", len(msgs) == 2 and msgs[0]["role"] == "system" and msgs[1]["role"] == "user")
check("RAG 系统提示含忠实度", "faithfulness" in msgs[0]["content"])
check("RAG 用户提示注入上下文", "片段A" in msgs[1]["content"])
agt_s = make_sample("agent", question="任务", tool_calls="trace-1", answer="输出")
msgs2 = judge.build_messages(agt_s)
check("Agent 消息组装", len(msgs2) == 2 and "trace-1" in msgs2[1]["content"])
check("Agent 系统提示含任务完成度", "task_completion" in msgs2[0]["content"])

# ---------- 4. 分数解析 ----------
ok_json = judge.parse_score(
    '{"scores": {"faithfulness": 8, "relevance": 7, "completeness": 6, "citation": 9, "clarity": 8}, "total": 7.6, "reason": "整体良好"}',
    "rag",
)
check("JSON 解析维度", ok_json["scores"].get("faithfulness") == 8.0 and ok_json["scores"].get("clarity") == 8.0)
check("JSON 解析 total", ok_json["total"] == 7.6)
check("JSON 解析 reason", ok_json["reason"] == "整体良好")

codeblock = judge.parse_score('```json\n{"scores": {"relevance": 5, "correctness": 0}}\n```', "rag")
check("代码块包裹容错", codeblock["scores"].get("relevance") == 5.0)

zh_fallback = judge.parse_score("忠实度：7 分，相关性：8 分，完整性：6 分，引用准确性：9 分，清晰性：8 分", "rag")
check("中文别名兜底 RAG", zh_fallback["scores"].get("faithfulness") == 7.0 and zh_fallback["scores"].get("citation") == 9.0)

agent_fallback = judge.parse_score("任务完成度：8，工具调用正确性：9，规划合理性：7，错误处理：6，效率：8", "agent")
check(
    "中文别名兜底 Agent",
    agent_fallback["scores"].get("task_completion") == 8.0
    and agent_fallback["scores"].get("tool_accuracy") == 9.0
    and agent_fallback["scores"].get("efficiency") == 8.0,
)

# ---------- 5. 加权总分（Decimal 四舍五入） ----------
rag_scores = {"faithfulness": 8, "relevance": 7, "completeness": 6, "citation": 9, "clarity": 8}
total_rag = judge.compute_total(rag_scores, "rag")
exp_rag = round(0.30 * 8 + 0.25 * 7 + 0.20 * 6 + 0.15 * 9 + 0.10 * 8, 1)
check("RAG 加权总分", abs(total_rag - exp_rag) < 1e-9, f"{total_rag} vs {exp_rag}")

# 7.35 -> 7.4 场景（Decimal 修正）：0.30*7+0.25*8+0.20*6+0.15*7+0.10*10 = 7.35
half = judge.compute_total({"faithfulness": 7, "relevance": 8, "completeness": 6, "citation": 7, "clarity": 10}, "rag")
check("四舍五入边界 (7.35->7.4)", half == 7.4, f"{half}")

agt_scores = {"task_completion": 8, "tool_accuracy": 7, "planning": 6, "recovery": 9, "efficiency": 8}
total_agt = judge.compute_total(agt_scores, "agent")
exp_agt = round(0.30 * 8 + 0.25 * 7 + 0.20 * 6 + 0.15 * 9 + 0.10 * 8, 1)
check("Agent 加权总分", abs(total_agt - exp_agt) < 1e-9, f"{total_agt} vs {exp_agt}")

# ---------- 6. 汇总（分组） ----------
res = [
    judge.JudgeResult(sample=rag_s, scores=dict(rag_scores), total=total_rag, reason="r", ok=True),
    judge.JudgeResult(sample=make_sample("rag", question="q2"), scores=dict(rag_scores), total=total_rag, reason="r", ok=True),
    judge.JudgeResult(sample=agt_s, scores=dict(agt_scores), total=total_agt, reason="a", ok=True),
    judge.JudgeResult(sample=make_sample("agent", question="q3"), ok=False, error="timeout"),
]
summary = judge.summarize(res)
check("汇总总数 4", summary["total_samples"] == 4)
check("成功 3 失败 1", summary["judged_ok"] == 3 and summary["judged_fail"] == 1)
check("场景分组 2 个", set(summary["scenes"].keys()) == {"rag", "agent"})
check("RAG 场景均分", summary["scenes"]["rag"]["score"] == total_rag)
check("Agent 失败统计", summary["scenes"]["agent"]["judged_fail"] == 1)
check("RAG 分布合计 2", sum(summary["scenes"]["rag"]["distribution"].values()) == 2)
check("最佳最差样本存在", "best_sample" in summary["scenes"]["rag"] and "worst_sample" in summary["scenes"]["rag"])

# ---------- 7. 输出文件 ----------
with tempfile.TemporaryDirectory() as td:
    out = Path(td)
    csv_p = judge.write_results(res, out, summary)
    jp, mp = judge.write_summary(summary, out)
    check("CSV 已生成", csv_p.exists() and csv_p.stat().st_size > 0)
    check("JSON 已生成", jp.exists())
    check("MD 已生成", mp.exists() and "RAG" in mp.read_text(encoding="utf-8"))
    csv_txt = csv_p.read_text(encoding="utf-8-sig")
    check("CSV 含场景列", "type" in csv_txt.splitlines()[0])
    check("CSV 含全部维度列", all(d in csv_txt.splitlines()[0] for d in ("faithfulness", "task_completion", "efficiency")))
    check("CSV 含失败行", "timeout" in csv_txt)

# ---------- 8. 主入口缺配置 ----------
class _A:
    api_key = ""
    base_url = ""
    model = ""


# parse_args 直接注入缺配置参数
try:
    rc = judge.main(["--input", "samples_demo.jsonl", "--output", "tmp_out"])
    check("缺配置返回 1", rc == 1)
except SystemExit as e:
    check("缺配置返回 1（SystemExit）", e.code == 1)

print(f"\n========== 全部 dry-run 测试通过（{PASS} 项）==========" if FAIL == 0 else f"\n========== {FAIL} 项失败 ==========")
sys.exit(1 if FAIL else 0)
