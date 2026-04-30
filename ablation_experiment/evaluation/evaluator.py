"""
Evaluation Module

Uses Qwen/Qwen3.5-122B-A10B on SiliconFlow to evaluate generated reports
using a 5-dimension scoring rubric (each 0-20 points, total 0-100).

Dimensions:
1. Methodology & Analysis Framework Rigor
2. Strategic Alignment & External Environment Fit
3. Logical Coherence & Strategic Closed-Loop Thinking
4. KPI Setting & Practical Implementation Value
5. Implicit Constraint Insight & Organizational Governance Depth
"""

import re
import json
import csv
import time
from typing import Dict, List
from openai import OpenAI

from ablation_experiment.config import (
    EVAL_API_KEY, EVAL_BASE_URL, EVAL_MODEL, EVAL_TIMEOUT,
    RESULTS_DIR, NUM_RUNS, NUM_EVAL_ROUNDS, EVAL_TEMPERATURE,
    EVAL_INTERVAL, OUTLIER_METHOD, TRIM_COUNT
)


# ============================================================================
# SCORING RUBRIC PROMPT (optimized for reliable JSON output)
# ============================================================================

SCORING_SYSTEM_PROMPT = """你是一位拥有20年经验的国际顶尖战略咨询合伙人，专门评估战略规划报告。

你必须严格按JSON格式输出评分，不要输出任何其他文字、分析或解释。不要使用markdown代码块。直接输出纯JSON。

## 评分维度（每项0-20分，允许0.5分精度）

### 维度一：方法论运用与分析框架严谨度 (0-20)
评估各章节是否正确、深入地运用了指定的战略分析模型（PEST/SWOT/BCG/五力/BSC/安索夫/7S/ESG等）。
- 18-20: 模型框架运用精准、维度完整、各要素间逻辑衔接紧密
- 14-17: 模型框架基本正确，但维度覆盖不完整或分析深度不够
- 10-13: 模型框架使用有偏差，维度缺失或混淆
- 0-9: 未使用指定模型或使用完全错误

### 维度二：战略一致性与外部环境契合度 (0-20)
评估报告是否紧密结合政策导向（交通强国、交通强省、国企改革），对外部环境认知是否准确深入。
- 18-20: 政策引用精准、外部环境认知深刻、战略定位高度契合
- 14-17: 政策引用较准确但深度不足，或战略定位有偏差
- 10-13: 政策引用泛泛，缺乏针对性分析
- 0-9: 脱离政策背景，战略定位模糊

### 维度三：逻辑连贯性与战略闭环思维 (0-20)
评估诊断阶段→战略推演→实施举措之间是否形成完整闭环，章节间逻辑是否连贯。
- 18-20: 诊断-战略-举措完美闭环，问题-对策一一对应，章节间高度连贯
- 14-17: 基本形成闭环，但部分对策缺乏诊断依据，或章节间有脱节
- 10-13: 闭环不完整，诊断与举措脱节
- 0-9: 各章节独立无关联，无闭环思维

### 维度四：KPI设定与实操落地指导价值 (0-20)
评估KPI是否符合SMART原则，实施路径是否有具体的资源、时间表、责任人。
- 18-20: KPI完全符合SMART，资源路径清晰到预算和人天级别
- 14-17: KPI基本可量化，但实施路径颗粒度不够
- 10-13: KPI过于宏观，缺乏量化指标和实施路径
- 0-9: 几乎无可操作的KPI和实施路径

### 维度五：隐性约束洞察与组织治理深度 (0-20)
评估报告是否识别了组织内部的隐性约束（利益相关方博弈、变革阻力、文化惯性），治理策略是否深入。
- 18-20: 深刻洞察组织摩擦力，提出具体可行的变革管理策略
- 14-17: 识别了部分组织约束，但应对策略偏宏观
- 10-13: 对组织约束的认知停留在表面
- 0-9: 忽视组织内部约束，治理建议空泛"""


def _extract_chapter_content(report_text: str) -> str:
    """Extract only chapter content from a report, stripping structural overhead.

    Removes: cover page, TOC, executive summary, blueprint appendix,
    consistency notes. Keeps only the actual chapter body text so the
    evaluator sees the maximum amount of analytical content.
    """
    import re

    lines = report_text.split('\n')
    chapter_lines = []
    in_chapter = False
    in_appendix = False

    for line in lines:
        stripped = line.strip()

        # Skip until we hit the first chapter heading
        if not in_chapter:
            # Detect first chapter: starts with # and contains 第X章 or 第x章
            if re.match(r'^#\s*第[一二三四五六七八九十\d]+章', stripped):
                in_chapter = True
                chapter_lines.append(line)
            continue

        # Stop at appendix section
        if stripped.startswith('#') and ('附录' in stripped or 'Appendix' in stripped
                                          or '战略蓝图' in stripped):
            in_appendix = True
            continue

        # Stop at consistency review notes
        if '一致性审查' in stripped or 'consistency' in stripped.lower():
            break

        if in_appendix:
            continue

        # Skip section separators
        if stripped == '---':
            continue

        chapter_lines.append(line)

    content = '\n'.join(chapter_lines).strip()

    # If extraction yielded too little (malformed report), fall back to original
    if len(content) < len(report_text) * 0.3:
        return report_text

    return content


def _build_scoring_prompt(report_text: str) -> str:
    """Build the user prompt for evaluation, keeping it compact."""

    # Strip structural overhead first so only chapter content is evaluated
    chapter_content = _extract_chapter_content(report_text)

    # Truncate to fit context window - use generous budget for better discrimination
    if len(chapter_content) > 25000:
        eval_text = chapter_content[:18000] + "\n\n...[中间章节内容省略]...\n" + chapter_content[-7000:]
    else:
        eval_text = chapter_content

    return f"""请评估以下战略规划报告，直接输出JSON，不要输出其他任何文字。

报告内容:
{eval_text}

输出格式（严格遵守）:
{{"d1_score":0,"d1_analysis":"简评","d2_score":0,"d2_analysis":"简评","d3_score":0,"d3_analysis":"简评","d4_score":0,"d4_analysis":"简评","d5_score":0,"d5_analysis":"简评","total_score":0,"suggestions":"建议"}}

说明: d1=方法论, d2=战略一致, d3=逻辑闭环, d4=KPI实操, d5=组织治理。每项0-20分，total=五项之和，analysis和suggestions各限50字以内。"""


# ============================================================================
# EVALUATOR
# ============================================================================

class ReportEvaluator:
    """Evaluates reports using Qwen model with 5-dimension rubric."""

    def __init__(self):
        if not EVAL_API_KEY:
            raise ValueError("SILICONFLOW_API_KEY environment variable not set")

        self.client = OpenAI(
            api_key=EVAL_API_KEY,
            base_url=EVAL_BASE_URL,
            timeout=EVAL_TIMEOUT
        )
        self.model = EVAL_MODEL

    def evaluate(self, report_text: str) -> Dict:
        """
        Evaluate a single report.

        Args:
            report_text: Full report text to evaluate

        Returns:
            Dict with dimension scores, total, and improvement suggestions
        """
        prompt = _build_scoring_prompt(report_text)

        for attempt in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=EVAL_TEMPERATURE,
                    # Key fix: large enough for Qwen thinking tokens + JSON output
                    max_tokens=8192,
                    # Disable thinking mode on SiliconFlow to save tokens
                    extra_body={"enable_thinking": False} if "siliconflow" in EVAL_BASE_URL else None,
                )

                result_text = response.choices[0].message.content

                # Check if response was truncated
                finish_reason = response.choices[0].finish_reason
                if finish_reason == "length":
                    print(f"  [WARN] Response truncated at max_tokens (attempt {attempt+1}), retrying with shorter input...")
                    # Retry with even shorter chapter content
                    if attempt < 4:
                        from ablation_experiment.evaluation.evaluator import _extract_chapter_content
                        shorter = _extract_chapter_content(report_text)
                        prompt = _build_scoring_prompt(shorter[:8000])
                        continue

                parsed = self._parse_evaluation(result_text)
                if parsed.get("total_score", 0) > 0:
                    return parsed
                else:
                    print(f"  [WARN] Parsed score is 0, retrying... (attempt {attempt+1}/5)")
                    if attempt < 4:
                        time.sleep(3)

            except Exception as e:
                err_str = str(e)
                wait = 2 ** attempt
                # Longer wait for rate limiting (429)
                if "429" in err_str or "rate" in err_str.lower():
                    wait = max(wait, 15)
                    print(f"  [WARN] Rate limited (attempt {attempt+1}/5), waiting {wait}s...")
                else:
                    print(f"  [WARN] Evaluation attempt {attempt+1}/5 failed: {e}")
                if attempt < 4:
                    time.sleep(wait)

        # All attempts failed, try one more time with minimal prompt
        return self._emergency_evaluate(report_text)

    def _parse_evaluation(self, response: str) -> Dict:
        """Parse evaluation response into structured dict."""
        response = response.strip()

        # Remove <think/> blocks (Qwen thinking mode)
        response = re.sub(r'<think[^>]*>.*?</think[^>]*>', '', response, flags=re.DOTALL)

        # Remove markdown code blocks
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        # Try direct JSON parse
        result = self._try_parse_json(response)
        if result is not None:
            return self._normalize_result(result)

        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = self._try_parse_json(json_match.group())
            if result is not None:
                return self._normalize_result(result)

        return self._get_fallback_evaluation("JSON parse failed")

    def _try_parse_json(self, text: str) -> dict:
        """Try to parse JSON, return None on failure."""
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

    def _normalize_result(self, result: dict) -> Dict:
        """Normalize various JSON formats into the standard output format."""

        # Check for compact format (d1_score, d2_score, etc.)
        if "d1_score" in result:
            return self._normalize_compact(result)

        # Check for full Chinese dimension names
        dim_names = [
            "维度一_方法论运用与分析框架严谨度",
            "维度二_战略一致性与外部环境契合度",
            "维度三_逻辑连贯性与战略闭环思维",
            "维度四_KPI设定与实操落地指导价值",
            "维度五_隐性约束洞察与组织治理深度",
        ]

        scores = {}
        for dim in dim_names:
            if dim in result and isinstance(result[dim], dict):
                score = result[dim].get("score", 0)
                scores[dim] = {
                    "score": max(0, min(20, int(score))),
                    "analysis": result[dim].get("analysis", "")
                }
            else:
                scores[dim] = {"score": 0, "analysis": "评分缺失"}

        total = sum(s["score"] for s in scores.values())
        improvement = result.get("improvement_suggestions", "")
        if not improvement:
            improvement = result.get("suggestions", "")

        return {
            "scores": scores,
            "total_score": total,
            "improvement_suggestions": improvement
        }

    def _normalize_compact(self, result: dict) -> Dict:
        """Normalize the compact JSON format (d1_score, d2_score, ...)."""
        dim_mapping = {
            "d1": "维度一_方法论运用与分析框架严谨度",
            "d2": "维度二_战略一致性与外部环境契合度",
            "d3": "维度三_逻辑连贯性与战略闭环思维",
            "d4": "维度四_KPI设定与实操落地指导价值",
            "d5": "维度五_隐性约束洞察与组织治理深度",
        }

        scores = {}
        for key, dim_name in dim_mapping.items():
            score = result.get(f"{key}_score", 0)
            analysis = result.get(f"{key}_analysis", "")
            scores[dim_name] = {
                "score": max(0, min(20, int(score))),
                "analysis": analysis
            }

        total = result.get("total_score", 0)
        if total == 0:
            total = sum(s["score"] for s in scores.values())

        suggestions = result.get("suggestions", "")
        if not suggestions:
            suggestions = result.get("improvement_suggestions", "")

        return {
            "scores": scores,
            "total_score": total,
            "improvement_suggestions": suggestions
        }

    def _emergency_evaluate(self, report_text: str) -> Dict:
        """Last resort: evaluate with minimal prompt to maximize success."""
        short_text = report_text[:4000]

        prompt = f"""给以下报告打分(纯JSON，不要其他文字):
{{"d1":0,"d2":0,"d3":0,"d4":0,"d5":0,"total":0,"sug":"建议"}}
d1-d5各0-20分。报告摘要:{short_text[:2000]}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "只输出JSON。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2048,
                extra_body={"enable_thinking": False} if "siliconflow" in EVAL_BASE_URL else None,
            )
            raw = response.choices[0].message.content
            # Strip think blocks
            raw = re.sub(r'<think[^>]*>.*?</think[^>]*>', '', raw, flags=re.DOTALL)
            result = self._try_parse_json(raw)
            if result:
                # Convert short keys to compact format
                compact = {}
                for i in range(1, 6):
                    key = f"d{i}"
                    if key in result:
                        compact[f"d{i}_score"] = max(0, min(20, int(result[key])))
                        compact[f"d{i}_analysis"] = ""
                compact["total_score"] = result.get("total", sum(compact.get(f"d{i}_score", 0) for i in range(1, 6)))
                compact["suggestions"] = result.get("sug", "")
                return self._normalize_compact(compact)
        except Exception:
            pass

        return self._get_fallback_evaluation("All evaluation attempts failed")

    # ========================================================================
    # MULTI-ROUND EVALUATION (repeated scoring for statistical robustness)
    # ========================================================================

    def evaluate_with_repetition(self, report_text: str, num_rounds: int = None) -> Dict:
        """对同一份报告独立打分 num_rounds 次，返回截尾均值聚合结果。

        Args:
            report_text: 完整报告文本
            num_rounds: 打分轮次（默认从配置读取 NUM_EVAL_ROUNDS=10）

        Returns:
            Dict with aggregated scores, std, 95% CI, stability flags, raw values
        """
        if num_rounds is None:
            num_rounds = NUM_EVAL_ROUNDS

        raw_scores = []
        for round_id in range(num_rounds):
            print(f"    评估轮次 {round_id + 1}/{num_rounds}...", end="", flush=True)
            result = self.evaluate(report_text)
            total = result.get("total_score", 0)
            print(f" 总分={total}")

            if total > 0:  # 只保留成功的评估
                raw_scores.append(result)

            if round_id < num_rounds - 1:
                time.sleep(EVAL_INTERVAL)

        if not raw_scores:
            return self._get_fallback_evaluation("所有评估轮次均失败")

        return self._aggregate_scores(raw_scores)

    def _aggregate_scores(self, raw_scores: List[Dict]) -> Dict:
        """聚合多次打分结果：截尾均值 + 标准差 + 95%置信区间 + 稳定性标记。"""
        import statistics

        dim_keys = [
            "维度一_方法论运用与分析框架严谨度",
            "维度二_战略一致性与外部环境契合度",
            "维度三_逻辑连贯性与战略闭环思维",
            "维度四_KPI设定与实操落地指导价值",
            "维度五_隐性约束洞察与组织治理深度",
        ]

        n = len(raw_scores)
        trim = min(TRIM_COUNT, max(0, n // 2 - 1))  # 安全边界

        aggregated = {}
        all_totals = []

        for dim_key in dim_keys:
            # 收集该维度所有轮次的分数，排序
            values = sorted([
                s.get("scores", {}).get(dim_key, {}).get("score", 0)
                for s in raw_scores
            ])

            all_totals.append(values)

            # 截尾均值：去掉最高最低各 trim 个
            trimmed = values[trim : n - trim] if trim > 0 else values
            trimmed_n = len(trimmed)

            mean_val = statistics.mean(trimmed)
            std_val = statistics.stdev(trimmed) if trimmed_n > 1 else 0

            # 95% 置信区间（t 分布）
            if trimmed_n > 1 and std_val > 0:
                # 内联 t 值，避免 scipy 依赖
                # df=4→2.776, df=5→2.571, df=6→2.447, df=7→2.365, df=8→2.306
                t_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776,
                           5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
                           9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086}
                df = trimmed_n - 1
                t_val = t_table.get(df, 2.262 if df > 10 else 2.776)
                ci = t_val * (std_val / (trimmed_n ** 0.5))
            else:
                ci = 0

            # 稳定性标记：标准差 ≤ 维度满分(20)的20% = 4分
            is_stable = std_val <= 4.0

            # 取中位数轮次的分析评语
            mid_idx = n // 2
            analysis = raw_scores[mid_idx].get("scores", {}).get(dim_key, {}).get("analysis", "")

            aggregated[dim_key] = {
                "score": round(mean_val, 1),
                "std": round(std_val, 2),
                "ci_95": round(ci, 2),
                "stable": is_stable,
                "raw_values": values,
                "analysis": analysis,
            }

        total_mean = round(sum(v["score"] for v in aggregated.values()), 1)

        # 取中位数轮次的改进建议
        mid_idx = n // 2
        suggestions = raw_scores[mid_idx].get("improvement_suggestions", "")

        return {
            "scores": aggregated,
            "total_score": total_mean,
            "num_rounds": n,
            "trim_count": trim,
            "outlier_method": OUTLIER_METHOD,
            "improvement_suggestions": suggestions,
        }

    def _get_fallback_evaluation(self, error_msg: str) -> Dict:
        """Return fallback evaluation when scoring fails."""
        dimensions = [
            "维度一_方法论运用与分析框架严谨度",
            "维度二_战略一致性与外部环境契合度",
            "维度三_逻辑连贯性与战略闭环思维",
            "维度四_KPI设定与实操落地指导价值",
            "维度五_隐性约束洞察与组织治理深度",
        ]

        scores = {}
        for dim in dimensions:
            scores[dim] = {"score": 0, "analysis": f"评分失败: {error_msg}"}

        return {
            "scores": scores,
            "total_score": 0,
            "improvement_suggestions": f"评分过程出错: {error_msg}"
        }


# ============================================================================
# RESULTS EXPORT
# ============================================================================

def save_results_csv(results: List[Dict], filepath: str = None) -> str:
    """
    Save evaluation results to CSV file with confidence intervals and raw values.
    """
    if filepath is None:
        filepath = str(RESULTS_DIR / "evaluation_report.csv")

    dim_keys = [
        ("维度一_方法论", "维度一_方法论运用与分析框架严谨度"),
        ("维度二_战略一致", "维度二_战略一致性与外部环境契合度"),
        ("维度三_逻辑闭环", "维度三_逻辑连贯性与战略闭环思维"),
        ("维度四_KPI实操", "维度四_KPI设定与实操落地指导价值"),
        ("维度五_组织治理", "维度五_隐性约束洞察与组织治理深度"),
    ]

    # Build fieldnames from dim_keys to ensure exact match with row dict keys
    fieldnames = ["group", "run", "num_rounds", "total_score"]
    for short_key, _ in dim_keys:
        fieldnames.extend([
            f"{short_key}", f"{short_key}_std",
            f"{short_key}_ci95", f"{short_key}_stable", f"{short_key}_raw",
        ])
    fieldnames.append("improvement_suggestions")

    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            scores = r.get("scores", {})
            row = {
                "group": r.get("group", ""),
                "run": r.get("run", ""),
                "num_rounds": r.get("num_rounds", ""),
                "total_score": r.get("total_score", 0),
            }

            for short_key, full_key in dim_keys:
                dim_data = scores.get(full_key, {})
                row[f"{short_key}"] = dim_data.get("score", 0)
                row[f"{short_key}_std"] = dim_data.get("std", 0)
                row[f"{short_key}_ci95"] = dim_data.get("ci_95", 0)
                row[f"{short_key}_stable"] = dim_data.get("stable", "")
                row[f"{short_key}_raw"] = str(dim_data.get("raw_values", []))

            row["improvement_suggestions"] = r.get("improvement_suggestions", "")[:500]
            writer.writerow(row)

    return filepath


def print_summary(results: List[Dict]) -> str:
    """
    Print and return a summary of evaluation results with mean, std, CI per group.
    """
    import statistics

    group_names = {
        "group0": "组0: 基线RAG (无工作流/无记忆)",
        "group1": "组1: 单Agent+记忆 (无工作流/有记忆)",
        "group2": "组2: 多Agent无记忆 (有工作流/无记忆)",
        "group3": "组3: 完整系统 (有工作流/有记忆)",
    }

    dim_short = {
        "维度一_方法论运用与分析框架严谨度": "方法论",
        "维度二_战略一致性与外部环境契合度": "战略一致",
        "维度三_逻辑连贯性与战略闭环思维": "逻辑闭环",
        "维度四_KPI设定与实操落地指导价值": "KPI实操",
        "维度五_隐性约束洞察与组织治理深度": "组织治理",
    }

    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("  消融实验评估结果汇总 (多轮重复评估)")
    lines.append("=" * 80)

    for group_id in ["group0", "group1", "group2", "group3"]:
        group_results = [r for r in results if r.get("group") == group_id]
        if not group_results:
            continue

        label = group_names.get(group_id, group_id)
        lines.append(f"\n{'─' * 60}")
        lines.append(f"  {label}")
        lines.append(f"{'─' * 60}")

        # Collect scores
        totals = [r.get("total_score", 0) for r in group_results]
        dim_scores = {}
        dim_stds = {}
        for dim_key in dim_short:
            dim_scores[dim_key] = [
                r.get("scores", {}).get(dim_key, {}).get("score", 0)
                for r in group_results
            ]
            dim_stds[dim_key] = [
                r.get("scores", {}).get(dim_key, {}).get("std", 0)
                for r in group_results
            ]

        # Print individual run scores with CI
        for i, r in enumerate(group_results):
            num_rounds = r.get("num_rounds", "?")
            lines.append(f"  Run {i+1}: 总分={r.get('total_score', 0)}/100 "
                         f"(基于{num_rounds}轮打分的截尾均值)")

        # Print averages with confidence intervals
        if totals:
            mean_total = statistics.mean(totals)
            std_total = statistics.stdev(totals) if len(totals) > 1 else 0
            lines.append(f"\n  平均总分: {mean_total:.1f} ± {std_total:.1f}")

        for dim_key, dim_label in dim_short.items():
            scores = dim_scores.get(dim_key, [])
            stds = dim_stds.get(dim_key, [])
            if scores:
                mean_s = statistics.mean(scores)
                std_s = statistics.stdev(scores) if len(scores) > 1 else 0
                avg_eval_std = statistics.mean(stds) if stds else 0
                lines.append(f"  {dim_label}: {mean_s:.1f} ± {std_s:.1f} "
                             f"(评估内平均波动: {avg_eval_std:.1f})")

    lines.append(f"\n{'=' * 80}")
    lines.append("  实验完成!")
    lines.append(f"{'=' * 80}\n")

    summary = "\n".join(lines)
    print(summary)

    # Also save summary to file
    summary_path = str(RESULTS_DIR / "evaluation_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)

    return summary
