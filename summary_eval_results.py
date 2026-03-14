# summary_eval_results.py
import os
import json
import argparse


def load_json(path):
    if not os.path.exists(path):
        print(f"[WARN] 文件不存在: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if cur is None or not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def fmt_float(x, digits=4):
    if x is None:
        return "-"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def build_model_row(model_name, summary):
    if summary is None:
        return {
            "Model": model_name,
            "Valid Decision Ratio": "-",
            "Optimal Decision Accuracy": "-",
            "Avg Env Reward": "-",
            "Avg Latency (ms)": "-",
            "Avg Energy (J)": "-",
            "Num Samples": "-"
        }

    return {
        "Model": model_name,
        "Valid Decision Ratio": fmt_float(summary.get("valid_decision_ratio")),
        "Optimal Decision Accuracy": fmt_float(summary.get("optimal_decision_accuracy")),
        "Avg Env Reward": fmt_float(summary.get("avg_env_reward")),
        "Avg Latency (ms)": fmt_float(summary.get("avg_latency_ms")),
        "Avg Energy (J)": fmt_float(summary.get("avg_energy_j")),
        "Num Samples": summary.get("num_samples", "-")
    }


def render_markdown_table(rows):
    if not rows:
        return "无数据"

    headers = list(rows[0].keys())
    lines = []

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for row in rows:
        values = [str(row.get(h, "-")) for h in headers]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_eval", type=str, default="output/sft_eval.json")
    parser.add_argument("--grpo_eval", type=str, default="output/grpo_eval.json")
    parser.add_argument("--rm_analysis", type=str, default="output/rm_score_analysis.json")
    parser.add_argument("--output_md", type=str, default="output/eval_summary.md")
    parser.add_argument("--output_json", type=str, default="output/eval_summary.json")
    args = parser.parse_args()

    sft_data = load_json(args.sft_eval)
    grpo_data = load_json(args.grpo_eval)
    rm_data = load_json(args.rm_analysis)

    sft_summary = safe_get(sft_data, "summary")
    grpo_summary = safe_get(grpo_data, "summary")

    compare_rows = [
        build_model_row("SFT", sft_summary),
        build_model_row("SFT + GRPO", grpo_summary),
    ]

    rm_summary = {
        "num_samples": safe_get(rm_data, "num_samples"),
        "pairwise_accuracy": safe_get(rm_data, "pairwise_accuracy"),
        "avg_margin": safe_get(rm_data, "avg_margin"),
    }

    # 计算增益
    gain_summary = {}
    if sft_summary is not None and grpo_summary is not None:
        def diff(key):
            a = sft_summary.get(key)
            b = grpo_summary.get(key)
            if a is None or b is None:
                return None
            return b - a

        gain_summary = {
            "valid_decision_ratio_gain": diff("valid_decision_ratio"),
            "optimal_decision_accuracy_gain": diff("optimal_decision_accuracy"),
            "avg_env_reward_gain": diff("avg_env_reward"),
            "avg_latency_ms_gain": diff("avg_latency_ms"),
            "avg_energy_j_gain": diff("avg_energy_j"),
        }

    # Markdown 汇总
    md_lines = []
    md_lines.append("# Evaluation Summary\n")
    md_lines.append("## 1. Policy Comparison\n")
    md_lines.append(render_markdown_table(compare_rows))
    md_lines.append("")

    md_lines.append("## 2. RM Analysis\n")
    rm_rows = [{
        "Metric": "Pairwise Accuracy",
        "Value": fmt_float(rm_summary["pairwise_accuracy"])
    }, {
        "Metric": "Average Margin",
        "Value": fmt_float(rm_summary["avg_margin"])
    }, {
        "Metric": "Num Samples",
        "Value": rm_summary["num_samples"] if rm_summary["num_samples"] is not None else "-"
    }]
    md_lines.append(render_markdown_table(rm_rows))
    md_lines.append("")

    md_lines.append("## 3. Gain Summary (GRPO - SFT)\n")
    if gain_summary:
        gain_rows = [
            {"Metric": "Valid Decision Ratio Gain", "Value": fmt_float(gain_summary["valid_decision_ratio_gain"])},
            {"Metric": "Optimal Decision Accuracy Gain", "Value": fmt_float(gain_summary["optimal_decision_accuracy_gain"])},
            {"Metric": "Avg Env Reward Gain", "Value": fmt_float(gain_summary["avg_env_reward_gain"])},
            {"Metric": "Avg Latency (ms) Gain", "Value": fmt_float(gain_summary["avg_latency_ms_gain"])},
            {"Metric": "Avg Energy (J) Gain", "Value": fmt_float(gain_summary["avg_energy_j_gain"])},
        ]
        md_lines.append(render_markdown_table(gain_rows))
    else:
        md_lines.append("未检测到完整的 SFT / GRPO 评估结果，无法计算增益。")
    md_lines.append("")

    # 给你面试可直接引用的话术
    md_lines.append("## 4. Interview-ready Notes\n")
    if sft_summary is not None and grpo_summary is not None:
        md_lines.append(
            "- 若 `Optimal Decision Accuracy` 和 `Avg Env Reward` 提升，可表述为："
            "“在固定模拟测试集上，经过 GRPO 对齐后，模型的最优决策命中率和环境回报均优于 SFT 基线。”"
        )
        md_lines.append(
            "- 若 `Valid Decision Ratio` 提升，可表述为："
            "“GRPO 不仅优化了策略质量，也提升了输出格式稳定性，使模型更稳定地产生可解析调度决策。”"
        )
        md_lines.append(
            "- 若 `Avg Latency (ms)` 下降、`Avg Energy (J)` 下降，可表述为："
            "“策略优化后，模型在延迟与能耗权衡上更接近物理模拟器给出的最优策略。”"
        )
    else:
        md_lines.append("- 当前只完成了部分结果文件汇总，等 SFT / GRPO 评估结果齐全后可自动生成对比结论。")

    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    output_json = {
        "policy_comparison": compare_rows,
        "rm_analysis": rm_summary,
        "gain_summary": gain_summary
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)

    print("✅ 汇总完成")
    print(f"📄 Markdown: {args.output_md}")
    print(f"📄 JSON: {args.output_json}")
    print("\n===== Policy Comparison =====")
    print(render_markdown_table(compare_rows))
    print("\n===== RM Analysis =====")
    print(render_markdown_table(rm_rows))
    if gain_summary:
        print("\n===== Gain Summary (GRPO - SFT) =====")
        print(render_markdown_table(gain_rows))


if __name__ == "__main__":
    main()