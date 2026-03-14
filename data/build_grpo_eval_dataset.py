# build_grpo_eval_dataset.py
import json
import random
from tqdm import tqdm

def calculate_v2x_cost(task_size, cpu_cycles, bandwidth, local_cpu_freq, edge_cpu_freq, decision):
    P_LOCAL = 0.5
    P_TRANS = 1.2

    if decision == "local":
        latency = (cpu_cycles / local_cpu_freq) * 1000.0
        energy = P_LOCAL * (cpu_cycles / local_cpu_freq)
    else:
        trans_time = task_size / bandwidth
        comp_time = cpu_cycles / edge_cpu_freq
        latency = (trans_time + comp_time) * 1000.0
        energy = P_TRANS * trans_time

    return latency, energy


def build_prompt(task_size, cpu_cycles, bandwidth, local_cpu_freq, edge_cpu_freq):
    prompt = (
        f"作为V2X调度Agent，请针对以下任务做出调度决策：\n"
        f"- 任务数据量: {task_size:.2f} MB\n"
        f"- 所需计算力: {cpu_cycles/1e9:.2f} G cycles\n"
        f"- 当前网络带宽: {bandwidth:.2f} MB/s\n"
        f"- 本地CPU频率: {local_cpu_freq/1e9:.2f} GHz\n"
        f"- 边缘服务器频率: {edge_cpu_freq/1e9:.2f} GHz\n"
        f"请在 '本地计算' 和 '边缘卸载' 中选择，并给出推理过程。"
    )
    return prompt


def generate_eval_sample():
    task_size = random.uniform(1.0, 5.0)
    cpu_cycles = random.uniform(0.5e9, 2e9)
    bandwidth = random.uniform(5.0, 20.0)
    local_cpu_freq = random.uniform(1e9, 1.5e9)
    edge_cpu_freq = random.uniform(5e9, 10e9)

    lat_local, eng_local = calculate_v2x_cost(
        task_size, cpu_cycles, bandwidth, local_cpu_freq, edge_cpu_freq, "local"
    )
    lat_edge, eng_edge = calculate_v2x_cost(
        task_size, cpu_cycles, bandwidth, local_cpu_freq, edge_cpu_freq, "edge"
    )

    score_local = -(0.7 * lat_local + 0.3 * eng_local * 100.0)
    score_edge = -(0.7 * lat_edge + 0.3 * eng_edge * 100.0)

    optimal_action = "local" if score_local > score_edge else "edge"

    return {
        "prompt": build_prompt(task_size, cpu_cycles, bandwidth, local_cpu_freq, edge_cpu_freq),
        "state": {
            "task_size": task_size,
            "cpu_cycles": cpu_cycles,
            "bandwidth": bandwidth,
            "local_cpu_freq": local_cpu_freq,
            "edge_cpu_freq": edge_cpu_freq,
        },
        "oracle": {
            "optimal_action": optimal_action,
            "local": {
                "latency_ms": lat_local,
                "energy_j": eng_local,
                "score": score_local,
            },
            "edge": {
                "latency_ms": lat_edge,
                "energy_j": eng_edge,
                "score": score_edge,
            }
        }
    }


if __name__ == "__main__":
    random.seed(42)

    num_samples = 200
    output_file = "data/v2x_eval_prompts.jsonl"

    all_samples = []
    for _ in tqdm(range(num_samples), desc="Building eval dataset"):
        all_samples.append(generate_eval_sample())

    with open(output_file, "w", encoding="utf-8") as f:
        for item in all_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 固定测试集生成完成，共 {num_samples} 条")
    print(f"📁 已保存到: {output_file}")