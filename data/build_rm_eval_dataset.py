import json
import random
from tqdm import tqdm

def calculate_v2x_cost(task_size, cpu_cycles, bandwidth, local_cpu_freq, edge_cpu_freq, decision):
    """
    与训练集保持一致的 V2X 物理环境 cost 逻辑
    """
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


def generate_preference_pair():
    """
    构造一条 held-out preference pair
    与训练集格式保持完全一致：prompt / chosen / rejected
    """
    task_size = random.uniform(1.0, 5.0)
    cpu_cycles = random.uniform(0.5e9, 2e9)
    bandwidth = random.uniform(5.0, 20.0)
    local_cpu_freq = random.uniform(1e9, 1.5e9)
    edge_cpu_freq = random.uniform(5e9, 10e9)

    prompt = (
        f"作为V2X调度Agent，请针对以下任务做出调度决策：\n"
        f"- 任务数据量: {task_size:.2f} MB\n"
        f"- 所需计算力: {cpu_cycles/1e9:.2f} G cycles\n"
        f"- 当前网络带宽: {bandwidth:.2f} MB/s\n"
        f"- 本地CPU频率: {local_cpu_freq/1e9:.2f} GHz\n"
        f"- 边缘服务器频率: {edge_cpu_freq/1e9:.2f} GHz\n"
        f"请在 '本地计算' 和 '边缘卸载' 中选择，并给出推理过程。"
    )

    lat_local, eng_local = calculate_v2x_cost(
        task_size, cpu_cycles, bandwidth, local_cpu_freq, edge_cpu_freq, "local"
    )
    lat_edge, eng_edge = calculate_v2x_cost(
        task_size, cpu_cycles, bandwidth, local_cpu_freq, edge_cpu_freq, "edge"
    )

    score_local = -(0.7 * lat_local + 0.3 * eng_local * 100.0)
    score_edge = -(0.7 * lat_edge + 0.3 * eng_edge * 100.0)

    resp_local = (
        f"分析：本地计算延迟预计为{lat_local:.1f}ms，能耗为{eng_local:.2f}J。"
        f"边缘计算延迟预计为{lat_edge:.1f}ms，能耗为{eng_edge:.2f}J。"
        f"综合考量，我决定执行：[本地计算]。"
    )
    resp_edge = (
        f"分析：本地计算延迟预计为{lat_local:.1f}ms，能耗为{eng_local:.2f}J。"
        f"边缘计算延迟预计为{lat_edge:.1f}ms，能耗为{eng_edge:.2f}J。"
        f"由于边缘卸载性能更优，我决定执行：[边缘卸载]。"
    )

    if score_local > score_edge:
        chosen, rejected = resp_local, resp_edge
    else:
        chosen, rejected = resp_edge, resp_local

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }


if __name__ == "__main__":
    # 关键：和训练集使用不同随机种子，确保不是同一批样本
    random.seed(20260314)

    num_samples = 500
    output_file = "data/v2x_rm_preference_eval.jsonl"

    dataset = []
    for _ in tqdm(range(num_samples), desc="Building RM held-out eval dataset"):
        dataset.append(generate_preference_pair())

    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ RM held-out 测试集生成完成，共 {num_samples} 条")
    print(f"📁 保存到: {output_file}")