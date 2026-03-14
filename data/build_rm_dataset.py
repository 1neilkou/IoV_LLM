import json
import random
from tqdm import tqdm

def calculate_v2x_cost(task_size, cpu_cycles, bandwidth, local_cpu_freq, edge_cpu_freq, decision):
    """
    V2X 物理模拟器：计算不同调度决策的 Cost (延迟 + 能耗)
    decision: "local" (本地计算) 或 "edge" (边缘卸载)
    返回: (延迟 ms, 能耗 J)
    """
    # 物理常量预设
    P_LOCAL = 0.5   # 本地计算功率 (W)
    P_TRANS = 1.2   # 传输功率 (W)
    
    if decision == "local":
        # 本地计算：延迟 = 任务所需CPU周期 / 本地CPU频率
        latency = (cpu_cycles / local_cpu_freq) * 1000 # 转为 ms
        energy = P_LOCAL * (cpu_cycles / local_cpu_freq)
    else:
        # 边缘计算：延迟 = 传输时间 + 边缘计算时间
        trans_time = task_size / bandwidth
        comp_time = cpu_cycles / edge_cpu_freq
        latency = (trans_time + comp_time) * 1000 # 转为 ms
        energy = P_TRANS * trans_time
        
    return latency, energy

def generate_preference_pair():
    """生成一对 (Chosen, Rejected) 偏好数据"""
    # 1. 随机生成一个 V2X 任务环境状态 (State)
    task_size = random.uniform(1.0, 5.0)       # 任务大小 1-5 MB
    cpu_cycles = random.uniform(0.5e9, 2e9)    # 所需计算量 0.5G - 2G cycles
    bandwidth = random.uniform(5.0, 20.0)      # 当前可用带宽 5-20 MB/s
    local_cpu_freq = random.uniform(1e9, 1.5e9)# 终端CPU频率 1-1.5 GHz
    edge_cpu_freq = random.uniform(5e9, 10e9)  # 边缘节点CPU频率 5-10 GHz
    
    # 2. 构造 Prompt (大模型看到的题目)
    prompt = (
        f"作为V2X调度Agent，请针对以下任务做出调度决策：\n"
        f"- 任务数据量: {task_size:.2f} MB\n"
        f"- 所需计算力: {cpu_cycles/1e9:.2f} G cycles\n"
        f"- 当前网络带宽: {bandwidth:.2f} MB/s\n"
        f"- 本地CPU频率: {local_cpu_freq/1e9:.2f} GHz\n"
        f"- 边缘服务器频率: {edge_cpu_freq/1e9:.2f} GHz\n"
        f"请在 '本地计算' 和 '边缘卸载' 中选择，并给出推理过程。"
    )
    
    # 3. 模拟物理世界，计算两种选择的真实 Cost
    lat_local, eng_local = calculate_v2x_cost(task_size, cpu_cycles, bandwidth, local_cpu_freq, edge_cpu_freq, "local")
    lat_edge, eng_edge = calculate_v2x_cost(task_size, cpu_cycles, bandwidth, local_cpu_freq, edge_cpu_freq, "edge")
    
    # 假设我们的奖励函数是：权重 0.7 看延迟，0.3 看能耗
    score_local = -(0.7 * lat_local + 0.3 * eng_local * 100)
    score_edge = -(0.7 * lat_edge + 0.3 * eng_edge * 100)
    
    # 4. 构造标准回答 (CoT 推理过程 + 最终决策)
    resp_local = f"分析：本地计算延迟预计为{lat_local:.1f}ms，能耗为{eng_local:.2f}J。边缘计算延迟预计为{lat_edge:.1f}ms，能耗为{eng_edge:.2f}J。综合考量，我决定执行：[本地计算]。"
    resp_edge = f"分析：本地计算延迟预计为{lat_local:.1f}ms，能耗为{eng_local:.2f}J。边缘计算延迟预计为{lat_edge:.1f}ms，能耗为{eng_edge:.2f}J。由于边缘卸载性能更优，我决定执行：[边缘卸载]。"
    
    # 5. 判别 Chosen (胜利者) 和 Rejected (失败者)
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
    print("🚀 正在基于 V2X 物理引擎合成 RM 偏好数据集...")
    num_samples = 5000 # 生成 5000 条训练数据
    
    dataset = []
    for _ in tqdm(range(num_samples)):
        dataset.append(generate_preference_pair())
        
    output_file = "v2x_rm_preference.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"✅ 偏好数据集生成完毕！共 {num_samples} 条，已保存至 {output_file}")