import json
import random

# 基于 PPT 118-140 页的应用特征定义
APP_TYPES = {
    "视频流媒体": {"deadline": "2.0s", "pref": "GPU", "pref_val": 0.70},
    "AI推理": {"deadline": "0.8s", "pref": "NPU", "pref_val": 0.50},
    "游戏渲染": {"deadline": "0.016s", "pref": "GPU", "pref_val": 0.55},
    "自动驾驶": {"deadline": "0.1s", "pref": "NPU", "pref_val": 0.50},
    "AR导航": {"deadline": "0.05s", "pref": "GPU", "pref_val": 0.50},
    "Web浏览": {"deadline": "1.5s", "pref": "CPU", "pref_val": 0.75}
}

# 基于 PPT 191 页的设备规格
DEVICES = {
    "本地设备": {"cpu": 50, "gpu": 0.8, "npu": 12},
    "邻居车辆": {"cpu": 180, "gpu": 1.5, "npu": 144},
    "RSU标准型": {"cpu": 800, "gpu": 16, "npu": 35}
}

def generate_sample():
    app_name = random.choice(list(APP_TYPES.keys()))
    app_info = APP_TYPES[app_name]
    
    # 模拟环境状态：随机生成当前各路节点的负载情况
    local_load = random.uniform(0.1, 0.9)
    v2i_bandwidth = random.uniform(20, 100) # Mbps [cite: 214]
    
    # 逻辑判断（简化版）：如果本地负载高且任务偏好与本地不匹配，则尝试卸载
    if local_load > 0.6:
        decision = "卸载至 RSU标准型"
        reason = f"本地负载高达 {local_load:.1%}, 且任务对 {app_info['pref']} 有强偏好，远程 RSU 算力更充足。"
    else:
        decision = "本地执行"
        reason = f"本地负载较轻 ({local_load:.1%}), 能够满足该任务的时延要求。"

    # 构建大模型偏好的指令格式
    prompt = {
        "instruction": "你是一个车联网资源调度专家。请根据任务需求和当前异构算力环境，给出最优卸载决策。",
        "input": f"任务类型：{app_name}；截止时延：{app_info['deadline']}；算力偏好：{app_info['pref']}。当前本地CPU负载：{local_load:.1%}；V2I带宽：{v2i_bandwidth:.1f}Mbps。",
        "output": f"【决策】：{decision}。理由：{reason}"
    }
    return prompt

# 生成 500 条数据
dataset = [generate_sample() for _ in range(500)]

with open("data/train.jsonl", "w", encoding="utf-8") as f:
    for entry in dataset:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print("数据生成成功！已存入 IoV_LLM/data/train.jsonl")