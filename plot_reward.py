import json
import matplotlib.pyplot as plt
import os

def draw_advanced_grpo_curve(log_file):
    if not os.path.exists(log_file):
        print(f"❌ 找不到日志: {log_file}")
        return

    with open(log_file, "r") as f:
        data = json.load(f)

    steps, total_rewards, kl_values = [], [], []

    for log in data["log_history"]:
        if "reward" in log:
            steps.append(log["step"])
            total_rewards.append(log["reward"])
            # 抓取 KL。有些版本叫 'kl'，有些叫 'kl_divergence'，我们都试一下
            kl = log.get("kl") or log.get("kl_divergence") or 0
            kl_values.append(kl)

    if not steps:
        print("⚠️ 还没有数据，再多跑几步。")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制 Reward (左轴)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Reward Score', color='tab:red')
    ax1.plot(steps, total_rewards, color='tab:red', label='Total Reward', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, alpha=0.3)

    # 绘制 KL (右轴)
    ax2 = ax1.twinx() 
    ax2.set_ylabel('KL Divergence', color='tab:blue')
    ax2.plot(steps, kl_values, color='tab:blue', label='KL (Policy Drift)', linestyle=':')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title('V2X GRPO: Reward vs Policy Drift (KL)')
    fig.tight_layout()
    
    save_path = "./output/iov_advanced_plot.png"
    plt.savefig(save_path)
    print(f"✅ 双轴趋势图已生成：{save_path}")

if __name__ == "__main__":
    draw_advanced_grpo_curve("./output/iov_qwen_grpo/trainer_state.json")