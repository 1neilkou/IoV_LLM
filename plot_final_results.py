import json
import matplotlib.pyplot as plt
import os

def load_trainer_state(file_path):
    if not os.path.exists(file_path):
        print(f"⚠️ 未找到文件: {file_path}")
        return None
    with open(file_path, 'r') as f:
        return json.load(f)

def plot_real_results():
    # 路径配置
    rm_path = "./output/iov_qwen_rm/trainer_state.json"
    grpo_path = "./output/iov_qwen_grpo/checkpoint-70/trainer_state.json" # 请根据你实际产生的 checkpoint 编号修改

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # --- 1. 绘制真实 Reward Model Loss (实验一) ---
    rm_state = load_trainer_state(rm_path)
    if rm_state:
        steps = [log['step'] for log in rm_state['log_history'] if 'loss' in log]
        losses = [log['loss'] for log in rm_state['log_history'] if 'loss' in log]
        axes[0].plot(steps, losses, color='#3498db', marker='o', label='RM Ranking Loss')
        axes[0].set_yscale('log')
        axes[0].set_title("Real Exp 1: Reward Model Loss Convergence")
        axes[0].set_xlabel("Steps")
        axes[0].set_ylabel("Loss (Log Scale)")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

    # --- 2. 绘制真实 GRPO Reward 变化 (实验二) ---
    grpo_state = load_trainer_state(grpo_path)
    if grpo_state:
        # 提取 GRPO 里的 Reward 均值
        steps = [log['step'] for log in grpo_state['log_history'] if 'rewards/reward_function/mean' in log]
        rewards = [log['rewards/reward_function/mean'] for log in grpo_state['log_history'] if 'rewards/reward_function/mean' in log]
        
        if steps:
            axes[1].plot(steps, rewards, color='#2ecc71', marker='s', label='Mean Reward')
            axes[1].set_title("Real Exp 2: GRPO Reward Trajectory")
            axes[1].set_xlabel("Steps")
            axes[1].set_ylabel("Reward Score")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
        else:
            axes[1].text(0.5, 0.5, "Wait: Reward logs are zero in current state", ha='center')

    plt.tight_layout()
    plt.savefig('real_performance_report.png')
    print("🚀 真实数据报表已生成: real_performance_report.png")

if __name__ == "__main__":
    plot_real_results()