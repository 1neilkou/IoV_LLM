import json
import matplotlib.pyplot as plt
import os

# 路径指向你训练状态记录
log_path = "output/iov_qwen_lora/checkpoint-294/trainer_state.json"

if os.path.exists(log_path):
    with open(log_path, "r") as f:
        data = json.load(f)
    
    # 提取 log_history 中的 loss 数据
    # 注意：Transformers 会定期记录数据，我们只取包含 loss 的步数
    steps = [log["step"] for log in data["log_history"] if "loss" in log]
    losses = [log["loss"] for log in data["log_history"] if "loss" in log]

    if not steps:
        print("❌ 日志文件中尚未记录 Loss 数据。")
    else:
        # 开始绘图
        plt.figure(figsize=(10, 6))
        plt.plot(steps, losses, marker='o', linestyle='-', color='#2c3e50', markersize=4, label="SFT Loss")
        plt.title("IoV-LLM Training Loss (Qwen2.5-7B + LoRA)", fontsize=14)
        plt.xlabel("Training Steps", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.grid(True, which="both", ls="-.", alpha=0.5)
        plt.legend()
        
        # 保存图片
        plt.savefig("iov_loss_curve.png")
        print(f"✅ 成功！Loss 曲线已保存至: {os.path.abspath('iov_loss_curve.png')}")
        print(f"初始 Loss: {losses[0]:.4f}")
        print(f"最终 Loss: {losses[-1]:.4f}")
else:
    print(f"❌ 找不到文件: {log_path}。请检查 checkpoint 编号是否正确。")