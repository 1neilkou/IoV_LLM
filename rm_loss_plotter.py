import re
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

# ===================== 配置项 =====================
# 设置中文字体（避免中文乱码）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
plt.rcParams["figure.dpi"] = 150  # 图片清晰度
plt.rcParams["savefig.dpi"] = 300

def parse_training_log(log_file_path: str) -> dict:
    """
    解析训练日志文件，提取步数、loss、chosen_score、rejected_score
    """
    # 匹配日志行的正则表达式（兼容你之前的日志格式）
    pattern = r"Step: (\d+) \| Loss: ([\d\.]+) \| Chosen Score: ([\d\.]+) \| Rejected Score: ([\d\.]+)"
    
    metrics = {
        "steps": [],
        "losses": [],
        "chosen_scores": [],
        "rejected_scores": []
    }
    
    # 读取日志文件
    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ 错误：未找到日志文件 {log_file_path}")
        return metrics
    except Exception as e:
        print(f"❌ 读取日志文件失败：{e}")
        return metrics
    
    # 逐行匹配数据
    for line in lines:
        match = re.search(pattern, line)
        if match:
            step = int(match.group(1))
            loss = float(match.group(2))
            chosen = float(match.group(3))
            rejected = float(match.group(4))
            
            metrics["steps"].append(step)
            metrics["losses"].append(loss)
            metrics["chosen_scores"].append(chosen)
            metrics["rejected_scores"].append(rejected)
    
    if not metrics["steps"]:
        print("⚠️ 未从日志中提取到训练数据，请检查日志格式是否匹配")
    else:
        print(f"✅ 成功提取 {len(metrics['steps'])} 条训练数据")
    
    return metrics

def plot_loss_curve(metrics: dict, save_path: str = "rm_training_curve.png"):
    """
    绘制并保存 loss 曲线和得分曲线
    """
    if len(metrics["steps"]) == 0:
        print("❌ 无训练数据，无法绘制曲线")
        return
    
    # 创建双子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # ========== 子图1：Loss 曲线 ==========
    ax1.plot(metrics["steps"], metrics["losses"], 
             color="#e74c3c", linewidth=2, marker="o", markersize=4, label="原始 Loss")
    
    # 添加平滑 Loss 曲线（滑动平均，可选）
    if len(metrics["losses"]) >= 5:
        window_size = min(5, len(metrics["losses"]) // 2)
        smoothed_loss = np.convolve(metrics["losses"], np.ones(window_size)/window_size, mode="valid")
        smoothed_steps = metrics["steps"][window_size-1:]
        ax1.plot(smoothed_steps, smoothed_loss, 
                 color="#c0392b", linewidth=2.5, label=f"平滑 Loss (窗口={window_size})")
    
    ax1.set_ylabel("Loss 值", fontsize=12)
    ax1.set_title("Reward Model 训练 Loss 曲线", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim(bottom=0)  # Loss 最小值为0
    
    # ========== 子图2：得分曲线 ==========
    ax2.plot(metrics["steps"], metrics["chosen_scores"], 
             color="#2ecc71", linewidth=2, marker="s", markersize=4, label="Chosen 得分")
    ax2.plot(metrics["steps"], metrics["rejected_scores"], 
             color="#3498db", linewidth=2, marker="^", markersize=4, label="Rejected 得分")
    
    # 计算得分差
    score_diff = [c - r for c, r in zip(metrics["chosen_scores"], metrics["rejected_scores"])]
    ax2.plot(metrics["steps"], score_diff, 
             color="#9b59b6", linewidth=2, linestyle="--", label="得分差 (Chosen-Rejected)")
    
    ax2.set_xlabel("训练步数", fontsize=12)
    ax2.set_ylabel("奖励得分", fontsize=12)
    ax2.set_title("Chosen vs Rejected 奖励得分对比", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    print(f"✅ 曲线已保存至：{save_path}")
    
    # 显示图片（本地运行时）
    plt.show()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Reward Model 训练 Loss 曲线绘制工具")
    parser.add_argument("--log_path", type=str, required=True, help="训练日志文件路径（必填）")
    parser.add_argument("--save_path", type=str, default="rm_training_curve.png", 
                        help="图片保存路径（默认：rm_training_curve.png）")
    args = parser.parse_args()
    
    # 1. 解析日志
    print(f"📄 正在解析日志文件：{args.log_path}")
    metrics = parse_training_log(args.log_path)
    
    # 2. 绘制曲线
    if metrics["steps"]:
        print("🎨 正在绘制训练曲线...")
        plot_loss_curve(metrics, args.save_path)
    
    print("🎉 绘图完成！")

if __name__ == "__main__":
    main()