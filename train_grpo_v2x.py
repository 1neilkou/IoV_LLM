import os
import torch
import re
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

# 1. 环境配置
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True" # 5090 显存优化补丁
model_id = "./models/Qwen2.5-7B-Instruct"

# 2. 定义奖励函数 (必须在 Trainer 初始化前定义)
def reward_function(completions, **kwargs):
    rewards = []
    for content in completions:
        score = 0.0
        # 格式奖励：CoT 思维链
        if "分析" in content and "决定" in content:
            score += 1.0
        # 决策奖励：决策关键字 (逻辑对齐)
        if re.search(r"\[本地计算\]|\[边缘卸载\]", content):
            score += 1.0
        # 长度惩罚：防止模型刷分
        if len(content) < 20:
            score -= 1.0
        rewards.append(score)
    return rewards

def main():
    print("🔥 启动 5090 GRPO 强化学习引擎...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. 准备微调数据集
    raw_dataset = load_dataset("json", data_files="./data/v2x_domain_qa.jsonl", split="train")
    def format_dataset(example):
        return {"prompt": example["instruction"]}
    dataset = raw_dataset.map(format_dataset)
    print(f"📊 数据映射完成：{dataset.column_names}")

    # 4. 配置 GRPO 超参数
    training_args = GRPOConfig(
        output_dir="./output/iov_qwen_grpo",
        learning_rate=5e-6,
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=16,
        num_generations=2,            # 5090 建议设为 2
        max_completion_length=128,    # 限制生成长度
        num_train_epochs=1,
        bf16=True,
        logging_steps=1,
        gradient_checkpointing=True,  # 开启重计算
        report_to="none"              # 离线环境关闭汇报
    )

    # 5. 配置 LoRA (关键：必须在初始化时传入)
    peft_config = LoraConfig(
        r=8, 
        lora_alpha=16, 
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], 
        task_type="CAUSAL_LM"
    )

    # 6. 初始化唯一的 GRPOTrainer
    trainer = GRPOTrainer(
        model=model_id,
        reward_funcs=reward_function, # 传入上面定义的函数
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,     # 确保使用 LoRA
    )

    print("🚀 GRPO 对齐开始，正在压榨 5090 算力...")
    trainer.train()
    
    # 7. 保存结果
    save_path = "./output/iov_qwen_grpo/final_agent"
    trainer.save_model(save_path)
    print(f"✅ GRPO 训练完成！结果已保存至 {save_path}")

if __name__ == "__main__":
    main()