import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt # 导入绘图库
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

# 1. 环境与路径
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_id = "./models/Qwen2.5-7B-Instruct"
dataset_path = "./data/v2x_rm_preference.jsonl"
output_dir = "./output/iov_qwen_rm"

# 2. 自定义双路 Data Collator
@dataclass
class RewardDataCollator:
    tokenizer: Any
    max_length: int = 1024

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        chosen_texts = [f"{f['prompt']}\n\n{f['chosen']}" for f in features]
        rejected_texts = [f"{f['prompt']}\n\n{f['rejected']}" for f in features]
        
        c_batch = self.tokenizer(chosen_texts, max_length=self.max_length, truncation=True, padding=True, return_tensors="pt")
        r_batch = self.tokenizer(rejected_texts, max_length=self.max_length, truncation=True, padding=True, return_tensors="pt")
        
        return {
            "input_ids_chosen": c_batch["input_ids"],
            "attention_mask_chosen": c_batch["attention_mask"],
            "input_ids_rejected": r_batch["input_ids"],
            "attention_mask_rejected": r_batch["attention_mask"],
        }

# 3. 自定义 Reward Trainer
class CustomRewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        rewards_chosen = model(input_ids=inputs["input_ids_chosen"], 
                               attention_mask=inputs["attention_mask_chosen"])[0]
        rewards_rejected = model(input_ids=inputs["input_ids_rejected"], 
                                  attention_mask=inputs["attention_mask_rejected"])[0]
        
        loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        return (loss, rewards_chosen) if return_outputs else loss

# 4. 绘图辅助函数
def plot_loss(log_history, save_path):
    steps = [log["step"] for log in log_history if "loss" in log]
    losses = [log["loss"] for log in log_history if "loss" in log]
    
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, label="Training Loss", color="#2E86C1", linewidth=2)
    plt.title("Reward Model Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(save_path)
    print(f"📊 训练曲线图已保存至: {save_path}")

def main():
    print("🚀 启动 5090 深度优化版 Reward Model 训练引擎...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=1,
        torch_dtype=torch.bfloat16,
)

    model.to(device)

    if hasattr(model, "score") and hasattr(model.score, "weight"):
        print("🎯 检测到新分类头，初始化权重...")
        model.score.weight.data.normal_(mean=0.0, std=0.02)




    model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=3e-5,
        num_train_epochs=1,
        logging_steps=5,      # 每5步记录一次loss
        save_steps=50,
        bf16=True,
        gradient_checkpointing=True,
        # 增加这一行，解决 recompute 时的设备索引问题
        gradient_checkpointing_kwargs={"use_reentrant": False}, 
        # 确保只有一张卡时，分布式搜索关闭
        # ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
        remove_unused_columns=False,
        # logging_dir=f"{output_dir}/logs", # 日志目录
        report_to="none"      # 如果没有WandB，设为none
    )

    trainer = CustomRewardTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=RewardDataCollator(tokenizer=tokenizer)
    )

    print("🔥 引擎点火，开始训练...")
    train_result = trainer.train()

    # --- 记录与绘图部分 ---
    # 1. 保存最终模型
    trainer.save_model(os.path.join(output_dir, "final_rm"))
    
    # 2. 导出训练日志为 JSON (方便以后随时读取)
    log_path = os.path.join(output_dir, "trainer_state.json")
    with open(log_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=4)
    
    # 3. 立即生成 Loss 图片
    plot_loss(trainer.state.log_history, os.path.join(output_dir, "loss_curve.png"))
    
    print("✅ Reward Model 训练与记录完成！")

if __name__ == "__main__":
    main()