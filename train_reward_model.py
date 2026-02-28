import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

# 1. 环境与路径
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

# 3. 自定义 Reward Trainer (手动实现 Pairwise Ranking Loss)
class CustomRewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 分别计算 Chosen 和 Rejected 的得分 (Logits)
        rewards_chosen = model(input_ids=inputs["input_ids_chosen"], 
                               attention_mask=inputs["attention_mask_chosen"])[0]
        rewards_rejected = model(input_ids=inputs["input_ids_rejected"], 
                                 attention_mask=inputs["attention_mask_rejected"])[0]
        
        # 核心：Pairwise Ranking Loss (对数排名损失)
        # Loss = -log(sigmoid(r_chosen - r_rejected))
        loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        
        return (loss, rewards_chosen) if return_outputs else loss

def main():
    print("🚀 启动 5090 深度优化版 Reward Model 训练引擎...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # 加载模型并添加分类头 (num_labels=1)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=1, dtype=torch.bfloat16, device_map="auto"
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # LoRA 配置
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=dataset_path, split="train")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        num_train_epochs=1,
        logging_steps=5,
        save_steps=50,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False, # 关键：必须设为 False 才能保留自定义列
    )

    trainer = CustomRewardTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=RewardDataCollator(tokenizer=tokenizer)
    )

    print("🔥 引擎点火，开始训练...")
    trainer.train()
    trainer.save_model(os.path.join(output_dir, "final_rm"))
    print("✅ Reward Model 训练完成！")

if __name__ == "__main__":
    main()