import os
import torch
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# 1. 环境配置
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# 2. 路径设置
model_id = "./models/Qwen2.5-7B-Instruct"
dataset_path = "./data/iov_train_data.jsonl"
output_dir = "./output/iov_qwen_lora"

def process_func(example, tokenizer):
    """
    针对 Qwen 格式构建微调 Prompt Template
    """
    MAX_LENGTH = 512
    input_ids, labels = [], []
    
    # 构建车联网领域的指令格式
    instruction = tokenizer(
        f"<|im_start|>system\n你是一个车联网专家，请根据提供的论文内容回答问题。<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n",
        add_special_tokens=False
    )
    response = tokenizer(f"{example['output']}<|im_end|>\n", add_special_tokens=False)
    
    input_ids = instruction["input_ids"] + response["input_ids"]
    # 标签中，指令部分用 -100 忽略，只计算回答部分的 Loss
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
        
    return {
        "input_ids": input_ids,
        "labels": labels
    }

def main():
    print("🚀 正在初始化 5090 实验环境...")

    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 加载数据集
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    dataset = Dataset.from_list(data)
    tokenized_id = dataset.map(lambda x: process_func(x, tokenizer), remove_columns=dataset.column_names)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model.enable_input_require_grads() # 开启梯度检查点必需

    # LoRA 配置
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,   
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # 覆盖全量线性层提升效果
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 训练参数设置 - 针对 5090 (32GB) 优化
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,        # 5090 显存充裕，可设为 4-8
        gradient_accumulation_steps=4,        # 等效 batch_size = 16
        logging_steps=10,
        num_train_epochs=3,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,          # 进一步节省显存
        bf16=True,                            # 5090 必须开启 bf16
        report_to="none"
    )

    # 启动 Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    print("\n✅ LoRA 准备就绪，当前显存占用：{:.2f} GB".format(torch.cuda.memory_allocated() / 1024**3))
    print("开始微调车联网大脑...")
    
    trainer.train()
    
    # 保存结果
    trainer.save_model(output_dir)
    print(f"🎉 训练完成！模型已保存至: {output_dir}")

if __name__ == "__main__":
    main()