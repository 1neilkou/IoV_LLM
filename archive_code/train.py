import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# 1. 基础配置
model_id = "Qwen/Qwen2.5-3B-Instruct"
dataset_path = "data/train.jsonl"

# 2. 加载数据集
dataset = load_dataset("json", data_files=dataset_path, split="train")

# 3. 配置 LoRA (对应 PPT 提到的任务-算力匹配先验学习)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# 4. 加载模型与分词器
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token # Qwen 必须设置 pad_token
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# 5. 设置训练参数 (体现工程优化)
training_args = TrainingArguments(
    output_dir="./output/iov_llm_lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    save_steps=100,
    logging_steps=10,
    bf16=True, # 字节等大厂常用 bf16 提升训练效率
    push_to_hub=False,
    report_to="none"
)

# 6. 启动 SFT 训练
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="instruction", # 这里根据你 jsonl 的格式调整，建议将 input/output 拼接
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
)

trainer.train()
trainer.save_model("./output/iov_llm_lora_final")
print("训练完成！模型已保存至 ./output/iov_llm_lora_final")