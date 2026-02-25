import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

# 引入我们之前写好的数据打包模块
from utils.data_utils import V2XDataProcessor

# 1. 设置 GPU 与离线模式
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# 2. 路径配置 (注意：这里使用我们刚混合好的最终数据集)
model_id = "./models/Qwen2.5-7B-Instruct"
dataset_path = "./data/iov_train_data_final.jsonl" # 👈 修改为你合并 Alpaca 后的数据

def main():
    print("🚀 正在初始化 5090 满血训练环境...")

    # ================= 1. 加载模型与 Tokenizer =================
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token 

    # 核心优化 1：针对 5090 开启 Flash Attention 2
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id, 
    #     torch_dtype=torch.bfloat16, 
    #     device_map="auto",
    #     attn_implementation="flash_attention_2" # 👈 面试必考：将 O(N^2) 复杂度显著降低，大幅提速
    # )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        dtype=torch.bfloat16,             # 👈 修复点1：解决 torch_dtype 报废警告
        device_map="auto",
        attn_implementation="sdpa"        # 👈 修复点2：弃用报错的 FA2，改用原生 SDPA 加速
    )

    # ================= 2. 配置进阶版 LoRA =================
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,             # 核心优化 2：Rank 提升到 64，增强复杂协议的拟合能力
        lora_alpha=128,   # alpha 通常设置为 r 的 2 倍
        lora_dropout=0.05,
        # 核心优化 3：不要只加 q 和 v，把 MLP 层也加上，对齐效果更好
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print(f"📊 LoRA 注入成功！当前 5090 基础显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # ================= 3. 数据加载与 Packing =================
    print(f"📦 开始加载并打包训练数据: {dataset_path} ...")
    raw_dataset = load_dataset("json", data_files=dataset_path)["train"]
    
    # 核心优化 4：使用 Data Packing 压榨硬件效率
    processor = V2XDataProcessor(tokenizer, max_seq_length=2048)
    packed_train_dataset = processor.pack_dataset(raw_dataset)

    # ================= 4. 配置训练引擎 (Trainer) =================
    print("⚙️ 配置训练参数...")
    training_args = TrainingArguments(
        output_dir="./output/iov_qwen_lora",
        per_device_train_batch_size=1,   # 5090 的 32G 显存，配合 Packing 可以开到 4 甚至 8,显存爆了改为1
        gradient_accumulation_steps=16,   # 模拟全局 Batch Size = 16
        # 👇 修改点 3 (终极核武器)：开启梯度检查点 (用大约 20% 的训练时间换取 50% 的显存空间)
        gradient_checkpointing=True,
        learning_rate=2e-4,              # LoRA 适用稍大的学习率
        num_train_epochs=3,              # 遍历数据 3 次
        logging_steps=51,                 # 每 5 步打印一次 Loss，数据少
        save_steps=50,                   # 每 50 步保存一次 Checkpoint
        bf16=True,                       # 5090 必备，防 NaN
        optim="adamw_torch",             # 使用脱耦的 AdamW 优化器
        report_to="none"                 # 保持离线，不连 wandb
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=packed_train_dataset,
    )

    # ================= 5. 开始微调 =================
    print("🔥 引擎点火，开始在 RTX 5090 上微调...")
    trainer.train()
    
    # 保存最终模型
    trainer.model.save_pretrained("./output/iov_qwen_lora/final_model")
    tokenizer.save_pretrained("./output/iov_qwen_lora/final_model")
    print("✅ 训练完美收官，Checkpoint 已保存！")

if __name__ == "__main__":
    main()