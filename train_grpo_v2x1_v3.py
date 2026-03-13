import os
import torch
import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, PeftModel

# 1. 显存与设备映射
# 逻辑 cuda:0 = 物理 GPU 1 (干净卡)，逻辑 cuda:1 = 物理 GPU 0 (队友卡)
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0" 
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

model_id = "./models/Qwen2.5-7B-Instruct"
sft_adapter_path = "./output/iov_qwen_lora/final_model" # 👈 SFT 知识包路径
rm_adapter_path = "./output/iov_qwen_rm/final_rm"      # 👈 RM 判官包路径

# --- 2. 在逻辑 cuda:1 (物理 GPU 0) 部署【专家判官】 ---
print("⚖️ 正在部署 RM 奖励模型...")
rm_tokenizer = AutoTokenizer.from_pretrained(model_id)
rm_tokenizer.pad_token = rm_tokenizer.eos_token
rm_tokenizer.padding_side = "right"  # 统一padding方向（避免模型混淆）

base_rm = AutoModelForSequenceClassification.from_pretrained(
    model_id, 
    num_labels=1, 
    torch_dtype=torch.bfloat16,
    device_map={"": 1} 
)
rm_model = PeftModel.from_pretrained(base_rm, rm_adapter_path)
# 🔥 核心修复点 2：显式告诉模型 pad_token_id 是多少
rm_model.config.pad_token_id = rm_tokenizer.pad_token_id
rm_model.eval()

# --- 3. 奖励函数定义 ---
def rm_reward_function(completions, **kwargs):
    inputs = rm_tokenizer(completions, return_tensors="pt", padding=True, truncation=True).to(rm_model.device)
    with torch.no_grad():
        logits = rm_model(**inputs).logits
        scores = logits.cpu().detach().float().numpy().flatten().tolist()
    return scores

def reward_function(completions, **kwargs):
    rewards = []
    for content in completions:
        score = 0.0
        # 逻辑奖惩：关键词匹配
        if "分析" in content and "延迟" in content: score += 2.0
        if "边缘" in content or "卸载" in content: score += 2.0
        # 格式奖惩：强制中括号
        if "[边缘卸载]" in content or "[本地计算]" in content: 
            score += 5.0 
        else:
            score -= 2.0 # 格式不对，小惩大诫
        rewards.append(score)
    return rewards

def main():
    print("🚀 启动 5090 GRPO 满血进化引擎...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # 4. 加载数据
    raw_dataset = load_dataset("json", data_files="./data/v2x_domain_qa.jsonl", split="train")
    dataset = raw_dataset.map(lambda x: {"prompt": x["instruction"]})

    # --- 5. 【核心修改】在逻辑 cuda:0 (物理 GPU 1) 构建“懂行”的 Policy ---
    print("🧠 正在构建基于 SFT 知识的 Policy 模型...")
    # 第一步：加载原始基座
    raw_base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": 0}
    )
    # 第二步：挂载 SFT 适配器并合并（Merge）
    # 这样做能让模型“永久记住”SFT 知识，且腾出 LoRA 槽位给 GRPO
    sft_model = PeftModel.from_pretrained(raw_base_model, sft_adapter_path)
    print("📦 正在合并 SFT 权重，打造‘车联网专家’基座...")
    model = sft_model.merge_and_unload() 

    # 强制标记并行，防止镜像到 GPU 0
    model.is_parallelizable = True
    model.model_parallel = True 

    # 6. 配置 GRPO (针对 5090 优化)
    training_args = GRPOConfig(
        output_dir="./output/iov_qwen_grpo",
        learning_rate=1e-6,             # 稳健的学习率
        per_device_train_batch_size=1, 
        num_generations=8,              # 👈 关键：增加组内采样数，平滑奖励波动
        max_completion_length=256,       
        num_train_epochs=5,
        beta=0.01,                      # 👈 鼓励模型大胆尝试新策略
        gradient_accumulation_steps=8,  # 总 batch 为 8
        bf16=True,
        logging_steps=1,
        gradient_checkpointing=True,
        report_to="none",
        ddp_find_unused_parameters=False,
    )
    training_args._n_gpu = 1 # 锁死单卡模式

    # 7. 为 GRPO 开启新的 LoRA 训练（在 SFT 基座上进化）
    peft_config = LoraConfig(
        r=16, lora_alpha=32, 
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], 
        task_type="CAUSAL_LM"
    )

    # 8. 初始化 Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_function, rm_reward_function],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print("🔥 引擎点火！由 SFT 专家带队，RM 严密监督。")
    torch.cuda.empty_cache()
    trainer.train()
    
    # 9. 保存最终成果
    print("💾 正在持久化‘究极进化版’智能体...")
    trainer.state.save_to_json("./output/iov_qwen_grpo/trainer_state.json")
    trainer.save_model("./output/iov_qwen_grpo/final_agent")
    print("✅ 进化完成！")

if __name__ == "__main__":
    main()