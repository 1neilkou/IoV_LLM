import os
import torch
import re
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, PeftModel

# 1. 【核心修改】重新映射物理显卡索引
# 这样设置后：逻辑 cuda:0 对应物理 GPU 1 (干净卡)，逻辑 cuda:1 对应物理 GPU 0 (队友卡)
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0" 
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

model_id = "./models/Qwen2.5-7B-Instruct"
rm_path = "./output/iov_qwen_rm/final_rm"

# --- 2. 在物理 GPU 0 (逻辑 cuda:1) 上部署判官 RM ---
print("⚖️ 正在【物理 GPU 0】部署奖励模型 (利用队友剩下的 26GB 空间)...")
rm_tokenizer = AutoTokenizer.from_pretrained(model_id)
rm_tokenizer.pad_token = rm_tokenizer.eos_token

base_rm = AutoModelForSequenceClassification.from_pretrained(
    model_id, 
    num_labels=1, 
    torch_dtype=torch.bfloat16,
    device_map={"": 1}  # 逻辑 1 = 物理 0
)
rm_model = PeftModel.from_pretrained(base_rm, rm_path)
rm_model.config.pad_token_id = rm_tokenizer.pad_token_id
rm_model.eval()

# --- 3. 奖励函数定义 ---
def rm_reward_function(completions, **kwargs):
    inputs = rm_tokenizer(completions, return_tensors="pt", padding=True, truncation=True).to(rm_model.device)
    with torch.no_grad():
        logits = rm_model(**inputs).logits
        if logits.dim() > 1 and logits.shape[-1] > 1:
            scores = logits[:, 0].cpu().detach().float().numpy().tolist()
        else:
            scores = logits.cpu().detach().float().numpy().flatten().tolist()
    return scores

def reward_function(completions, **kwargs):
    rewards = []
    for content in completions:
        score = 0.0
        if "分析" in content and "延迟" in content: score += 2.0
        if "边缘" in content or "卸载" in content: score += 2.0
        if "[边缘卸载]" in content: score += 3.0
        if "[本地计算]" in content: score += 3.0
        if not re.search(r"\[.*\]", content): score -= 5.0
        rewards.append(score)
    return rewards

def main():
    print("🚀 启动 5090 GRPO 引擎 (物理卡 1 独占模式)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # 4. 数据准备
    raw_dataset = load_dataset("json", data_files="./data/v2x_domain_qa.jsonl", split="train")
    dataset = raw_dataset.map(lambda x: {"prompt": x["instruction"]})

    # --- 5. 在物理 GPU 1 (逻辑 cuda:0) 加载训练主体 ---
    print("🧠 正在【物理 GPU 1】加载 Policy 模型 (独享 32GB 显存)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": 0}  # 逻辑 0 = 物理 1
    )
    
    # 强制标记为已并行化，防止 Trainer 自动向逻辑 cuda:1 (物理 0) 广播权重
    model.is_parallelizable = True
    model.model_parallel = True 

    # 6. 配置 GRPO (显存安全配置)
    training_args = GRPOConfig(
        output_dir="./output/iov_qwen_grpo",
        learning_rate=1e-6,
        per_device_train_batch_size=1, 
        num_generations=2,              
        max_completion_length=128,       # 留出足够的显存 buffer
        num_train_epochs=10,
        gradient_accumulation_steps=16, 
        bf16=True,
        logging_steps=1,
        gradient_checkpointing=True,
        report_to="none",
        ddp_find_unused_parameters=False,
    )
    # 🔥 【核心补丁】强制覆盖进程感知的显卡数量
    # 这一步能彻底关掉 DataParallel，阻止它往 GPU 0 镜像数据
    training_args._n_gpu = 1

    # 7. LoRA
    peft_config = LoraConfig(
        r=16, lora_alpha=32, 
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], 
        task_type="CAUSAL_LM"
    )

    # 8. 初始化
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_function, rm_reward_function],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    print("🔥 启动！GPU 1 练大脑，GPU 0 做裁判。")
    torch.cuda.empty_cache() # 临门一脚，排空碎屑
    trainer.train()
    
    # 9. 保存
    trainer.save_model("./output/iov_qwen_grpo/final_agent")
    print("✅ 进化完成！")

if __name__ == "__main__":
    main()