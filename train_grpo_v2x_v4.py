# train_grpo_v2x1_v4.py
import os
import re
import math
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, PeftModel

# =========================
# 0. 环境配置
# =========================
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

model_id = "./models/Qwen2.5-7B-Instruct"
sft_adapter_path = "./output/iov_qwen_lora/final_model"
rm_adapter_path = "./output/iov_qwen_rm/final_rm"
grpo_data_path = "./data/v2x_grpo_prompts.jsonl"

# =========================
# 1. 部署 RM 模型
# =========================
print("⚖️ 正在部署 RM 奖励模型...")
rm_tokenizer = AutoTokenizer.from_pretrained(model_id)
rm_tokenizer.pad_token = rm_tokenizer.eos_token
rm_tokenizer.padding_side = "right"

base_rm = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=1,
    torch_dtype=torch.bfloat16,
    device_map={"": 1},   # RM放逻辑cuda:1
)
rm_model = PeftModel.from_pretrained(base_rm, rm_adapter_path)
rm_model.config.pad_token_id = rm_tokenizer.pad_token_id
rm_model.eval()


# =========================
# 2. 工具函数
# =========================
def zscore_normalize(values, eps=1e-6):
    """对一个 batch 内 reward 做标准化，缓解不同 reward 尺度不一致"""
    arr = np.array(values, dtype=np.float32)
    mean = arr.mean()
    std = arr.std()
    if std < eps:
        return arr.tolist()
    return ((arr - mean) / (std + eps)).tolist()


def extract_prompts_from_kwargs(kwargs, num_samples):
    """
    兼容 TRL/GRPOTrainer 传参差异：
    常见可能是 prompts / prompt。
    """
    prompts = None

    if "prompts" in kwargs and kwargs["prompts"] is not None:
        prompts = kwargs["prompts"]
    elif "prompt" in kwargs and kwargs["prompt"] is not None:
        prompts = kwargs["prompt"]

    if prompts is None:
        # 最差兜底：给空串，至少不报错
        return [""] * num_samples

    # 兼容单个字符串
    if isinstance(prompts, str):
        return [prompts] * num_samples

    # 兼容 list[str]
    if isinstance(prompts, list):
        return prompts

    # 兜底
    return [""] * num_samples


def calculate_v2x_cost(task_size, cpu_cycles, bandwidth, local_cpu_freq, edge_cpu_freq, decision):
    """
    直接复用你 RM 数据构造里的物理逻辑：
    延迟 + 能耗
    """
    P_LOCAL = 0.5
    P_TRANS = 1.2

    if decision == "local":
        latency = (cpu_cycles / local_cpu_freq) * 1000.0
        energy = P_LOCAL * (cpu_cycles / local_cpu_freq)
    else:
        trans_time = task_size / bandwidth
        comp_time = cpu_cycles / edge_cpu_freq
        latency = (trans_time + comp_time) * 1000.0
        energy = P_TRANS * trans_time

    return latency, energy


def parse_prompt_state(prompt):
    """
    从 prompt 中解析状态参数。
    prompt 来自 build_rm_dataset.py 的固定模板，因此这里用正则足够。
    """
    patterns = {
        "task_size": r"任务数据量:\s*([\d.]+)\s*MB",
        "cpu_cycles": r"所需计算力:\s*([\d.]+)\s*G cycles",
        "bandwidth": r"当前网络带宽:\s*([\d.]+)\s*MB/s",
        "local_cpu_freq": r"本地CPU频率:\s*([\d.]+)\s*GHz",
        "edge_cpu_freq": r"边缘服务器频率:\s*([\d.]+)\s*GHz",
    }

    state = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, prompt)
        if not match:
            return None
        state[key] = float(match.group(1))

    # 单位换回原始量纲
    state["cpu_cycles"] *= 1e9
    state["local_cpu_freq"] *= 1e9
    state["edge_cpu_freq"] *= 1e9
    return state


def extract_decision(completion):
    """
    从模型输出中抽取最终决策。
    兼容多种说法，尽量鲁棒。
    """
    text = completion.strip()

    if "[边缘卸载]" in text or "边缘卸载" in text:
        return "edge"
    if "[本地计算]" in text or "本地计算" in text:
        return "local"

    return None


# =========================
# 3. 三个 reward 分量
# =========================
def rm_reward_function(completions, **kwargs):
    """
    修复点1：
    RM 不能只看 completion，必须看 prompt + completion。
    """
    prompts = extract_prompts_from_kwargs(kwargs, len(completions))
    texts = [p + "\n\n" + c for p, c in zip(prompts, completions)]

    inputs = rm_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(rm_model.device)

    with torch.no_grad():
        logits = rm_model(**inputs).logits

    scores = logits.float().view(-1).cpu().tolist()
    scores = zscore_normalize(scores)
    return [float(x) for x in scores]


def env_reward_function(completions, **kwargs):
    """
    修复点2：
    引入真实环境 reward，而不是只奖励“会说术语”。
    """
    prompts = extract_prompts_from_kwargs(kwargs, len(completions))
    rewards = []

    for prompt, completion in zip(prompts, completions):
        state = parse_prompt_state(prompt)
        decision = extract_decision(completion)

        # 没解析出状态，轻微惩罚
        if state is None:
            rewards.append(-1.0)
            continue

        # 没明确给出决策，惩罚
        if decision is None:
            rewards.append(-2.0)
            continue

        latency, energy = calculate_v2x_cost(
            task_size=state["task_size"],
            cpu_cycles=state["cpu_cycles"],
            bandwidth=state["bandwidth"],
            local_cpu_freq=state["local_cpu_freq"],
            edge_cpu_freq=state["edge_cpu_freq"],
            decision=decision,
        )

        # 和 RM 构造阶段一致：越小越好
        raw_score = -(0.7 * latency + 0.3 * energy * 100.0)
        rewards.append(raw_score)

    rewards = zscore_normalize(rewards)
    return [float(x) for x in rewards]


def rule_reward_function(completions, **kwargs):
    """
    修复点3：
    规则奖励保留，但降权，避免模型刷关键词。
    这里只做“格式和基本推理结构”的轻量引导。
    """
    rewards = []

    for content in completions:
        score = 0.0
        text = content.strip()

        # 1) 有最终决策标签，加分
        if "[边缘卸载]" in text or "[本地计算]" in text:
            score += 1.0

        # 2) 结论放在末尾，再加一点
        if text.endswith("]"):
            score += 0.5

        # 3) 有基本分析痕迹
        if "分析" in text or "根据" in text or "由于" in text or "因为" in text:
            score += 0.5

        # 4) 出现数字，说明不是纯胡说
        if re.search(r"\d", text):
            score += 0.5

        # 5) 太长惩罚，防止水字数
        if len(text) > 300:
            score -= 1.0

        rewards.append(score)

    rewards = zscore_normalize(rewards)
    return [float(x) for x in rewards]


def combined_reward_function(completions, **kwargs):
    """
    修复点4：
    显式控制三种 reward 的权重。
    推荐：环境 > RM > 规则
    """
    rm_scores = rm_reward_function(completions, **kwargs)
    env_scores = env_reward_function(completions, **kwargs)
    rule_scores = rule_reward_function(completions, **kwargs)

    final_scores = []
    for r_rm, r_env, r_rule in zip(rm_scores, env_scores, rule_scores):
        score = 0.35 * r_rm + 0.50 * r_env + 0.15 * r_rule
        final_scores.append(float(score))

    return final_scores


# =========================
# 4. 主训练流程
# =========================
def main():
    print("🚀 启动 GRPO v4 训练引擎...")

    # -------- tokenizer --------
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 生成更稳

    # -------- dataset --------
    print("📦 正在加载 GRPO prompt 数据...")
    raw_dataset = load_dataset("json", data_files=grpo_data_path, split="train")
    dataset = raw_dataset.map(lambda x: {"prompt": x["prompt"]})

    # -------- policy model：先融合SFT，再挂新的LoRA做GRPO --------
    print("🧠 正在构建基于 SFT 的初始 Policy...")
    raw_base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},  # Policy 放逻辑cuda:0
    )
    sft_model = PeftModel.from_pretrained(raw_base_model, sft_adapter_path)
    model = sft_model.merge_and_unload()

    model.is_parallelizable = True
    model.model_parallel = True

    # -------- training args --------
    training_args = GRPOConfig(
        output_dir="./output/iov_qwen_grpo_v4",
        learning_rate=1e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=8,              # 先用8更稳更省显存；显存足够再改16
        generation_batch_size=8,
        max_completion_length=384,      # 比你原来256更适合解释+结论
        num_train_epochs=3,
        beta=0.02,                      # 比 0.001 更稳，减少策略漂移
        bf16=True,
        logging_steps=1,
        gradient_checkpointing=True,
        report_to="none",
        ddp_find_unused_parameters=False,
    )
    training_args._n_gpu = 1

    # -------- GRPO LoRA --------
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    # -------- trainer --------
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=combined_reward_function,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("🔥 开始训练：SFT 基座 + RM + Env Reward + 轻量规则奖励")
    torch.cuda.empty_cache()
    trainer.train()

    # -------- save --------
    print("💾 保存最终模型...")
    final_path = training_args.output_dir
    os.makedirs(final_path, exist_ok=True)

    trainer.state.save_to_json(os.path.join(final_path, "trainer_state.json"))
    trainer.save_model(os.path.join(final_path, "final_agent"))

    print(f"✅ GRPO v4 训练完成，模型保存至: {final_path}")


if __name__ == "__main__":
    main()