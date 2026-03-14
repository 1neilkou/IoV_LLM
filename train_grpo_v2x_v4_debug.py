# train_grpo_v2x1_v4_debug.py
import os
import re
import math
import random
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
# 1. Debug 配置
# =========================
DEBUG_MODE = True                  # 总开关
DEBUG_REWARD_EVERY = 10           # 每多少次 reward 调用打印一次统计
DEBUG_SAMPLE_EVERY = 50           # 每多少次 reward 调用打印一次样例
DEBUG_MAX_PROMPT_PREVIEW = 220
DEBUG_MAX_COMPLETION_PREVIEW = 260
DEBUG_RANDOM_SAMPLE = False       # True=随机打印样例；False=固定打印第一个
DEBUG_LIMIT_DATASET = 100       # 如 100 / 500；None 表示全量

_reward_debug_state = {
    "call_count": 0
}

# =========================
# 2. 部署 RM 模型
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
# 3. 工具函数
# =========================
def zscore_normalize(values, eps=1e-6):
    arr = np.array(values, dtype=np.float32)
    mean = arr.mean()
    std = arr.std()
    if std < eps:
        return arr.tolist()
    return ((arr - mean) / (std + eps)).tolist()


def summarize_values(name, values):
    arr = np.array(values, dtype=np.float32)
    return (
        f"{name}: "
        f"mean={arr.mean():.4f}, std={arr.std():.4f}, "
        f"min={arr.min():.4f}, max={arr.max():.4f}"
    )


def safe_preview(text, max_len=200):
    text = text.replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[:max_len] + " ..."


def extract_prompts_from_kwargs(kwargs, num_samples):
    prompts = None

    if "prompts" in kwargs and kwargs["prompts"] is not None:
        prompts = kwargs["prompts"]
    elif "prompt" in kwargs and kwargs["prompt"] is not None:
        prompts = kwargs["prompt"]

    if prompts is None:
        return [""] * num_samples

    if isinstance(prompts, str):
        return [prompts] * num_samples

    if isinstance(prompts, list):
        return prompts

    return [""] * num_samples


def calculate_v2x_cost(task_size, cpu_cycles, bandwidth, local_cpu_freq, edge_cpu_freq, decision):
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

    state["cpu_cycles"] *= 1e9
    state["local_cpu_freq"] *= 1e9
    state["edge_cpu_freq"] *= 1e9
    return state


def extract_decision(completion):
    text = completion.strip()

    if "[边缘卸载]" in text or "边缘卸载" in text:
        return "edge"
    if "[本地计算]" in text or "本地计算" in text:
        return "local"

    return None


# =========================
# 4. reward 分量
# =========================
def rm_reward_function(completions, **kwargs):
    prompts = extract_prompts_from_kwargs(kwargs, len(completions))
    texts = [p + "\n\n" + c for p, c in zip(prompts, completions)]

    try:
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

    except Exception as e:
        print(f"[RM_REWARD_ERROR] {type(e).__name__}: {e}")
        return [0.0] * len(completions)


def env_reward_function(completions, **kwargs):
    prompts = extract_prompts_from_kwargs(kwargs, len(completions))
    rewards = []

    for prompt, completion in zip(prompts, completions):
        try:
            state = parse_prompt_state(prompt)
            decision = extract_decision(completion)

            if state is None:
                rewards.append(-1.0)
                continue

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

            raw_score = -(0.7 * latency + 0.3 * energy * 100.0)
            rewards.append(raw_score)

        except Exception as e:
            print(f"[ENV_REWARD_ERROR] {type(e).__name__}: {e}")
            rewards.append(-2.0)

    rewards = zscore_normalize(rewards)
    return [float(x) for x in rewards]


def rule_reward_function(completions, **kwargs):
    rewards = []

    for content in completions:
        try:
            score = 0.0
            text = content.strip()

            if "[边缘卸载]" in text or "[本地计算]" in text:
                score += 1.0

            if text.endswith("]"):
                score += 0.5

            if "分析" in text or "根据" in text or "由于" in text or "因为" in text:
                score += 0.5

            if re.search(r"\d", text):
                score += 0.5

            if len(text) > 300:
                score -= 1.0

            rewards.append(score)

        except Exception as e:
            print(f"[RULE_REWARD_ERROR] {type(e).__name__}: {e}")
            rewards.append(0.0)

    rewards = zscore_normalize(rewards)
    return [float(x) for x in rewards]


def debug_print_reward_info(
    prompts,
    completions,
    rm_scores,
    env_scores,
    rule_scores,
    final_scores,
):
    _reward_debug_state["call_count"] += 1
    call_id = _reward_debug_state["call_count"]

    should_print_stats = DEBUG_MODE and (call_id % DEBUG_REWARD_EVERY == 0)
    should_print_sample = DEBUG_MODE and (call_id % DEBUG_SAMPLE_EVERY == 0)

    if should_print_stats:
        print("\n" + "=" * 100)
        print(f"[REWARD DEBUG] call_count = {call_id}")
        print(summarize_values("RM", rm_scores))
        print(summarize_values("ENV", env_scores))
        print(summarize_values("RULE", rule_scores))
        print(summarize_values("FINAL", final_scores))

        # 简单诊断
        arr_rm = np.array(rm_scores, dtype=np.float32)
        arr_env = np.array(env_scores, dtype=np.float32)
        arr_rule = np.array(rule_scores, dtype=np.float32)
        arr_final = np.array(final_scores, dtype=np.float32)

        rm_zero_ratio = float(np.mean(np.isclose(arr_rm, 0.0, atol=1e-6)))
        env_zero_ratio = float(np.mean(np.isclose(arr_env, 0.0, atol=1e-6)))
        rule_zero_ratio = float(np.mean(np.isclose(arr_rule, 0.0, atol=1e-6)))

        print(
            f"[REWARD DEBUG] zero_ratio -> "
            f"RM={rm_zero_ratio:.2%}, ENV={env_zero_ratio:.2%}, RULE={rule_zero_ratio:.2%}"
        )

        valid_decision_ratio = np.mean([
            1.0 if extract_decision(c) is not None else 0.0
            for c in completions
        ])
        print(f"[REWARD DEBUG] valid_decision_ratio = {valid_decision_ratio:.2%}")

    if should_print_sample and len(completions) > 0:
        idx = random.randint(0, len(completions) - 1) if DEBUG_RANDOM_SAMPLE else 0

        prompt_preview = safe_preview(prompts[idx], DEBUG_MAX_PROMPT_PREVIEW)
        completion_preview = safe_preview(completions[idx], DEBUG_MAX_COMPLETION_PREVIEW)
        decision = extract_decision(completions[idx])
        state = parse_prompt_state(prompts[idx])

        print("-" * 100)
        print(f"[SAMPLE DEBUG] call_count={call_id}, sample_idx={idx}")
        print(f"[PROMPT] {prompt_preview}")
        print(f"[COMPLETION] {completion_preview}")
        print(f"[PARSED_DECISION] {decision}")
        print(f"[PARSED_STATE] {state}")
        print(
            f"[REWARD BREAKDOWN] "
            f"RM={rm_scores[idx]:.4f}, "
            f"ENV={env_scores[idx]:.4f}, "
            f"RULE={rule_scores[idx]:.4f}, "
            f"FINAL={final_scores[idx]:.4f}"
        )
        print("=" * 100 + "\n")


def combined_reward_function(completions, **kwargs):
    prompts = extract_prompts_from_kwargs(kwargs, len(completions))

    rm_scores = rm_reward_function(completions, **kwargs)
    env_scores = env_reward_function(completions, **kwargs)
    rule_scores = rule_reward_function(completions, **kwargs)

    final_scores = []
    for r_rm, r_env, r_rule in zip(rm_scores, env_scores, rule_scores):
        score = 0.35 * r_rm + 0.50 * r_env + 0.15 * r_rule
        final_scores.append(float(score))

    if DEBUG_MODE:
        debug_print_reward_info(
            prompts=prompts,
            completions=completions,
            rm_scores=rm_scores,
            env_scores=env_scores,
            rule_scores=rule_scores,
            final_scores=final_scores,
        )

    return final_scores


# =========================
# 5. 主训练流程
# =========================
def main():
    print("🚀 启动 GRPO v4 DEBUG 训练引擎...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("📦 正在加载 GRPO prompt 数据...")
    raw_dataset = load_dataset("json", data_files=grpo_data_path, split="train")

    if DEBUG_LIMIT_DATASET is not None:
        debug_n = min(DEBUG_LIMIT_DATASET, len(raw_dataset))
        raw_dataset = raw_dataset.select(range(debug_n))
        print(f"🧪 DEBUG 模式：仅使用前 {debug_n} 条样本进行训练")

    dataset = raw_dataset.map(lambda x: {"prompt": x["prompt"]})

    print("🧠 正在构建基于 SFT 的初始 Policy...")
    raw_base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    sft_model = PeftModel.from_pretrained(raw_base_model, sft_adapter_path)
    model = sft_model.merge_and_unload()

    model.is_parallelizable = True
    model.model_parallel = True

    training_args = GRPOConfig(
        output_dir="./output/iov_qwen_grpo_v4_debug",
        learning_rate=1e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=8,
        generation_batch_size=8,
        max_completion_length=384,
        num_train_epochs=3,
        beta=0.02,
        bf16=True,
        logging_steps=1,
        gradient_checkpointing=True,
        report_to="none",
        ddp_find_unused_parameters=False,
    )
    training_args._n_gpu = 1

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=combined_reward_function,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("🔥 开始训练：Debug 日志已开启")
    torch.cuda.empty_cache()
    trainer.train()

    print("💾 保存最终模型...")
    final_path = training_args.output_dir
    os.makedirs(final_path, exist_ok=True)

    trainer.state.save_to_json(os.path.join(final_path, "trainer_state.json"))
    trainer.save_model(os.path.join(final_path, "final_agent"))

    print(f"✅ GRPO v4 DEBUG 训练完成，模型保存至: {final_path}")


if __name__ == "__main__":
    main()