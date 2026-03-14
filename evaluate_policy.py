# evaluate_policy.py
import os
import re
import json
import math
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

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


def extract_decision(text: str):
    text = text.strip()
    if "[边缘卸载]" in text or "边缘卸载" in text:
        return "edge"
    if "[本地计算]" in text or "本地计算" in text:
        return "local"
    return None


def generate_completion(model, tokenizer, prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion = full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text
    return completion


def load_model(model_id, adapter_path=None, merge_adapter=False):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    if adapter_path:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        if merge_adapter:
            model = model.merge_and_unload()
    else:
        model = base_model

    model.eval()
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="./models/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--merge_adapter", action="store_true")
    parser.add_argument("--eval_file", type=str, default="data/v2x_eval_prompts.jsonl")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer, model = load_model(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        merge_adapter=args.merge_adapter
    )

    with open(args.eval_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    if args.max_samples is not None:
        data = data[:args.max_samples]

    results = []
    valid_count = 0
    correct_count = 0
    total_env_reward = 0.0
    total_latency = 0.0
    total_energy = 0.0

    for item in tqdm(data, desc="Evaluating"):
        prompt = item["prompt"]
        state = item["state"]
        oracle = item["oracle"]

        completion = generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens
        )

        pred_action = extract_decision(completion)

        if pred_action is not None:
            valid_count += 1
            latency, energy = calculate_v2x_cost(
                task_size=state["task_size"],
                cpu_cycles=state["cpu_cycles"],
                bandwidth=state["bandwidth"],
                local_cpu_freq=state["local_cpu_freq"],
                edge_cpu_freq=state["edge_cpu_freq"],
                decision=pred_action,
            )
            env_reward = -(0.7 * latency + 0.3 * energy * 100.0)

            if pred_action == oracle["optimal_action"]:
                correct_count += 1
        else:
            latency, energy = None, None
            env_reward = -9999.0  # 无法解析时强惩罚

        if latency is not None:
            total_latency += latency
            total_energy += energy
            total_env_reward += env_reward

        results.append({
            "prompt": prompt,
            "completion": completion,
            "pred_action": pred_action,
            "oracle_action": oracle["optimal_action"],
            "latency_ms": latency,
            "energy_j": energy,
            "env_reward": env_reward,
            "is_correct": pred_action == oracle["optimal_action"] if pred_action else False
        })

    valid_ratio = valid_count / len(data) if len(data) else 0.0
    acc = correct_count / valid_count if valid_count else 0.0
    avg_latency = total_latency / valid_count if valid_count else None
    avg_energy = total_energy / valid_count if valid_count else None
    avg_env_reward = total_env_reward / valid_count if valid_count else None

    summary = {
        "num_samples": len(data),
        "valid_decision_ratio": valid_ratio,
        "optimal_decision_accuracy": acc,
        "avg_latency_ms": avg_latency,
        "avg_energy_j": avg_energy,
        "avg_env_reward": avg_env_reward
    }

    output = {
        "summary": summary,
        "details": results
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("✅ 评估完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()