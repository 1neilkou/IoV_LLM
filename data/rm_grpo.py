import json

# 输入：你训练 RM 用的偏好数据
rm_preference_file = "v2x_rm_preference.jsonl"
# 输出：专门给 GRPO 用的 Prompt 集
grpo_dataset_file = "v2x_grpo_prompts.jsonl"

grpo_prompts = []

with open(rm_preference_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        # 只保留 prompt，因为 GRPO 不需要参考答案，它要模型自己生成
        grpo_prompts.append({"prompt": data["prompt"]})

# 写入新文件
with open(grpo_dataset_file, "w", encoding="utf-8") as f:
    for item in grpo_prompts:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 成功提取 {len(grpo_prompts)} 条 GRPO 专用考题！")