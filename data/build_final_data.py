import json
import random
import os

# 1. 加载你的私有车联网数据
v2x_data = []
if os.path.exists("v2x_domain_qa.jsonl"):
    with open("v2x_domain_qa.jsonl", "r", encoding="utf-8") as f:
        v2x_data = [json.loads(line) for line in f]
else:
    print("找不到 v2x_domain_qa.jsonl，请确认路径！")
    exit()

# 2. 加载 Alpaca 通用数据
with open("alpaca_data.json", "r", encoding="utf-8") as f:
    alpaca_full = json.load(f)

# Alpaca 数据格式转换：把 input 拼接到 instruction 里
alpaca_formatted = []
for item in alpaca_full:
    inst = item["instruction"] + ("\n" + item["input"] if item.get("input", "") != "" else "")
    alpaca_formatted.append({"instruction": inst, "output": item["output"]})

# 3. 动态计算混合比例 (Data Mixture)
# 面试谈资：通常领域数据和通用数据保持 8:2 或 9:1 的比例最佳
# 这里我们因为测试数据只有 15 条，我们至少混入 10 条 Alpaca 数据感受一下
sample_size = max(int(len(v2x_data) * 0.2), 10) 
alpaca_sampled = random.sample(alpaca_formatted, sample_size)

# 4. 混合并彻底打乱分布
final_mix = v2x_data + alpaca_sampled
random.shuffle(final_mix)

# 5. 保存为最终给 5090 训练的文件
output_file = "iov_train_data_final.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for item in final_mix:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 数据混合完成！")
print(f"📊 数据分布 -> V2X专业数据: {len(v2x_data)} 条 | 通用数据 (Alpaca): {len(alpaca_sampled)} 条")
print(f"💾 最终微调数据集已保存为: {output_file}")