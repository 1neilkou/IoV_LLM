import json
import random

def mix_datasets(domain_file, general_file, output_file, domain_ratio=0.9):
    # 1. 加载清洗后的垂直领域数据 (车联网)
    with open(domain_file, 'r', encoding='utf-8') as f:
        domain_data = [json.loads(line) for line in f]
        
    # 2. 加载开源的通用高质量数据 (比如下载一部分 OpenOrca 或 Magpie 数据)
    with open(general_file, 'r', encoding='utf-8') as f:
        general_data = [json.loads(line) for line in f]
        
    # 3. 计算需要抽样的通用数据量
    # 假设 domain 占 90%，那么 general 应该占 10%
    target_general_count = int(len(domain_data) / domain_ratio * (1 - domain_ratio))
    
    # 随机抽样并混合
    sampled_general = random.sample(general_data, min(target_general_count, len(general_data)))
    mixed_data = domain_data + sampled_general
    
    # 4. 彻底打乱数据分布，防止模型在一个 Epoch 内先学通用再学领域
    random.shuffle(mixed_data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in mixed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"✅ 混合完成！垂直领域: {len(domain_data)}条，通用领域: {len(sampled_general)}条。")

if __name__ == "__main__":
    # 运行此脚本前，请先去 HuggingFace 下载几千条通用指令数据保存为 general_data.jsonl
    mix_datasets('cleaned_v2x_data.jsonl', 'general_data.jsonl', 'final_mixed_data.jsonl')