import json
import re
from collections import Counter

def is_high_quality(item):
    instruction = item.get('instruction', '')
    output = item.get('output', '')
    text = instruction + output
    
    # 规则1：长度过滤 (太短学不到知识，太长可能截断)
    if len(text) < 20 or len(text) > 4000:
        return False
        
    # 规则2：车联网特有噪音过滤 (比如连续超过 5 个十六进制无意义填充码)
    if len(re.findall(r'(0x[0-9a-fA-F]{2}\s*){5,}', text)) > 0:
        return False
        
    # 规则3：重复度惩罚 (N-gram 极度重复通常是爬虫抓取的脏数据)
    words = text.split()
    if len(words) > 50:
        word_counts = Counter(words)
        # 如果出现频率最高的词占比超过 30%，说明是车轱辘话
        if word_counts.most_common(1)[0][1] / len(words) > 0.3:
            return False
            
    return True

def clean_data(input_file, output_file):
    valid_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if is_high_quality(item):
                valid_data.append(item)
                
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"✅ 清洗完成！保留高质量样本：{len(valid_data)} 条。")

if __name__ == "__main__":
    # 假设你原始抓取的协议数据在这里
    clean_data('raw_v2x_data.jsonl', 'cleaned_v2x_data.jsonl')