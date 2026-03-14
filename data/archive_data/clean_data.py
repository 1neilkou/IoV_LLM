import json
import re

input_file = "data/iov_train_data.jsonl"
output_file = "data/iov_train_data_cleaned.jsonl"

def is_garbage(text):
    # 过滤明显的论文参考文献模式
    patterns = [
        r'vol\.\s*\d+', 
        r'pp\.\s*\d+–\d+', 
        r'\[\d+\]', 
        r'IEEE', 
        r'Conference',
        r'http'
    ]
    matches = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
    # 如果一条回答里包含了过多的引用特征，判定为噪声
    return matches >= 2

with open(input_file, 'r', encoding='utf-8') as f, \
     open(output_file, 'w', encoding='utf-8') as out:
    count = 0
    for line in f:
        data = json.loads(line)
        if not is_garbage(data['output']):
            out.write(json.dumps(data, ensure_ascii=False) + '\n')
        else:
            count += 1

print(f"✅ 清理完成！删除了 {count} 条参考文献垃圾数据。")