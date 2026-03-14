import json

input_file = "iov_train_data.jsonl"
output_file = "iov_train_data_v3.jsonl"

# 定义垃圾特征词
garbage_keywords = ['vol.', 'pp.', 'pages', 'issue', 'doi', 'journal', 'proceedings', 'et al.']

with open(input_file, 'r', encoding='utf-8') as f, \
     open(output_file, 'w', encoding='utf-8') as out:
    keep_count = 0
    drop_count = 0
    
    for line in f:
        data = json.loads(line)
        text = data['output'].lower()
        
        # 计算垃圾关键词出现的密度
        hit_count = sum(1 for kw in garbage_keywords if kw in text)
        
        # 如果一条简短的回答里出现了 3 个以上论文索引特征词，大概率是垃圾
        if hit_count >= 3:
            drop_count += 1
            continue
            
        out.write(json.dumps(data, ensure_ascii=False) + '\n')
        keep_count += 1

print(f"🚨 暴力清理结果：")
print(f"   - 丢弃疑似文献垃圾：{drop_count} 条")
print(f"   - 保留相对干净条目：{keep_count} 条")