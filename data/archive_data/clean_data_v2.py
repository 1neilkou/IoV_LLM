import json
import re

input_file = "iov_train_data.jsonl"
output_file = "iov_train_data_v2.jsonl"

def surgical_clean(text):
    # 寻找参考文献可能开始的标志性位置
    # 比如看到 [1] 或者 References 或者是大量出现 vol. 的地方
    markers = [
        r'References', 
        r'BIBLIOGRAPHY',
        r'\[1\]\s+[A-Z]', # 匹配形如 [1] 后跟大写字母的文献起始
        r'\n\d+\.\s+[A-Z]' # 匹配形如 1. 后跟大写字母的列表
    ]
    
    cleaned_text = text
    for marker in markers:
        parts = re.split(marker, cleaned_text, flags=re.IGNORECASE)
        if len(parts) > 1:
            # 只保留标志位之前的内容，即真正的正文
            cleaned_text = parts[0]
            
    # 如果处理后太短（说明整条都是垃圾），则标记为丢弃
    return cleaned_text.strip()

with open(input_file, 'r', encoding='utf-8') as f, \
     open(output_file, 'w', encoding='utf-8') as out:
    dropped = 0
    modified = 0
    for line in f:
        data = json.loads(line)
        original_text = data['output']
        new_text = surgical_clean(original_text)
        
        if len(new_text) < 30: # 剔除过短的无意义回复
            dropped += 1
            continue
        
        if len(new_text) < len(original_text):
            modified += 1
            
        data['output'] = new_text
        out.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"📊 清洗报告：修改了 {modified} 条数据的尾部噪声，丢弃了 {dropped} 条纯垃圾条目。")