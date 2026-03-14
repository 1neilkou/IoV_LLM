import json
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

def offline_pack(input_file, output_file, model_id, max_seq_length=2048):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    all_token_ids = []
    
    print("⏳ 正在进行 Tokenize...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            item = json.loads(line)
            # 严格对齐 Qwen2.5 的 Chat Template
            text = f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>\n"
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_token_ids.extend(tokens)
            
    # 切分成固定长度的 Block
    total_len = (len(all_token_ids) // max_seq_length) * max_seq_length
    packed_dataset = []
    
    print("📦 正在拼接成固定长度 Block...")
    for i in range(0, total_len, max_seq_length):
        chunk = all_token_ids[i : i + max_seq_length]
        packed_dataset.append(torch.tensor(chunk, dtype=torch.long))
        
    # 直接保存为 PyTorch 张量文件
    torch.save(packed_dataset, output_file)
    print(f"✅ 离线 Packing 完成！生成了 {len(packed_dataset)} 个 {max_seq_length} 长度的张量块。")

if __name__ == "__main__":
    # 将最终混合好的数据打包，直接供 5090 极速读取
    offline_pack('final_mixed_data.jsonl', 'packed_v2x_tensor.pt', '../models/Qwen2.5-7B-Instruct')