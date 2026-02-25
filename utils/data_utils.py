import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from typing import List, Dict

class V2XDataProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def pack_dataset(self, dataset: List[Dict[str, str]]) -> List[Dict[str, torch.Tensor]]:
        """
        将 V2X 原始数据打包成固定长度的块 (Data Packing)
        """
        all_token_ids = []
        
        # 1. 将所有样本序列化并合并
        for item in tqdm(dataset, desc="Tokenizing and joining"):
            # 这里的 Prompt 模板要和你的推理模板保持一致
            text = f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>"
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            # 添加 EOS 符号，作为不同样本间的物理隔断
            all_token_ids.extend(tokens + [self.tokenizer.eos_token_id])

        # 2. 分桶 (Chunking)
        total_len = len(all_token_ids)
        # 抛弃不足一个窗口的尾部数据
        num_blocks = total_len // self.max_seq_length
        total_len = num_blocks * self.max_seq_length

        packed_dataset = []
        for i in range(0, total_len, self.max_seq_length):
            chunk = all_token_ids[i : i + self.max_seq_length]
            
            # 在 Packing 模式下，labels 通常等于 input_ids
            # 模型通过内部的 Causal Mask 自动处理预测逻辑
            packed_dataset.append({
                "input_ids": torch.tensor(chunk, dtype=torch.long),
                "labels": torch.tensor(chunk, dtype=torch.long),
                "attention_mask": torch.ones(self.max_seq_length, dtype=torch.long)
            })

        print(f"✅ Packing 完成: 产出 {len(packed_dataset)} 个训练 Block (长度: {self.max_seq_length})")
        return packed_dataset

# 以后可以扩展 RAG 数据预处理等功能