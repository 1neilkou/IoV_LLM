import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 锁定空闲的 5090

import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 加载模型（建议用 Qwen2.5-7B，它对中文和垂直领域理解极强）
model_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# 5090 显存够，直接用 bf16 精度，速度最快
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# 2. 定义 LoRA 配置（你的简历核心：参数高效微调）
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=16, # 秩，决定了 Adapter 的参数量
    lora_alpha=32, 
    lora_dropout=0.1
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters() # 打印你会发现只训练了不到 1% 的参数