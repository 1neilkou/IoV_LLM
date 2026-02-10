import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# 配置路径
base_model_path = "./models/Qwen2.5-7B-Instruct"
adapter_path = "./output/iov_qwen_lora"
save_path = "./models/Qwen2.5-7B-IoV-Final"

print(f"正在加载基础模型: {base_model_path}")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="cpu", # 合并建议在内存中进行，防止显存溢出
    trust_remote_code=True
)

print(f"正在加载 LoRA 权重: {adapter_path}")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("正在进行权重熔焊 (Merging)...")
# merge_and_unload 会将 LoRA 权重合并进主模型并卸载 PEFT 结构
model = model.merge_and_unload()

print(f"正在保存完整模型至: {save_path}")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("🎉 合并完成！你现在拥有了一个独立的 15GB 车联网专家模型。")