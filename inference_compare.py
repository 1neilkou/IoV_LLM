import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. 基础配置
model_path = "./models/Qwen2.5-7B-Instruct"
adapter_path = "./output/iov_qwen_lora"
device = "cuda:0"  # 还是用你那张强大的 5090

print("正在加载原始模型与 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
# 加载原始模型
base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map={"": device}
)

# 2. 核心：加载微调后的插件
print("正在加载 LoRA 专家插件...")
peft_model = PeftModel.from_pretrained(base_model, adapter_path)

def ask_expert(prompt):
    messages = [
        {"role": "system", "content": "你是一个车联网专家。"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 生成回答
    generated_ids = peft_model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7
    )
    
    # 过滤掉 input 部分，只输出回答
    response = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return response

# 3. 现场考题
test_questions = [
    "请详细描述 Fan 等人在 2024 年提出的车辆边缘计算 (VEC) 任务卸载方案。",
    "在 6G 算力网络中，Di 等人提到的 'Pooling Contribution-Aware' 分配算法是如何工作的？",
    "算力网络中，如何平衡负载均衡与时延？请结合 Feng 等人的研究回答。"
]

print("\n" + "="*50)
for i, q in enumerate(test_questions):
    print(f"\n测试题目 {i+1}: {q}")
    print("-" * 20)
    print(f"【IoV 专家模型回答】:\n{ask_expert(q)}")
    print("="*50)