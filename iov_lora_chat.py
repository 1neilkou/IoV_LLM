import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. 设置环境变量，保持离线加载习惯
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 2. 路径设置
base_model_id = "./models/Qwen2.5-7B-Instruct"
lora_model_path = "./output/iov_qwen_lora/final_model"

def main():
    print("🚀 正在启动 5090 推理引擎...")

    # ================= 1. 加载 Tokenizer =================
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # ================= 2. 加载 Base 模型 =================
    print("📦 正在加载 Qwen2.5 基座模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa" # 推理时也可以用 sdpa 加速
    )

    # ================= 3. 动态挂载 LoRA 权重 =================
    print(f"🔗 正在挂载 V2X 领域 LoRA 适配器: {lora_model_path} ...")
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    
    # 面试加分项：在实际部署中，通常会把 LoRA 权重 merge 进 base 模型以提升推理速度
    # model = model.merge_and_unload() 

    model.eval() # 切换到评估模式
    print("✅ V2X 专家模型已就绪！(输入 'quit' 或 'exit' 退出)")
    print("-" * 50)

    # ================= 4. 终端多轮对话 Loop =================
    messages = [
        {"role": "system", "content": "你是一个6G车联网与算网一体化领域的资深专家，请用专业、严谨的语气回答用户的问题。"}
    ]

    while True:
        user_input = input("\n🧑‍💻 你: ")
        if user_input.lower() in ["quit", "exit"]:
            print("👋 专家已下线，再见！")
            break
        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})

        # 使用 Qwen 官方的 Chat Template 构建 Prompt
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 生成回答
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.3, # 垂直领域知识建议温度调低，减少幻觉
                top_p=0.85,
                repetition_penalty=1.05
            )
            
        # 截取新生成的部分（去掉 prompt）
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"\n🤖 V2X专家: {response}")
        
        # 将回答加入历史上下文，支持多轮对话
        messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
    