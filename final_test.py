from transformers import pipeline
# 加载你刚刚熔载完成的 15GB 完整权重
pipe = pipeline("text-generation", model="./models/Qwen2.5-7B-IoV-Final", device=0, torch_dtype="bfloat16")
# 考问一个论文中的核心逻辑
prompt = "你是一个车联网专家，请解释 6G 算力网络中 RSU 如何协同处理高时延任务？"
# 验证输出是否带有微调注入的专业语感
print(pipe(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n", max_new_tokens=200)[0]['generated_text'])