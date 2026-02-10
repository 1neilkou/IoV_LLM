import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. 加载配置
model_path = "./models/Qwen2.5-7B-IoV-Final"  # 你合并后的成品模型
index_path = "faiss_iov_index"
device = "cuda:0"

print("正在初始化专家大脑与数字图书馆...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5", model_kwargs={'device': 'cuda'})
vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

def chat_with_rag(question):
    # 2. 检索：从 3384 个分片中找出最相关的 3 个
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n---\n".join([doc.page_content for doc in docs])
    source_papers = set([doc.metadata['source'] for doc in docs])

    # 3. 构造增强 Prompt (让模型基于参考资料回答)
    prompt = f"""你是一个车联网专家。请根据以下提供的参考资料，专业地回答用户的问题。
如果参考资料中没有相关信息，请结合你的专业知识回答，但要注明“基于通用知识”。

【参考资料】：
{context}

【用户问题】：
{question}
"""

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 4. 生成回答
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=True, temperature=0.7)
    response = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    return response, source_papers

# --- 现场测试 ---
question = "请解释一下 VIMA 架构是如何处理多模态提示（Multimodal Prompting）的？"
print(f"\n提问: {question}")
ans, sources = chat_with_rag(question)
print(f"\n【专家回答】:\n{ans}")
print(f"\n【参考来源】: {sources}")