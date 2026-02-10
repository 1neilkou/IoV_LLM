import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 1. 核心资源加载 (复用之前的逻辑)
model_path = "./models/Qwen2.5-7B-IoV-Final"
index_path = "faiss_iov_index"
device = "cuda:0"

print("正在启动 Web 专家系统...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5", model_kwargs={'device': 'cuda'})
vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

def predict(message, history):
    # RAG 检索逻辑
    docs = vectorstore.similarity_search(message, k=3)
    context = "\n---\n".join([f"【片段{i+1}来自 {doc.metadata['source']}】\n{doc.page_content}" for i, doc in enumerate(docs)])
    
    prompt = f"你是一个车联网专家。请根据参考资料回答问题。资料如下：\n{context}\n\n问题：{message}"
    
    # 推理
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=True, temperature=0.7)
    response = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    # 返回回答和检索到的参考资料
    return response + "\n\n" + "--- 检索参考 ---\n" + context

# 2. 构造 Gradio 界面
demo = gr.ChatInterface(
    predict,
    title="IoV-LLM: 车联网/机器人多模态专家系统",
    description="基于 Qwen2.5-7B 微调 + RAG 检索增强。支持 B5G/6G 协议分析与 VIMA 架构咨询。",
    examples=["VIMA 如何处理多模态提示？", "解释 6G 算力网络中 RSU 的协同逻辑", "RT-2 模型是如何进行机器人控制的？"],
    textbox=gr.Textbox(placeholder="请输入您的技术问题...", container=False, scale=7),
)

if __name__ == "__main__":
    # share=True 会生成一个公网地址，方便你在新加坡访问
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)