import os
import json
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. 保持离线与 GPU 隔离
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 这里不强制离线，因为 BGE 模型可能还需要走镜像，但为了求稳，你也可以在终端再次注入 HF_ENDPOINT

# 2. 路径配置
BASE_MODEL_PATH = "./models/Qwen2.5-7B-Instruct"
LORA_PATH = "./output/iov_qwen_lora/final_model"
EMBEDDING_MODEL_ID = "BAAI/bge-large-zh-v1.5"
INDEX_FILE = "./data/faiss_iov_index.bin"
METADATA_FILE = "./data/faiss_metadata.json"

def load_rag_components():
    """加载 RAG 的三大核心组件：向量模型、FAISS 库、文本元数据"""
    print("🔍 [1/3] 正在唤醒 BGE 向量检索引擎...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_ID, device='cuda')
    
    print("🗄️ [2/3] 正在挂载 FAISS 知识库...")
    index = faiss.read_index(INDEX_FILE)
    
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    return embedder, index, metadata

def load_llm():
    """加载基座模型与微调权重"""
    print("🧠 [3/3] 正在加载 Qwen2.5 基座与 V2X LoRA 适配器...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa"
    )
    
    # 挂载微调后的 LoRA
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()
    return tokenizer, model

def search_context(query, embedder, index, metadata, top_k=3):
    """
    检索召回逻辑：将用户提问向量化，去 FAISS 里计算内积，取 Top-K
    """
    # 针对 BGE 模型，用户的检索 query 建议加上前缀，以提升相似度计算的精度
    instruction = "为这个句子生成表示以用于检索相关文章："
    query_vector = embedder.encode([instruction + query], normalize_embeddings=True)
    
    # FAISS 检索：返回相似度得分 (D) 和 索引位置 (I)
    distances, indices = index.search(np.array(query_vector).astype('float32'), top_k)
    
    retrieved_texts = []
    for idx in indices[0]:
        if idx != -1 and idx < len(metadata):
            retrieved_texts.append(metadata[idx])
            
    return retrieved_texts

def main():
    print("🚀 正在启动 V2X 算网一体化 RAG 专家系统...")
    
    # 初始化所有组件
    embedder, index, metadata = load_rag_components()
    tokenizer, model = load_llm()
    
    print("-" * 60)
    print("✅ RAG 专家已上线！(输入 'quit' 退出)")
    print("💡 提示：你可以问一些 6G 车联网、资源调度、协议相关的问题。")
    print("-" * 60)

    while True:
        user_input = input("\n🧑‍💻 你: ")
        if user_input.lower() in ["quit", "exit"]:
            print("👋 专家已下线，再见！")
            break
        if not user_input.strip():
            continue

        # 1. 从 FAISS 检索相关文献片段
        contexts = search_context(user_input, embedder, index, metadata, top_k=3)
        context_str = "\n\n---\n\n".join(contexts)
        
        print(f"\n[RAG 正在从文献库中检索到 {len(contexts)} 段强相关背景知识...]")

        # 2. 组装 RAG Prompt (大厂标配的 RAG 模板)
        rag_prompt = f"""你是一个6G车联网领域的资深专家。请基于以下【参考资料】来回答用户的问题。
要求：
1. 回答要严谨专业，逻辑清晰。
2. 如果参考资料中无法完全解答该问题，你可以结合你自身的专业知识进行补充。

【参考资料】：
{context_str}

【用户提问】：
{user_input}
"""
        
        # 3. 输入大模型生成回答
        messages = [{"role": "user", "content": rag_prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=800,
                temperature=0.2, # 降低温度，防止大模型胡编乱造，强迫它多看参考资料
                top_p=0.85
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"\n🤖 V2X RAG专家: {response}")

if __name__ == "__main__":
    main()