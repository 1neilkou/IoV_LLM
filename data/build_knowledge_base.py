import os
import json
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# 1. 配置路径与模型
PDF_FILE = "pdfs/v2x_paper_1.pdf"  # 替换为你的 6G/V2X 参考文献
INDEX_FILE = "faiss_iperov_index.bin"
METADATA_FILE = "faiss_metadata.json"

# 面试重点：选用 BAAI (智源研究院) 的 BGE 模型，它是目前中文/中英混合检索的霸主
EMBEDDING_MODEL_ID = "BAAI/bge-large-zh-v1.5" 

def extract_and_chunk_pdf(pdf_path, chunk_size=500, overlap=50):
    """
    工业级切块策略 (Chunking with Overlap)
    加入 overlap 防止切断关键上下文（如一段协议横跨两页）
    """
    print(f"📄 正在解析 PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text().replace('\n', ' ') # 简单清洗换行符
    
    chunks = []
    # 使用滑动窗口进行切块
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 50: # 过滤掉太短的无意义碎块
            chunks.append(chunk)
            
    print(f"✂️ 切块完成！共生成 {len(chunks)} 个文本块 (Chunk Size: {chunk_size}, Overlap: {overlap})")
    return chunks

def main():
    # ================= 1. 文献切片 =================
    if not os.path.exists(PDF_FILE):
        print(f"❌ 找不到文件 {PDF_FILE}，请先准备好你的 PDF！")
        return
        
    chunks = extract_and_chunk_pdf(PDF_FILE)

    # ================= 2. 加载 BGE 向量模型 =================
    print(f"🧠 正在加载 Embedding 模型: {EMBEDDING_MODEL_ID} ...")
    # 自动调用 5090 GPU 加速向量化
    model = SentenceTransformer(EMBEDDING_MODEL_ID, device='cuda')

    # ================= 3. 生成文本向量 =================
    print("🌊 正在将文本块转化为高维稠密向量 (Dense Vectors)...")
    # BGE 要求对于检索库（被检索的文档），不需要加特殊前缀
    embeddings = model.encode(chunks, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    
    # 获取向量维度 (BGE-large 通常是 1024 维)
    dim = embeddings.shape[1]
    print(f"📐 向量维度: {dim}")

    # ================= 4. 构建 FAISS 向量数据库 =================
    print("🗄️ 正在构建 FAISS 索引...")
    # 使用内积 (Inner Product) 索引，因为我们上面做了 normalize_embeddings=True
    # 这在数学上等价于计算余弦相似度 (Cosine Similarity)，是大厂检索标配
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings).astype('float32'))

    # ================= 5. 落盘保存 =================
    faiss.write_index(index, INDEX_FILE)
    
    # 必须把文本块也存下来，因为 FAISS 只存向量，不存原文
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
        
    print(f"✅ 知识库构建成功！")
    print(f"💽 FAISS 索引已保存至: {INDEX_FILE}")
    print(f"💽 原文 Metadata 已保存至: {METADATA_FILE}")

if __name__ == "__main__":
    main()