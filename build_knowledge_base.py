from langchain_community.document_loaders import PyPDFLoader
# 修改前：from langchain.text_splitter import RecursiveCharacterTextSplitter
# 修改后（推荐用法）：
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
# 保持之前的降级兼容写法
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# 1. 加载你的车联网论文库
pdf_folder = "./data/pdfs"  # 假设你把论文都放在这
documents = []
for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_folder, file))
        documents.extend(loader.load())

# 2. 文本分段（Chunking）
# 为什么设为 500？为了适应车联网论文中紧凑的公式和定义
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

# 3. 向量化（Embedding）
# 建议使用 BGE 模型，这在简历里是加分项
model_name = "BAAI/bge-small-zh-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# 4. 构建并保存 Faiss 索引
print("正在构建向量索引...")
vectorstore = FAISS.from_documents(splits, embeddings)
vectorstore.save_local("faiss_iov_index")
print("✅ 知识库构建完成！索引已保存至 faiss_iov_index")