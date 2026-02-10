from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# 1. 配置路径
pdf_folder = "./data/pdfs"
model_name = "BAAI/bge-small-zh-v1.5"
index_path = "faiss_iov_index"

# 2. 检查 PDF 文件夹是否存在
if not os.path.exists(pdf_folder) or not os.listdir(pdf_folder):
    print(f"❌ 错误：文件夹 {pdf_folder} 为空，请先上传 PDF 论文！")
    exit()

# 3. 加载并清理文档
print("正在读取 PDF 并进行预处理...")
documents = []
for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        try:
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            documents.extend(loader.load())
        except Exception as e:
            print(f"⚠️ 跳过无法读取的文件 {file}: {e}")

# 4. 文本切分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

# 5. 【关键修复】确保所有内容都是纯字符串，并过滤掉无效分片
texts = [str(doc.page_content) for doc in splits if len(doc.page_content.strip()) > 10]
metadatas = [doc.metadata for doc in splits if len(doc.page_content.strip()) > 10]

print(f"有效分片数: {len(texts)}")

# 6. 初始化嵌入模型（强制使用 CPU/GPU 兼容模式）
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cuda'},  # 充分利用你的 5090
    encode_kwargs={'normalize_embeddings': True}
)

# 7. 构建索引
print("正在构建向量索引，这可能需要一分钟...")
vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
vectorstore.save_local(index_path)

print(f"✅ 知识库构建完成！索引已保存至 {index_path}")