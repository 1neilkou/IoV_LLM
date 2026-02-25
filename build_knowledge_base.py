from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import tqdm

# 1. 配置
pdf_folder = "./data/pdfs"
model_name = "BAAI/bge-small-zh-v1.5"
index_path = "faiss_iov_index"

# 2. 读取 PDF
documents = []
for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        try:
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            documents.extend(loader.load())
        except Exception as e: print(f"跳过 {file}: {e}")

# 3. 切分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(documents)

# 4. 【关键】严格清洗数据，确保完全是纯净字符串
texts = []
metadatas = []
for doc in splits:
    content = doc.page_content.strip()
    if len(content) > 10:
        # 强制剔除掉非 ASCII 或非正常字符，防止 tokenizer 崩溃
        clean_content = "".join(i for i in content if i.isprintable())
        texts.append(clean_content)
        metadatas.append(doc.metadata)

print(f"有效分片数: {len(texts)}")

# 5. 初始化 Embedding
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cuda'}
)

# 6. 【核心改变】手动分批构建，避免底层 Union 类型错误
print("开始分批构建索引...")
batch_size = 100
vectorstore = None

for i in tqdm.tqdm(range(0, len(texts), batch_size)):
    batch_texts = texts[i:i + batch_size]
    batch_metadatas = metadatas[i:i + batch_size]
    
    if vectorstore is None:
        vectorstore = FAISS.from_texts(batch_texts, embeddings, metadatas=batch_metadatas)
    else:
        # 采用添加模式，不触发大的列表校验
        vectorstore.add_texts(batch_texts, metadatas=batch_metadatas)

# 7. 保存
vectorstore.save_local(index_path)
print(f"✅ 知识库构建完成！已处理 {len(texts)} 条数据并保存至 {index_path}")