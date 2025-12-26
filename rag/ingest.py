import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 設定區 ---
# 建議使用本地的 Embedding 模型，這樣就完全不用連網，Demo 時反應也快
# 推薦模型: "sentence-transformers/all-MiniLM-L6-v2" (輕巧好用)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def ingest_thesis(file_path, student_name):
    print(f"正在處理 {student_name} 的論文: {file_path}...")
    
    # 1. 載入 PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # 2. 切片 (Chunking)
    # 針對上百頁論文，建議 chunk_size 設為 800~1000，重疊 100
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(pages)

    # 3. 注入 Metadata (這是最關鍵的一步)
    # 讓每個切片都知道自己屬於哪個畢業生
    for chunk in splits:
        chunk.metadata["student_name"] = student_name
    
    # 4. 存入並持久化 ChromaDB
    # persist_directory 指定資料要存放在哪個資料夾
    vector_db = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./thesis_db" 
    )
    
    # 在舊版 Chroma 需要調用 persist()，新版會自動儲存
    print(f"✅ {student_name} 的論文處理完成，已存入 ./thesis_db")

if __name__ == "__main__":
    # Demo 範例：存入兩位畢業生的論文
    ingest_thesis("/home/ntust_spark/Downloads/attention.pdf", "畢業生A")
    ingest_thesis("/home/ntust_spark/Downloads/sam2.pdf", "畢業生B")