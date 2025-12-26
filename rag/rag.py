import os
import time
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

app = FastAPI()

# 解決跨網域問題
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. 初始化模型
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./thesis_db", embedding_function=embeddings)

# 2. 接入 GPT-OSS 20B (!!! 修正了 http://http:// 錯誤 !!!)
llm = ChatOpenAI(
    # 修正重點：埠口必須是 8355，且建議改用 127.0.0.1 (因為在同一台主機且用了 --network host)
    base_url="http://127.0.0.1:8355/v1", 
    api_key="none",
    model="gpt-oss-20b"
)

# --- 讓 Open WebUI 偵測成功的介面 ---

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "gpt-oss-rag-expert",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "user"
            }
        ]
    }

# 補上這個介面，解決你看到的 405 錯誤
@app.get("/v1/openapi.json")
async def get_openapi():
    return {}

# --- 聊天核心邏輯 ---

@app.post("/v1/chat/completions")
async def chat_proxy(request: Request):
    body = await request.json()
    messages = body.get('messages', [])
    user_query = messages[-1]['content']
    
    print(f"收到問題: {user_query}") # 在終端機看進度

    # RAG 檢索
    target_student = "畢業生A" if "畢業生A" in user_query else "畢業生B"
    docs = vector_db.similarity_search(user_query, k=5, filter={"student_name": target_student})
    
    context = "\n".join([f"[第{d.metadata.get('page','?')}頁]: {d.page_content}" for d in docs])

    # 構造學術摘要 Prompt
    rag_prompt = f"請根據以下論文內容進行摘要回答：\n\n{context}\n\n問題：{user_query}"
    
    # 呼叫 20B 模型生成回答
    try:
        response = llm.invoke(rag_prompt)
        content = response.content
    except Exception as e:
        content = f"後端 GPT-OSS 20B 連線錯誤: {str(e)}"
    
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "gpt-oss-rag-expert",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)