"""
app.py — Chainlit RAG chatbot for Sổ Tay Sinh Viên
Run with: chainlit run app.py

Fixes applied vs previous version:
1. Fixed imports: create_history_aware_retriever and friends live in `langchain`
   and `langchain.chains.combine_documents`, NOT in `langchain_classic`
2. Moved heavy loading (embeddings + vectorstore) to @cl.on_chat_start with
   progress messages so the app never appears frozen
3. cl.cache decorators removed from functions with arguments — they caused
   silent failures; caching is handled via module-level globals instead
"""

import os
from dotenv import load_dotenv
from typing import List, Optional

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ✅ Correct imports for LangChain 1.x — NOT langchain_classic
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

import chainlit as cl

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

VECTORSTORE_PATH = "./vectorstore"
EMBED_MODEL = "BAAI/bge-m3"
GROQ_MODEL = "llama-3.3-70b-versatile"
COLLECTION_NAME = "sotay_sinh_vien"

# ── Module-level cache (shared across all sessions, loaded once) ──────────────
# We use globals instead of @cl.cache to avoid argument-hashing issues.

_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectorstore: Optional[Chroma] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        print("🧠 Loading embedding model (one-time, ~570MB)...")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("   ✓ Embedding model loaded")
    return _embeddings


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        print("📦 Loading vector store...")
        _vectorstore = Chroma(
            persist_directory=VECTORSTORE_PATH,
            embedding_function=get_embeddings(),
            collection_name=COLLECTION_NAME,
        )
        print("   ✓ Vector store loaded")
    return _vectorstore


# ── Prompts ───────────────────────────────────────────────────────────────────

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Dựa vào lịch sử hội thoại và câu hỏi mới nhất của người dùng, "
     "hãy diễn đạt lại câu hỏi thành một câu hỏi độc lập, rõ ràng "
     "mà không cần lịch sử hội thoại để hiểu. "
     "KHÔNG trả lời câu hỏi, chỉ diễn đạt lại nếu cần, "
     "còn không thì giữ nguyên câu hỏi."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Bạn là trợ lý AI hỗ trợ sinh viên tra cứu thông tin từ Sổ Tay Sinh Viên.\n"
     "Hãy trả lời câu hỏi dựa HOÀN TOÀN vào ngữ cảnh được cung cấp bên dưới.\n\n"
     "Nếu thông tin không có trong tài liệu, hãy nói rõ: "
     "'Tôi không tìm thấy thông tin này trong Sổ Tay Sinh Viên.'\n"
     "Đừng bịa đặt hoặc suy đoán.\n"
     "Trả lời bằng tiếng Việt, ngắn gọn và có cấu trúc rõ ràng.\n\n"
     "Ngữ cảnh:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


# ── Chainlit lifecycle ────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    # Check vectorstore exists before trying to load anything
    if not os.path.exists(VECTORSTORE_PATH):
        await cl.Message(
            content=(
                "❌ Vector store chưa được tạo.\n\n"
                "Hãy chạy lệnh này trước:\n```\npython ingest.py\n```"
            )
        ).send()
        return

    # Show loading message so the user knows the app isn't frozen
    loading_msg = cl.Message(content="⏳ Đang khởi động, vui lòng đợi...")
    await loading_msg.send()

    # Load heavy resources (runs once, subsequent sessions reuse globals)
    # This is synchronous but shows the loading message first
    vectorstore = get_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7},
    )

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.1,
        streaming=True,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    # Modern LCEL chain (LangChain 1.x) —
    # Step 1: reformulate question with chat history context
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, CONTEXTUALIZE_PROMPT
    )
    # Step 2: answer using retrieved docs
    qa_chain = create_stuff_documents_chain(llm, QA_PROMPT)
    # Combine
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # Store per-user state in session
    cl.user_session.set("rag_chain", rag_chain)
    cl.user_session.set("chat_history", [])

    # Replace loading message with welcome
    loading_msg.content = (
        "👋 Xin chào! Tôi là trợ lý AI hỗ trợ tra cứu **Sổ Tay Sinh Viên**.\n\n"
        "Bạn có thể hỏi tôi về:\n"
        "- 💰 Học phí và các khoản phí\n"
        "- 📋 Quy chế học vụ\n"
        "- 🎓 Chính sách học bổng\n"
        "- 📅 Lịch học và thi cử\n"
        "- Và nhiều nội dung khác trong Sổ Tay Sinh Viên\n\n"
        "Hỏi bất cứ điều gì bạn cần biết!"
    )
    await loading_msg.update()


@cl.on_message
async def on_message(message: cl.Message):
    rag_chain = cl.user_session.get("rag_chain")
    chat_history: List = cl.user_session.get("chat_history")

    if rag_chain is None:
        await cl.Message(content="⚠️ Session lỗi. Hãy tải lại trang.").send()
        return

    response_msg = cl.Message(content="")
    await response_msg.send()

    res = await rag_chain.ainvoke(
        {"input": message.content, "chat_history": chat_history},
        config={"callbacks": [cl.LangchainCallbackHandler()]},
    )

    answer = res["answer"]
    response_msg.content = answer
    await response_msg.update()

    # Update history (keep last 10 messages = 5 turns)
    chat_history.append(HumanMessage(content=message.content))
    chat_history.append(AIMessage(content=answer))
    cl.user_session.set("chat_history", chat_history[-10:])

    # Show source documents
    source_docs = res.get("context", [])
    if source_docs:
        sources_text = "\n\n".join([
            f"📄 **Nguồn {i+1}:** ...{doc.page_content[:300]}..."
            for i, doc in enumerate(source_docs[:3])
        ])
        await cl.Message(
            content=f"**Tài liệu tham khảo:**\n\n{sources_text}",
            parent_id=response_msg.id,
        ).send()