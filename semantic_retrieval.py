import json
import time
import os
import torch  # Thêm thư viện để check GPU
from typing import List, Dict, Any
from operator import itemgetter

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages, HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough

# --- 1. DATA LOADING FUNCTIONS (Giữ nguyên) ---

def load_json_data(json_path: str) -> List[Document]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    for item in data:
        content = item['content']
        metadata = item['metadata']
        metadata['id'] = item['id']
        
        if metadata.get('content_type') == 'paragraph':
            heading = " > ".join(metadata['heading_path'][-2:])
            page_content = f"{heading}\n{content}"
        else:
            page_content = content
        
        documents.append(Document(page_content=page_content, metadata=metadata))
    
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    MAX_CHAR_LENGTH_DIEU = 2900
    MAX_CHAR_LENGTH_PARA = 1200

    dieu_docs = [doc for doc in documents if doc.metadata.get('content_type') == 'dieu']
    para_docs = [doc for doc in documents if doc.metadata.get('content_type') == 'paragraph']
    other_docs = [doc for doc in documents if doc.metadata.get('content_type') not in ['dieu', 'paragraph']]

    splitters = {
        'dieu': RecursiveCharacterTextSplitter(
            chunk_size=2500, chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""]
        ),
        'paragraph': RecursiveCharacterTextSplitter(
            chunk_size=1100, chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        ),
        'other': RecursiveCharacterTextSplitter(
            chunk_size=1200, chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    }

    split_dieu = splitters['dieu'].split_documents(dieu_docs) if dieu_docs else []
    split_para = splitters['paragraph'].split_documents(para_docs) if para_docs else []
    split_other = splitters['other'].split_documents(other_docs) if other_docs else []

    def enforce_max_length(docs: List[Document], max_len: int) -> List[Document]:
        splitter = CharacterTextSplitter(chunk_size=max_len, chunk_overlap=50)
        result = []
        for doc in docs:
            if len(doc.page_content) > max_len:
                result.extend(splitter.split_documents([doc]))
            else:
                result.append(doc)
        return result

    split_dieu = enforce_max_length(split_dieu, MAX_CHAR_LENGTH_DIEU)
    split_para = enforce_max_length(split_para, MAX_CHAR_LENGTH_PARA)
    split_other = enforce_max_length(split_other, MAX_CHAR_LENGTH_PARA)

    return split_dieu + split_para + split_other

# --- 2. RETRIEVER & LLM SETUP (Đã tối ưu GPU) ---

def create_vector_retriever(documents: List[Document], model_name: str, k: int = 5, cache_dir: str = None) -> FAISS:
    # Tự động chọn thiết bị (GPU nếu có)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing Embeddings on device: {device}")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device, 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )

    if cache_dir and os.path.exists(os.path.join(cache_dir, 'faiss_index')):
        try:
            vectorstore = FAISS.load_local(
                os.path.join(cache_dir, 'faiss_index'),
                embeddings,
                allow_dangerous_deserialization=True
            )
            return vectorstore.as_retriever(search_kwargs={"k": k})
        except:
            pass

    # Xử lý documents an toàn (giữ nguyên logic cũ của bạn)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        max_tokens = min(8000, getattr(tokenizer, 'model_max_length', 8192))
        has_tokenizer = True
    except:
        tokenizer = None
        max_tokens = 8192
        has_tokenizer = False

    safe_documents = []
    for doc in documents:
        content = doc.page_content
        if not isinstance(content, str): continue
        if has_tokenizer:
            try:
                tokens = tokenizer.encode(content)
                if len(tokens) > max_tokens:
                    avg_chars_per_token = len(content) / len(tokens)
                    truncated = content[:int(max_tokens * avg_chars_per_token * 0.95)]
                    safe_documents.append(Document(page_content=truncated, metadata=doc.metadata))
                else:
                    safe_documents.append(doc)
            except:
                safe_documents.append(doc if len(content) <= 20000 else Document(page_content=content[:20000], metadata=doc.metadata))
        else:
            safe_documents.append(doc if len(content) <= 30000 else Document(page_content=content[:30000], metadata=doc.metadata))

    clean_docs = [doc for doc in safe_documents if isinstance(doc.page_content, str) and doc.page_content.strip()]
    if not clean_docs: return None

    # Tạo vectorstore theo batch
    batch_size = 50
    vectorstore = FAISS.from_documents(clean_docs[:batch_size], embeddings)
    for i in range(batch_size, len(clean_docs), batch_size):
        batch = clean_docs[i:i+batch_size]
        try:
            vectorstore.add_documents(batch)
        except:
            continue

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        try:
            vectorstore.save_local(os.path.join(cache_dir, 'faiss_index'))
        except:
            pass

    return vectorstore.as_retriever(search_kwargs={"k": k})

def load_llm(model_path: str) -> ChatLlamaCpp:
    """Load LLM with optimized settings"""
    # Tự động tính số threads tối ưu
    n_threads = os.cpu_count() - 2 if os.cpu_count() > 2 else 1
    
    llm = ChatLlamaCpp(
        model_path=model_path,
        n_ctx=4096,
        temperature=0.3,
        max_tokens=2048,
        n_batch=512, # Tăng batch size để xử lý nhanh hơn
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.3,
        verbose=False,
        streaming=True,
        use_mlock=True,
        use_mmap=True,
        f16_kv=True,
        n_threads=n_threads,
        n_gpu_layers=-1 if torch.cuda.is_available() else 0 # Đẩy full layers lên GPU nếu có
    )
    return llm

# --- 3. CHAINS SETUP (Fix Trimmer & Variable Mismatch) ---

contextualize_q_system_prompt = (
    "Dựa trên lịch sử hội thoại và câu hỏi mới nhất của người dùng, "
    "nếu câu hỏi có tham chiếu đến ngữ cảnh trong lịch sử, "
    "hãy viết lại thành một câu hỏi độc lập, đầy đủ ý nghĩa. "
    "Không trả lời câu hỏi, chỉ tái định dạng câu hỏi."
)

def create_qa_chain(retriever, llm, history_store) -> RunnableWithMessageHistory:
    # 1. Prompt Rephrase: Sử dụng 'trimmed_history'
    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("trimmed_history"), # Quan trọng: Dùng biến đã trim
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, rephrase_prompt
    )
    
    # 2. Prompt Trả lời RAG: Sử dụng 'trimmed_history'
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là trợ lý ảo hỗ trợ sinh viên dựa trên tài liệu được cung cấp.
        Dựa vào thông tin sau: {context}
        Trả lời câu hỏi ngắn gọn, chính xác và thân thiện.
        Nếu không tìm thấy thông tin trong ngữ cảnh, hãy nói: "Xin lỗi, tài liệu hiện tại không có thông tin về vấn đề này." """),
        MessagesPlaceholder("trimmed_history"), # Quan trọng: Dùng biến đã trim
        ("human", "{input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    # 3. Cấu hình Trimmer: Giữ lại khoảng 4-5 cặp câu hỏi gần nhất
    trimmer = trim_messages(
        max_tokens=1500, # Đủ cho khoảng 4-5 vòng hội thoại + context
        strategy="last",
        token_counter=llm,
        include_system=True,
        allow_partial=False
    )
    
    # 4. Pipeline xử lý: Lấy History gốc -> Trim -> Chạy RAG
    chain_with_trimming = (
        RunnablePassthrough.assign(trimmed_history=itemgetter("chat_history") | trimmer)
        | rag_chain
    )
    
    # 5. Kết nối với History Store bên ngoài
    return RunnableWithMessageHistory(
        runnable=chain_with_trimming,
        get_session_history=lambda session_id: history_store[session_id],
        input_messages_key="input",
        history_messages_key="chat_history", # Key gốc trong store
        output_messages_key="answer"
    )

# --- 4. MAIN CLASS (Đã bỏ Non-RAG & Fix History Sync) ---

class SemanticRAGChatbot:
    """RAG-only Chatbot for student handbook"""
    
    def __init__(
        self,
        json_path: str,
        llm_path: str,
        embedding_model_name: str = "dangvantuan/vietnamese-document-embedding",
        retriever_k: int = 5,
        cache_dir: str = "data/cache",
        force_rebuild: bool = False
    ):
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            
        # Store quản lý lịch sử tập trung
        self.history_store = {} 
        self.default_session_id = "user_session"
        self.history_store[self.default_session_id] = ChatMessageHistory()

        # Load Document logic (như cũ)
        cache_exists = (
            cache_dir and 
            os.path.exists(os.path.join(cache_dir, 'faiss_index')) and 
            os.path.exists(os.path.join(cache_dir, 'processed_docs.json'))
        )
        need_new_db = force_rebuild or not cache_exists
        
        if need_new_db:
            print("Loading and processing documents...")
            self.documents = load_json_data(json_path)
            self.documents = split_documents(self.documents)
            if cache_dir:
                serializable_docs = [
                    {'page_content': doc.page_content, 'metadata': doc.metadata}
                    for doc in self.documents
                ]
                with open(os.path.join(cache_dir, 'processed_docs.json'), 'w', encoding='utf-8') as f:
                    json.dump(serializable_docs, f, ensure_ascii=False, indent=2)
        else:
            print("Loading processed documents from cache...")
            with open(os.path.join(cache_dir, 'processed_docs.json'), 'r', encoding='utf-8') as f:
                cached_docs = json.load(f)
            self.documents = [
                Document(page_content=doc['page_content'], metadata=doc['metadata'])
                for doc in cached_docs
            ]
        
        print(f"Creating semantic vector retriever...")
        self.retriever = create_vector_retriever(
            self.documents, 
            embedding_model_name,
            k=retriever_k,
            cache_dir=cache_dir
        )
        
        print(f"Loading LLM from {llm_path}")
        self.llm = load_llm(llm_path)
        
        # Khởi tạo chuỗi RAG duy nhất
        self.qa_chain = create_qa_chain(self.retriever, self.llm, self.history_store)
        
        print("Semantic RAG chatbot (Single Mode) initialized successfully!")

    def answer_question(self, query: str) -> Dict[str, Any]:
        query = query.strip()
        if not query:
            return {"answer": "Bạn chưa nhập câu hỏi.", "intent": "none"}

        start_time = time.time()
        full_answer = ""
        
        # Gọi Chain với session_id cố định
        try:
            for chunk in self.qa_chain.stream(
                {"input": query},
                config={"configurable": {"session_id": self.default_session_id}}
            ):
                if isinstance(chunk, dict) and "answer" in chunk:
                    full_answer += chunk["answer"]
        except Exception as e:
            return {"answer": f"Đã xảy ra lỗi: {str(e)}", "intent": "error"}

        end_time = time.time()
        # print(f"Response time: {end_time - start_time:.2f}s")

        return {
            "answer": full_answer.strip(),
            "sources": ["Student Handbook"],
            "intent": "student_handbook"
        }

    def save_conversation_history(self, file_path: str):
        """Lưu lịch sử RAG ra file JSON"""
        history_obj = self.history_store[self.default_session_id]
        messages_dict = [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content} 
            for m in history_obj.messages
        ]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(messages_dict, f, ensure_ascii=False, indent=2)
        print(f"Conversation history saved to {file_path}")

    def load_conversation_history(self, file_path: str):
        """Load lịch sử từ JSON vào lại Memory của LangChain"""
        if not os.path.exists(file_path):
            print("History file not found.")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Xóa cũ và nạp mới
        history_obj = self.history_store[self.default_session_id]
        history_obj.clear()
        
        for msg in data:
            if msg['role'] == 'user':
                history_obj.add_message(HumanMessage(content=msg['content']))
            else:
                history_obj.add_message(AIMessage(content=msg['content']))
        
        print(f"Loaded {len(data)} messages into conversation history.")
