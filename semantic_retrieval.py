import json
import time
import os
from typing import List, Dict, Any
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
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnablePassthrough

# Load JSON data
def load_json_data(json_path: str) -> List[Document]:
    """Load JSON chứa các đoạn văn bản + metadata, chuyển thành list Document.
    Tích hợp heading_path vào content không hiển thị tags [HEADING] và [CONTENT].
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    for item in data:
        content = item['content']  # Văn bản chính
        metadata = item['metadata']  # Metadata bổ sung
        metadata['id'] = item['id']  # Gắn ID vào metadata
        
        # Tích hợp heading_path mà không hiển thị tags
        if metadata.get('content_type') == 'paragraph':
            heading = " > ".join(metadata['heading_path'][-2:])  # Lấy 2 cấp tiêu đề cuối
            page_content = f"{heading}\n{content}"
        else:
            page_content = content
        
        documents.append(Document(page_content=page_content, metadata=metadata))
    
    return documents


# Split documents
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

# Create vector retriever
def create_vector_retriever(documents: List[Document], model_name: str, k: int = 5, cache_dir: str = None) -> FAISS:
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu', 'trust_remote_code': True},
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
        if not isinstance(content, str):
            continue
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
    if not clean_docs:
        return None

    batch_size = 50
    vectorstore = FAISS.from_documents(clean_docs[:batch_size], embeddings)
    for i in range(batch_size, len(clean_docs), batch_size):
        batch = clean_docs[i:i+batch_size]
        try:
            vectorstore.add_documents(batch)
        except:
            for doc in batch:
                try:
                    vectorstore.add_documents([doc])
                except:
                    continue

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        try:
            vectorstore.save_local(os.path.join(cache_dir, 'faiss_index'))
        except:
            pass

    return vectorstore.as_retriever(search_kwargs={"k": k})


# Load LLM
def load_llm(model_path: str) -> ChatLlamaCpp:
    """Load a local LLM using ChatLlamaCpp"""
    llm = ChatLlamaCpp(
        model_path=model_path,
        n_ctx=4096,
        temperature=0.3,
        max_tokens=2048,
        n_batch=256,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.3,
        verbose=False,
        streaming=True,
        use_mlock=True,
        use_mmap=True,
        f16_kv=True,
        n_threads= os.cpu_count() or 4
    )
    return llm

contextualize_q_system_prompt = (
    "Dựa trên lịch sử hội thoại và câu hỏi mới nhất của người dùng, "
    "nếu câu hỏi có tham chiếu đến ngữ cảnh trong lịch sử, "
    "hãy tạo một câu hỏi độc lập, đầy đủ ý nghĩa mà không cần tham chiếu lịch sử. "
    "Không trả lời câu hỏi, chỉ tái định dạng nếu cần, nếu không thì giữ nguyên."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create QA Chain with history management
def create_qa_chain(retriever, llm) -> RunnableWithMessageHistory:
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    doc_prompt = ChatPromptTemplate.from_messages([
        ("system", """Bạn là chuyên gia tư vấn học tập cho sinh viên đại học.
        Dựa vào thông tin sau: {context}
        Trả lời câu hỏi ngắn gọn và chính xác. Nếu không tìm thấy thông tin, nói: 
        "Tôi không tìm thấy thông tin về vấn đề này trong cơ sở dữ liệu." """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    document_chain = create_stuff_documents_chain(llm, doc_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)
    
    trimmer = trim_messages(
        max_tokens=1000,
        strategy="last",
        token_counter=llm,
        include_system=True,
        allow_partial=False
    )
    
    trim_history = RunnablePassthrough.assign(history=trimmer)
    chain_with_trimming = trim_history | retrieval_chain
    
    history = ChatMessageHistory()
    conversational_chain = RunnableWithMessageHistory(
        runnable=chain_with_trimming,
        get_session_history=lambda session_id: history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    
    return conversational_chain

def create_academic_support_chain(llm) -> RunnableWithMessageHistory:
    """Create chain for academic support with contextualized question"""
    contextualize_chain = contextualize_q_prompt | llm
    
    academic_prompt = ChatPromptTemplate.from_messages([
    ("system", """Bạn là một chuyên gia tư vấn học tập cho sinh viên đại học, hỗ trợ về học thuật, kỹ năng học, và định hướng nghề nghiệp. Trả lời ngắn gọn, khích lệ, thân thiện, và sử dụng Chain of Thought (CoT) đơn giản để suy nghĩ từng bước, đảm bảo câu trả lời rõ ràng, logic.

    **Hướng dẫn chung**:
    1. **Dựa vào ngữ cảnh**: Xem lịch sử hội thoại để hiểu câu hỏi và trả lời phù hợp.
    2. **Áp dụng CoT đơn giản**:
    - Bước 1: Xác định loại câu hỏi (học thuật, kỹ năng học, định hướng nghề nghiệp, hay đối thoại thông thường).
    - Bước 2: Kiểm tra thông tin từ lịch sử hội thoại hoặc kiến thức chung để trả lời.
    - Bước 3: Đưa ra câu trả lời ngắn gọn, cụ thể, kèm lời động viên.
    3. **Trường hợp đặc biệt**:
    - Nếu không đủ thông tin: Nói: "Mình cần thêm chi tiết để trả lời tốt hơn. Bạn có thể nói rõ hơn không? Mình sẽ giúp ngay!"
    - Nếu câu hỏi mơ hồ: Hỏi lại để làm rõ (VD: "Bạn muốn hỏi về môn học cụ thể hay kỹ năng học chung?").
    - Nếu câu hỏi về xu hướng nghề nghiệp: Chỉ đưa ra gợi ý chung dựa trên kiến thức phổ biến và khuyến khích người dùng cung cấp thêm thông tin.

    **Xử lý phản hồi từ người dùng**:
    - **Nhận xét tích cực**: Cảm ơn và mời hỏi thêm (VD: "Cảm ơn bạn! Có gì cần hỗ trợ nữa không?").
    - **Nhận xét trung tính**: Xác nhận và khuyến khích tiếp tục (VD: "OK, bạn muốn tìm hiểu gì nữa?").
    - **Nhận xét tiêu cực**: Xin lỗi và đề nghị làm rõ (VD: "Mình xin lỗi nếu chưa rõ, bạn muốn mình giải thích thêm gì?").

    **Xử lý các loại câu hỏi**:
    - **Học thuật**: Đưa ra mẹo cụ thể (VD: cách ghi chú, lập kế hoạch học) và động viên.
    - **Kỹ năng học**: Gợi ý phương pháp đơn giản (VD: Pomodoro, tóm tắt ý chính) và khuyến khích thử.
    - **Định hướng nghề nghiệp**: Đưa ra bước hành động chung (VD: tìm hiểu ngành, làm CV) và động viên.
    - **Chào hỏi/đối thoại**: Trả lời thân thiện, mời hỏi thêm (VD: "Xin chào! Bạn cần giúp gì hôm nay?").

    **Ví dụ**:
    - Câu hỏi: "Làm sao để học tốt môn Toán?"
    - Bước 1: Câu hỏi học thuật về môn Toán.
    - Bước 2: Dựa trên kiến thức chung, gợi ý phương pháp học.
    - Bước 3: Trả lời: "Học Toán tốt, bạn thử làm bài tập từ dễ đến khó và ôn công thức mỗi ngày. Dành 20-30 phút luyện tập là tiến bộ ngay! Bạn đã thử cách nào chưa?"
    - Câu hỏi: "Ngành CNTT có triển vọng không?"
    - Bước 1: Câu hỏi định hướng nghề nghiệp.
    - Bước 2: Dựa trên kiến thức chung, CNTT thường có triển vọng.
    - Bước 3: Trả lời: "CNTT là ngành rất tiềm năng, đặc biệt ở lập trình và AI. Bạn có thể bắt đầu học kỹ năng cơ bản như code. Bạn muốn tìm hiểu thêm về lĩnh vực nào? Cố lên nhé!"

    Hãy giữ câu trả lời tích cực, thực tế, và khuyến khích người dùng tiến bộ!"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
    
    answer_chain = academic_prompt | llm
    
    full_chain = (
        RunnablePassthrough.assign(input=contextualize_chain) | answer_chain
    )
    
    history = ChatMessageHistory()
    conversational_chain = RunnableWithMessageHistory(
        runnable=full_chain,
        get_session_history=lambda session_id: history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    
    return conversational_chain

class SemanticRAGChatbot:
    """Semantic Retrieval-based chatbot for student handbook and academic support with separate histories for RAG and non-RAG modes"""
    
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
        self.force_rebuild = force_rebuild
        self.mode = None  # Lưu chế độ: 'rag' hoặc 'non_rag'
        self.rag_history = []  # Lịch sử cho chế độ RAG
        self.non_rag_history = []  # Lịch sử cho chế độ non-RAG

        # Load và xử lý tài liệu
        cache_exists = (
            cache_dir and 
            os.path.exists(os.path.join(cache_dir, 'faiss_index')) and 
            os.path.exists(os.path.join(cache_dir, 'processed_docs.json'))
        )
        need_new_db = force_rebuild or not cache_exists
        
        if need_new_db:
            print("Loading documents...")
            self.documents = load_json_data(json_path)
            print("Splitting documents...")
            self.documents = split_documents(self.documents)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                serializable_docs = [
                    {'page_content': doc.page_content, 'metadata': doc.metadata}
                    for doc in self.documents
                ]
                with open(os.path.join(cache_dir, 'processed_docs.json'), 'w', encoding='utf-8') as f:
                    json.dump(serializable_docs, f, ensure_ascii=False, indent=2)
        else:
            print("Loading processed documents from cache...")
            try:
                with open(os.path.join(cache_dir, 'processed_docs.json'), 'r', encoding='utf-8') as f:
                    cached_docs = json.load(f)
                self.documents = [
                    Document(page_content=doc['page_content'], metadata=doc['metadata'])
                    for doc in cached_docs
                ]
            except FileNotFoundError:
                print("Cache file not found, rebuilding documents...")
                self.documents = load_json_data(json_path)
                self.documents = split_documents(self.documents)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                    serializable_docs = [
                        {'page_content': doc.page_content, 'metadata': doc.metadata}
                        for doc in self.documents
                    ]
                    with open(os.path.join(cache_dir, 'processed_docs.json'), 'w', encoding='utf-8') as f:
                        json.dump(serializable_docs, f, ensure_ascii=False, indent=2)
        
        print(f"Creating semantic vector retriever")
        self.retriever = create_vector_retriever(
            self.documents, 
            embedding_model_name,
            k=retriever_k,
            cache_dir=cache_dir
        )
        
        print(f"Loading LLM from {llm_path}")
        self.llm = load_llm(llm_path)
        
        # Khởi tạo chuỗi cho student_handbook (RAG)
        self.qa_chain = create_qa_chain(self.retriever, self.llm)
        # Khởi tạo chuỗi cho academic_support (non-RAG)
        self.academic_support_chain = create_academic_support_chain(self.llm)
        
        print("Semantic RAG chatbot initialized successfully!")

    def answer_question(self, query: str) -> Dict[str, Any]:
        query = query.strip()
        if not query:
            return {"answer": "Bạn chưa nhập câu hỏi, hãy thử lại nhé!", "intent": "none"}

        start_time = time.time()
        full_answer = ""
        sources = []

        if self.mode == "rag":
            for chunk in self.qa_chain.stream(
                {"input": query},
                config={"configurable": {"session_id": "default"}}
            ):
                if isinstance(chunk, dict) and "answer" in chunk:
                    full_answer += chunk["answer"]
                    # print(chunk["answer"], end="", flush=True)
                else:
                    continue
            sources.append("Student Handbook")
            intent = "student_handbook"
            # Thêm vào lịch sử RAG
            self.rag_history.append({"role": "user", "content": query})
            self.rag_history.append({"role": "assistant", "content": full_answer})
        else:  # non_rag
            for chunk in self.academic_support_chain.stream(
                {"input": query},
                config={"configurable": {"session_id": "default"}}
            ):
                if hasattr(chunk, 'content') and chunk.content:
                    full_answer += chunk.content
                    # print(chunk.content, end="", flush=True)
                elif isinstance(chunk, dict) and "content" in chunk and chunk["content"]:
                    full_answer += chunk["content"]
                    # print(chunk["content"], end="", flush=True)
                elif isinstance(chunk, str) and chunk:
                    full_answer += chunk
                    # print(chunk, end="", flush=True)
                else:
                    continue
            sources.append("LLM Internal Knowledge")
            intent = "academic_support"
            # Thêm vào lịch sử non-RAG
            self.non_rag_history.append({"role": "user", "content": query})
            self.non_rag_history.append({"role": "assistant", "content": full_answer})

        end_time = time.time()
        response_time = end_time - start_time
        # print(f"\nThời gian trả lời: {response_time:.2f} giây")

        return {
            "answer": full_answer.strip(),
            "sources": sources,
            "intent": intent
        }

    def save_conversation_history(self, rag_file_path: str, non_rag_file_path: str):
        """Lưu lịch sử hội thoại riêng cho RAG và non-RAG"""
        # Lưu lịch sử RAG
        with open(rag_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.rag_history, f, ensure_ascii=False, indent=2)
        print(f"RAG conversation history saved to {rag_file_path}")
        
        # Lưu lịch sử non-RAG
        with open(non_rag_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.non_rag_history, f, ensure_ascii=False, indent=2)
        print(f"Non-RAG conversation history saved to {non_rag_file_path}")

    def load_conversation_history(self, rag_file_path: str, non_rag_file_path: str):
        """Tải lịch sử hội thoại riêng cho RAG và non-RAG"""
        # Tải lịch sử RAG
        if os.path.exists(rag_file_path):
            with open(rag_file_path, 'r', encoding='utf-8') as f:
                self.rag_history = json.load(f)
            print(f"RAG conversation history loaded from {rag_file_path}")
        else:
            print(f"No RAG conversation history found at {rag_file_path}")
        
        # Tải lịch sử non-RAG
        if os.path.exists(non_rag_file_path):
            with open(non_rag_file_path, 'r', encoding='utf-8') as f:
                self.non_rag_history = json.load(f)
            print(f"Non-RAG conversation history loaded from {non_rag_file_path}")
        else:
            print(f"No non-RAG conversation history found at {non_rag_file_path}")