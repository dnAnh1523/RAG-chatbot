import streamlit as st
import os
from semantic_retrieval import SemanticRAGChatbot
import time
import re

# Hàm stream text được cải tiến để bảo toàn format markdown
def stream_text_improved(text: str, delay: float = 0.02):
    """Stream text theo cách bảo toàn hoàn toàn định dạng markdown"""
    # Chia theo các đơn vị logic: đoạn văn, dòng, câu
    # Bảo toàn các ký tự đặc biệt markdown như -, *, #, etc.
    
    # Pattern để chia văn bản thành các phần có ý nghĩa
    patterns = [
        r'(\n\n)',  # Đoạn văn mới
        r'(\n(?=[-*•]\s))',  # Dòng bắt đầu bằng bullet point
        r'(\n(?=\d+\.\s))',  # Dòng bắt đầu bằng số
        r'(\n(?=\*\s))',  # Dòng bắt đầu bằng *
        r'([.!?]\s+)',  # Kết thúc câu
        r'([,;:]\s+)',  # Dấu phẩy, chấm phẩy
    ]
    
    # Chia văn bản thành các chunk
    chunks = [text]
    for pattern in patterns:
        new_chunks = []
        for chunk in chunks:
            new_chunks.extend(re.split(pattern, chunk))
        chunks = [c for c in new_chunks if c and c.strip()]
    
    # Stream từng chunk
    for chunk in chunks:
        if chunk.strip():
            yield chunk
            time.sleep(delay)

# Hàm stream đơn giản hơn - chia theo từ nhưng bảo toàn markdown
def stream_text_simple(text: str, delay: float = 0.03):
    """Stream text theo từ nhưng bảo toàn các ký tự markdown đặc biệt"""
    # Tách các dòng trước
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        if line.strip():
            # Nếu dòng bắt đầu bằng ký tự đặc biệt markdown, giữ nguyên
            if line.strip().startswith(('-', '*', '•', '+')):
                yield line
                if i < len(lines) - 1:  # Không phải dòng cuối
                    yield '\n'
            else:
                # Chia thành từ cho dòng bình thường
                words = line.split()
                for j, word in enumerate(words):
                    yield word
                    if j < len(words) - 1:  # Không phải từ cuối
                        yield ' '
                if i < len(lines) - 1:  # Không phải dòng cuối
                    yield '\n'
        else:
            # Dòng trống
            yield '\n'
        
        time.sleep(delay)

# Khởi tạo session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'rag_display_history' not in st.session_state:
    st.session_state.rag_display_history = [{"role": "assistant", "content": "Hỏi tôi về thông tin trong sổ tay sinh viên nhé!"}]
if 'non_rag_display_history' not in st.session_state:
    st.session_state.non_rag_display_history = [{"role": "assistant", "content": "Hỏi tôi về học tập hoặc cần tư vấn gì nhé!"}]
if 'mode' not in st.session_state:
    st.session_state.mode = "rag"  # Mặc định là RAG
if 'history_paths' not in st.session_state:
    st.session_state.history_paths = {
        'rag': "data/cache/rag_history.json",
        'non_rag': "data/cache/non_rag_history.json"
    }

# Hàm khởi tạo chatbot với thanh tiến trình
def initialize_chatbot():
    json_path = "data/JSON/SO_TAY_SINH_VIEN.json"
    llm_path = "models/meta/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    embedding_model = "dangvantuan/vietnamese-document-embedding"
    cache_dir = "data/cache"

    # Tạo container cho thanh tiến trình và thông báo
    progress_container = st.empty()
    status_text = st.empty()
    
    with progress_container:
        progress_bar = st.progress(0)
        
    status_text.write("Đang khởi tạo...")

    # Giả lập các bước khởi tạo
    progress_bar.progress(10)  # Bắt đầu
    time.sleep(0.1)  # Độ trễ giả lập

    # Khởi tạo chatbot
    chatbot = SemanticRAGChatbot(
        json_path=json_path,
        llm_path=llm_path,
        embedding_model_name=embedding_model,
        retriever_k=5,
        cache_dir=cache_dir,
        force_rebuild=False
    )
    progress_bar.progress(50)  # Hoàn thành khởi tạo chatbot
    time.sleep(0.1)

    chatbot.mode = st.session_state.mode
    st.session_state.chatbot = chatbot
    
    progress_bar.progress(80)  # Chuẩn bị tải lịch sử
    time.sleep(0.1)

    # Tải lịch sử hội thoại để contextualize
    chatbot.load_conversation_history(
        st.session_state.history_paths['rag'],
        st.session_state.history_paths['non_rag']
    )
    
    # Khi đạt 100%, đổi thông báo thành "Khởi tạo hoàn tất"
    progress_bar.progress(100)
    status_text.write("Khởi tạo hoàn tất")
    
    # Đợi 1 giây sau đó xóa cả progress bar và thông báo
    time.sleep(1)
    progress_container.empty()
    status_text.empty()

# Hàm đặt lại lịch sử hiển thị
def reset_display_history():
    if st.session_state.mode == "rag":
        st.session_state.rag_display_history = [{"role": "assistant", "content": "Hỏi tôi về thông tin trong sổ tay sinh viên nhé!"}]
    else:
        st.session_state.non_rag_display_history = [{"role": "assistant", "content": "Hỏi tôi về học tập hoặc cần tư vấn gì nhé!"}]
    st.success("Đã bắt đầu cuộc trò chuyện mới!")

# Tiêu đề ứng dụng
st.title("🤖 RAG-based chatbot")
st.caption("Hoạt động với hai chế độ: tra cứu sổ tay sinh viên (RAG) và tư vấn học tập (non-RAG)")

# Khởi tạo chatbot nếu chưa có
if st.session_state.chatbot is None:
    initialize_chatbot()

# Thanh bên để chọn chế độ và quản lý lịch sử
with st.sidebar:
    st.header("Cài đặt")
    mode_options = ["RAG (Sổ tay sinh viên)", "Non-RAG (Tư vấn học tập)"]
    selected_mode = st.selectbox(
        "Chọn chế độ trò chuyện:",
        mode_options,
        index=0 if st.session_state.mode == "rag" else 1
    )
    
    # Cập nhật chế độ
    new_mode = "rag" if selected_mode == "RAG (Sổ tay sinh viên)" else "non_rag"
    if new_mode != st.session_state.mode:
        st.session_state.mode = new_mode
        st.session_state.chatbot.mode = new_mode
        st.write(f"Đã chuyển sang chế độ {selected_mode}.")

    # Nút cuộc trò chuyện mới
    if st.button("Cuộc trò chuyện mới"):
        reset_display_history()

    # Nút lưu lịch sử
    if st.button("Lưu lịch sử hội thoại"):
        st.session_state.chatbot.rag_history = st.session_state.rag_display_history
        st.session_state.chatbot.non_rag_history = st.session_state.non_rag_display_history
        st.session_state.chatbot.save_conversation_history(
            st.session_state.history_paths['rag'],
            st.session_state.history_paths['non_rag']
        )
        st.success("Đã lưu lịch sử hội thoại!")

# Hiển thị lịch sử trò chuyện
history = st.session_state.rag_display_history if st.session_state.mode == "rag" else st.session_state.non_rag_display_history
for msg in history:
    st.chat_message(msg["role"]).write(msg["content"])

# Ô nhập câu hỏi
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    start_time = time.time()
    
    st.chat_message("user").write(prompt)
    
    # Stream response với hiệu ứng real-time
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        stream_started = False
        stream_completed = False
        
        # Khởi tạo status sau khi đã setup chat message
        with st.status("Đang tìm thông tin...", expanded=False) as status:
            try:
                # Thêm delay để người dùng thấy được status đầu tiên
                time.sleep(0.5)
                status.update(label="Đang tạo phản hồi...", state="running")
                
                # Tạo stream generator
                if st.session_state.mode == "rag":
                    stream_generator = st.session_state.chatbot.qa_chain.stream(
                        {"input": prompt},
                        config={"configurable": {"session_id": "default"}}
                    )
                else:  # non_rag mode
                    stream_generator = st.session_state.chatbot.academic_support_chain.stream(
                        {"input": prompt},
                        config={"configurable": {"session_id": "default"}}
                    )
                
                # Thêm delay trước khi bắt đầu stream
                time.sleep(0.75)
                status.update(label="Đang phản hồi...", state="running")
                
                # Xử lý stream chunks và hiển thị real-time
                chunk_count = 0
                for chunk in stream_generator:
                    chunk_count += 1
                    if not stream_started:
                        stream_started = True
                        # Chỉ cập nhật status nếu thực sự có content
                        if chunk_count == 1:
                            status.update(label="Đang tạo phản hồi...", state="running")
                    
                    chunk_text = ""
                    
                    if st.session_state.mode == "rag":
                        # Xử lý chunk cho RAG mode
                        if isinstance(chunk, dict) and "answer" in chunk:
                            chunk_text = chunk["answer"]
                    else:
                        # Xử lý chunk cho non-RAG mode
                        if hasattr(chunk, 'content') and chunk.content:
                            chunk_text = chunk.content
                        elif isinstance(chunk, dict) and "content" in chunk and chunk["content"]:
                            chunk_text = chunk["content"]
                        elif isinstance(chunk, str) and chunk:
                            chunk_text = chunk
                    
                    # Cập nhật response và hiển thị ngay lập tức
                    if chunk_text:
                        full_response += chunk_text
                        # Hiển thị với cursor để tạo hiệu ứng typing
                        message_placeholder.markdown(full_response)
                        time.sleep(0.01)  # Delay nhỏ để tạo hiệu ứng mượt
                
                # Đánh dấu stream đã hoàn thành
                stream_completed = True
                
                # Xóa cursor và hiển thị kết quả cuối cùng
                message_placeholder.markdown(full_response + "|")
                
                # Chỉ cập nhật status thành complete khi stream thực sự hoàn thành
                if stream_completed and full_response.strip():
                    status.update(label="Hoàn tất", state="complete")
                else:
                    status.update(label="Không có phản hồi", state="error")
                
            except Exception as e:
                message_placeholder.error(f"Có lỗi xảy ra: {str(e)}")
                full_response = "Xin lỗi, có lỗi xảy ra khi xử lý câu hỏi của bạn."
                status.update(label="Lỗi xử lý", state="error")
        
        end_time = time.time()
        total_time = end_time - start_time
        st.caption(f"{total_time:.2f}s")
    
    # Cập nhật lịch sử chỉ khi có phản hồi hợp lệ
    if full_response.strip():
        if st.session_state.mode == "rag":
            st.session_state.rag_display_history.append({"role": "user", "content": prompt})
            st.session_state.rag_display_history.append({"role": "assistant", "content": full_response})
            st.session_state.chatbot.rag_history.append({"role": "user", "content": prompt})
            st.session_state.chatbot.rag_history.append({"role": "assistant", "content": full_response})
        else:
            st.session_state.non_rag_display_history.append({"role": "user", "content": prompt})
            st.session_state.non_rag_display_history.append({"role": "assistant", "content": full_response})
            st.session_state.chatbot.non_rag_history.append({"role": "user", "content": prompt})
            st.session_state.chatbot.non_rag_history.append({"role": "assistant", "content": full_response})