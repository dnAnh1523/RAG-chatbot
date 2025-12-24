import streamlit as st
import os
import time
from semantic_retrieval import SemanticRAGChatbot

# Thiết lập page config
st.set_page_config(page_title="Trợ lý Sổ tay Sinh viên", page_icon="🎓")

# Khởi tạo session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Mình là trợ lý ảo hỗ trợ tra cứu Sổ tay Sinh viên. Bạn cần tìm thông tin gì hôm nay?"}
    ]
if 'history_path' not in st.session_state:
    st.session_state.history_path = "data/cache/chat_history.json"

# Hàm khởi tạo chatbot
def initialize_chatbot():
    json_path = "data/JSON/SO_TAY_SINH_VIEN.json"
    llm_path = "models/meta/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    embedding_model = "dangvantuan/vietnamese-document-embedding"
    cache_dir = "data/cache"

    # Container loading
    with st.status("Đang khởi động hệ thống...", expanded=True) as status:
        st.write("🔄 Đang tải mô hình ngôn ngữ và dữ liệu...")
        
        # Khởi tạo chatbot
        chatbot = SemanticRAGChatbot(
            json_path=json_path,
            llm_path=llm_path,
            embedding_model_name=embedding_model,
            retriever_k=5,
            cache_dir=cache_dir,
            force_rebuild=False
        )
        
        st.write("📂 Đang khôi phục lịch sử hội thoại...")
        chatbot.load_conversation_history(st.session_state.history_path)
        
        st.session_state.chatbot = chatbot
        status.update(label="Sẵn sàng!", state="complete", expanded=False)

# Hàm xóa lịch sử
def reset_conversation():
    st.session_state.messages = [
        {"role": "assistant", "content": "Xin chào! Mình là trợ lý ảo hỗ trợ tra cứu Sổ tay Sinh viên. Bạn cần tìm thông tin gì hôm nay?"}
    ]
    if st.session_state.chatbot:
        # Xóa memory trong LangChain
        st.session_state.chatbot.history_store[st.session_state.chatbot.default_session_id].clear()
    st.toast("Đã bắt đầu cuộc trò chuyện mới!", icon="🧹")

# --- UI CHÍNH ---

st.title("🎓 Trợ lý Sổ tay Sinh viên")
st.caption("Hỏi đáp thông tin quy chế, đào tạo và công tác sinh viên dựa trên tài liệu chính thức.")

# Khởi tạo chatbot nếu chưa có
if st.session_state.chatbot is None:
    initialize_chatbot()

# Sidebar công cụ
with st.sidebar:
    st.header("⚙️ Công cụ")
    
    if st.button("Lưu lịch sử hội thoại", use_container_width=True):
        if st.session_state.chatbot:
            st.session_state.chatbot.save_conversation_history(st.session_state.history_path)
            st.success("Đã lưu lịch sử thành công!")

    if st.button("Xóa hội thoại", type="primary", use_container_width=True):
        reset_conversation()
        st.rerun()

    st.divider()
    st.info("💡 **Mẹo:** Bạn có thể hỏi về quy chế thi, học bổng, điểm rèn luyện, hoặc các thủ tục hành chính...")

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Xử lý input người dùng
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    # 1. Hiển thị câu hỏi người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 2. Xử lý trả lời
    if st.session_state.chatbot:
        start_time = time.time()
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Status indicator
            with st.spinner("Đang tra cứu tài liệu..."):
                try:
                    # Gọi stream từ Chain
                    # Quan trọng: Dùng đúng session_id mặc định của chatbot class
                    stream_generator = st.session_state.chatbot.qa_chain.stream(
                        {"input": prompt},
                        config={"configurable": {"session_id": st.session_state.chatbot.default_session_id}}
                    )
                    
                    for chunk in stream_generator:
                        if isinstance(chunk, dict) and "answer" in chunk:
                            token = chunk["answer"]
                            full_response += token
                            message_placeholder.markdown(full_response + "▌")
                            
                    # Hoàn tất
                    message_placeholder.markdown(full_response)
                    
                except Exception as e:
                    st.error(f"Lỗi hệ thống: {str(e)}")
                    full_response = "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau."
            
            # Hiển thị thời gian phản hồi
            process_time = time.time() - start_time
            st.caption(f"⏱️ Phản hồi trong {process_time:.2f}s")

        # 3. Lưu câu trả lời vào session state UI
        st.session_state.messages.append({"role": "assistant", "content": full_response})
