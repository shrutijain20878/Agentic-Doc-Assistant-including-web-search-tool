import streamlit as st
from datetime import datetime
from chat_storage import load_sessions, add_message
from rag.ingest import ingest_files
from agent.graph import graph

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Agentic-Doc-Assistant")

# Ensure the Title is visible and properly formatted
st.markdown("# 🚀 Agentic-Doc-Assistant")
st.divider()

# --- 1. Load Data ---
sessions = load_sessions()
session_ids = sorted(list(sessions.keys()), reverse=True)

# --- 2. Sidebar Layout ---
with st.sidebar:
    st.header("📁 Upload Document")
    
    # Requirement: Upload PDF
    uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
    
    # Requirement: Change button text dynamically
    btn_label = "Upload Document" if uploaded_files else "Update Knowledge Base"
    
    if uploaded_files:
        if st.button(btn_label, use_container_width=True, type="secondary"):
            with st.spinner("Ingesting..."):
                ingest_files(uploaded_files)
                st.toast("Documents processed!", icon="✅")

    st.divider()
    
    # Requirement: New Chat Button ABOVE history
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.session_picker = None 
        st.session_state.current_session = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.messages = []
        st.rerun()
        
    st.subheader("📜 Recent Chats")
    selected_session = st.radio(
        "Select a session",
        session_ids,
        index=None,
        key="session_picker",
        label_visibility="collapsed"
    )

# --- 3. Session Initialization Logic ---
if selected_session:
    st.session_state.current_session = selected_session
    st.session_state.messages = sessions[selected_session]
elif "current_session" not in st.session_state:
    st.session_state.current_session = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.messages = []

# --- 4. Display Chat History ---
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# --- 5. User Input & Streaming ---
if prompt := st.chat_input("Ask your documents anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    add_message(st.session_state.current_session, "user", prompt)

    with chat_container:
        with st.chat_message("user"):
            st.write(prompt)

    with chat_container:
        with st.chat_message("assistant"):
            answer_placeholder = st.empty() 
            full_answer = ""

            # Run the graph
            for chunk in graph.stream({"question": prompt}, stream_mode="updates"):
                for node_name, output in chunk.items():
                    if "answer" in output:
                        res = output["answer"]
                        # --- THE FIX ---
                        # If the output is our internal signal, skip displaying it
                        if res == "NOT_FOUND":
                            continue
                        # Handle Streaming (Generators)
                        if hasattr(res, "__iter__") and not isinstance(res, str):
                            for token in res:
                                # Ollama/LangChain 2026 tokens often have a .content attribute
                                content = token.content if hasattr(token, 'content') else str(token)
                                full_answer += content
                                answer_placeholder.markdown(full_answer + "▌")
                        else:
                            full_answer = str(res)
                            answer_placeholder.markdown(full_answer)

            answer_placeholder.markdown(full_answer)
            st.session_state.messages.append({"role": "assistant", "content": full_answer})
            add_message(st.session_state.current_session, "assistant", full_answer)