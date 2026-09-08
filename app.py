import streamlit as st
from datetime import datetime

from chat_storage import load_sessions, add_message
from rag.ingest import ingest_files
from agent.graph import graph


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    layout="wide",
    page_title="Agentic-Doc-Assistant"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def remove_thinking(text: str) -> str:
    """
    Remove <think>...</think> reasoning blocks from model output.

    This is useful for models such as Qwen that may return
    reasoning tags together with the final answer.
    """

    if not text:
        return ""

    cleaned = text

    # Remove complete thinking blocks
    while "<think>" in cleaned and "</think>" in cleaned:
        start = cleaned.find("<think>")
        end = cleaned.find("</think>") + len("</think>")

        cleaned = cleaned[:start] + cleaned[end:]

    # If an unfinished thinking block exists, remove everything
    # from <think> onward.
    if "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]

    # Remove any stray closing tag
    cleaned = cleaned.replace("</think>", "")

    return cleaned.strip()


# ============================================================
# PAGE TITLE
# ============================================================

st.markdown("# 🚀 Agentic-Doc-Assistant")
st.divider()


# ============================================================
# LOAD CHAT DATA
# ============================================================

sessions = load_sessions()

session_ids = sorted(
    list(sessions.keys()),
    reverse=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📁 Upload Document")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    # Dynamic button label
    btn_label = (
        "Upload Document"
        if uploaded_files
        else "Update Knowledge Base"
    )

    # ----------------------------
    # PDF INGESTION
    # ----------------------------

    if uploaded_files:

        if st.button(
            btn_label,
            use_container_width=True,
            type="secondary"
        ):

            with st.spinner("Processing documents..."):

                try:

                    ingest_files(uploaded_files)

                    st.toast(
                        "Documents processed successfully!",
                        icon="✅"
                    )

                except Exception as e:

                    st.error(
                        f"Document processing failed: {e}"
                    )

    st.divider()

    # ----------------------------
    # NEW CHAT
    # ----------------------------

    if st.button(
        "➕ New Chat",
        use_container_width=True,
        type="primary"
    ):

        st.session_state.session_picker = None

        st.session_state.current_session = (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

        st.session_state.messages = []

        st.rerun()

    # ----------------------------
    # CHAT HISTORY
    # ----------------------------

    st.subheader("📜 Recent Chats")

    selected_session = st.radio(
        "Select a session",
        session_ids,
        index=None,
        key="session_picker",
        label_visibility="collapsed"
    )


# ============================================================
# SESSION INITIALIZATION
# ============================================================

if selected_session:

    st.session_state.current_session = selected_session

    st.session_state.messages = sessions[selected_session]

elif "current_session" not in st.session_state:

    st.session_state.current_session = (
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    st.session_state.messages = []


# ============================================================
# DISPLAY EXISTING CHAT HISTORY
# ============================================================

chat_container = st.container()

with chat_container:

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):

            st.write(msg["content"])


# ============================================================
# USER INPUT
# ============================================================

if prompt := st.chat_input(
    "Ask your documents anything..."
):

    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    add_message(
        st.session_state.current_session,
        "user",
        prompt
    )

    # --------------------------------------------------------
    # DISPLAY USER MESSAGE
    # --------------------------------------------------------

    with chat_container:

        with st.chat_message("user"):

            st.write(prompt)

    # --------------------------------------------------------
    # ASSISTANT RESPONSE
    # --------------------------------------------------------

    with chat_container:

        with st.chat_message("assistant"):

            answer_placeholder = st.empty()

            # Complete raw response from the model
            full_answer = ""

            # ------------------------------------------------
            # RUN LANGGRAPH
            # ------------------------------------------------

            try:

                for chunk in graph.stream(
                    {"question": prompt},
                    stream_mode="updates"
                ):

                    for node_name, output in chunk.items():

                        # Make sure output is a dictionary
                        if not isinstance(output, dict):
                            continue

                        # We only care about nodes returning "answer"
                        if "answer" not in output:
                            continue

                        res = output["answer"]

                        # ------------------------------------
                        # INTERNAL RAG SIGNAL
                        # ------------------------------------

                        if res == "NOT_FOUND":
                            continue

                        # ------------------------------------
                        # STREAMING RESPONSE
                        # ------------------------------------

                        if (
                            hasattr(res, "__iter__")
                            and not isinstance(res, str)
                        ):

                            for token in res:

                                # LangChain message object
                                if hasattr(token, "content"):
                                    content = token.content

                                else:
                                    content = str(token)

                                if not content:
                                    continue

                                # Add raw model output
                                full_answer += content

                                # Clean thinking tags before
                                # showing anything to the user
                                visible_answer = remove_thinking(
                                    full_answer
                                )

                                # Only display visible answer
                                if visible_answer:

                                    answer_placeholder.markdown(
                                        visible_answer + "▌"
                                    )

                        # ------------------------------------
                        # NON-STREAMING RESPONSE
                        # ------------------------------------

                        else:

                            full_answer = str(res)

                            visible_answer = remove_thinking(
                                full_answer
                            )

                            answer_placeholder.markdown(
                                visible_answer
                            )

                # ------------------------------------------------
                # FINAL CLEAN RESPONSE
                # ------------------------------------------------

                clean_answer = remove_thinking(
                    full_answer
                )

                # If the model returned nothing visible
                if not clean_answer:

                    clean_answer = (
                        "I couldn't generate a response. "
                        "Please try again."
                    )

                # Display final answer without cursor
                answer_placeholder.markdown(
                    clean_answer
                )

                # ------------------------------------------------
                # SAVE CLEAN ANSWER
                # ------------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": clean_answer
                    }
                )

                add_message(
                    st.session_state.current_session,
                    "assistant",
                    clean_answer
                )

            # ------------------------------------------------
            # ERROR HANDLING
            # ------------------------------------------------

            except Exception as e:

                error_message = (
                    "⚠️ Something went wrong while generating "
                    "the response."
                )

                answer_placeholder.error(
                    error_message
                )

                print(
                    f"[ERROR] Agent execution failed: {e}"
                )