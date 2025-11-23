import streamlit as st
import requests
import time
from datetime import datetime

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="💬 SSA Chatbot", layout="wide", initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Chat container */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Header */
    .header-container {
        background: white;
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        text-align: center;
    }

    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }

    .header-subtitle {
        color: #718096;
        font-size: 1.1rem;
    }

    /* Source cards */
    .source-card {
        background: #f7fafc;
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

API_URL = "http://127.0.0.1:8000/chat"

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("# ⚙️ Settings")

    # RAG Parameters only
    st.markdown("### 🔎 RAG Parameters")
    top_k = st.slider(
        "Top K Results", 1, 10, 3, help="Number of relevant documents to retrieve"
    )
    score_threshold = st.slider(
        "Similarity Threshold",
        0.0,
        1.0,
        0.1,
        0.05,
        help="Minimum similarity score for retrieval",
    )

    st.divider()


# ---------------- HEADER ----------------
st.markdown(
    """
<div class="header-container">
    <div class="header-title">💬 SSA Chatbot</div>
    <div class="header-subtitle">
        Ask me anything about Social Security Administration or SSN documents
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------- QUICK QUESTIONS ----------------
st.markdown("### 💡 Quick Questions")
col1, col2, col3, col4 = st.columns(4)

quick_questions = [
    "How to apply for SSN?",
    "What are SSA benefits?",
    "Disability requirements",
    "Retirement age info",
]

# When a quick question is clicked, trigger a message directly
selected_question = None
for idx, (col, question) in enumerate(zip([col1, col2, col3, col4], quick_questions)):
    with col:
        if st.button(question, key=f"quick_{idx}", use_container_width=True):
            selected_question = question
            st.toast(f"💡 You asked: {question}")  # Small visual feedback

# If a quick question is selected, treat it like user input
if selected_question:
    user_input = selected_question
else:
    user_input = st.chat_input("Let's chat...", key="chat_input")

# ---------------- DISPLAY CHAT HISTORY ----------------
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.info(
            "👋 Welcome! Start by asking a question about SSA or use the quick questions above."
        )

    for idx, msg in enumerate(st.session_state.messages):
        role = msg["role"]
        with st.chat_message(role, avatar="🤖" if role == "assistant" else "👤"):
            st.markdown(msg["content"])
            if role == "assistant" and "sources" in msg and msg["sources"]:
                with st.expander(f"📚 View {len(msg['sources'])} Source(s)"):
                    for i, s in enumerate(msg["sources"], 1):
                        st.markdown(
                            f"""
                        <div class="source-card">
                            <strong>Source {i}:</strong> {s.get('source', 'Unknown')}<br>
                            <strong>Relevance Score:</strong> {s.get('score', 0):.3f}<br>
                            <strong>Preview:</strong> {s.get('preview', 'N/A')}
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

# ---------------- CHAT INPUT ----------------
if user_input:
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.total_queries += 1

    # Query backend
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤔 Thinking...")

        try:
            start_time = time.time()
            res = requests.post(
                API_URL,
                json={
                    "query": user_input,
                    "top_k": top_k,
                    "score_threshold": score_threshold,
                },
                timeout=180,
            )
            response_time = time.time() - start_time

            data = res.json()
            answer = data.get("answer", "No answer found.")
            sources = data.get("sources", [])

            message_placeholder.markdown(answer)
            st.caption(f"⏱️ Response time: {response_time:.2f}s")

            # Show retrieved sources
            if sources:
                with st.expander(f"📚 View {len(sources)} Source(s)"):
                    for i, s in enumerate(sources, 1):
                        st.markdown(
                            f"""
                        <div class="source-card">
                            <strong>Source {i}:</strong> {s.get('source', 'Unknown')}<br>
                            <strong>Relevance Score:</strong> {s.get('score', 0):.3f}<br>
                            <strong>Preview:</strong> {s.get('preview', 'N/A')}
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

        except requests.exceptions.Timeout:
            message_placeholder.error("⏰ Request timed out. Please try again.")
            answer, sources = "Timeout error", []
        except requests.exceptions.ConnectionError:
            message_placeholder.error(
                "❌ Cannot connect to backend. Please ensure the API is running on port 8000."
            )
            answer, sources = "Connection error", []
        except Exception as e:
            message_placeholder.error(f"⚠️ Error: {str(e)}")
            answer, sources = f"Error: {str(e)}", []

    # Save response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
