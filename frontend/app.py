import streamlit as st
import requests
import time
<<<<<<< HEAD
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
=======

# Config
st.set_page_config(
    page_title="GovAssist AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);}
    
>>>>>>> d3f36b6 (update file)
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
<<<<<<< HEAD

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
=======
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
    }
    
    [data-testid="stSidebar"] * {color: white !important;}
    
>>>>>>> d3f36b6 (update file)
    .source-card {
        background: #f7fafc;
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
<<<<<<< HEAD
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
=======
    
    .source-card-web {
        background: #fff5f5;
        border-left: 4px solid #f56565;
    }
    
    .category-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
    
    .category-ssa {background: #0066CC; color: white;}
    .category-uscis {background: #006633; color: white;}
    .category-travel_state {background: #CC0000; color: white;}
    
    .header-container {
        background: white;
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8000/chat"
HEALTH_URL = "http://localhost:8000/health"

def check_backend():
    try:
        r = requests.get(HEALTH_URL, timeout=3)
        return r.status_code == 200, r.json() if r.status_code == 200 else None
    except:
        return False, None

def get_source_type_badge(source_type: str) -> str:
    if source_type == "web":
        return '<span class="category-badge" style="background: #f56565;">🌐 Web Search</span>'
    else:
        return '<span class="category-badge" style="background: #48bb78;">📚 Knowledge Base</span>'

def get_category_badge(category: str) -> str:
    category_names = {
        "ssa": "Social Security",
        "uscis": "Immigration", 
        "travel_state": "Travel & Passports",
        "general": "General"
    }
    name = category_names.get(category, category)
    return f'<span class="category-badge category-{category}">{name}</span>'

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "queries" not in st.session_state:
    st.session_state.queries = 0

connected, health = check_backend()

# Sidebar
with st.sidebar:
    st.markdown("# ⚙️ Settings")
    
    # Backend status
    st.markdown("### Backend Status")
    if connected:
        st.success("✅ Connected")
        if health:
            st.metric("📚 Documents", health.get("documents", 0))
            
            # Show web search status
            web_search = health.get("web_search", False)
            if web_search:
                st.success("🌐 Web Search: Enabled")
            else:
                st.warning("⚠️ Web Search: Disabled")
            
            # Show config
            config = health.get("config", {})
            if config:
                st.info(f"🎯 Default Threshold: {config.get('score_threshold', 'N/A')}")
                st.info(f"🔍 Embedding: {config.get('embedding_model', 'N/A')}")
    else:
        st.error("❌ Disconnected")
        st.info("Start backend:\n```bash\ncd backend\nuvicorn app:app --reload\n```")
    
    st.divider()
    
    # RAG Parameters - THESE NOW WORK!
    st.markdown("### 🔍 Search Settings")
    st.caption("Adjust these to tune retrieval quality")
    
    top_k = st.slider(
        "Results to retrieve", 
        min_value=1, 
        max_value=10, 
        value=5,
        help="Number of document chunks to retrieve. More chunks = more context but slower."
    )
    
    score_threshold = st.slider(
        "Similarity threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.4,  # Default to 0.4 as requested
        step=0.05,
        help="Minimum similarity score (0.4 recommended). Higher = stricter filtering."
    )
    
    enable_web_search = st.checkbox(
        "Enable Web Search",
        value=True,
        help="Allow web search for current information when RAG confidence is low"
    )
    
    st.divider()
    
    # Actions
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.queries = 0
        st.rerun()
    
    st.divider()
    
    # Stats
    st.markdown("### 📊 Session Stats")
    st.metric("Total Queries", st.session_state.queries)
    
    if st.session_state.messages:
        web_searches = sum(1 for msg in st.session_state.messages 
                          if msg.get("role") == "assistant" and msg.get("used_web_search"))
        st.metric("Web Searches Used", web_searches)
        
        # Average response time
        times = [msg.get("time", 0) for msg in st.session_state.messages if msg.get("role") == "assistant"]
        if times:
            avg_time = sum(times) / len(times)
            st.metric("Avg Response Time", f"{avg_time:.2f}s")

# Header
st.markdown("""
<div class="header-container">
    <div class="header-title">GovAssist AI</div>
    <p style="color: #718096; font-size: 1.1rem;">
        Your Intelligent Guide to U.S. Government Services
    </p>
    <p style="color: #a0aec0; font-size: 0.9rem;">
        Social Security • Immigration • Travel & Passports • Web Search Enabled
    </p>
</div>
""", unsafe_allow_html=True)

if not connected:
    st.warning("⚠️ Backend not running. Start it with: `cd backend && uvicorn app:app --reload`")
    st.stop()

# Quick Questions - Simplified without category filter
st.markdown("### 💡 Quick Questions")

quick_questions = [
    "Hi! How can you help me?",
    "How do I apply for Social Security?",
    "What are the latest USCIS updates?",
    "How do I renew my passport?",
]

cols = st.columns(4)

for idx, (col, question) in enumerate(zip(cols, quick_questions)):
    with col:
        if st.button(question, key=f"quick_{idx}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.queries += 1
            
            try:
                start = time.time()
                
                # Build conversation history (last 3 Q&A pairs = 6 messages)
                history = []
                if len(st.session_state.messages) > 0:
                    # Get last 6 messages (3 Q&A pairs)
                    recent_messages = st.session_state.messages[-6:]
                    for msg in recent_messages:
                        history.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                
                res = requests.post(
                    API_URL,
                    json={
                        "query": question, 
                        "top_k": top_k, 
                        "score_threshold": score_threshold,
                        "enable_web_search": enable_web_search,
                        "conversation_history": history  # Last 3 Q&A pairs
                    },
                    timeout=180
                )
                elapsed = time.time() - start
                
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data.get("answer", "No answer"),
                        "sources": data.get("sources", []),
                        "time": elapsed,
                        "used_web_search": data.get("used_web_search", False)
                    })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Error: {e}",
                    "sources": []
                })
            
            st.rerun()

st.divider()

# Chat history
if not st.session_state.messages:
    st.info("👋 Welcome! Ask a question about U.S. government services or just say hi!")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant":
            # Show timing and web search indicator
            caption_parts = []
            if "time" in msg:
                caption_parts.append(f"⏱️ {msg['time']:.2f}s")
            if msg.get("used_web_search"):
                caption_parts.append("🌐 Web Search Used")
            
            if caption_parts:
                st.caption(" • ".join(caption_parts))
            
            if "sources" in msg and msg["sources"]:
                with st.expander(f"📚 View {len(msg['sources'])} Source(s)"):
                    for i, src in enumerate(msg["sources"], 1):
                        source_type = src.get("type", "rag")
                        category = src.get("category", "unknown")
                        
                        # Different styling for web vs RAG sources
                        card_class = "source-card-web" if source_type == "web" else "source-card"
                        
                        type_badge = get_source_type_badge(source_type)
                        category_badge = get_category_badge(category) if source_type == "rag" else ""
                        
                        st.markdown(f"""
                        <div class="source-card {card_class}">
                            {type_badge}{category_badge}<br>
                            <strong>Source {i}:</strong> {src.get('source', 'Unknown')}<br>
                            <strong>Title:</strong> {src.get('title', 'N/A')}<br>
                            <strong>Score:</strong> {src.get('score', 0):.3f}<br>
                            <strong>Preview:</strong> {src.get('preview', 'N/A')[:200]}...
                        </div>
                        """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask anything about Social Security, Immigration, or Passports..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.queries += 1
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🤔 Thinking...")
        
        try:
            start = time.time()
            
            # Build conversation history (last 3 Q&A pairs = 6 messages)
            history = []
            if len(st.session_state.messages) > 0:
                # Get last 6 messages (3 Q&A pairs) excluding current query
                recent_messages = st.session_state.messages[-6:]
                for msg in recent_messages:
                    history.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # IMPORTANT: Use the slider values here!
            res = requests.post(
                API_URL,
                json={
                    "query": prompt, 
                    "top_k": top_k,  # ← From slider
                    "score_threshold": score_threshold,  # ← From slider
                    "enable_web_search": enable_web_search,  # ← From checkbox
                    "conversation_history": history  # ← Last 3 Q&A pairs
                },
                timeout=180
            )
            elapsed = time.time() - start
            
            if res.status_code == 200:
                data = res.json()
                answer = data.get("answer", "No answer")
                sources = data.get("sources", [])
                used_web_search = data.get("used_web_search", False)
                
                placeholder.markdown(answer)
                
                caption_parts = [f"⏱️ {elapsed:.2f}s"]
                if used_web_search:
                    caption_parts.append("🌐 Web Search Used")
                st.caption(" • ".join(caption_parts))
                
                if sources:
                    with st.expander(f"📚 View {len(sources)} Source(s)"):
                        for i, src in enumerate(sources, 1):
                            source_type = src.get("type", "rag")
                            category = src.get("category", "unknown")
                            
                            card_class = "source-card-web" if source_type == "web" else "source-card"
                            type_badge = get_source_type_badge(source_type)
                            category_badge = get_category_badge(category) if source_type == "rag" else ""
                            
                            st.markdown(f"""
                            <div class="source-card {card_class}">
                                {type_badge}{category_badge}<br>
                                <strong>Source {i}:</strong> {src.get('source', 'Unknown')}<br>
                                <strong>Title:</strong> {src.get('title', 'N/A')}<br>
                                <strong>Score:</strong> {src.get('score', 0):.3f}<br>
                                <strong>Preview:</strong> {src.get('preview', 'N/A')[:200]}...
                            </div>
                            """, unsafe_allow_html=True)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "time": elapsed,
                    "used_web_search": used_web_search
                })
            else:
                error = f"❌ Error: {res.status_code}"
                placeholder.error(error)
                st.session_state.messages.append({"role": "assistant", "content": error, "sources": []})
                
        except Exception as e:
            error = f"❌ Error: {e}"
            placeholder.error(error)
            st.session_state.messages.append({"role": "assistant", "content": error, "sources": []})
    
    st.rerun()
>>>>>>> d3f36b6 (update file)
