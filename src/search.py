import os
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from src.vectorstore import query_vectorstore
import google.genai as genai
from dotenv import load_dotenv

<<<<<<< HEAD

# ---------------- GEMINI CLIENT ----------------
def init_gemini_client():
    """
    Initialize Gemini API client from environment variable.
    Make sure .env contains GEMINI_API_KEY
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")

    client = genai.Client(api_key=api_key)
    print("Gemini client initialized.")
    return client
=======
# Import web search functionality
try:
    from tools.web_search import search_web, should_use_web_search, format_search_results
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    print("⚠️ Web search not available. Install tavily-python: pip install tavily-python")
    WEB_SEARCH_AVAILABLE = False

# ✅ LANGFUSE INTEGRATION - FIXED
try:
    from langfuse.decorators import observe, langfuse_context
    from langfuse import Langfuse
    
    # Load environment variables first
    load_dotenv()
    
    # Initialize Langfuse client with API keys
    langfuse_client = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )
    
    LANGFUSE_AVAILABLE = True
    print("✅ Langfuse monitoring enabled")
except ImportError:
    print("⚠️ Langfuse not available. Install: pip install langfuse")
    LANGFUSE_AVAILABLE = False
    # Create dummy decorator
    def observe():
        def decorator(func):
            return func
        return decorator
except Exception as e:
    print(f"⚠️ Langfuse initialization failed: {e}")
    LANGFUSE_AVAILABLE = False
    def observe():
        def decorator(func):
            return func
        return decorator


# ---------------- GEMINI CLIENT ----------------
def init_gemini_client():
    """Initialize Gemini API client from environment variable"""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if api_key:
        api_key = api_key.strip('"').strip("'")

    if not api_key or len(api_key) < 20:
        raise ValueError("GEMINI_API_KEY not found or invalid in environment variables.")

    print("Initializing Gemini client...")
    
    try:
        client = genai.Client(api_key=api_key)
        print("Gemini client initialized (will test on first use).")
        return client
    except Exception as e:
        print(f"Failed to initialize Gemini client: {e}")
        raise


# ---------------- QUERY CLASSIFICATION ----------------
@observe()  # ✅ LANGFUSE: Track query classification
def classify_query_type(query: str) -> str:
    """Classify query type for appropriate handling"""
    query_lower = query.lower().strip()
    
    # Greetings
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 
                 'good evening', 'how are you', 'what can you do', 
                 'help me', 'can you help', 'what is this', 'who are you']
    
    if any(greeting in query_lower for greeting in greetings):
        return 'greeting'
    
    # Follow-up commands (EXPANDED)
    followup_keywords = [
        "try again", "retry", "search again", "look it up", "look again",
        "check again", "search properly", "search web", "google it",
        "find it", "previous question", "last question", "that question",
        "fix answer", "wrong answer", "try properly", "details", "more info",
        "tell me more", "explain more", "elaborate", "look into"
    ]
    if any(kw in query_lower for kw in followup_keywords):
        return 'follow_up'
    
    # Temporal keywords
    temporal_keywords = ["latest", "recent", "current", "today", "update", "2024", "2025", "now", "new"]
    if any(kw in query_lower for kw in temporal_keywords):
        return 'temporal'
    
    # Location keywords (EXPANDED)
    location_keywords = [
        "near", "nearest", "closest", "address", "location", "where is",
        "center", "office", "branch", "facility", "local ssn", "ssn office",
        "uscis office", "passport office", "find office", "lookup office",
        "office in", "office at", "office for"
    ]
    if any(kw in query_lower for kw in location_keywords):
        return 'location'
    
    # Check if web search is needed
    if should_use_web_search(query):
        return 'web_search'
    
    return 'factual'


# ---------------- GOVERNMENT RELEVANCE CHECK ----------------
@observe()  # ✅ LANGFUSE: Track relevance check
def is_government_related(query: str, qtype: str) -> bool:
    """Check if query is related to U.S. government services (Medium-Strict)"""
    # Always allow follow-ups and greetings
    if qtype in ["greeting", "follow_up"]:
        return True

    q = query.lower()

    # Core government keywords
    core_keywords = [
        "ssn", "social security", "ssa", "retirement", "medicare", "medicaid",
        "uscis", "immigration", "green card", "citizenship", "visa", "visas",
        "passport", "passports", "u.s. embassy", "consulate", "travel document",
        "work permit", "employment authorization", "ead"
    ]
    if any(k in q for k in core_keywords):
        return True

    # Extended government keywords
    extended_keywords = [
        "benefits", "federal", "government office", "official documents",
        "form", "application", "public services", "child benefits",
        "tax", "irs", "disability", "survivor benefits", "refugee", "asylum"
    ]
    if any(k in q for k in extended_keywords):
        return True

    # CLEARLY NON-GOVERNMENT → Strict rejection
    non_gov = [
        "messi", "ronaldo", "football", "soccer", "sports", "movie", "song",
        "music", "concert", "album", "recipe", "restaurant", "food", "pizza",
        "burger", "weather", "stock", "crypto", "bitcoin", "python",
        "javascript", "coding", "programming", "gaming", "iphone", "android"
    ]
    if any(k in q for k in non_gov):
        return False

    # Default: allow (medium strict)
    return True


# ---------------- CONVERSATION CONTEXT ----------------
def extract_conversation_context(history: List[Dict]) -> str:
    """Extract last 5 user queries with brief responses for context"""
    if not history:
        return ""

    context = []
    user_turns = []

    for i, msg in enumerate(history):
        if msg["role"] == "user":
            answer = ""
            if i + 1 < len(history) and history[i + 1]["role"] == "assistant":
                answer = history[i + 1]["content"][:200]
                answer = answer.replace("**", "").replace("*", "").replace("#", "")

            user_turns.append((msg["content"], answer))

    # Take last 5 Q&A pairs
    last_5 = user_turns[-5:] if len(user_turns) > 5 else user_turns

    for i, (q, a) in enumerate(last_5, 1):
        context.append(f"Q{i}: {q}")
        if a:
            context.append(f"A{i}: {a}")

    return "\n".join(context)
>>>>>>> d3f36b6 (update file)


# ---------------- BASIC RAG SEARCH ----------------
def rag_simple(
    query: str,
<<<<<<< HEAD
    collection,
=======
    collection: Dict,
>>>>>>> d3f36b6 (update file)
    model: SentenceTransformer,
    client,
    top_k: int = 3,
):
<<<<<<< HEAD
    """
    Retrieve top-k context chunks from vectorstore, send to Gemini, return concise answer.
    """
    print(f"RAG Search for: '{query}'")

    # Step 1: Generate query embedding
    query_embedding = model.encode([query])[0]

    # Step 2: Retrieve top chunks
    results = query_vectorstore(collection, query_embedding, top_k=top_k)
=======
    """Retrieve top-k context chunks from FAISS, send to Gemini"""
    print(f"RAG Search for: '{query}'")

    query_embedding = model.encode([query])[0]
    results = query_vectorstore(collection["index"], collection["metadata"], query_embedding, top_k=top_k)
>>>>>>> d3f36b6 (update file)
    context = "\n\n".join([doc["content"] for doc in results]) if results else ""

    if not context:
        return "No relevant context found."

<<<<<<< HEAD
    # Step 3: Build LLM prompt
    prompt = f"""
Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {query}

Answer:
"""

    # Step 4: Query Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    # Step 5: Return answer
=======
    prompt = f"""You are a knowledgeable assistant specializing in U.S. government services including Social Security, Immigration (USCIS), and Travel/Passport services.

Use the following verified government information to answer the user's question accurately and helpfully.

CONTEXT FROM OFFICIAL SOURCES:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
- Provide accurate, helpful information based on the context above
- Use clear, simple language accessible to all users
- If the context doesn't fully answer the question, say what you know and what information is missing
- Include relevant details like requirements, deadlines, or contact information when applicable
- Be professional but friendly in tone

ANSWER:"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )

>>>>>>> d3f36b6 (update file)
    answer = getattr(response, "text", "No answer generated.").strip()
    return answer


<<<<<<< HEAD
# ---------------- ADVANCED RAG SEARCH ----------------
def rag_advanced(
    query: str,
    collection,
    model: SentenceTransformer,
    client,
    top_k: int = 5,
    score_threshold: float = 0.1,
    return_context: bool = False,
) -> Dict[str, Any]:
    """
    Advanced RAG pipeline returning answer, sources, confidence, and optionally context.
    """
    print(f"Running advanced RAG pipeline for query: '{query}'")

    # Step 1: Embed query
    query_embedding = model.encode([query])[0]

    # Step 2: Retrieve documents
    results = query_vectorstore(collection, query_embedding, top_k=top_k)

    if not results:
        return {
            "answer": "No relevant context found.",
            "sources": [],
            "confidence": 0.0,
            "context": "",
        }

    # Step 3: Filter by threshold
    filtered_results = [r for r in results if r["similarity"] >= score_threshold]
    if not filtered_results:
        return {
            "answer": "No relevant documents above threshold.",
            "sources": [],
            "confidence": 0.0,
            "context": "",
        }

    # Step 4: Build context and sources
    context = "\n\n".join([r["content"] for r in filtered_results])
    sources = [
        {
            "source": r["metadata"].get("source", "unknown"),
            "score": r["similarity"],
            "preview": r["content"][:120] + "...",
        }
        for r in filtered_results
    ]

    confidence = max(r["similarity"] for r in filtered_results)

    # Step 5: Build Gemini prompt
    prompt = f"""
Use the following context to answer the question concisely and accurately.

Context:
{context}

Question: {query}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    output = {
        "answer": getattr(response, "text", "No answer generated.").strip(),
        "sources": sources,
        "confidence": confidence,
    }

    if return_context:
        output["context"] = context

    print("RAG answer generated successfully.")
    return output
=======
# ---------------- ADVANCED RAG SEARCH WITH WEB INTEGRATION + LANGFUSE ----------------
@observe()  # ✅ LANGFUSE: Track entire RAG pipeline
def rag_advanced(
    query: str,
    collection: Dict,
    model: SentenceTransformer,
    client,
    top_k: int = 8,
    score_threshold: float = 0.25,
    return_context: bool = False,
    enable_web_search: bool = True,
    conversation_history: List[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Advanced RAG pipeline with FAISS + optional web search fallback + Langfuse monitoring
    """
    
    # ✅ LANGFUSE: Update trace with query metadata
    if LANGFUSE_AVAILABLE:
        langfuse_context.update_current_trace(
            name="rag_query",
            metadata={
                "query": query,
                "top_k": top_k,
                "score_threshold": score_threshold,
                "enable_web_search": enable_web_search,
                "has_conversation_history": len(conversation_history) > 0 if conversation_history else False
            }
        )
    
    print(f"\n{'='*50}")
    print(f"📝 Query: {query}")
    print(f"{'='*50}")

    # Classify query type
    qtype = classify_query_type(query)
    print(f"🏷️  Type: {qtype}")
    
    # ✅ LANGFUSE: Log query type
    if LANGFUSE_AVAILABLE:
        langfuse_context.update_current_observation(metadata={"query_type": qtype})

    # Handle greetings
    if qtype == 'greeting':
        answer = (
            "Hello! I'm **GovAssist AI**, and I can help with U.S. government services:\n\n"
            "• **Social Security (SSA)** - SSN applications, benefits, retirement, disability, offices\n"
            "• **Immigration (USCIS)** - Green cards, visas, citizenship, work permits, forms\n"
            "• **U.S. Passports** - Applications, renewals, travel documents, consulates\n\n"
            "**Try asking:**\n"
            "• *'nearest SSN office to Logan, Utah'*\n"
            "• *'documents needed for SSN application'*\n"
            "• *'how to apply for a green card'*\n"
        )
        
        if LANGFUSE_AVAILABLE:
            langfuse_context.update_current_trace(output={"answer": answer, "type": "greeting"})
        
        return {
            "answer": answer,
            "sources": [],
            "confidence": 1.0,
            "used_web_search": False,
            "context": ""
        }

    # Extract conversation context
    ctx = extract_conversation_context(conversation_history or [])
    if ctx:
        print(f"💬 Using conversation context ({len(ctx)} chars)")

    # Handle follow-up commands
    if qtype == "follow_up":
        last_users = [m for m in (conversation_history or []) if m["role"] == "user"]
        if len(last_users) < 2:
            return {
                "answer": "I don't have a previous question to refer to. Could you please ask your question again?",
                "sources": [],
                "confidence": 0.0,
                "used_web_search": False,
                "context": ""
            }
        
        prev_query = last_users[-2]["content"]
        print(f"🔄 Follow-up detected → Using previous query: '{prev_query}'")
        query = prev_query
        qtype = classify_query_type(prev_query)

    # Check if government-related
    if not is_government_related(query, qtype):
        print("⛔ REJECTED: Non-government topic")
        
        answer = (
            "I specialize **only** in U.S. government services (Social Security, Immigration, Passports).\n\n"
            f"The question **'{query[:60]}...'** is outside those areas.\n\n"
            "**Try asking:**\n"
            "• How to apply for Social Security benefits\n"
            "• USCIS form requirements\n"
        )
        
        if LANGFUSE_AVAILABLE:
            langfuse_context.update_current_trace(
                output={"answer": answer, "rejected": True},
                metadata={"rejection_reason": "non_government"}
            )
        
        return {
            "answer": answer,
            "sources": [],
            "confidence": 0.0,
            "used_web_search": False,
            "context": ""
        }

    print("✅ Government-related query accepted")

    # ═══════════════════════════════════════════════════════
    # STEP 1: RAG SEARCH with Langfuse tracking
    # ═══════════════════════════════════════════════════════
    print(f"🔍 RAG Search (top_k={top_k}, threshold={score_threshold})...")
    
    if LANGFUSE_AVAILABLE:
        langfuse_context.update_current_observation(
            name="rag_search",
            metadata={"top_k": top_k, "threshold": score_threshold}
        )
    
    query_embedding = model.encode([query])[0]
    results = query_vectorstore(
        collection["index"], 
        collection["metadata"], 
        query_embedding, 
        top_k=top_k
    )
    
    # Filter by threshold
    rag_results = [r for r in results if r["similarity"] >= score_threshold]
    rag_conf = max([r["similarity"] for r in rag_results]) if rag_results else 0.0
    
    print(f"Retrieved {len(results)} results, {len(rag_results)} above threshold")
    if rag_results:
        print(f"✅ RAG confidence: {rag_conf:.3f}")
    
    # ✅ LANGFUSE: Log RAG results
    if LANGFUSE_AVAILABLE:
        langfuse_context.update_current_observation(
            output={
                "total_results": len(results),
                "filtered_results": len(rag_results),
                "max_confidence": float(rag_conf)
            }
        )
    
    # Build RAG context and sources
    rag_context = ""
    rag_sources = []
    
    if rag_results:
        rag_context = "\n\n".join([r["content"] for r in rag_results])
        rag_sources = [
            {
                "type": "rag",
                "source": r["metadata"].get("source", "unknown"),
                "category": r["metadata"].get("category", "general"),
                "title": r["metadata"].get("title", ""),
                "score": r["similarity"],
                "preview": r["content"][:200] + "...",
            }
            for r in rag_results
        ]

    # ═══════════════════════════════════════════════════════
    # STEP 2: WEB SEARCH with Langfuse tracking
    # ═══════════════════════════════════════════════════════
    use_web = False
    web_context = ""
    web_sources = []
    
    if enable_web_search and WEB_SEARCH_AVAILABLE:
        if qtype in ['temporal', 'location', 'web_search'] or rag_conf < 0.30:
            print(f"🌐 Web Search triggered (type={qtype}, rag_conf={rag_conf:.3f})...")
            use_web = True
            
            if LANGFUSE_AVAILABLE:
                langfuse_context.update_current_observation(
                    name="web_search",
                    metadata={
                        "trigger_reason": qtype if qtype in ['temporal', 'location', 'web_search'] else "low_rag_confidence",
                        "rag_confidence": float(rag_conf)
                    }
                )
            
            try:
                search_query = query
                if qtype == 'location' and 'logan' in query.lower() and 'utah' in query.lower():
                    search_query = "Social Security office Ogden Utah address"
                
                web_results = search_web(search_query, max_results=5)
                
                if web_results:
                    web_context = format_search_results(web_results)
                    
                    for result in web_results:
                        if result["type"] == "search_result":
                            web_sources.append({
                                "type": "web",
                                "source": result.get("url", "Web Search"),
                                "title": result.get("title", "Web Result"),
                                "category": "web_search",
                                "score": result.get("score", 0.5),
                                "preview": result.get("content", "")[:200] + "..."
                            })
                    
                    print(f"✅ Found {len(web_sources)} web sources")
                    
                    # ✅ LANGFUSE: Log web results
                    if LANGFUSE_AVAILABLE:
                        langfuse_context.update_current_observation(
                            output={"web_results_count": len(web_sources)}
                        )
            except Exception as e:
                print(f"❌ Web search error: {e}")
                use_web = False

    # ═══════════════════════════════════════════════════════
    # STEP 3: Combine Contexts
    # ═══════════════════════════════════════════════════════
    combined = ""
    if rag_context:
        combined += f"--- KNOWLEDGE BASE (Internal Documents) ---\n{rag_context}\n"
    if web_context:
        combined += f"--- WEB SEARCH RESULTS (Current Information) ---\n{web_context}\n"
    
    if not combined:
        return {
            "answer": "I couldn't find specific information about that.",
            "sources": [],
            "confidence": 0.0,
            "used_web_search": use_web,
            "context": ""
        }

    # ═══════════════════════════════════════════════════════
    # STEP 4: Generate Answer with Gemini + Langfuse
    # ═══════════════════════════════════════════════════════
    prompt = f"""You are **GovAssist AI**, an expert assistant for U.S. government services.

{f"**CONVERSATION HISTORY:**{chr(10)}{ctx}{chr(10)}{chr(10)}" if ctx else ""}
**INFORMATION SOURCES:**
{combined}

**USER QUESTION:** {query}

**INSTRUCTIONS:**
1. Use BOTH knowledge base and web search results if available
2. For **location questions** - provide specific address if found, otherwise SSA locator
3. Be **clear, direct, and helpful**
4. Use conversation history for context

**YOUR ANSWER:**"""

    try:
        # ✅ LANGFUSE: Track LLM generation
        if LANGFUSE_AVAILABLE:
            generation = langfuse_context.generation(
                name="gemini_generate",
                model="gemini-2.0-flash",
                input=prompt,
                metadata={"query_type": qtype}
            )
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        answer = response.text
        print("✅ Generated answer with Gemini")
        
        # ✅ LANGFUSE: Log generation output
        if LANGFUSE_AVAILABLE:
            generation.end(output=answer)
        
    except Exception as e:
        print(f"❌ LLM generation error: {e}")
        answer = "I encountered an error generating the response. Please try rephrasing your question."
        
        if LANGFUSE_AVAILABLE:
            langfuse_context.update_current_trace(
                metadata={"error": str(e), "error_type": "llm_generation"}
            )

    # ═══════════════════════════════════════════════════════
    # STEP 5: Post-processing fallback for location queries
    # ═══════════════════════════════════════════════════════
    if qtype == "location" and "logan" in query.lower() and "utah" in query.lower():
        if "ssa.gov" not in answer.lower() and "1-800-772-1213" not in answer:
            answer += (
                "\n\n**Find Your Local Office:**\n"
                "• SSA Office Locator: https://secure.ssa.gov/ICON/main.jsp\n"
                "• Call SSA: 1-800-772-1213 (TTY: 1-800-325-0778)"
            )

    # ═══════════════════════════════════════════════════════
    # STEP 6: Return Complete Response + Langfuse tracking
    # ═══════════════════════════════════════════════════════
    all_sources = rag_sources + web_sources
    confidence = max([s["score"] for s in all_sources]) if all_sources else 0.0

    print(f"✅ Final confidence: {confidence:.3f}")
    print(f"📊 Sources: {len(rag_sources)} RAG + {len(web_sources)} Web")
    print(f"{'='*50}\n")

    output = {
        "answer": answer,
        "sources": all_sources,
        "confidence": confidence,
        "used_web_search": use_web,
    }

    if return_context:
        output["context"] = combined[:1000]
    
    # ✅ LANGFUSE: Log final output metrics
    if LANGFUSE_AVAILABLE:
        langfuse_context.update_current_trace(
            output={
                "answer_length": len(answer),
                "total_sources": len(all_sources),
                "confidence": float(confidence),
                "used_web_search": use_web
            },
            metadata={
                "rag_sources": len(rag_sources),
                "web_sources": len(web_sources),
                "query_type": qtype
            }
        )

    return output


# ---------------- GREETING HANDLER ----------------
def handle_greeting(query: str, client) -> str:
    """Handle conversational greetings"""
    greeting_prompt = f"""You are GovAssist AI, a helpful assistant specializing in U.S. government services.

USER: {query}

Respond warmly and briefly explain what you can help with. Mention you can assist with:
- Social Security benefits and services
- Immigration and USCIS information  
- Passport and international travel

Keep it friendly, concise (2-3 sentences), and inviting.

RESPONSE:"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=greeting_prompt,
        )
        return response.text.strip()
    except:
        return "Hello! I'm GovAssist AI, your guide to U.S. government services. I can help you with Social Security, Immigration (USCIS), and Passport/Travel questions. What would you like to know?"


# ---------------- CONFIGURATION INFO ----------------
def get_config_info() -> Dict[str, Any]:
    """Return current configuration settings"""
    return {
        "default_top_k": 8,
        "default_score_threshold": 0.25,
        "web_search_available": WEB_SEARCH_AVAILABLE,
        "langfuse_available": LANGFUSE_AVAILABLE,  # ✅ NEW
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": "gemini-2.0-flash",
        "conversation_history": 5,
        "strictness": "medium",
        "chunking": {
            "ssa": {"chunk_size": 800, "overlap": 150},
            "uscis": {"chunk_size": 900, "overlap": 150},
            "travel_state": {"chunk_size": 700, "overlap": 100},
            "general": {"chunk_size": 800, "overlap": 150}
        }
    }
>>>>>>> d3f36b6 (update file)
