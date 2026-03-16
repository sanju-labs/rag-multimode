import streamlit as st

# --- Page Config ---
st.set_page_config(page_title="Bing AI", page_icon="🤖", layout="wide")

# --- UI Styling ---
st.markdown("""
<style>
/* Base Theme */
.stApp {
    background-color: #0e0e0e;
    color: #e3e3e3;
}

/* Header Typography */
.greeting {
    font-size: 3.5rem;
    font-weight: 850;
    text-align: center;
    background: linear-gradient(120deg, #4285f4, #9b72cb, #d96570);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0px;
}

.sub-greeting {
    text-align: center;
    color: #888;
    font-size: 1.1rem;
    margin-bottom: 40px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0c0f14;
    border-right: 1px solid #2a2f36;
}

.bts-card {
    background: linear-gradient(145deg, #161b22, #1c2229);
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #2f363d;
    margin-bottom: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}

/* Suggestion Buttons */
.stButton>button {
    border-radius: 12px;
    background: #161b22;
    border: 1px solid #30363d;
    color: #e3e3e3;
    padding: 14px 18px;
    transition: all .2s ease;
}

.stButton>button:hover {
    border: 1px solid #4285f4;
    background: #1c2229;
}

/* --- INTEGRATED CHAT INPUT STYLING --- */

/* Removes the default white/grey bar at the bottom */
[data-testid="stBottom"] {
    background-color: transparent !important;
    border-top: none !important;
}

/* Targets the chat input container */
[data-testid="stChatInput"] {
    background-color: transparent !important;
    border: none !important;
    padding-bottom: 40px !important;
}

/* Styles the actual text area box */
[data-testid="stChatInput"] textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 16px !important;
    color: #e3e3e3 !important;
    padding: 16px 20px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5) !important;
}

/* Hover and Focus effect */
[data-testid="stChatInput"] textarea:hover {
    border: 1px solid #4285f4 !important;
}

[data-testid="stChatInput"] textarea:focus {
    border: 1px solid #4285f4 !important;
    box-shadow: 0 0 15px rgba(66, 133, 244, 0.2), 0 10px 30px rgba(0, 0, 0, 0.5) !important;
}

/* Style the 'Send' arrow button */
[data-testid="stChatInput"] button {
    background-color: #4285f4 !important;
    border-radius: 10px !important;
    bottom: 12px !important;
    right: 12px !important;
}

/* Thinking Animation */
.thinking {
    display: flex;
    gap: 6px;
    align-items: center;
}

.thinking span {
    width: 8px;
    height: 8px;
    background: #9b72cb;
    border-radius: 50%;
    animation: bounce 1.4s infinite;
}

.thinking span:nth-child(2) { animation-delay: .2s; }
.thinking span:nth-child(3) { animation-delay: .4s; }

@keyframes bounce {
    0%, 80%, 100% { transform: scale(0); opacity: .4; }
    40% { transform: scale(1); opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# --- Load RAG ---
@st.cache_resource
def load_rag_system():
    try:
        from rag import ask
        from vectorstore import load_vectorstore
        load_vectorstore()
        return {"ok": True, "ask": ask}
    except Exception as e:
        return {"ok": False, "error": str(e)}

rag_system = load_rag_system()

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h3 style='color:#4285f4; font-weight:700;'>Behind The Scenes</h3>", unsafe_allow_html=True)
    st.write("Curious about how the AI thinks? Toggle below.")
    bts_on = st.toggle("Show AI Logic")
    st.markdown("---")

    if bts_on and st.session_state.last_result:
        res = st.session_state.last_result
        st.markdown(f'<div class="bts-card"><b>Semantic Search</b><br>Checked {res.get("chunks_checked",5)} chunks</div>', unsafe_allow_html=True)

# --- Main UI ---
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown('<p class="greeting">Hi, I\'m Bing</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-greeting">Ask anything about the uploaded documents</p>', unsafe_allow_html=True)

# Chat History
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Suggestions
selected_query = None
if not st.session_state.messages:
    c1, c2 = st.columns(2)
    qs = ["What is the company’s name?", "What is the company’s revenue?", "Compare revenue year-over-year", "Where is the company located?"]
    for i, q in enumerate(qs):
        if (c1 if i % 2 == 0 else c2).button(f"✨ {q}", use_container_width=True):
            selected_query = q

# Chat Input
prompt = st.chat_input("Ask Bing about the documents...")
final_query = selected_query if selected_query else prompt

# Response Logic
if final_query:
    st.session_state.messages.append({"role": "user", "content": final_query})
    with st.chat_message("assistant", avatar="🤖"):
        thinking = st.empty()
        thinking.markdown('<div class="thinking"><span></span><span></span><span></span>&nbsp;&nbsp;<i>Thinking...</i></div>', unsafe_allow_html=True)
        
        try:
            result = rag_system["ask"](final_query)
            st.session_state.last_result = result
            answer = result.get("answer", "No data found.")
            thinking.empty()
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()
        except Exception as e:
            thinking.empty()
            st.error(e)