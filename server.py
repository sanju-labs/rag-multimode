# server.py
# Run this instead of query.py for fast responses.
# The embedding model loads ONCE when the server starts, then stays in memory.
#
# Start server:  uvicorn server:app --reload
# Open browser:  http://localhost:8000
# Ask question:  http://localhost:8000/ask?q=What+was+Apple+revenue

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import time

app = FastAPI()

# Load everything once at startup — this is the key
# All queries after this will be fast
print("Loading embedding model and index...")
from rag import ask_stream, ask
from vectorstore import load_vectorstore
load_vectorstore()   # pre-loads FAISS index into memory
print("Ready.")


@app.get("/", response_class=HTMLResponse)
def home():
    """Simple UI to test queries in the browser."""
    return """
    <html>
    <body style="font-family: Arial; max-width: 600px; margin: 50px auto;">
        <h2>FinRAG — Financial Document Q&A</h2>
        <input id="q" style="width:100%; padding:10px; font-size:16px"
               placeholder="What was Apple's revenue in 2025?" />
        <br><br>
        <button onclick="ask()" style="padding:10px 20px; font-size:16px">Ask</button>
        <br><br>
        <div id="result" style="background:#f4f4f4; padding:15px; min-height:50px; white-space:pre-wrap;"></div>
        <div id="meta" style="color:gray; font-size:13px; margin-top:8px;"></div>
        <script>
        async function ask() {
            const q = document.getElementById('q').value;
            document.getElementById('result').innerText = 'Thinking...';
            const r = await fetch('/ask?q=' + encodeURIComponent(q));
            const data = await r.json();
            document.getElementById('result').innerText = data.answer;
            document.getElementById('meta').innerText =
                'Routed to: ' + data.routed_to +
                ' | Chunks: ' + data.chunks_checked +
                ' | Time: ' + data.time_seconds + 's';
        }
        document.getElementById('q').addEventListener('keydown', e => {
            if (e.key === 'Enter') ask();
        });
        </script>
    </body>
    </html>
    """


@app.get("/ask")
def ask_endpoint(q: str):
    """API endpoint — returns JSON answer."""
    t = time.time()
    result = ask(q)
    result["time_seconds"] = round(time.time() - t, 2)
    return result
