# backend/app/main.py
import os
import json
import uuid
import asyncio
from typing import Dict, Any, List
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import numpy as np
from openai import OpenAI
from .utils import DOCUMENTS, top_k_by_embedding, simple_overlap_rank, safe_parse_metadata_from_text
from datetime import datetime
from llm_guard.input_scanners import PromptInjection

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", "3"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))  # Only include docs above this similarity score

# Check for valid API key (must be non-empty and look like a real key)
def has_valid_api_key():
    if not OPENAI_API_KEY:
        return False
    key = OPENAI_API_KEY.strip()
    # OpenAI keys typically start with 'sk-' and are at least 20 characters
    return bool(key and len(key) > 20 and key.startswith('sk-'))

MOCK_MODE = not has_valid_api_key()
if has_valid_api_key():
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None

app = FastAPI(title="Nextflow Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",     # Local development
        "http://127.0.0.1:3000",     # Alternative localhost
        "http://0.0.0.0:3000",       # Docker frontend
        "https://nextflow-chatbot.vercel.app/" # Deployed front end
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSIONS: Dict[str, List[Dict[str, str]]] = {}

# Initialize prompt injection scanner
prompt_injection_scanner = PromptInjection()

def _now():
    return datetime.utcnow().isoformat()

# Compute embeddings for DOCUMENTS at startup if an API key exists
@app.on_event("startup")
def prepare_documents():
    print(f"[{_now()}] Starting up - MOCK_MODE: {MOCK_MODE}")
    if MOCK_MODE:
        print(f"[{_now()}] MOCK_MODE: skipping embeddings")
        return
    try:
        for d in DOCUMENTS:
            resp = client.embeddings.create(input=d["text"], model=EMBEDDING_MODEL)
            emb = np.array(resp.data[0].embedding, dtype=float)
            if np.linalg.norm(emb) > 0:
                emb = emb / np.linalg.norm(emb)
            d["embedding"] = emb
        print(f"[{_now()}] Computed embeddings for DOCUMENTS")
    except Exception as e:
        # If something fails, we still run but without embeddings (fallback to simple overlap)
        print(f"[{_now()}] Warning: failed to compute embeddings at startup: {e}")

    # Run smoke test
    try:
        from smoke_test import smoke_test
        asyncio.create_task(smoke_test())
    except ImportError:
        print(f"[{_now()}] Warning: smoke test not available")

@app.get("/api/status")
def get_status():
    key_info = "NOT SET"
    if OPENAI_API_KEY:
        key_info = f"SET (length: {len(OPENAI_API_KEY)}, starts with: {OPENAI_API_KEY[:10] if len(OPENAI_API_KEY) >= 10 else OPENAI_API_KEY}...)"

    print(f"[{_now()}] Status check: MOCK_MODE={MOCK_MODE}, OPENAI_API_KEY={key_info}")
    status = {
        "mock_mode": MOCK_MODE,
        "message": "Running in Mock Mode - responses will echo your input" if MOCK_MODE else "Running with OpenAI integration"
    }
    print(f"[{_now()}] Returning status: {status}")
    return status

@app.post("/api/session")
def create_session():
    sid = str(uuid.uuid4())
    SESSIONS[sid] = []
    return {"session_id": sid}

def build_system_prompt(top_docs: List[Dict[str, Any]]) -> str:
    sources_text = "\n\n".join([f"[{d.get('title','')}]({d.get('url')})\n{d.get('text')}" for d in top_docs])
    return f"""
You are an assistant that should answer user questions using the provided SOURCES when appropriate.
Do not hallucinate citations. Prefer to cite the sources if they support the answer.

Below are some common sources of confusion for the user, as well as information on how to respond:

1) Confusion around how to check the version of Nextflow - if the user makes reference to wanting
to know the version of Nextflow, tell them to run the following command in the terminal: nextflow -v

2) Confusion around DSL1 vs DSL2 syntax - the user might be confused between DSL1 syntax and DSL2
syntax. Here are some common differences between the two syntaxs:

* DSL1 does not require a declaration. DSL2 does require a delcaration at the top of the script
* DSL1 only makes use of top-level processes. DSL2 also supports subworkflows.
* DSL1 pipelines are written as a single script. DSL2 allows for modules and workflows.

SOURCES:
{sources_text}
"""

@app.post("/api/chat/stream")
async def chat_stream(request: Request):
    body = await request.json()
    session_id = body.get("session_id")
    user_message = (body.get("message") or "").strip()
    if not user_message:
        return JSONResponse({"error": "Empty message"}, status_code=400)
    if not session_id:
        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = []

    print(f"[backend] User asked: {user_message}")  # <-- log question

    # Check for prompt injection
    sanitized_prompt, is_valid, risk_score = prompt_injection_scanner.scan(user_message)
    if not is_valid:
        async def injection_response():
            response = "Prompt injection detected, please rephrase your question and try again."
            yield f"data: {response}\n\n"
            SESSIONS[session_id].append({"role": "assistant", "content": response})
        return StreamingResponse(injection_response(), media_type="text/event-stream")

    # Handle mock mode
    if MOCK_MODE:
        async def mock_event_gen():
            mock_response = f"Your message was: {user_message}"
            print(f"[backend] Mock response: {mock_response}")

            # Stream the mock response character by character for realism
            for char in mock_response:
                yield f"data: {char}\n\n"
                await asyncio.sleep(0.01)

            # Send mock metadata
            mock_metadata = {
                "mock_mode": True,
                "citations": [],
                "reasoning_summary": "Mock mode - echoing user input",
                "used_docs": []
            }
            yield f"event: metadata\ndata: {json.dumps(mock_metadata)}\n\n"

            # Store in session
            SESSIONS[session_id].append({"role": "assistant", "content": mock_response})

        return StreamingResponse(mock_event_gen(), media_type="text/event-stream")

    # Retrieve relevant documents using RAG
    try:
        qemb_resp = client.embeddings.create(input=user_message, model=EMBEDDING_MODEL)
        qvec = np.array(qemb_resp.data[0].embedding, dtype=float)
        if np.linalg.norm(qvec) > 0:
            qvec = qvec / np.linalg.norm(qvec)
        # use top_k_by_embedding only if docs have embeddings
        if any("embedding" in d for d in DOCUMENTS):
            top_docs = top_k_by_embedding(qvec, DOCUMENTS, k=TOP_K, threshold=SIMILARITY_THRESHOLD)
        else:
            # For word overlap, use a minimum of 1 overlapping word as threshold
            top_docs = simple_overlap_rank(user_message, DOCUMENTS, k=TOP_K, threshold=1)
    except Exception as e:
        print(f"[backend] Warning: query embedding failed, falling back to overlap rank: {e}", flush=True)
        top_docs = simple_overlap_rank(user_message, DOCUMENTS, k=TOP_K, threshold=1)

    print(f"[backend] Retrieved {len(top_docs)} documents for query (threshold: {SIMILARITY_THRESHOLD})", flush=True)
    for i, doc in enumerate(top_docs):
        similarity_score = doc.get('similarity_score', doc.get('overlap_score', 'N/A'))
        print(f"[backend] Doc {i+1}: {doc.get('title', 'No title')} (score: {similarity_score})", flush=True)

    system_prompt = build_system_prompt(top_docs)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    async def event_gen():
        assistant_text = ""
        try:
            # Run the blocking OpenAI stream in a background thread
            loop = asyncio.get_event_loop()
            stream = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=GEN_MODEL,
                    messages=messages,
                    temperature=0.2,
                    stream=True,
                ),
            )

            streamed_text = ""
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        assistant_text += delta.content

                        # Check if we've hit the metadata section and stop streaming content
                        if "<|metadata|>" in assistant_text:
                            # Only stream up to the metadata marker
                            clean_part = assistant_text.split("<|metadata|>")[0]
                            remaining = clean_part[len(streamed_text):]
                            if remaining:
                                yield f"data: {remaining}\n\n"
                                streamed_text += remaining
                            break
                        else:
                            yield f"data: {delta.content}\n\n"
                            streamed_text += delta.content
                        await asyncio.sleep(0)  # let event loop flush

            print(f"[backend] Assistant replied: {assistant_text}")  # <-- log response

            # Try to extract metadata appended by the model
            metadata = safe_parse_metadata_from_text(assistant_text)
            if not metadata:
                metadata = {
                    "mock_mode": False,
                    "citations": [{"id": d.get("id"), "title": d.get("title"), "url": d.get("url")} for d in top_docs],
                    "reasoning_summary": "Used RAG to retrieve relevant documents.",
                    "used_docs": [d.get("id") for d in top_docs]
                }

            # Clean the assistant text by removing metadata tags if present
            clean_text = assistant_text
            if "<|metadata|>" in clean_text and "<|endmetadata|>" in clean_text:
                clean_text = clean_text.split("<|metadata|>")[0].strip()

            # Store the cleaned version in session history
            SESSIONS[session_id].append({"role": "assistant", "content": clean_text})

            yield f"event: metadata\ndata: {json.dumps(metadata)}\n\n"

        except Exception as e:
            # Check if it's an OpenAI authentication error
            error_str = str(e).lower()
            if "authentication" in error_str or "api_key" in error_str or "unauthorized" in error_str:
                # Fallback to mock mode on auth error
                mock_response = f"Your message was: {user_message}"
                print(f"[backend] OpenAI auth error, falling back to mock: {e}")
                yield f"data: {mock_response}\n\n"

                mock_metadata = {
                    "mock_mode": True,
                    "citations": [],
                    "reasoning_summary": "Auth error - switched to mock mode",
                    "used_docs": []
                }
                yield f"event: metadata\ndata: {json.dumps(mock_metadata)}\n\n"

                # Store in session
                SESSIONS[session_id].append({"role": "assistant", "content": mock_response})
            else:
                yield f"data: Sorry — an error occurred: {str(e)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
