from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from ingest import ingest_pdf  # Our ingestion function
from agent import app as rag_graph  # The LangGraph app
from database import create_conversation, add_message, get_conversation_messages
import os
import tempfile
from typing import Optional
from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI(title="Company RAG Assistant")

# Allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None  # This allows null or missing

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), user_id: str = Form("demo_user")):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files allowed")
    
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        ingest_pdf(tmp_path, file.filename, user_id)
        os.unlink(tmp_path)
        return {"status": "success", "message": f"{file.filename} indexed successfully"}
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(500, str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    user_id = "demo_user"
    
    # Create conversation if new
    if not request.conversation_id:
        conv_id = create_conversation(user_id, request.question[:50])
    else:
        conv_id = request.conversation_id
    
    # Load history
    history = get_conversation_messages(conv_id)
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": request.question})
    
    # Prepare inputs for agent
    inputs = {"messages": messages, "retrieved_docs": []}
    
    # Save user message immediately
    add_message(conv_id, "user", request.question)

    async def event_generator():
        full_response = ""
        try:
            # Stream from LangGraph + LLM
            async for event in rag_graph.astream_events(inputs, version="v2"):
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        full_response += content
                        yield f"data: {json.dumps({'delta': content})}\n\n"
                        await asyncio.sleep(0.01)  # slight delay for smooth UX

            # After stream ends → save full assistant message
            add_message(conv_id, "assistant", full_response)
            yield f"data: {json.dumps({'conversation_id': conv_id, 'done': True})}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)