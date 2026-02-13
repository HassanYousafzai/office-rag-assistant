import streamlit as st
import os
import uuid
from dotenv import load_dotenv
from groq import Groq
from ingest import ingest_pdf
from retrieval import retrieve_relevant_chunks

# Load .env 
load_dotenv()  

# Debug: show if key is loaded
GROK_API_KEY = os.getenv("GROK_API_KEY")
if not GROK_API_KEY:
    st.error("GROK_API_KEY not found. Debug info:")
    st.write(f"Current working directory: {os.getcwd()}")
    st.write("Files in directory:", os.listdir("."))
    st.stop()

# Now initialize Groq
from groq import Groq
groq_client = Groq(api_key=GROK_API_KEY)

# ────────────────────────────────────────────────
# Session state
# ────────────────────────────────────────────────
if 'user_id' not in st.session_state:
    st.session_state.user_id = f"user_{uuid.uuid4().hex[:8]}"  # Unique per browser session
if 'messages' not in st.session_state:
    st.session_state.messages = []

# ────────────────────────────────────────────────
# Sidebar: Upload & User Info
# ────────────────────────────────────────────────
st.sidebar.title("Office Document Assistant")
st.sidebar.markdown(f"**Your User ID:** `{st.session_state.user_id[:12]}...` (your documents only)")
st.sidebar.markdown("Upload PDFs to add them to your personal knowledge base.")

uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"], help="Upload company policies, reports, contracts, etc.")

if uploaded_file is not None:
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.sidebar.spinner("Processing document..."):
        try:
            ingest_pdf(
                file_path=temp_path,
                file_name=uploaded_file.name,
                user_id=st.session_state.user_id
            )
            st.sidebar.success(f"Successfully uploaded & processed: {uploaded_file.name}")
        except Exception as e:
            st.sidebar.error(f"Upload failed: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

# ────────────────────────────────────────────────
# Main Chat Interface
# ────────────────────────────────────────────────
st.title("Office Document Assistant")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Retrieve relevant chunks
            chunks = retrieve_relevant_chunks(
                query=prompt,
                top_k=16,
                user_id=st.session_state.user_id
            )

            # Build context
            context = "\n\n".join([
                f"[Page ~{c['page_number']}] {c['content']}"
                for c in chunks
            ])

            # Prepare messages for Grok
            messages = [
                {
                    "role": "system",
                    "content": """You are a helpful office document assistant.
Use ONLY the provided context from the documents to answer questions.
If the user asks to quote something exactly (clause, section, policy text, etc.), quote it verbatim in quotation marks and include the page/section reference if available.
Be concise, accurate, professional, and helpful. Do not make up information.When quoting any clause, article, or text, ALWAYS provide the COMPLETE sentence/paragraph from the context — do NOT cut off mid-sentence even if the chunk ends there."""
                },
                {
                    "role": "user",
                    "content": f"""Context from your uploaded documents:
{context}

Question: {prompt}"""
                }
            ]

            try:
                completion = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",  # Current working model (Feb 2025)
                    messages=messages,
                    temperature=0.3,
                    max_tokens=800,
                    top_p=0.9
                )
                response = completion.choices[0].message.content.strip()

            except Exception as e:
                response = f"Error calling Groq API: {str(e)}\n\nPlease check your API key or try again later."
                st.error(response)

            st.markdown(response)

    # Save assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})