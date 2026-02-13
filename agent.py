from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from database import create_conversation, add_message, get_conversation_messages
from retrieval import retrieve_relevant_chunks
from typing import List, Dict, TypedDict, Annotated
import operator
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env

# Updated model (current & excellent for reasoning)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Best available on your account
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

class AgentState(TypedDict):
    messages: Annotated[List[Dict], operator.add]
    retrieved_docs: List[Dict]

def retrieval_tool(state: AgentState):
    last_question = state["messages"][-1]["content"]
    docs = retrieve_relevant_chunks(last_question, top_k=6)
    return {"retrieved_docs": docs}

def generate_answer(state: AgentState):
    question = state["messages"][-1]["content"]
    docs = state["retrieved_docs"]

    context = "\n\n".join(
        f"SOURCE [{i+1}] (Page ~{doc['page_number']}): {doc['content'][:1500]}..."
        for i, doc in enumerate(docs)
    )

    prompt = f"""
You are an expert assistant answering questions strictly based on the company's internal documents. When quoting any clause, article, or text, ALWAYS provide the COMPLETE sentence/paragraph from the context — do NOT cut off mid-sentence even if the chunk ends there.

Question: {question}

Relevant Sources (use ONLY these):
{context}

Instructions:
- Answer clearly and accurately using only the sources.
- Cite sources inline with [1], [2], etc. at the end of relevant sentences.
- List full citations at the end.
- If no relevant info, say "No information found in documents."
- Be professional and concise.

Answer:
"""

    response = llm.invoke(prompt)
    answer = response.content

    # Append citations preview
    citations = "\n\nSources:\n" + "\n".join(
    f"[{i+1}] Page ~{doc['page_number']} (Similarity: {doc['similarity']:.3f}): {doc['content'][:400]}..."
    for i, doc in enumerate(docs)
)
    full_answer = answer + citations

    return {"messages": [{"role": "assistant", "content": full_answer}]}

# Graph
graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieval_tool)
graph.add_node("generate", generate_answer)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
app = graph.compile()

# New ask function with history
def ask_question_with_history(question: str, conversation_id: str = None):
    if not conversation_id:
        conversation_id = create_conversation(title=question[:50])

    # Load history
    history = get_conversation_messages(conversation_id)
    messages = [{"role": m["role"], "content": m["content"]} for m in history]

    # Add new question
    messages.append({"role": "user", "content": question})

    inputs = {
        "messages": messages,
        "retrieved_docs": []
    }
    result = app.invoke(inputs)

    # Save user message and assistant response
    add_message(conversation_id, "user", question)
    assistant_content = result["messages"][-1]["content"]
    add_message(conversation_id, "assistant", assistant_content, sources=[{"doc": "sample.pdf"}])  # Can enhance

    print(f"\nConversation ID: {conversation_id}")
    print(f"Question: {question}\n")
    print(assistant_content)
    print("\n" + "="*60 + "\n")

    return conversation_id

if __name__ == "__main__":
    conv_id = None
    questions = [
        "What are the fundamental rights?",
        "Give more details about freedom of assembly.",
        "What about freedom of speech?",
        "What is the national language?"
    ]
    for q in questions:
        conv_id = ask_question_with_history(q, conv_id)