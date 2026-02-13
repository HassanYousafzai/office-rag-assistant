from supabase import create_client, Client
from dotenv import load_dotenv
import os

load_dotenv()

# Admin client (server-side, full access - use carefully)
supabase_admin: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
)

# Public client (for future frontend use)
supabase_public: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_ANON_KEY")
)

def test_connection():
    try:
        # Simple test query on documents table
        response = supabase_admin.table("documents").select("*").limit(1).execute()
        print("Supabase connected successfully!")
        print("Response:", response.data)
    except Exception as e:
        print("Connection error:", str(e))

if __name__ == "__main__":
    test_connection()

def create_conversation(user_id: str = "demo_user", title: str = "New Chat"):
    response = supabase_admin.table("conversations").insert({
        "user_id": user_id,
        "title": title
    }).execute()
    if response.data:
        return response.data[0]["id"]
    raise Exception("Failed to create conversation")

def add_message(conversation_id: str, role: str, content: str, sources: list = None):
    supabase_admin.table("messages").insert({
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "sources": sources or []
    }).execute()

def get_conversation_messages(conversation_id: str):
    response = supabase_admin.table("messages").select("*").eq("conversation_id", conversation_id).order("created_at").execute()
    return response.data