import uuid
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from database import supabase_admin

# CONFIG
embedder = SentenceTransformer("all-MiniLM-L6-v2")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
    keep_separator=True,
    add_start_index=True,
    strip_whitespace=True
)

def ingest_pdf(file_path: str, file_name: str, user_id: str = "demo_user"):
    """
    Ingest a single PDF file into Supabase.
    - Extracts text page by page
    - Chunks intelligently
    - Estimates page numbers
    - Embeds chunks
    - Stores document + chunks with user_id
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"Starting ingestion of '{file_name}' for user '{user_id}'...")

    try:
        # 1. Parse PDF per page
        reader = PdfReader(file_path)
        page_contents = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                page_contents.append({"page": page_num, "content": text.strip()})

        if not page_contents:
            raise ValueError("No readable text extracted from PDF")

        print(f"Extracted text from {len(page_contents)} pages")

        # Optional: debug check for specific content (remove or customize)
        full_text = "\n\n".join([p["content"] for p in page_contents])
        if "251. National language" in full_text:
            print("SUCCESS: Article 251 text is present")
        else:
            print("Note: No Article 251 text found in this PDF")

        # 2. Split into chunks
        langchain_docs = text_splitter.create_documents(
            [full_text],
            metadatas=[{"source_file": file_name}]
        )

        print(f"Created {len(langchain_docs)} chunks")

        # 3. Accurate page estimation
        enriched_chunks = []
        cumulative_chars = 0
        page_boundaries = [0]
        for p in page_contents:
            cumulative_chars += len(p["content"])
            page_boundaries.append(cumulative_chars)

        for i, doc in enumerate(langchain_docs):
            start_idx = doc.metadata.get("start_index", 0)

            page_estimate = 1
            for p_idx, boundary in enumerate(page_boundaries[1:], 1):
                if start_idx < boundary:
                    page_estimate = page_contents[p_idx-1]["page"]
                    break

            page_estimate = max(1, min(page_estimate, len(page_contents)))

            first_line = doc.page_content.split("\n", 1)[0].strip()
            section_hint = first_line if len(first_line) < 80 and (first_line.isupper() or "." in first_line) else ""

            enriched_chunks.append({
                "content": doc.page_content,
                "metadata": {
                    "source": file_name,
                    "page_number": page_estimate,
                    "start_index": start_idx,
                    "section_hint": section_hint,
                    "chunk_index": i,
                    "total_chunks": len(langchain_docs)
                }
            })

        # Debug: page distribution
        page_counts = {}
        for c in enriched_chunks:
            p = c["metadata"]["page_number"]
            page_counts[p] = page_counts.get(p, 0) + 1
        print("Page distribution:", sorted(page_counts.items()))

        # 4. Embed
        texts = [c["content"] for c in enriched_chunks]
        print("Generating embeddings...")
        embeddings = embedder.encode(texts, normalize_embeddings=True, batch_size=32).tolist()

        # 5. Insert document record
        doc_response = supabase_admin.table("documents").insert({
            "user_id": user_id,
            "file_name": file_name,
            "file_path": file_path
        }).execute()

        if not doc_response.data:
            raise Exception("Failed to create document record")

        document_id = doc_response.data[0]["id"]
        print(f"Document created: {document_id} (user: {user_id})")

        # 6. Batch insert chunks
        chunk_records = []
        for i, chunk_data in enumerate(enriched_chunks):
            chunk_records.append({
                "id": str(uuid.uuid4()),
                "document_id": document_id,
                "content": chunk_data["content"],
                "embedding": embeddings[i],
                "page_number": chunk_data["metadata"]["page_number"],
                "metadata": chunk_data["metadata"]
            })

        batch_size = 200
        for start in range(0, len(chunk_records), batch_size):
            batch = chunk_records[start:start + batch_size]
            resp = supabase_admin.table("chunks").insert(batch).execute()
            if resp.data:
                print(f"Inserted {len(batch)} chunks (batch {start//batch_size + 1})")
            else:
                print("Chunk batch insert failed:", resp)

        print(f"Ingestion complete! {len(enriched_chunks)} chunks stored for '{file_name}' (user: {user_id})")
        print("-" * 80)

    except Exception as e:
        print(f"Ingestion failed for '{file_name}': {str(e)}")
        raise

    finally:
        # Clean up temp file if it exists
        if os.path.exists(file_path) and "temp_" in file_path:
            try:
                os.remove(file_path)
            except:
                pass


if __name__ == "__main__":
    # Example: ingest multiple files (update paths as needed)
    pdf_files = [
        {"path": "sample.pdf", "name": "sample.pdf"},
        # Add your own PDFs here:
        # {"path": "my_documents/Company_Policy_2025.pdf", "name": "Company_Policy_2025.pdf"},
        # {"path": "my_documents/HR_Guidelines.pdf", "name": "HR_Guidelines.pdf"},
    ]

    user_id = "demo_user"  # Change this to your test user ID, or pass dynamically

    for pdf in pdf_files:
        if os.path.exists(pdf["path"]):
            ingest_pdf(pdf["path"], pdf["name"], user_id=user_id)
        else:
            print(f"File not found: {pdf['path']}")