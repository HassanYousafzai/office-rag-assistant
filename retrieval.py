from database import supabase_admin
from sentence_transformers import SentenceTransformer

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Global variable to remember the last known page of Article 251 (per document type if needed later)
LAST_ARTICLE_251_PAGE = None

def retrieve_relevant_chunks(query: str, top_k: int = 8, user_id: str = "demo_user"):
    """
    Retrieve top relevant chunks from Supabase pgvector with dynamic page learning.
    Returns list of dicts with content, page, similarity, etc.
    """
    global LAST_ARTICLE_251_PAGE

    try:
        print(f"Retrieving for user_id: {user_id} | Query: '{query}'")

        # 1. Embed the query
        query_embedding = embedder.encode([query], normalize_embeddings=True).tolist()[0]

        # 2. Prepare params
        is_quote_or_clause = "quote" in query.lower() or "clause" in query.lower()
        match_count = top_k * 6 if is_quote_or_clause else top_k * 4
        params = {
            "query_embedding": query_embedding,
            "match_count": match_count,
            "filter_user_id": user_id
        }

        query_lower = query.lower()
        keyword = None

        # Keyword priority (stronger for English clause and quotes)
        if "251" in query_lower or "article 251" in query_lower:
            keyword = "251"
            print("Using keyword filter: '251'")
        elif "national language" in query_lower or "urdu" in query_lower:
            keyword = "national language"
            print("Using keyword filter: 'national language'")
        elif "english" in query_lower and ("251" in query_lower or "clause" in query_lower):
            keyword = "English language"
            print("Using keyword filter: 'English language' (clause 2 boost)")
        elif "clause (2)" in query_lower or "clause 2" in query_lower:
            keyword = "English language"
            print("Using keyword filter: 'English language' for clause (2) quote")

        if keyword:
            params["keyword"] = keyword

        # Dynamic fallback range if we learned a page
        if LAST_ARTICLE_251_PAGE is not None and any(x in query_lower for x in ["251", "national language", "english", "clause"]):
            min_page = max(1, LAST_ARTICLE_251_PAGE - 40)
            max_page = LAST_ARTICLE_251_PAGE + 40
            params["min_page"] = min_page
            params["max_page"] = max_page
            print(f"Using dynamic fallback range {min_page}–{max_page} (learned from previous retrieval)")

        # 3. First attempt
        response = supabase_admin.rpc("match_chunks", params).execute()

        # 4. If no results and page restriction was used → retry without
        if not response.data and ("min_page" in params or "max_page" in params):
            print("No results with page restriction — retrying without page filter...")
            params.pop("min_page", None)
            params.pop("max_page", None)
            response = supabase_admin.rpc("match_chunks", params).execute()

        if not response.data:
            print("No chunks found even after fallback.")
            return []

        # 5. Process results
        results = []
        for row in response.data:
            results.append({
                "content": row["content"],
                "page_number": row.get("page_number", "?"),
                "similarity": row.get("similarity", 0.0),
                "source": row.get("metadata", {}).get("source", "unknown"),
                "section_hint": row.get("metadata", {}).get("section_hint", ""),
                "chunk_id": row.get("id", "?")
            })

        # Sort by similarity & take top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        results = results[:top_k]

        # Debug print
        print("\nRetrieved chunks:")
        for r in results:
            snippet = r['content'][:150].replace("\n", " ").strip()
            print(f"  Page ~{r['page_number']} | Sim {r['similarity']:.3f} | {snippet}...")

        # Learn: update page if high-confidence match (lower threshold for quotes)
        for r in results:
            if r["similarity"] > 0.28 and any(x in r["content"] for x in ["251", "National language", "English language"]):
                LAST_ARTICLE_251_PAGE = r["page_number"]
                print(f"Learned Article 251 is on page ~{LAST_ARTICLE_251_PAGE} for future queries")
                break

        return results

    except Exception as e:
        print(f"Retrieval error for user {user_id}: {str(e)}")
        return []


# ────────────────────────────────────────────────
# Quick test
# ────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "What is the national language?",
        "What does Article 251 say about English?",
        "Quote clause (2) of Article 251 exactly",
        "What are the restrictions on freedom of speech?",
        "Summarize the fundamental rights"
    ]

    for q in test_queries:
        print(f"\n{'='*60}\nQuery: {q}")
        results = retrieve_relevant_chunks(q, top_k=6)
        for i, res in enumerate(results, 1):
            snippet = res["content"][:300].replace("\n", " ").strip()
            print(f"{i}. Page ~{res['page_number']} | Sim {res['similarity']:.3f}")
            print(f"   {snippet}...\n")