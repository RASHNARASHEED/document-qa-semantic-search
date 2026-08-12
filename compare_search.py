from app import load_documents, chunk_text, build_vector_store, search


def keyword_search(chunks, query, top_k=3):
    query_words = query.lower().split()
    
    scored_chunks = []
    for chunk in chunks:
        chunk_text_lower = chunk["text"].lower()
        match_count = sum(1 for word in query_words if word in chunk_text_lower)
        
        if match_count > 0:
            scored_chunks.append((match_count, chunk))
    
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    return scored_chunks[:top_k]


if __name__ == "__main__":
    docs = load_documents()
    all_chunks = []
    for filename, text in docs:
        chunks = chunk_text(text, filename)
        all_chunks.extend(chunks)
    
    print(f"Loaded {len(docs)} documents, {len(all_chunks)} chunks\n")
    
    print("Building semantic vector store...")
    collection = build_vector_store(all_chunks)
    print("Done.\n")
    
    test_queries = [
        "Can I work from home?",
        "How many vacation days do I get?",
        "What happens if I lose my laptop?",
    ]
    
    for query in test_queries:
        print("=" * 60)
        print(f"QUERY: {query}")
        print("=" * 60)
        
        print("\n--- KEYWORD SEARCH ---")
        kw_results = keyword_search(all_chunks, query, top_k=3)
        if not kw_results:
            print("No matches found.")
        else:
            for match_count, chunk in kw_results:
                print(f"[{chunk['source']}] (matched {match_count} words): {chunk['text'][:100]}...")
        
        print("\n--- SEMANTIC SEARCH ---")
        sem_results = search(collection, query, top_k=3)
        for i in range(len(sem_results["documents"][0])):
            source = sem_results["metadatas"][0][i]["source"]
            distance = sem_results["distances"][0][i]
            text = sem_results["documents"][0][i]
            print(f"[{source}] (distance {distance:.4f}): {text[:100]}...")
        
        print()