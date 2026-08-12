import streamlit as st
from app import load_documents, chunk_text, build_vector_store, search, get_embedding

st.title("Document Q&A - Semantic Search")

@st.cache_resource
def get_vector_store():
    docs = load_documents()
    all_chunks = []
    for filename, text in docs:
        chunks = chunk_text(text, filename)
        all_chunks.extend(chunks)
    collection = build_vector_store(all_chunks)
    return collection, len(docs), len(all_chunks)

collection, num_docs, num_chunks = get_vector_store()
st.write(f"Indexed {num_docs} documents into {num_chunks} chunks.")

query = st.text_input("Ask a question about the policies:")

if query:
    results = search(collection, query, top_k=3)

    for i in range(len(results["documents"][0])):
        chunk_text_result = results["documents"][0][i]
        distance = results["distances"][0][i]
        metadata = results["metadatas"][0][i]

        st.write(f"**Result {i+1}** — source: `{metadata['source']}` — distance: `{distance:.4f}`")
        st.write(chunk_text_result)
        st.write("---")

st.write("---")
st.subheader("Upload a new document")

uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])

if uploaded_file is not None:
    file_text = uploaded_file.read().decode("utf-8")

    new_chunks = chunk_text(file_text, uploaded_file.name)

    for chunk in new_chunks:
        vector = get_embedding(chunk["text"])
        collection.add(
            embeddings=[vector],
            documents=[chunk["text"]],
            metadatas=[{"source": chunk["source"], "chunk_id": chunk["chunk_id"]}],
            ids=[f"{chunk['source']}_{chunk['chunk_id']}"]
        )

    st.success(f"Added {uploaded_file.name} — {len(new_chunks)} chunks indexed.")