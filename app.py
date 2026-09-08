import os
import chromadb
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

load_dotenv()
client = OpenAI()

DOCUMENTS_FOLDER = "documents"


def load_documents():
    documents = []

    for folder_path, subfolders, filenames in os.walk(DOCUMENTS_FOLDER):
        for filename in filenames:
            filepath = os.path.join(folder_path, filename)

            if filename.endswith(".txt") or filename.endswith(".md"):
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                documents.append((filename, text))

            elif filename.endswith(".pdf"):
                reader = PdfReader(filepath)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                documents.append((filename, text))

    return documents


def chunk_text(text, filename, max_chunk_size=300, overlap_sentences=1):
    sentences = [s.strip() for s in text.split(".") if s.strip()]

    chunks = []
    chunk_index = 0
    current_chunk_sentences = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length > max_chunk_size and current_chunk_sentences:
            chunk_text_value = ". ".join(current_chunk_sentences) + "."
            chunks.append({
                "text": chunk_text_value,
                "source": filename,
                "chunk_id": chunk_index
            })
            chunk_index += 1

            current_chunk_sentences = current_chunk_sentences[-overlap_sentences:]
            current_length = sum(len(s) for s in current_chunk_sentences)

        current_chunk_sentences.append(sentence)
        current_length += sentence_length

    if current_chunk_sentences:
        chunk_text_value = ". ".join(current_chunk_sentences) + "."
        chunks.append({
            "text": chunk_text_value,
            "source": filename,
            "chunk_id": chunk_index
        })

    return chunks


def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


@st.cache_resource(show_spinner=False)
def build_vector_store():
    docs = load_documents()

    all_chunks = []
    for filename, text in docs:
        chunks = chunk_text(text, filename)
        all_chunks.extend(chunks)

    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="policy_docs")

    for chunk in all_chunks:
        vector = get_embedding(chunk["text"])
        collection.add(
            embeddings=[vector],
            documents=[chunk["text"]],
            metadatas=[{"source": chunk["source"], "chunk_id": chunk["chunk_id"]}],
            ids=[f"{chunk['source']}_{chunk['chunk_id']}"]
        )

    return collection, len(docs), len(all_chunks)


def search(collection, query_text, top_k=3):
    query_vector = get_embedding(query_text)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )
    return results


def generate_answer(results, query_text, distance_threshold=1.5):
    if not results["documents"][0]:
        return "I don't know. No relevant information was found.", []

    top_distance = results["distances"][0][0]
    if top_distance > distance_threshold:
        return "I don't know. No relevant information was found in the documents.", []

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    context = ""
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
        context += f"[Chunk {i+1} - Source: {meta['source']}, chunk {meta['chunk_id']}]\n{chunk}\n\n"

    prompt = f"""You are a helpful assistant answering questions about internal company policy documents.

Use the context below to answer the question as completely as possible.
The user's question may be phrased as a statement or informally (e.g. "work from home policy is available")
instead of a full question - treat it as asking for relevant information on that topic.
Cite the source filename after each fact, like (Source: filename).
Only respond "I don't know" if the context truly does not contain any information relevant to the question.

Context:
{context}

Question: {query_text}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    answer = response.choices[0].message.content
    return answer, chunks


st.set_page_config(page_title="RAG Q&A Bot", page_icon="📄", layout="centered")

st.title("RAG-Powered Q&A Bot")
st.caption("Ask a question about the company policy documents. Answers are grounded in retrieved chunks, with sources cited.")

with st.spinner("Loading documents and building vector store..."):
    collection, num_docs, num_chunks = build_vector_store()

st.success(f"Loaded {num_docs} documents, {num_chunks} chunks indexed.")

query = st.text_input("Your question:", placeholder="What is the company's policy on remote work?")

if st.button("Ask", type="primary"):
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Retrieving relevant chunks and generating answer..."):
            results = search(collection, query, top_k=3)
            answer, used_chunks = generate_answer(results, query)

        st.subheader("Answer")
        st.markdown(answer)

        st.subheader("Retrieved chunks (for transparency)")
        if results["documents"][0]:
            for i in range(len(results["documents"][0])):
                chunk_text_result = results["documents"][0][i]
                distance = results["distances"][0][i]
                metadata = results["metadatas"][0][i]

                with st.expander(f"Result {i+1} — source: {metadata['source']} (distance: {distance:.4f})"):
                    st.write(chunk_text_result)
        else:
            st.write("No chunks retrieved.")