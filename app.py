import os
import chromadb
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI

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


def build_vector_store(chunks):
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="policy_docs")
    
    for chunk in chunks:
        vector = get_embedding(chunk["text"])
        
        collection.add(
            embeddings=[vector],
            documents=[chunk["text"]],
            metadatas=[{"source": chunk["source"], "chunk_id": chunk["chunk_id"]}],
            ids=[f"{chunk['source']}_{chunk['chunk_id']}"]
        )
    
    return collection

def search(collection, query_text, top_k=3):
    query_vector = get_embedding(query_text)
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )
    
    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    
    all_chunks = []
    for filename, text in docs:
        chunks = chunk_text(text, filename)
        all_chunks.extend(chunks)
    
    print(f"\nTotal chunks created: {len(all_chunks)}")
    
    print("\nBuilding vector store (embedding all chunks)...")
    collection = build_vector_store(all_chunks)
    print(f"Vector store built. Total items stored: {collection.count()}")
    
    # Test search
    # Interactive search
    while True:
        test_query = input("\nEnter your question (or type 'exit' to quit): ")
        
        if test_query.lower() == "exit":
            break
        
        results = search(collection, test_query, top_k=3)
        
        for i in range(len(results["documents"][0])):
            chunk_text_result = results["documents"][0][i]
            distance = results["distances"][0][i]
            metadata = results["metadatas"][0][i]
            
            print(f"\nResult {i+1} (source: {metadata['source']}, distance: {distance:.4f})")
            print(f"Text: {chunk_text_result}")