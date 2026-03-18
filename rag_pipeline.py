import ollama
from sentence_transformers import SentenceTransformer
from load_documents import load_documents
from chunk_documents import chunk_text
from create_embeddings import create_embeddings
from vector_store import store_vectors
from retriever import retrieve_documents


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Step 1: Load documents
text = load_documents("data")

# Step 2: Chunk documents
chunks = chunk_text(text)

# Step 3: Create embeddings
embeddings = create_embeddings(chunks)

# Step 4: Store embeddings in ChromaDB
collection = store_vectors(chunks, embeddings)


print("RAG System Ready! Ask your questions.")

while True:

    query = input("\nEnter your question: ")

    # Create query embedding
    query_embedding = model.encode(query)

    # Retrieve relevant documents
    docs = retrieve_documents(collection, query_embedding)

    # ✅ Limit context (VERY IMPORTANT)
    top_docs = docs[:2]
    context = " ".join(top_docs)

    # ✅ Strong prompt
    prompt = f"""
You are a healthcare assistant.

Answer the question ONLY based on the context below.
Rules:
- Use simple and correct English
- Fix any spelling or grammar mistakes
- Keep the answer short (2-3 lines)
- Do not merge words incorrectly

Context:
{context}

Question:
{query}
"""

    # Send to LLM
    response = ollama.chat(
        model="phi3:mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("\nAnswer:", response["message"]["content"])


    ##To run RAG_project use python rag_pipeline.py##
