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
text = load_documents("data.txt")

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

    # Combine context
    context = " ".join(docs)

    # Send to LLM
    response = ollama.chat(
        model="phi3:mini",
        messages=[
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {query}"
            }
        ]
    )

    print("\nAnswer:", response["message"]["content"])