from sentence_transformers import SentenceTransformer
from load_documents import load_documents
from chunk_documents import chunk_text  

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings


if __name__ == "__main__":
    # ✅ Load all text files from 'data' folder
    text = load_documents("data")   # ❌ removed "data.txt"
    
    # ✅ Create chunks
    chunks = chunk_text(text)

    # ✅ Generate embeddings
    embeddings = create_embeddings(chunks)

    print("Number of chunks:", len(chunks))
    print("Embedding size:", len(embeddings[0]))