from sentence_transformers import SentenceTransformer
from load_documents import load_documents
from chunk_documents import chunk_text  

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings


if __name__ == "__main__":
    text = load_documents("data.txt")
    chunks = chunk_text(text)

    embeddings = create_embeddings(chunks)

    print("Number of chunks:", len(chunks))
    print("Embedding size:", len(embeddings[0]))