import chromadb
from load_documents import load_documents
from chunk_documents import chunk_text
from create_embeddings import create_embeddings

def store_vectors(chunks, embeddings):

    client = chromadb.Client()

    collection = client.create_collection(name="rag_collection")

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i].tolist()],
            ids=[str(i)]
        )

    return collection


if __name__ == "__main__":

    text = load_documents("data.txt")

    chunks = chunk_text(text)

    embeddings = create_embeddings(chunks)

    collection = store_vectors(chunks, embeddings)

    print("Embeddings stored successfully in ChromaDB")