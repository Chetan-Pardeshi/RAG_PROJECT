from load_documents import load_documents


def chunk_text(text, chunk_size=200):
    """
    Split text into smaller chunks
    """
    chunks = []
    
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    return chunks


if __name__ == "__main__":
    text = load_documents("data.txt")
    
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:\n{chunk}")