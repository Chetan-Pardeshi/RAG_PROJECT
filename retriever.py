def retrieve_documents(collection, query_embedding, n_results=2):

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )

    return results["documents"][0]