#Healthcare Information Assistant using RAG


def load_documents(file_path):
    """
    Load text data from the file
    """
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()
    
    return data


if __name__ == "__main__":
    documents = load_documents("data.txt")
    print(documents)