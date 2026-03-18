import os

def load_documents(folder_path):
    all_text = ""
    
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                all_text += f.read() + "\n"
    
    return all_text