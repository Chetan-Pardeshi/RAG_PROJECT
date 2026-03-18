import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    text = re.sub(r"[^a-zA-Z0-9., ]", "", text)  # remove special chars
    text = text.replace("suit", "")  # fix weird words if needed
    return text.strip()