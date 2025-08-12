import os, hashlib, datetime
from pypdf import PdfReader
from bs4 import BeautifulSoup

def read_pdf(path):
    reader = PdfReader(path)
    text = "\n".join([p.extract_text() or "" for p in reader.pages])
    return text

def read_md(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def read_html(path):
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "lxml")
    return soup.get_text(separator="\n")

def load_raw_docs(raw_dir="data/raw"):
    docs = []
    for root, _, files in os.walk(raw_dir):
        for fn in files:
            p = os.path.join(root, fn)
            if fn.lower().endswith(".pdf"):
                text = read_pdf(p)
            elif fn.lower().endswith(".md"):
                text = read_md(p)
            elif fn.lower().endswith((".html", ".htm")):
                text = read_html(p)
            else:
                continue
            docs.append({
                "id": hashlib.md5(p.encode()).hexdigest(),
                "path": p,
                "title": os.path.splitext(fn)[0],
                "text": text,
                "ingested_at": datetime.datetime.utcnow().isoformat()
            })
    return docs