# pip install ddgs requests pandas qdrant-client openai \
# llama-index llama-index-readers-web llama-index-readers-file \
# llama-index-vector-stores-qdrant
# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------

import os
import json
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from ddgs import DDGS
from openai import AzureOpenAI

from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.file import PDFReader

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore

from qdrant_client import QdrantClient


# ------------------------------------------------------------
# AZURE OPENAI CONFIG
# ------------------------------------------------------------

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

AZURE_API_VERSION = "2024-02-01"

client = AzureOpenAI(
    api_key=AZURE_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION
)


# ------------------------------------------------------------
# DIRECTORIES
# ------------------------------------------------------------

DOWNLOAD_DIR = Path("downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# VECTOR DATABASE (IN MEMORY QDRANT)
# ------------------------------------------------------------

qdrant_client = QdrantClient(":memory:")

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="standards_collection"
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)


# ------------------------------------------------------------
# SEARCH CONFIG
# ------------------------------------------------------------

FILE_FORMATS = ["pdf", "docx", "doc", "xlsx"]

QUERY_TEMPLATES = [
    '"{standard}" framework filetype:{format}',
    '"{standard}" standard filetype:{format}',
    '"{standard}" specification filetype:{format}',
    '"{standard}" framework'
]


# ------------------------------------------------------------
# GENERATE SEARCH QUERIES
# ------------------------------------------------------------

def generate_queries(standard):

    queries = []

    for fmt in FILE_FORMATS:
        for template in QUERY_TEMPLATES[:3]:
            queries.append(template.format(
                standard=standard,
                format=fmt
            ))

    queries.append(QUERY_TEMPLATES[-1].format(
        standard=standard
    ))

    return queries


# ------------------------------------------------------------
# WEB SEARCH
# ------------------------------------------------------------

def search_web(query, max_results=10):

    urls = []

    with DDGS() as ddgs:

        results = ddgs.text(query, max_results=max_results)

        for r in results:
            urls.append({
                "title": r["title"],
                "url": r["href"],
                "snippet": r.get("body", "")
            })

    return urls


# ------------------------------------------------------------
# DOWNLOAD DOCUMENT
# ------------------------------------------------------------

def download_file(url):

    filename = url.split("/")[-1]

    if not filename or "." not in filename:
        filename = "document.html"

    filepath = DOWNLOAD_DIR / filename

    r = requests.get(url, timeout=30)
    r.raise_for_status()

    with open(filepath, "wb") as f:
        f.write(r.content)

    return filepath


# ------------------------------------------------------------
# PREVIEW TEXT FOR LLM
# ------------------------------------------------------------

def get_preview(url, size=5000):

    try:

        r = requests.get(url, timeout=10)

        content = r.content[:size]

        return content.decode("utf-8", errors="ignore")

    except:
        return ""


# ------------------------------------------------------------
# LLM CHOOSES CANONICAL DOCUMENT
# ------------------------------------------------------------

def choose_document_llm(standard, candidates):

    previews = []

    for c in candidates[:6]:

        preview = get_preview(c["url"])

        previews.append({
            "url": c["url"],
            "title": c["title"],
            "preview": preview[:800]
        })

    prompt = f"""
Identify the official primary document for the standard.

Standard: {standard}

Ignore:
- quick start guides
- vendor documentation
- blogs
- implementation guides

Return ONLY the URL.

Candidates:
{previews}
"""

    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You identify canonical standards documents."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    answer = response.choices[0].message.content.strip()

    for c in candidates:
        if c["url"] in answer:
            return c["url"]

    return None


# ------------------------------------------------------------
# LOAD DOCUMENT
# ------------------------------------------------------------

def load_document(url, local_file):

    if local_file.suffix.lower() == ".pdf":

        reader = PDFReader()

        docs = reader.load_data(file=local_file)

        return docs

    reader = SimpleWebPageReader(html_to_text=True)

    docs = reader.load_data(urls=[url])

    return docs


# ------------------------------------------------------------
# BUILD VECTOR INDEX
# ------------------------------------------------------------

def build_index(documents):

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )

    return index


# ------------------------------------------------------------
# RETRIEVE CHUNKS
# ------------------------------------------------------------

def retrieve_chunks(index, query, k=10):

    retriever = index.as_retriever(
        similarity_top_k=k
    )

    nodes = retriever.retrieve(query)

    return [n.text for n in nodes]


# ------------------------------------------------------------
# REQUIREMENT EXTRACTION
# ------------------------------------------------------------

def extract_requirements_llm(standard, chunk):

    prompt = f"""
Extract compliance requirements from the text.

Return JSON list.

Format:

[
 {{
  "source_name": "{standard}",
  "topic": "...",
  "sub_topic": "...",
  "section_number": "...",
  "requirement": "..."
 }}
]

Text:
{chunk}
"""

    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "Extract structured compliance requirements."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    text = response.choices[0].message.content.strip()

    try:
        return json.loads(text)
    except:
        return []


# ------------------------------------------------------------
# PARALLEL EXTRACTION
# ------------------------------------------------------------

def extract_from_chunks(standard, chunks):

    results = []

    with ThreadPoolExecutor(max_workers=4) as executor:

        futures = []

        for c in chunks:
            futures.append(
                executor.submit(
                    extract_requirements_llm,
                    standard,
                    c
                )
            )

        for f in futures:
            try:
                results.extend(f.result())
            except:
                pass

    return results


# ------------------------------------------------------------
# DEDUPLICATE REQUIREMENTS
# ------------------------------------------------------------

def deduplicate_requirements(data):

    seen = set()

    unique = []

    for r in data:

        key = r.get("requirement", "").strip()

        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique


# ------------------------------------------------------------
# EXPORT TO EXCEL
# ------------------------------------------------------------

def export_to_excel(data, filename="requirements.xlsx"):

    if not data:
        print("No requirements extracted.")
        return

    df = pd.DataFrame(data)

    df.to_excel(filename, index=False)

    print("Excel created:", filename)


# ------------------------------------------------------------
# PIPELINE
# ------------------------------------------------------------

def fetch_standard_document(standard):

    queries = generate_queries(standard)

    candidates = []

    for q in queries:

        print("Searching:", q)

        candidates.extend(search_web(q))

    seen = set()
    unique = []

    for c in candidates:
        if c["url"] not in seen:
            seen.add(c["url"])
            unique.append(c)

    unique = unique[:15]

    print("\nCandidate URLs")

    for u in unique:
        print(u["url"])

    best_url = choose_document_llm(standard, unique)

    if not best_url:
        print("No canonical document found.")
        return

    print("\nSelected:", best_url)

    local_file = download_file(best_url)

    docs = load_document(best_url, local_file)

    print("Documents loaded:", len(docs))

    index = build_index(docs)

    queries = [
        "security requirements",
        "compliance requirements",
        "control requirements",
        "mandatory requirements",
        "policy requirements"
    ]

    chunks = []

    for q in queries:
        chunks.extend(retrieve_chunks(index, q))

    chunks = list(set(chunks))

    print("Retrieved chunks:", len(chunks))

    requirements = extract_from_chunks(
        standard,
        chunks
    )

    requirements = deduplicate_requirements(
        requirements
    )

    print("Extracted requirements:", len(requirements))

    with open("requirements.json", "w") as f:
        json.dump(requirements, f, indent=2)

    export_to_excel(requirements)

    return requirements


# ------------------------------------------------------------
# ENTRY
# ------------------------------------------------------------

if __name__ == "__main__":

    standard = input("Enter standard/framework name: ")

    fetch_standard_document(standard)

    print("\nPipeline completed.")
