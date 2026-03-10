# ------------------------------------------------------------
# Requirements
# pip install ddgs requests llama-index llama-index-readers-web llama-index-readers-file openai
# Ensure wget installed (winget install GnuWin32.Wget)
# ------------------------------------------------------------

import os
import subprocess
import requests
from pathlib import Path
from ddgs import DDGS
from openai import AzureOpenAI

# LlamaIndex readers
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.file import PDFReader


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
# CONFIG
# ------------------------------------------------------------

DOWNLOAD_DIR = Path("downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)

FILE_FORMATS = ["pdf", "docx", "doc", "xlsx", "xls"]

QUERY_TEMPLATES = [
    '"{standard}" framework filetype:{format}',
    '"{standard}" standard filetype:{format}',
    '"{standard}" specification filetype:{format}',
    '"{standard}" framework'
]


# ------------------------------------------------------------
# QUERY GENERATION
# ------------------------------------------------------------

def generate_queries(standard):

    queries = []

    for fmt in FILE_FORMATS:
        for template in QUERY_TEMPLATES[:3]:
            queries.append(template.format(standard=standard, format=fmt))

    queries.append(QUERY_TEMPLATES[-1].format(standard=standard))

    return queries


# ------------------------------------------------------------
# SEARCH
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
# LLM SELECTS CANONICAL DOCUMENT
# ------------------------------------------------------------

def choose_best_document(standard, candidates):

    formatted = "\n".join([
        f"{i+1}. {c['title']} | {c['url']} | {c['snippet']}"
        for i, c in enumerate(candidates)
    ])

    prompt = f"""
Select the URL that is the official primary document for the standard.

Standard: {standard}

Ignore:
- quick start guides
- mappings
- vendor whitepapers
- summaries

Return ONLY the URL.

Candidates:
{formatted}
"""

    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "Identify official standards documents."},
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
# DOWNLOAD WITH WGET
# ------------------------------------------------------------

def wget_download(url):

    filename = url.split("/")[-1]

    if not filename:
        filename = "document.html"

    filepath = DOWNLOAD_DIR / filename

    subprocess.run([
        "wget",
        "--timeout=30",
        "--tries=3",
        "--user-agent=Mozilla/5.0",
        "-O",
        str(filepath),
        url
    ], check=True)

    return filepath


# ------------------------------------------------------------
# LOAD WITH LLAMAINDEX
# ------------------------------------------------------------

def load_document(url, local_path=None):

    # If PDF downloaded locally
    if local_path and local_path.suffix == ".pdf":

        reader = PDFReader()
        docs = reader.load_data(file=local_path)

        return docs

    # HTML page
    reader = SimpleWebPageReader(html_to_text=True)
    docs = reader.load_data(urls=[url])

    return docs


# ------------------------------------------------------------
# PIPELINE
# ------------------------------------------------------------

def fetch_standard_document(standard):

    queries = generate_queries(standard)

    candidates = []

    for q in queries:

        print("\nSearching:", q)

        results = search_web(q)

        candidates.extend(results)

    # remove duplicates
    seen = set()
    unique = []

    for c in candidates:
        if c["url"] not in seen:
            seen.add(c["url"])
            unique.append(c)

    unique = unique[:15]

    print("\nCandidates:")
    for c in unique:
        print(c["url"])

    best_url = choose_best_document(standard, unique)

    if not best_url:
        print("\nNo canonical document identified.")
        return None

    print("\nSelected:", best_url)

    # Download document using wget
    local_file = wget_download(best_url)

    print("\nDownloaded to:", local_file)

    # Load using LlamaIndex
    docs = load_document(best_url, local_file)

    print("\nLoaded documents:", len(docs))

    return docs


# ------------------------------------------------------------
# ENTRY
# ------------------------------------------------------------

if __name__ == "__main__":

    standard = input("Enter standard/framework name: ")

    documents = fetch_standard_document(standard)

    if documents:
        print("\nDocument ready for indexing or RAG.")
    else:
        print("\nFailed.")
