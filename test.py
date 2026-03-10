# ------------------------------------------------------------
# Requirements
# pip install ddgs requests llama-index llama-index-readers-web llama-index-readers-file openai
# ------------------------------------------------------------

import os
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
# SEARCH WEB
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
# DOWNLOAD USING REQUESTS
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
# PREVIEW EXTRACTION FOR LLM ANALYSIS
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
You must identify the official primary document for the following standard.

Standard: {standard}

Ignore:
- quick start guides
- implementation guides
- vendor documents
- mappings
- blog posts

Return ONLY the URL that corresponds to the primary framework or specification.

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
# LLAMAINDEX DOCUMENT LOADER
# ------------------------------------------------------------

def load_document(url, local_file):

    # Handle PDF
    if local_file.suffix.lower() == ".pdf":

        reader = PDFReader()
        docs = reader.load_data(file=local_file)

        return docs

    # Handle HTML page
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

    # deduplicate
    seen = set()
    unique = []

    for c in candidates:
        if c["url"] not in seen:
            seen.add(c["url"])
            unique.append(c)

    unique = unique[:15]

    print("\nCandidate URLs:")
    for c in unique:
        print(c["url"])

    best_url = choose_document_llm(standard, unique)

    if not best_url:
        print("\nNo canonical document identified.")
        return None

    print("\nSelected document:", best_url)

    local_file = download_file(best_url)

    print("\nDownloaded:", local_file)

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
        print("\nDocument ready for processing.")
    else:
        print("\nFailed to retrieve document.")
