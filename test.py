# ------------------------------------------------------------
# Requirements
# pip install ddgs requests llama-index llama-index-readers-web llama-index-readers-file openai
# ------------------------------------------------------------

import os
from pathlib import Path
from ddgs import DDGS

# Azure OpenAI
from openai import AzureOpenAI

# LlamaIndex Readers
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.file import PDFReader
from llama_index.core import Document


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
# CONFIGURATION
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
# GENERATE SEARCH QUERIES
# ------------------------------------------------------------

def generate_queries(standard):

    queries = []

    for fmt in FILE_FORMATS:
        for template in QUERY_TEMPLATES[:3]:
            queries.append(template.format(standard=standard, format=fmt))

    queries.append(QUERY_TEMPLATES[-1].format(standard=standard))

    return queries


# ------------------------------------------------------------
# SEARCH ENGINE
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
# LLM SELECTION
# ------------------------------------------------------------

def choose_best_document(standard, candidates):

    formatted = "\n".join([
        f"{i+1}. {c['title']} | {c['url']} | {c['snippet']}"
        for i, c in enumerate(candidates)
    ])

    prompt = f"""
You are helping identify the official canonical document for a standard.

Standard: {standard}

Select the ONE URL that most likely represents the PRIMARY specification
or official framework document.

Ignore guides, vendor summaries, mappings, or reference materials.

Return ONLY the URL.

Candidates:
{formatted}
"""

    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "Identify canonical standards documents."},
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
# LLAMAINDEX DOCUMENT LOADING
# ------------------------------------------------------------

def load_document(url):

    # PDF handling
    if url.lower().endswith(".pdf"):

        pdf_path = DOWNLOAD_DIR / url.split("/")[-1]

        import requests
        r = requests.get(url)

        with open(pdf_path, "wb") as f:
            f.write(r.content)

        reader = PDFReader()
        docs = reader.load_data(file=pdf_path)

        return docs

    # HTML / webpage handling
    else:

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

    # limit pool
    unique = unique[:15]

    print("\nCandidates:")
    for c in unique:
        print(c["url"])

    best_url = choose_best_document(standard, unique)

    if not best_url:
        print("\nNo canonical document identified.")
        return None

    print("\nSelected:", best_url)

    docs = load_document(best_url)

    print("\nLoaded documents:", len(docs))

    return docs


# ------------------------------------------------------------
# ENTRY
# ------------------------------------------------------------

if __name__ == "__main__":

    standard = input("Enter standard/framework name: ")

    documents = fetch_standard_document(standard)

    if documents:
        print("\nDocument successfully loaded.")
    else:
        print("\nFailed.")
