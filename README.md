# ⚖️ Regulatory Expert Consultant

An AI-powered compliance analysis system built with **DSPy**, **Streamlit**, and local RAG. Acts as an expert compliance consultant capable of discovering regulatory frameworks, analyzing documents, building control libraries, and answering compliance questions.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd regulatory_consultant
pip install -r requirements.txt
```

> **System dependency for OCR (optional):** `pdf2image` requires `poppler`:

> - Windows: Download from [poppler releases](https://github.com/oschwartz10612/poppler-windows/releases)
>
> If not installed, the system automatically falls back to **PyMuPDF** for PDF rendering.

### 2. Run the Application

```bash
streamlit run main.py
```

Or directly:

```bash
streamlit run app/ui/streamlit_app.py
```

### 3. Open in Browser

Navigate to: `http://localhost:8501`

---

## 🔑 Configuration

Enter your **OpenAI API key** in the sidebar. The system uses `gpt-4o-mini` for cost efficiency.

No other API keys or external services are required.

---

## 📁 Project Structure

```
regulatory_consultant/
├── main.py                          # Entry point
├── requirements.txt
├── README.md
│
├── app/
│   ├── agent/
│   │   ├── dspy_agent.py            # Main orchestrator
│   │   ├── signatures.py            # DSPy I/O signatures
│   │   ├── modules.py               # DSPy Chain-of-Thought modules
│   │   └── reflection.py            # Quality reflection + agent memory
│   │
│   ├── processing/
│   │   └── pdf_processor.py         # PyPDF + multimodal OCR pipeline
│   │
│   ├── rag/
│   │   └── embeddings.py            # SentenceTransformers + FAISS store
│   │
│   ├── topic_modeling/
│   │   └── bertopic_model.py        # BERTopic + TF-IDF fallback
│   │
│   ├── workbook/
│   │   └── excel_generator.py       # Multi-sheet Excel workbook generator
│   │
│   ├── search/
│   │   ├── duckduckgo_search.py     # Free web search
│   │   └── document_downloader.py  # HTTP document downloader
│   │
│   └── ui/
│       └── streamlit_app.py         # Full Streamlit interface
│
├── data/
│   ├── raw_documents/               # Downloaded + uploaded documents
│   ├── processed_documents/         # Extracted text files
│   └── agent_memory.json            # Persistent agent learning memory
│
├── indexes/
│   ├── faiss_index.bin              # FAISS vector index
│   ├── metadata.json                # Chunk metadata
│   └── topic_model/                 # Saved BERTopic model
│
└── outputs/
    └── workbooks/                   # Generated Excel workbooks
```

---

## 🧠 Architecture

### DSPy Agent Design

The system uses **DSPy** for all LLM interactions, with:

| Signature                   | Purpose                                             |
| --------------------------- | --------------------------------------------------- |
| `IdentifyFrameworks`        | Discovers regulatory frameworks for an industry     |
| `RetrieveDocumentURLs`      | Finds official document sources                     |
| `ParseDocumentStructure`    | Extracts requirements into structured records       |
| `GenerateTopicLabels`       | Labels BERTopic clusters with human-readable themes |
| `BuildControlLibrary`       | Normalizes requirements into a control library      |
| `AnswerComplianceQuestion`  | RAG-grounded Q&A                                    |
| `EvaluateExtractionQuality` | Self-reflection quality scoring                     |
| `RefineExtractionPrompt`    | Improves extractions based on quality feedback      |

All modules use **ChainOfThought** for intermediate reasoning.

### Reflection Loop

After each document extraction:

1. LLM evaluates extraction quality (0-10 score)
2. Identifies specific issues (missing sections, poor formatting, etc.)
3. If score < 7: re-runs extraction with targeted refinements
4. Results logged to `data/reflection_log.json`
5. Quality history tracked per framework in `data/agent_memory.json`

### RAG Pipeline

1. Documents chunked into ~800 character segments with 100-char overlap
2. Embedded using `all-MiniLM-L6-v2` (SentenceTransformers)
3. Stored in local FAISS `IndexFlatIP` (cosine similarity)
4. At query time: retrieve top-6 chunks → feed to DSPy `AnswerComplianceQuestion`

---

## 📋 User Workflow

### Step 1: Framework Discovery

1. Select your industry from the sidebar dropdown
2. Click **"Discover Frameworks"**
3. Agent searches DuckDuckGo + uses GPT-4o-mini to identify applicable regulatory frameworks
4. Results displayed in a sortable table

### Step 2: Choose Analysis Type

- **Build Compliance Workbook** — Generates Excel workbook with framework summaries (fast)
- **Full Document Analysis** — Downloads, parses, and analyzes actual regulatory documents (thorough)
- **Ask Questions** — Jump to RAG Q&A mode

### Step 3: Document Analysis (Full Pipeline)

1. Agent downloads official documents from the web
2. PyPDF extracts text; OCR fallback if text extraction fails
3. DSPy parses documents into structured requirement records
4. Reflection loop evaluates and improves extraction quality
5. BERTopic models themes across all requirements
6. GPT labels topics with compliance-domain names
7. Control library generated from normalized requirements
8. All data indexed in FAISS for Q&A
9. Excel workbook generated with 5 sheets

### Step 4: Q&A

Ask natural language questions:

- _"What are the access control requirements under ISO 27001?"_
- _"How should incidents be classified and reported?"_
- _"What encryption standards are required for data at rest?"_

---

## 📊 Workbook Sheets

| Sheet              | Contents                                                     |
| ------------------ | ------------------------------------------------------------ |
| 📊 Summary         | Dashboard with key statistics                                |
| 📋 Frameworks      | Framework overview with authority, year, and summary         |
| 📄 Requirements    | Extracted requirements with section numbers and topics       |
| 🛡️ Control Library | Normalized controls with test procedures and risk narratives |
| 🔍 Themes          | BERTopic-discovered compliance themes with keywords          |

---

## ⚙️ Configuration Options

Edit `app/agent/dspy_agent.py` to change:

- **LLM model**: Default `gpt-4o-mini` (change to `gpt-4o` for better quality)
- **Max frameworks**: Number of frameworks to analyze in full pipeline
- **Reflection retries**: Quality reflection attempts before accepting extraction

---

## 🔧 Troubleshooting

**"No frameworks identified"**

- Check your API key is valid
- DuckDuckGo search may rate-limit; the system will use LLM knowledge as fallback

**"OCR failed"**

- Install `poppler` for `pdf2image`, or ensure `PyMuPDF` is installed
- The system will still extract text-layer content from PDFs without OCR

**"FAISS index empty"**

- Run document analysis (Tab 2) before using Q&A
- Or upload PDFs using the sidebar uploader

**BERTopic fails**

- System automatically falls back to TF-IDF + KMeans clustering
- Ensure `scikit-learn` is installed

---

## 📦 Technology Stack

| Component        | Library                                    |
| ---------------- | ------------------------------------------ |
| Agent Framework  | DSPy (ChainOfThought, Signatures, Modules) |
| LLM              | OpenAI GPT-4o-mini                         |
| UI               | Streamlit                                  |
| Web Search       | DuckDuckGo Search (free, no API key)       |
| PDF Processing   | PyPDF + pdf2image/PyMuPDF                  |
| OCR              | GPT-4o-mini Vision API                     |
| HTML Parsing     | BeautifulSoup4                             |
| Embeddings       | SentenceTransformers (all-MiniLM-L6-v2)    |
| Vector Store     | FAISS (local, CPU)                         |
| Topic Modeling   | BERTopic (+ TF-IDF fallback)               |
| Data Processing  | pandas, scikit-learn                       |
| Excel Generation | openpyxl                                   |

All data stored **locally** — no external databases.
