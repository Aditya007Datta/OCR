"""
Main DSPy Agent Orchestrator for the Regulatory Expert Consultant.
Coordinates all modules and manages the full analysis workflow.
"""

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

import dspy

from app.agent.modules import (
    ComplianceQAEngine,
    ControlLibraryBuilder,
    DocumentStructureParser,
    DocumentURLRetriever,
    ExtractionQualityReflector,
    FrameworkIdentifier,
    TopicLabelGenerator,
)
from app.agent.reflection import AgentMemory, ReflectionLogger
from app.processing.pdf_processor import PDFProcessor
from app.rag.embeddings import EmbeddingStore
from app.search.duckduckgo_search import DuckDuckGoSearcher
from app.search.document_downloader import DocumentDownloader
from app.topic_modeling.bertopic_model import TopicModeler
from app.workbook.excel_generator import ExcelGenerator

logger = logging.getLogger(__name__)


class RegulatoryAgent:
    """
    Main agent that orchestrates the full regulatory compliance analysis workflow.
    Uses DSPy modules for LLM reasoning and reflection for self-improvement.
    """

    def __init__(self, api_key: str, log_callback: Optional[Callable[[str], None]] = None):
        """
        Initialize the agent with an OpenAI API key.

        Args:
            api_key: OpenAI API key
            log_callback: Optional callback for streaming logs to the UI
        """
        self.api_key = api_key
        self.log_callback = log_callback or (lambda msg: None)
        self.session_id = str(uuid.uuid4())[:8]

        # Configure DSPy LLM
        self._configure_dspy(api_key)

        # Initialize all modules
        self.framework_identifier = FrameworkIdentifier()
        self.url_retriever = DocumentURLRetriever()
        self.doc_parser = DocumentStructureParser()
        self.topic_generator = TopicLabelGenerator()
        self.control_builder = ControlLibraryBuilder()
        self.qa_engine = ComplianceQAEngine()
        self.reflector = ExtractionQualityReflector(max_retries=2)

        # Initialize tools
        self.searcher = DuckDuckGoSearcher()
        self.downloader = DocumentDownloader()
        self.pdf_processor = PDFProcessor(api_key=api_key)
        self.embedding_store = EmbeddingStore()
        self.topic_modeler = TopicModeler()
        self.excel_generator = ExcelGenerator()

        # Memory and reflection
        self.memory = AgentMemory()
        self.reflection_logger = ReflectionLogger()

        # Runtime state
        self.current_frameworks: list[dict] = []
        self.current_industry: str = ""
        self.current_records: list[dict] = []
        self.current_controls: list[dict] = []
        self.current_topics: list[dict] = []

    def _configure_dspy(self, api_key: str):
        """Configure DSPy with OpenAI."""
        os.environ["OPENAI_API_KEY"] = api_key
        lm = dspy.LM(model="openai/gpt-4o-mini", api_key=api_key, max_tokens=4000)
        dspy.configure(lm=lm)
        logger.info("DSPy configured with GPT-4o-mini")

    def _log(self, message: str, level: str = "INFO"):
        """Log to both Python logger and UI callback."""
        if level == "INFO":
            logger.info(message)
        elif level == "WARNING":
            logger.warning(message)
        elif level == "ERROR":
            logger.error(message)
        self.log_callback(message)

    # ─────────────────────────────────────────────
    # STEP 1: Framework Discovery
    # ─────────────────────────────────────────────

    def identify_frameworks(self, industry: str) -> list[dict]:
        """
        Step 1: Identify regulatory frameworks for a given industry.
        Uses web search + LLM reasoning.
        """
        self._log(f"🔍 Searching for regulatory frameworks applicable to: {industry}")
        self.current_industry = industry

        # Web search for context
        self._log("📡 Querying DuckDuckGo for regulatory landscape...")
        search_query = f"regulatory compliance frameworks standards {industry} industry requirements 2024"
        search_results = self.searcher.search(search_query, max_results=8)
        search_text = self._format_search_results(search_results)

        # LLM identification
        self._log("🧠 Analyzing search results with DSPy FrameworkIdentifier...")
        result = self.framework_identifier(industry=industry, search_results=search_text)
        frameworks = result.get("frameworks", [])

        if not frameworks:
            self._log("⚠️  LLM returned no frameworks, using fallback search...", "WARNING")
            frameworks = self._fallback_framework_search(industry)

        self.current_frameworks = frameworks
        self._log(f"✅ Identified {len(frameworks)} regulatory frameworks")
        self.memory.record_session(industry, [f["name"] for f in frameworks], self.session_id)

        return frameworks

    def _fallback_framework_search(self, industry: str) -> list[dict]:
        """Fallback hardcoded frameworks for common industries."""
        common_frameworks = {
            "fintech": [
                {"name": "PCI-DSS", "year": 2004, "authority": "PCI Security Standards Council",
                 "industries": ["Fintech", "Payments", "Banking"],
                 "summary": "Payment Card Industry Data Security Standard protects cardholder data through technical and operational requirements. Applicable to all entities that store, process, or transmit payment card data. Version 4.0 released in 2022 with enhanced flexibility and continuous compliance focus. Mandatory for merchants and service providers handling card payments."},
                {"name": "GDPR", "year": 2018, "authority": "European Union",
                 "industries": ["Fintech", "All EU businesses"],
                 "summary": "General Data Protection Regulation governs data privacy and protection for EU residents. Establishes rights for individuals and obligations for data controllers and processors. Imposes significant penalties for non-compliance up to 4% of global annual turnover. Applies to any organization processing EU resident data regardless of location."},
            ],
            "healthcare": [
                {"name": "HIPAA", "year": 1996, "authority": "U.S. Department of Health & Human Services",
                 "industries": ["Healthcare", "Health Insurance", "Health IT"],
                 "summary": "Health Insurance Portability and Accountability Act protects patient health information privacy and security. Establishes standards for electronic health data transactions and code sets. The Security Rule requires administrative, physical, and technical safeguards. Covered entities and business associates must comply with all applicable provisions."},
            ],
        }
        return common_frameworks.get(industry.lower(), [])

    def _format_search_results(self, results: list[dict]) -> str:
        """Format search results into a text string for the LLM."""
        if not results:
            return "No search results available."
        lines = []
        for r in results[:6]:
            lines.append(f"Title: {r.get('title', '')}")
            lines.append(f"URL: {r.get('url', '')}")
            lines.append(f"Snippet: {r.get('snippet', '')}")
            lines.append("---")
        return "\n".join(lines)

    # ─────────────────────────────────────────────
    # STEP 2: Document Retrieval
    # ─────────────────────────────────────────────

    def retrieve_documents(self, framework_names: list[str]) -> dict[str, list[str]]:
        """
        Step 2: Find and download source documents for selected frameworks.
        Returns dict mapping framework name to list of local file paths.
        """
        downloaded = {}

        for framework_name in framework_names:
            self._log(f"🌐 Searching for official documents: {framework_name}")

            # Search for document URLs
            search_query = f"{framework_name} official PDF download compliance standard document"
            search_results = self.searcher.search(search_query, max_results=5)
            search_text = self._format_search_results(search_results)

            # LLM extracts best URLs
            urls = self.url_retriever(framework_name=framework_name, search_results=search_text)
            self._log(f"📎 Found {len(urls)} candidate URLs for {framework_name}")

            local_paths = []
            for url in urls[:3]:  # Limit to top 3 URLs
                self._log(f"⬇️  Downloading: {url[:80]}...")
                try:
                    local_path = self.downloader.download(url, framework_name)
                    if local_path:
                        local_paths.append(local_path)
                        self._log(f"✅ Saved to: {local_path}")
                except Exception as e:
                    self._log(f"⚠️  Download failed for {url[:60]}: {e}", "WARNING")

            downloaded[framework_name] = local_paths

        return downloaded

    # ─────────────────────────────────────────────
    # STEP 3: Document Processing
    # ─────────────────────────────────────────────

    def process_document(self, file_path: str, framework_name: str) -> dict[str, Any]:
        """
        Step 3: Process a document file (PDF or HTML) into structured records.
        Includes reflection loop for quality improvement.
        """
        self._log(f"📄 Processing document: {Path(file_path).name}")

        # Extract text
        self._log("🔤 Extracting text from document...")
        text = self.pdf_processor.extract_text(file_path)

        if not text or len(text) < 200:
            self._log("⚠️  Insufficient text, falling back to OCR pipeline...", "WARNING")
            text = self.pdf_processor.ocr_extract(file_path)

        self._log(f"✅ Extracted {len(text)} characters of text")

        # Parse document structure
        self._log("🏗️  Parsing document structure with DSPy...")
        records = self.doc_parser(document_text=text, framework_name=framework_name)
        self._log(f"📋 Initial extraction: {len(records)} requirement records")

        # Reflection loop
        self._log("🔄 Running quality reflection loop...")
        reflection_result = self.reflector(original_text=text, extracted_records=records)
        records = reflection_result["records"]
        quality_score = reflection_result["quality_score"]
        issues = reflection_result.get("issues", [])

        self.reflection_logger.log_reflection(
            step=f"document_parsing_{framework_name}",
            original_score=quality_score,
            final_score=quality_score,
            iterations=reflection_result["iterations"],
            improved=reflection_result["improved"],
            issues=issues,
        )
        self.memory.record_extraction_quality(framework_name, quality_score, issues)
        self.memory.record_document_processed(framework_name, file_path, len(records))

        self._log(f"✅ Extraction quality score: {quality_score}/10 | Records: {len(records)}")
        self.current_records.extend(records)

        return {
            "text": text,
            "records": records,
            "quality_score": quality_score,
            "framework": framework_name,
        }

    # ─────────────────────────────────────────────
    # STEP 4: Topic Modeling
    # ─────────────────────────────────────────────

    def run_topic_modeling(self, records: Optional[list[dict]] = None) -> list[dict]:
        """
        Step 4: Run BERTopic on document chunks to discover themes.
        """
        records = records or self.current_records
        if not records:
            self._log("⚠️  No records to model topics on", "WARNING")
            return []

        self._log(f"🧩 Running BERTopic on {len(records)} requirement records...")
        texts = [
            f"{r.get('topic', '')} {r.get('requirement_text', '')} {r.get('requirement_summary', '')}"
            for r in records
        ]
        texts = [t.strip() for t in texts if t.strip()]

        topic_results = self.topic_modeler.fit(texts)
        self._log(f"✅ Discovered {len(topic_results['topic_keywords'])} topic clusters")

        # Generate human-readable labels with LLM
        self._log("🏷️  Generating topic labels with DSPy...")
        framework_context = self.current_industry
        labeled_topics = self.topic_generator(
            topic_keywords=topic_results["topic_keywords"],
            framework_context=framework_context
        )

        # Merge BERTopic results with LLM labels
        topics = self._merge_topic_results(topic_results, labeled_topics, records)
        self.current_topics = topics
        self._log(f"✅ Topic modeling complete: {len(topics)} labeled themes")

        return topics

    def _merge_topic_results(
        self,
        topic_results: dict,
        labeled_topics: list[dict],
        records: list[dict]
    ) -> list[dict]:
        """Merge BERTopic results with LLM-generated labels."""
        merged = []
        label_map = {t["topic_id"]: t for t in labeled_topics}

        for topic_id, keywords in topic_results["topic_keywords"].items():
            label_data = label_map.get(topic_id, {})
            # Find associated sections
            associated = [
                r.get("section_number", "")
                for r, tid in zip(records, topic_results.get("doc_topics", []))
                if tid == topic_id
            ]
            merged.append({
                "topic_id": topic_id,
                "label": label_data.get("label", f"Topic {topic_id}"),
                "description": label_data.get("description", ""),
                "theme_category": label_data.get("theme_category", "Other"),
                "keywords": keywords[:10],
                "associated_sections": list(set(filter(None, associated)))[:10],
            })

        return merged

    # ─────────────────────────────────────────────
    # STEP 5: Control Library
    # ─────────────────────────────────────────────

    def build_control_library(
        self,
        records: Optional[list[dict]] = None,
        topics: Optional[list[dict]] = None,
        framework_name: str = "Multiple Frameworks"
    ) -> list[dict]:
        """
        Step 5: Build a normalized compliance control library.
        """
        records = records or self.current_records
        topics = topics or self.current_topics

        self._log(f"🏛️  Building control library from {len(records)} requirements...")
        controls = self.control_builder(
            structured_records=records,
            topic_labels=topics,
            framework_name=framework_name
        )

        self.current_controls = controls
        self.memory.record_controls_generated(len(controls))
        self._log(f"✅ Generated {len(controls)} compliance controls")

        return controls

    # ─────────────────────────────────────────────
    # STEP 6: Workbook Generation
    # ─────────────────────────────────────────────

    def generate_workbook(
        self,
        frameworks: Optional[list[dict]] = None,
        records: Optional[list[dict]] = None,
        controls: Optional[list[dict]] = None,
        topics: Optional[list[dict]] = None,
        output_name: str = "compliance_workbook"
    ) -> str:
        """
        Step 6: Generate an Excel compliance workbook.
        Returns the path to the generated file.
        """
        frameworks = frameworks or self.current_frameworks
        records = records or self.current_records
        controls = controls or self.current_controls
        topics = topics or self.current_topics

        self._log("📊 Generating Excel compliance workbook...")
        output_path = self.excel_generator.generate(
            frameworks=frameworks,
            records=records,
            controls=controls,
            topics=topics,
            output_name=output_name,
        )
        self._log(f"✅ Workbook saved: {output_path}")
        return output_path

    # ─────────────────────────────────────────────
    # STEP 7: RAG Indexing & Q&A
    # ─────────────────────────────────────────────

    def index_documents(self, records: Optional[list[dict]] = None):
        """Index document records into the FAISS vector store."""
        records = records or self.current_records
        if not records:
            self._log("⚠️  No records to index", "WARNING")
            return

        self._log(f"🗂️  Indexing {len(records)} records into FAISS vector store...")
        texts = [
            f"[{r.get('source', '')} | {r.get('section_number', '')}] "
            f"{r.get('requirement_text', '')} {r.get('requirement_summary', '')}"
            for r in records
        ]
        self.embedding_store.add_documents(texts, metadata=records)
        self._log(f"✅ Vector store indexed with {len(texts)} chunks")

    def answer_question(self, question: str) -> str:
        """Answer a compliance question using RAG."""
        self._log(f"❓ Processing question: {question[:80]}...")

        # Retrieve relevant chunks
        chunks = self.embedding_store.search(question, k=6)
        if not chunks:
            self._log("⚠️  No relevant documents found in index", "WARNING")
            context = "No documents have been indexed yet. Please analyze documents first."
        else:
            context_parts = []
            for chunk in chunks:
                meta = chunk.get("metadata", {})
                context_parts.append(
                    f"[{meta.get('source', 'Unknown')} | Section {meta.get('section_number', 'N/A')}]\n"
                    f"{chunk.get('text', '')}"
                )
            context = "\n\n".join(context_parts)

        self._log("🧠 Generating answer with DSPy ComplianceQAEngine...")
        answer = self.qa_engine(question=question, retrieved_context=context)
        self._log("✅ Answer generated")
        return answer

    # ─────────────────────────────────────────────
    # Full Workflow
    # ─────────────────────────────────────────────

    def run_full_analysis(
        self,
        industry: str,
        framework_names: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Run the complete analysis workflow end-to-end.
        """
        self._log(f"🚀 Starting full analysis for: {industry}")
        self._log("=" * 60)

        # Step 1: Identify frameworks
        frameworks = self.identify_frameworks(industry)
        if framework_names:
            frameworks = [f for f in frameworks if f["name"] in framework_names]

        # Step 2: Retrieve documents
        if frameworks:
            framework_list = [f["name"] for f in frameworks[:3]]  # Top 3 frameworks
            downloaded = self.retrieve_documents(framework_list)
        else:
            downloaded = {}

        # Step 3: Process documents
        all_records = []
        for framework_name, paths in downloaded.items():
            for path in paths[:2]:  # Process up to 2 docs per framework
                try:
                    result = self.process_document(path, framework_name)
                    all_records.extend(result["records"])
                except Exception as e:
                    self._log(f"⚠️  Error processing {path}: {e}", "WARNING")

        self.current_records = all_records

        # Step 4: Topic modeling
        if all_records:
            topics = self.run_topic_modeling(all_records)
        else:
            topics = []

        # Step 5: Control library
        if all_records:
            controls = self.build_control_library(all_records, topics, industry)
        else:
            controls = []

        # Step 6: Index for RAG
        if all_records:
            self.index_documents(all_records)

        # Step 7: Generate workbook
        workbook_path = self.generate_workbook(
            frameworks=frameworks,
            records=all_records,
            controls=controls,
            topics=topics,
            output_name=f"{industry.lower().replace(' ', '_')}_compliance"
        )

        self._log("=" * 60)
        self._log("🎉 Full analysis complete!")
        self._log(f"   Frameworks identified: {len(frameworks)}")
        self._log(f"   Documents processed:   {len(downloaded)}")
        self._log(f"   Requirements extracted: {len(all_records)}")
        self._log(f"   Topics discovered:     {len(topics)}")
        self._log(f"   Controls generated:    {len(controls)}")
        self._log(f"   Workbook:              {workbook_path}")

        return {
            "frameworks": frameworks,
            "records": all_records,
            "topics": topics,
            "controls": controls,
            "workbook_path": workbook_path,
            "stats": self.memory.get_stats(),
        }
