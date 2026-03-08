"""
DSPy Modules for the Regulatory Expert Consultant.
Each module wraps a signature with additional logic, chain-of-thought, or multi-step reasoning.
"""

import json
import logging
from typing import Any

import dspy

from app.agent.signatures import (
    AnswerComplianceQuestion,
    BuildControlLibrary,
    EvaluateExtractionQuality,
    GenerateTopicLabels,
    IdentifyFrameworks,
    ParseDocumentStructure,
    RefineExtractionPrompt,
    RetrieveDocumentURLs,
    SummarizeWorkflowStep,
)

logger = logging.getLogger(__name__)


class FrameworkIdentifier(dspy.Module):
    """Identifies regulatory frameworks relevant to a given industry."""

    def __init__(self):
        super().__init__()
        self.identify = dspy.ChainOfThought(IdentifyFrameworks)

    def forward(self, industry: str, search_results: str = "") -> dict[str, Any]:
        logger.info(f"Identifying frameworks for industry: {industry}")
        result = self.identify(industry=industry, search_results=search_results)
        try:
            frameworks = json.loads(result.frameworks_json)
            return {"frameworks": frameworks, "reasoning": result.rationale}
        except json.JSONDecodeError:
            logger.warning("Failed to parse frameworks JSON, returning raw")
            return {"frameworks": [], "raw": result.frameworks_json, "reasoning": getattr(result, "rationale", "")}


class DocumentURLRetriever(dspy.Module):
    """Finds official source URLs for regulatory framework documents."""

    def __init__(self):
        super().__init__()
        self.retrieve = dspy.ChainOfThought(RetrieveDocumentURLs)

    def forward(self, framework_name: str, search_results: str) -> list[str]:
        logger.info(f"Retrieving document URLs for: {framework_name}")
        result = self.retrieve(framework_name=framework_name, search_results=search_results)
        try:
            urls = json.loads(result.urls)
            return urls if isinstance(urls, list) else []
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse URLs for {framework_name}")
            return []


class DocumentStructureParser(dspy.Module):
    """Parses regulatory documents into structured requirement records."""

    def __init__(self):
        super().__init__()
        self.parse = dspy.ChainOfThought(ParseDocumentStructure)

    def forward(self, document_text: str, framework_name: str) -> list[dict]:
        logger.info(f"Parsing document structure for: {framework_name}")
        # Process in chunks if document is large
        max_chars = 12000
        if len(document_text) > max_chars:
            logger.info(f"Document is large ({len(document_text)} chars), processing in chunks")
            return self._parse_in_chunks(document_text, framework_name, max_chars)

        result = self.parse(document_text=document_text, framework_name=framework_name)
        return self._safe_parse_json(result.structured_records, [])

    def _parse_in_chunks(self, text: str, framework_name: str, chunk_size: int) -> list[dict]:
        """Parse a large document in overlapping chunks."""
        all_records = []
        words = text.split()
        chunk_words = chunk_size // 6  # Approximate words per chunk

        for i in range(0, len(words), chunk_words - 200):  # 200-word overlap
            chunk = " ".join(words[i: i + chunk_words])
            if not chunk.strip():
                continue
            try:
                result = self.parse(document_text=chunk, framework_name=framework_name)
                records = self._safe_parse_json(result.structured_records, [])
                all_records.extend(records)
            except Exception as e:
                logger.error(f"Error parsing chunk: {e}")

        return self._deduplicate_records(all_records)

    def _deduplicate_records(self, records: list[dict]) -> list[dict]:
        """Remove duplicate records based on requirement text."""
        seen = set()
        unique = []
        for r in records:
            key = r.get("requirement_text", "")[:100]
            if key and key not in seen:
                seen.add(key)
                unique.append(r)
        return unique

    def _safe_parse_json(self, text: str, default: Any) -> Any:
        try:
            # Strip markdown code blocks if present
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[1:-1])
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("JSON parse failed, returning default")
            return default


class TopicLabelGenerator(dspy.Module):
    """Generates human-readable labels for topic model clusters."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateTopicLabels)

    def forward(self, topic_keywords: dict, framework_context: str) -> list[dict]:
        logger.info("Generating topic labels")
        result = self.generate(
            topic_keywords=json.dumps(topic_keywords),
            framework_context=framework_context
        )
        try:
            cleaned = result.topic_labels.strip()
            if cleaned.startswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[1:-1])
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return []


class ControlLibraryBuilder(dspy.Module):
    """Builds a normalized compliance control library from parsed requirements."""

    def __init__(self):
        super().__init__()
        self.build = dspy.ChainOfThought(BuildControlLibrary)

    def forward(self, structured_records: list[dict], topic_labels: list[dict], framework_name: str) -> list[dict]:
        logger.info(f"Building control library for: {framework_name}")

        # Process in batches of 20 records
        batch_size = 20
        all_controls = []

        for i in range(0, len(structured_records), batch_size):
            batch = structured_records[i: i + batch_size]
            try:
                result = self.build(
                    structured_records=json.dumps(batch),
                    topic_labels=json.dumps(topic_labels[:10]),  # Top 10 topics for context
                    framework_name=framework_name
                )
                cleaned = result.control_library.strip()
                if cleaned.startswith("```"):
                    cleaned = "\n".join(cleaned.split("\n")[1:-1])
                controls = json.loads(cleaned)
                all_controls.extend(controls)
            except Exception as e:
                logger.error(f"Error building controls for batch {i}: {e}")

        return all_controls


class ComplianceQAEngine(dspy.Module):
    """Answers compliance questions using RAG-retrieved context."""

    def __init__(self):
        super().__init__()
        self.answer = dspy.ChainOfThought(AnswerComplianceQuestion)

    def forward(self, question: str, retrieved_context: str) -> str:
        logger.info(f"Answering compliance question: {question[:80]}...")
        result = self.answer(question=question, retrieved_context=retrieved_context)
        return result.answer


class ExtractionQualityReflector(dspy.Module):
    """
    Reflection module: evaluates extraction quality and refines if needed.
    This enables the agent's self-improvement loop.
    """

    def __init__(self, max_retries: int = 2):
        super().__init__()
        self.evaluate = dspy.ChainOfThought(EvaluateExtractionQuality)
        self.refine = dspy.ChainOfThought(RefineExtractionPrompt)
        self.max_retries = max_retries

    def forward(self, original_text: str, extracted_records: list[dict]) -> dict[str, Any]:
        """
        Evaluate extraction quality and optionally refine.
        Returns dict with: records, quality_score, iterations, improved
        """
        current_records = extracted_records
        text_sample = original_text[:3000]  # Use first 3000 chars as sample

        for iteration in range(self.max_retries):
            logger.info(f"Quality reflection iteration {iteration + 1}/{self.max_retries}")

            # Evaluate current extraction
            eval_result = self.evaluate(
                original_text_sample=text_sample,
                extracted_records=json.dumps(current_records[:5])  # Sample of records
            )

            try:
                quality = json.loads(eval_result.quality_score.strip())
                score = quality.get("score", 5)
                should_retry = quality.get("should_retry", False)
                issues = quality.get("issues", [])

                logger.info(f"Extraction quality score: {score}/10")

                if not should_retry or score >= 7:
                    return {
                        "records": current_records,
                        "quality_score": score,
                        "iterations": iteration + 1,
                        "improved": iteration > 0,
                        "issues": issues
                    }

                # Refine if quality is low
                logger.info(f"Refining extraction due to issues: {issues}")
                refine_result = self.refine(
                    original_records=json.dumps(current_records[:10]),
                    quality_issues=json.dumps(issues),
                    document_sample=text_sample
                )

                cleaned = refine_result.refined_records.strip()
                if cleaned.startswith("```"):
                    cleaned = "\n".join(cleaned.split("\n")[1:-1])
                refined = json.loads(cleaned)
                if refined:
                    current_records = refined

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Reflection parsing error: {e}")
                break

        return {
            "records": current_records,
            "quality_score": 5,
            "iterations": self.max_retries,
            "improved": True,
            "issues": []
        }


class WorkflowSummarizer(dspy.Module):
    """Generates friendly summaries of workflow steps for the UI."""

    def __init__(self):
        super().__init__()
        self.summarize = dspy.ChainOfThought(SummarizeWorkflowStep)

    def forward(self, step_name: str, step_results: str) -> str:
        result = self.summarize(step_name=step_name, step_results=step_results[:500])
        return result.user_summary
