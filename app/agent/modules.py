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


# ─── Shared JSON helper ────────────────────────────────────────────────────────

def safe_parse_json(text: str, default: Any) -> Any:
    """
    Robustly parse a JSON string returned by an LLM.

    Handles:
    - Markdown code fences (```json ... ```)
    - Preamble text before the first [ or {
    - Trailing text after the closing ] or }
    """
    if not text:
        return default
    try:
        cleaned = text.strip()

        # Strip markdown code fences
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first line (```json or ```) and last line (```)
            cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            cleaned = cleaned.strip()

        # Strip any preamble before the first JSON structure
        first_bracket = min(
            cleaned.find("[") if "[" in cleaned else len(cleaned),
            cleaned.find("{") if "{" in cleaned else len(cleaned),
        )
        if first_bracket > 0:
            cleaned = cleaned[first_bracket:]

        # Strip any trailing text after the last closing bracket
        last_bracket = max(cleaned.rfind("]"), cleaned.rfind("}"))
        if last_bracket >= 0:
            cleaned = cleaned[: last_bracket + 1]

        return json.loads(cleaned)

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"JSON parse failed: {e} — returning default")
        return default


# ─── Modules ──────────────────────────────────────────────────────────────────

class FrameworkIdentifier(dspy.Module):
    """Identifies regulatory frameworks relevant to a given industry."""

    def __init__(self):
        super().__init__()
        self.identify = dspy.ChainOfThought(IdentifyFrameworks)

    def forward(self, industry: str, search_results: str = "") -> dict[str, Any]:
        logger.info(f"Identifying frameworks for industry: {industry}")
        result = self.identify(industry=industry, search_results=search_results)

        # DSPy ChainOfThought stores reasoning as "reasoning", not "rationale"
        reasoning = getattr(result, "reasoning", "")

        frameworks = safe_parse_json(result.frameworks_json, [])
        if not isinstance(frameworks, list):
            logger.warning("frameworks_json did not parse to a list, resetting to []")
            frameworks = []

        return {"frameworks": frameworks, "reasoning": reasoning}


class DocumentURLRetriever(dspy.Module):
    """Finds official source URLs for regulatory framework documents."""

    def __init__(self):
        super().__init__()
        self.retrieve = dspy.ChainOfThought(RetrieveDocumentURLs)

    def forward(self, framework_name: str, search_results: str) -> list[str]:
        logger.info(f"Retrieving document URLs for: {framework_name}")
        result = self.retrieve(framework_name=framework_name, search_results=search_results)

        urls = safe_parse_json(result.urls, [])
        if not isinstance(urls, list):
            logger.warning(f"urls did not parse to a list for {framework_name}")
            return []

        # Filter out any non-string or empty entries
        return [u for u in urls if isinstance(u, str) and u.strip()]


class DocumentStructureParser(dspy.Module):
    """Parses regulatory documents into structured requirement records."""

    def __init__(self):
        super().__init__()
        self.parse = dspy.ChainOfThought(ParseDocumentStructure)

    def forward(self, document_text: str, framework_name: str) -> list[dict]:
        logger.info(f"Parsing document structure for: {framework_name}")
        max_chars = 12000

        if len(document_text) > max_chars:
            logger.info(f"Document is large ({len(document_text)} chars), processing in chunks")
            return self._parse_in_chunks(document_text, framework_name, max_chars)

        result = self.parse(document_text=document_text, framework_name=framework_name)
        records = safe_parse_json(result.structured_records, [])
        return records if isinstance(records, list) else []

    def _parse_in_chunks(self, text: str, framework_name: str, chunk_size: int) -> list[dict]:
        """Parse a large document in overlapping chunks."""
        all_records = []
        words = text.split()
        chunk_words = chunk_size // 6      # approx words per chunk
        overlap_words = 200                 # overlap between chunks

        for i in range(0, len(words), chunk_words - overlap_words):
            chunk = " ".join(words[i: i + chunk_words])
            if not chunk.strip():
                continue
            try:
                result = self.parse(document_text=chunk, framework_name=framework_name)
                records = safe_parse_json(result.structured_records, [])
                if isinstance(records, list):
                    all_records.extend(records)
            except Exception as e:
                logger.error(f"Error parsing chunk at word {i}: {e}")

        return self._deduplicate_records(all_records)

    def _deduplicate_records(self, records: list[dict]) -> list[dict]:
        """Remove duplicate records based on requirement text."""
        seen: set[str] = set()
        unique = []
        for r in records:
            key = r.get("requirement_text", "")[:100].strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(r)
        return unique


class TopicLabelGenerator(dspy.Module):
    """Generates human-readable labels for topic model clusters."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateTopicLabels)

    def forward(self, topic_keywords: dict, framework_context: str) -> list[dict]:
        logger.info("Generating topic labels")
        result = self.generate(
            topic_keywords=json.dumps(topic_keywords),
            framework_context=framework_context,
        )

        labels = safe_parse_json(result.topic_labels, [])
        if not isinstance(labels, list):
            logger.warning("topic_labels did not parse to a list")
            return []
        return labels


class ControlLibraryBuilder(dspy.Module):
    """Builds a normalized compliance control library from parsed requirements."""

    def __init__(self):
        super().__init__()
        self.build = dspy.ChainOfThought(BuildControlLibrary)

    def forward(
        self,
        structured_records: list[dict],
        topic_labels: list[dict],
        framework_name: str,
    ) -> list[dict]:
        logger.info(f"Building control library for: {framework_name}")

        batch_size = 20
        all_controls: list[dict] = []

        for i in range(0, len(structured_records), batch_size):
            batch = structured_records[i: i + batch_size]
            try:
                result = self.build(
                    structured_records=json.dumps(batch),
                    topic_labels=json.dumps(topic_labels[:10]),
                    framework_name=framework_name,
                )
                controls = safe_parse_json(result.control_library, [])
                if isinstance(controls, list):
                    all_controls.extend(controls)
                else:
                    logger.warning(f"control_library batch {i} did not parse to a list")
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
        return getattr(result, "answer", "No answer could be generated.")


class ExtractionQualityReflector(dspy.Module):
    """
    Reflection module: evaluates extraction quality and refines if needed.
    Enables the agent's self-improvement loop.
    """

    def __init__(self, max_retries: int = 2):
        super().__init__()
        self.evaluate = dspy.ChainOfThought(EvaluateExtractionQuality)
        self.refine = dspy.ChainOfThought(RefineExtractionPrompt)
        self.max_retries = max_retries

    def forward(self, original_text: str, extracted_records: list[dict]) -> dict[str, Any]:
        """
        Evaluate extraction quality and optionally refine.

        Returns:
            dict with keys: records, quality_score, iterations, improved, issues
        """
        current_records = extracted_records
        text_sample = original_text[:3000]

        for iteration in range(self.max_retries):
            logger.info(f"Quality reflection iteration {iteration + 1}/{self.max_retries}")

            try:
                eval_result = self.evaluate(
                    original_text_sample=text_sample,
                    extracted_records=json.dumps(current_records[:5]),
                )

                quality = safe_parse_json(
                    getattr(eval_result, "quality_score", "{}"),
                    {"score": 5, "should_retry": False, "issues": [], "suggestions": []},
                )

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
                        "issues": issues,
                    }

                # Quality is low — attempt refinement
                logger.info(f"Refining extraction due to issues: {issues}")
                refine_result = self.refine(
                    original_records=json.dumps(current_records[:10]),
                    quality_issues=json.dumps(issues),
                    document_sample=text_sample,
                )

                refined = safe_parse_json(
                    getattr(refine_result, "refined_records", "[]"),
                    [],
                )
                if isinstance(refined, list) and refined:
                    current_records = refined
                    logger.info(f"Refined to {len(current_records)} records")

            except Exception as e:
                logger.warning(f"Reflection iteration {iteration + 1} failed: {e}")
                break

        return {
            "records": current_records,
            "quality_score": 5,
            "iterations": self.max_retries,
            "improved": False,
            "issues": [],
        }


class WorkflowSummarizer(dspy.Module):
    """Generates friendly summaries of workflow steps for the UI."""

    def __init__(self):
        super().__init__()
        self.summarize = dspy.ChainOfThought(SummarizeWorkflowStep)

    def forward(self, step_name: str, step_results: str) -> str:
        result = self.summarize(step_name=step_name, step_results=step_results[:500])
        return getattr(result, "user_summary", "Step completed.")
