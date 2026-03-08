"""
DSPy Signatures for the Regulatory Expert Consultant.
Each signature defines the input/output contract for an LLM module.
"""

import dspy


class IdentifyFrameworks(dspy.Signature):
    """Identify relevant regulatory frameworks and compliance standards for a given industry."""

    industry: str = dspy.InputField(desc="The industry or domain (e.g., Fintech, Healthcare, Cloud SaaS)")
    search_results: str = dspy.InputField(desc="Web search results about regulations in this industry", default="")
    frameworks_json: str = dspy.OutputField(
        desc=(
            "JSON array of framework objects. Each object must have: "
            "name (string), year (int), authority (string), "
            "industries (list of strings), summary (string, 3-4 sentences). "
            "Return ONLY valid JSON, no markdown."
        )
    )


class RetrieveDocumentURLs(dspy.Signature):
    """Find official source URLs for a regulatory framework document."""

    framework_name: str = dspy.InputField(desc="Name of the regulatory framework")
    search_results: str = dspy.InputField(desc="Web search results for this framework's documents")
    urls: str = dspy.OutputField(
        desc="JSON array of URL strings pointing to official PDFs or HTML pages for this framework. Return ONLY valid JSON."
    )


class ExtractTextFromPage(dspy.Signature):
    """OCR: Extract all text from a document page image."""

    page_description: str = dspy.InputField(desc="Description of the page being processed")
    extracted_text: str = dspy.OutputField(
        desc="All text extracted from the page. Preserve headings, section numbers, and structure. Do not summarize."
    )


class ParseDocumentStructure(dspy.Signature):
    """Parse a regulatory document into structured requirement records."""

    document_text: str = dspy.InputField(desc="Full text of the regulatory document")
    framework_name: str = dspy.InputField(desc="Name of the framework (e.g., ISO 27001)")
    structured_records: str = dspy.OutputField(
        desc=(
            "JSON array of requirement objects. Each object must have: "
            "source (string), topic (string), subtopic (string), "
            "section_number (string), requirement_text (string), "
            "requirement_summary (string). "
            "Extract as many discrete requirements as possible. Return ONLY valid JSON."
        )
    )


class GenerateTopicLabels(dspy.Signature):
    """Generate human-readable labels and descriptions for BERTopic clusters."""

    topic_keywords: str = dspy.InputField(desc="Top keywords for each topic cluster as JSON")
    framework_context: str = dspy.InputField(desc="The regulatory framework these topics come from")
    topic_labels: str = dspy.OutputField(
        desc=(
            "JSON array of topic label objects. Each must have: "
            "topic_id (int), label (string), description (string, 1-2 sentences), "
            "theme_category (string from: Data Protection, Identity Management, "
            "Incident Response, Risk Management, Vendor Risk, Governance, "
            "Physical Security, Business Continuity, Audit & Compliance, Other). "
            "Return ONLY valid JSON."
        )
    )


class BuildControlLibrary(dspy.Signature):
    """Convert parsed document requirements into a normalized compliance control library."""

    structured_records: str = dspy.InputField(desc="JSON array of parsed document requirement records")
    topic_labels: str = dspy.InputField(desc="JSON array of topic/theme labels from topic modeling")
    framework_name: str = dspy.InputField(desc="Name of the source framework")
    control_library: str = dspy.OutputField(
        desc=(
            "JSON array of control objects. Each must have: "
            "control_theme (string), control_category (string), "
            "control_subcategory (string), control_requirement (string), "
            "control_description (string), test_procedure (string), "
            "risk_narrative (string), mapped_section (string), "
            "framework_source (string). "
            "Return ONLY valid JSON."
        )
    )


class AnswerComplianceQuestion(dspy.Signature):
    """Answer a compliance question using retrieved document context."""

    question: str = dspy.InputField(desc="The user's compliance question")
    retrieved_context: str = dspy.InputField(desc="Relevant document chunks retrieved from the vector store")
    answer: str = dspy.OutputField(
        desc="A thorough answer to the question. Include framework references and section numbers. Be specific and cite sources."
    )


class EvaluateExtractionQuality(dspy.Signature):
    """Evaluate the quality of document extraction and suggest improvements."""

    original_text_sample: str = dspy.InputField(desc="Sample of the original document text")
    extracted_records: str = dspy.InputField(desc="The structured records extracted from the document")
    quality_score: str = dspy.OutputField(
        desc="JSON object with: score (0-10), issues (list of strings), suggestions (list of strings), should_retry (boolean)"
    )


class RefineExtractionPrompt(dspy.Signature):
    """Refine the extraction approach based on quality evaluation feedback."""

    original_records: str = dspy.InputField(desc="The original extracted records")
    quality_issues: str = dspy.InputField(desc="Issues identified during quality evaluation")
    document_sample: str = dspy.InputField(desc="Sample of the original document")
    refined_records: str = dspy.OutputField(
        desc="Improved JSON array of requirement records, addressing the identified quality issues. Return ONLY valid JSON."
    )


class SummarizeWorkflowStep(dspy.Signature):
    """Generate a user-friendly summary of a completed workflow step."""

    step_name: str = dspy.InputField(desc="Name of the workflow step")
    step_results: str = dspy.InputField(desc="Raw results or data from the step")
    user_summary: str = dspy.OutputField(
        desc="A concise, friendly 2-3 sentence summary of what was accomplished in this step."
    )
