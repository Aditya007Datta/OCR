"""
Streamlit UI for the Regulatory Expert Consultant.
Provides a chat interface, document upload, and real-time agent logs.
"""

import json
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

# ─── Page Config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Regulatory Expert Consultant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1F4E79 0%, #2E75B6 100%);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p { color: #d0e8f7; margin: 5px 0 0; font-size: 1rem; }

    .log-panel {
        background: #0d1117;
        color: #58a6ff;
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        border-radius: 8px;
        padding: 12px;
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid #30363d;
    }
    .log-entry { margin: 2px 0; line-height: 1.4; }
    .log-info { color: #58a6ff; }
    .log-warn { color: #f8b500; }
    .log-error { color: #f85149; }
    .log-success { color: #3fb950; }

    .framework-card {
        background: #f0f7ff;
        border-left: 4px solid #2E75B6;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
    }
    .framework-card h4 { color: #1F4E79; margin: 0 0 4px; }
    .framework-card p { color: #555; margin: 2px 0; font-size: 0.9rem; }

    .metric-box {
        background: white;
        border: 1px solid #d0e4f7;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .metric-box .number { font-size: 2rem; font-weight: bold; color: #1F4E79; }
    .metric-box .label { color: #666; font-size: 0.85rem; }

    .chat-user { background: #e8f4fd; border-radius: 12px 12px 4px 12px; padding: 10px 14px; margin: 8px 0; }
    .chat-agent { background: #f0fff0; border-radius: 12px 12px 12px 4px; padding: 10px 14px; margin: 8px 0; border-left: 3px solid #2E75B6; }

    div[data-testid="stSidebar"] { background: #f8fafc; }
    .stButton > button { border-radius: 8px; font-weight: 500; }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(46, 117, 182, 0.2); }
</style>
""", unsafe_allow_html=True)


# ─── Logging Setup ──────────────────────────────────────────────────────────

class StreamlitLogHandler(logging.Handler):
    """Custom log handler that streams logs to a queue for Streamlit display."""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        msg = self.format(record)
        self.log_queue.put((record.levelname, msg))


def setup_logging(log_queue: queue.Queue):
    """Configure logging to also emit to Streamlit."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = StreamlitLogHandler(log_queue)
    handler.setFormatter(logging.Formatter("%(message)s"))
    # Avoid duplicate handlers
    if not any(isinstance(h, StreamlitLogHandler) for h in root_logger.handlers):
        root_logger.addHandler(handler)


# ─── Session State Initialization ───────────────────────────────────────────

def init_session_state():
    defaults = {
        "agent": None,
        "log_queue": queue.Queue(),
        "logs": [],
        "chat_history": [],
        "frameworks": [],
        "analysis_results": None,
        "industry": "",
        "api_key": "",
        "step": "welcome",  # welcome | identify | analyze | qa
        "selected_frameworks": [],
        "workbook_path": None,
        "processing": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ─── Agent Factory ───────────────────────────────────────────────────────────

def get_agent(api_key: str):
    """Create or return cached agent instance."""
    if st.session_state.agent is None or st.session_state.api_key != api_key:
        # Add project root to path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from app.agent.dspy_agent import RegulatoryAgent

        log_queue = st.session_state.log_queue

        def log_callback(msg: str):
            log_queue.put(("INFO", msg))

        st.session_state.agent = RegulatoryAgent(
            api_key=api_key,
            log_callback=log_callback,
        )
        st.session_state.api_key = api_key
        setup_logging(log_queue)
    return st.session_state.agent


# ─── Log Panel Component ─────────────────────────────────────────────────────

def render_log_panel():
    """Render the real-time agent log panel."""
    # Drain queue
    lq = st.session_state.log_queue
    while not lq.empty():
        try:
            level, msg = lq.get_nowait()
            st.session_state.logs.append((level, msg))
        except queue.Empty:
            break

    # Keep last 100 entries
    logs = st.session_state.logs[-100:]

    if not logs:
        return

    log_html_parts = ['<div class="log-panel">']
    for level, msg in logs:
        cls = "log-info"
        if "✅" in msg or "complete" in msg.lower() or "saved" in msg.lower():
            cls = "log-success"
        elif "⚠️" in msg or "warning" in msg.lower():
            cls = "log-warn"
        elif "❌" in msg or "error" in msg.lower() or "fail" in msg.lower():
            cls = "log-error"

        safe_msg = msg.replace("<", "&lt;").replace(">", "&gt;")
        log_html_parts.append(f'<div class="log-entry {cls}">{safe_msg}</div>')

    log_html_parts.append("</div>")
    st.markdown("".join(log_html_parts), unsafe_allow_html=True)


# ─── Framework Display ────────────────────────────────────────────────────────

def render_frameworks(frameworks: list[dict]):
    """Render framework cards in a nice grid."""
    if not frameworks:
        st.warning("No frameworks identified yet.")
        return

    # Summary metrics
    cols = st.columns(4)
    with cols[0]:
        st.metric("Frameworks Found", len(frameworks))
    with cols[1]:
        authorities = set(f.get("authority", "") for f in frameworks)
        st.metric("Regulatory Bodies", len(authorities))
    with cols[2]:
        years = [f.get("year", 0) for f in frameworks if f.get("year")]
        st.metric("Newest Standard", max(years) if years else "N/A")
    with cols[3]:
        industries_all = []
        for f in frameworks:
            ind = f.get("industries", [])
            if isinstance(ind, list):
                industries_all.extend(ind)
        st.metric("Industries Covered", len(set(industries_all)))

    st.divider()

    # Framework table
    df = pd.DataFrame([
        {
            "Framework": f.get("name", ""),
            "Year": f.get("year", ""),
            "Authority": f.get("authority", ""),
            "Industries": ", ".join(f.get("industries", [])) if isinstance(f.get("industries"), list) else "",
            "Summary": f.get("summary", "")[:120] + "..." if len(f.get("summary", "")) > 120 else f.get("summary", ""),
        }
        for f in frameworks
    ])
    st.dataframe(df, use_container_width=True, height=300)


# ─── Chat Interface ───────────────────────────────────────────────────────────

def add_chat_message(role: str, content: str):
    st.session_state.chat_history.append({"role": role, "content": content})


def render_chat_history():
    for msg in st.session_state.chat_history[-20:]:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            st.markdown(
                f'<div class="chat-user">👤 <strong>You:</strong> {content}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-agent">⚖️ <strong>Consultant:</strong> {content}</div>',
                unsafe_allow_html=True
            )


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    init_session_state()

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚖️ Regulatory Expert Consultant")
        st.caption("AI-powered compliance analysis")
        st.divider()

        # API Key
        api_key = st.text_input(
            "🔑 OpenAI API Key",
            type="password",
            value=st.session_state.api_key,
            placeholder="sk-...",
            help="Your OpenAI API key for GPT-4o-mini"
        )
        if api_key:
            st.session_state.api_key = api_key

        st.divider()

        # Industry Input
        st.markdown("### 🏭 Industry / Domain")
        industry_options = ["Fintech", "Healthcare", "Cloud SaaS", "Manufacturing",
                            "Insurance", "Government", "Education", "Retail", "Custom"]
        selected_industry = st.selectbox("Select industry", industry_options, index=0)

        if selected_industry == "Custom":
            custom_industry = st.text_input("Enter industry", placeholder="e.g., Aerospace")
            industry = custom_industry if custom_industry else "General"
        else:
            industry = selected_industry

        st.divider()

        # Document Upload
        st.markdown("### 📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Drag & drop PDFs here",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload regulatory documents for analysis"
        )

        st.divider()

        # Agent Memory Stats
        if st.session_state.api_key:
            try:
                sys.path.insert(0, str(Path(__file__).parent.parent.parent))
                from app.agent.reflection import AgentMemory
                mem = AgentMemory()
                stats = mem.get_stats()
                st.markdown("### 📈 Agent Memory")
                st.metric("Sessions", stats["total_sessions"])
                st.metric("Documents Processed", stats["total_documents"])
                st.metric("Controls Generated", stats["total_controls"])
            except Exception:
                pass

        # Clear session
        if st.button("🔄 New Session", use_container_width=True):
            for key in ["frameworks", "analysis_results", "chat_history",
                        "logs", "step", "workbook_path", "selected_frameworks"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    # ── Main Panel ────────────────────────────────────────────

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚖️ Regulatory Expert Consultant</h1>
        <p>AI-powered compliance analysis • DSPy Agent • Local RAG • Real-time insights</p>
    </div>
    """, unsafe_allow_html=True)

    # Validate API key
    if not st.session_state.api_key:
        st.info("👈 Enter your OpenAI API key in the sidebar to get started.")
        st.markdown("""
        ### What this system does:
        1. **Discovers** regulatory frameworks applicable to your industry
        2. **Downloads** and **analyzes** official regulatory documents
        3. **Extracts** structured compliance requirements using AI
        4. **Models** themes and topics across requirements (BERTopic)
        5. **Builds** a normalized control library
        6. **Generates** a professional Excel compliance workbook
        7. **Answers** compliance questions via RAG Q&A
        """)
        return

    # ── Tabs ─────────────────────────────────────────────────

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Framework Discovery",
        "📄 Document Analysis",
        "💬 Q&A Assistant",
        "⚙️ Agent Logs"
    ])

    # ── TAB 1: Framework Discovery ────────────────────────────
    with tab1:
        st.markdown("## 🔍 Regulatory Framework Discovery")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Target Industry:** `{industry}`")
        with col2:
            discover_btn = st.button("🚀 Discover Frameworks", type="primary", use_container_width=True)

        if discover_btn:
            with st.spinner("Identifying regulatory frameworks..."):
                try:
                    agent = get_agent(st.session_state.api_key)
                    frameworks = agent.identify_frameworks(industry)
                    st.session_state.frameworks = frameworks
                    st.session_state.industry = industry
                    add_chat_message("agent",
                        f"I've identified **{len(frameworks)}** regulatory frameworks for the **{industry}** industry. "
                        "Review the table below and select which ones to analyze in depth.")
                    st.success(f"✅ Found {len(frameworks)} frameworks!")
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.session_state.frameworks:
            render_frameworks(st.session_state.frameworks)

            st.divider()
            st.markdown("### Select Frameworks for Deep Analysis")

            fw_names = [f.get("name", "") for f in st.session_state.frameworks]
            selected = st.multiselect(
                "Choose frameworks to analyze:",
                fw_names,
                default=fw_names[:2] if len(fw_names) >= 2 else fw_names,
            )
            st.session_state.selected_frameworks = selected

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("📊 Build Compliance Workbook", use_container_width=True):
                    st.session_state.step = "workbook_only"
            with col_b:
                if st.button("🔬 Full Document Analysis", use_container_width=True, type="primary"):
                    st.session_state.step = "full_analysis"
            with col_c:
                if st.button("💬 Ask Questions (RAG)", use_container_width=True):
                    st.session_state.step = "qa"

    # ── TAB 2: Document Analysis ──────────────────────────────
    with tab2:
        st.markdown("## 📄 Document Analysis & Processing")

        # Handle uploaded PDFs
        if uploaded_files:
            st.markdown("### 📎 Uploaded Documents")
            for uf in uploaded_files:
                st.caption(f"• {uf.name} ({uf.size:,} bytes)")

            if st.button("🔬 Analyze Uploaded Documents", type="primary"):
                agent = get_agent(st.session_state.api_key)
                all_records = []
                progress = st.progress(0)
                status = st.empty()

                for i, uf in enumerate(uploaded_files):
                    status.text(f"Processing {uf.name}...")
                    try:
                        local_path = agent.pdf_processor.process_uploaded_file(
                            uf.read(), uf.name
                        )
                        # Determine framework name from filename
                        fw_name = Path(uf.name).stem.replace("_", " ").replace("-", " ").title()
                        result = agent.process_document(local_path, fw_name)
                        all_records.extend(result["records"])
                    except Exception as e:
                        st.error(f"Error processing {uf.name}: {e}")
                    progress.progress((i + 1) / len(uploaded_files))

                if all_records:
                    agent.current_records = all_records
                    st.session_state.analysis_results = {"records": all_records}

                    # Run topics
                    status.text("Running topic modeling...")
                    topics = agent.run_topic_modeling(all_records)
                    agent.current_topics = topics

                    # Build controls
                    status.text("Building control library...")
                    controls = agent.build_control_library(all_records, topics)
                    agent.current_controls = controls

                    # Index for RAG
                    status.text("Indexing for Q&A...")
                    agent.index_documents(all_records)

                    status.empty()
                    progress.empty()
                    st.success(f"✅ Processed {len(all_records)} requirements from {len(uploaded_files)} documents")

                    # Show results preview
                    st.markdown("### 📋 Extracted Requirements (Preview)")
                    df = pd.DataFrame(all_records[:20])
                    if not df.empty:
                        display_cols = [c for c in ["source", "section_number", "topic", "requirement_summary"]
                                       if c in df.columns]
                        st.dataframe(df[display_cols] if display_cols else df, use_container_width=True)

        # Full analysis trigger
        if st.session_state.step == "full_analysis":
            st.markdown("### 🔬 Full Analysis Pipeline")
            st.info(f"Running full analysis for: **{st.session_state.industry}** | "
                    f"Frameworks: {', '.join(st.session_state.selected_frameworks)}")

            with st.spinner("Running full analysis pipeline... This may take several minutes."):
                try:
                    agent = get_agent(st.session_state.api_key)
                    results = agent.run_full_analysis(
                        industry=st.session_state.industry,
                        framework_names=st.session_state.selected_frameworks if st.session_state.selected_frameworks else None,
                    )
                    st.session_state.analysis_results = results
                    st.session_state.workbook_path = results.get("workbook_path")
                    st.session_state.step = "done"

                    st.success("✅ Full analysis complete!")

                    # Stats
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Requirements", len(results.get("records", [])))
                    col2.metric("Topics Found", len(results.get("topics", [])))
                    col3.metric("Controls", len(results.get("controls", [])))
                    col4.metric("Frameworks", len(results.get("frameworks", [])))

                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.exception(e)

        # Workbook-only trigger
        if st.session_state.step == "workbook_only":
            if st.session_state.frameworks:
                with st.spinner("Generating compliance workbook..."):
                    try:
                        agent = get_agent(st.session_state.api_key)
                        path = agent.generate_workbook(
                            frameworks=st.session_state.frameworks,
                            output_name=f"{st.session_state.industry.lower()}_frameworks"
                        )
                        st.session_state.workbook_path = path
                        st.session_state.step = ""
                        st.success(f"✅ Workbook generated: {path}")
                    except Exception as e:
                        st.error(f"Workbook generation failed: {e}")
            else:
                st.warning("Discover frameworks first.")

        # Show workbook download if available
        if st.session_state.workbook_path and Path(st.session_state.workbook_path).exists():
            st.divider()
            st.markdown("### 📥 Download Workbook")
            with open(st.session_state.workbook_path, "rb") as f:
                workbook_bytes = f.read()
            st.download_button(
                label="⬇️ Download Compliance Workbook (.xlsx)",
                data=workbook_bytes,
                file_name=Path(st.session_state.workbook_path).name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        # Show analysis results if available
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results

            if results.get("topics"):
                st.markdown("### 🔍 Discovered Themes")
                topics_df = pd.DataFrame([
                    {
                        "Theme": t.get("label", ""),
                        "Category": t.get("theme_category", ""),
                        "Description": t.get("description", ""),
                        "Keywords": ", ".join(t.get("keywords", [])[:5]),
                    }
                    for t in results["topics"][:15]
                ])
                st.dataframe(topics_df, use_container_width=True)

            if results.get("controls"):
                st.markdown("### 🛡️ Control Library Preview")
                controls_df = pd.DataFrame([
                    {
                        "Theme": c.get("control_theme", ""),
                        "Category": c.get("control_category", ""),
                        "Requirement": c.get("control_requirement", "")[:80] + "...",
                        "Framework": c.get("framework_source", ""),
                    }
                    for c in results["controls"][:20]
                ])
                st.dataframe(controls_df, use_container_width=True)

    # ── TAB 3: Q&A Assistant ──────────────────────────────────
    with tab3:
        st.markdown("## 💬 Compliance Q&A Assistant")
        st.caption("Ask questions about regulatory requirements. Powered by RAG over indexed documents.")

        render_chat_history()

        # Sample questions
        with st.expander("💡 Sample questions"):
            samples = [
                "What are the access control requirements?",
                "What does this framework say about incident response?",
                "What encryption requirements are specified?",
                "What are the audit logging requirements?",
                "How should vendor risk be managed?",
            ]
            for s in samples:
                if st.button(s, key=f"sample_{s[:20]}"):
                    st.session_state["pending_question"] = s

        # Question input
        question = st.chat_input("Ask a compliance question...")

        # Handle sample question click
        if "pending_question" in st.session_state:
            question = st.session_state.pop("pending_question")

        if question:
            add_chat_message("user", question)

            try:
                agent = get_agent(st.session_state.api_key)

                # Check if documents are indexed
                if agent.embedding_store.document_count == 0:
                    # Use frameworks as context if no docs indexed
                    if st.session_state.frameworks:
                        fw_context = "\n".join([
                            f"{f['name']}: {f.get('summary', '')}"
                            for f in st.session_state.frameworks
                        ])
                        answer = agent.qa_engine(
                            question=question,
                            retrieved_context=f"Available framework summaries:\n{fw_context}"
                        )
                    else:
                        answer = ("I don't have any documents indexed yet. Please run a document "
                                  "analysis first, or upload regulatory PDFs for analysis.")
                else:
                    answer = agent.answer_question(question)

                add_chat_message("agent", answer)
                st.rerun()

            except Exception as e:
                add_chat_message("agent", f"I encountered an error: {e}. Please check your API key and try again.")
                st.rerun()

    # ── TAB 4: Agent Logs ─────────────────────────────────────
    with tab4:
        st.markdown("## ⚙️ Real-Time Agent Logs")
        st.caption("Live view of all agent operations, DSPy reasoning steps, and system events.")

        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button("🔄 Refresh Logs"):
                st.rerun()
        with col2:
            if st.button("🗑️ Clear Logs"):
                st.session_state.logs = []
                st.rerun()

        if not st.session_state.logs:
            st.info("Agent logs will appear here as the system runs.")
        else:
            st.markdown(f"**{len(st.session_state.logs)} log entries**")
            render_log_panel()

            # Expandable full log
            with st.expander("📜 Full log history"):
                log_text = "\n".join(f"[{level}] {msg}" for level, msg in st.session_state.logs)
                st.text_area("", value=log_text, height=400)
                st.download_button(
                    "⬇️ Download Logs",
                    data=log_text,
                    file_name="agent_logs.txt",
                    mime="text/plain"
                )

    # Drain logs after every render
    lq = st.session_state.log_queue
    while not lq.empty():
        try:
            level, msg = lq.get_nowait()
            st.session_state.logs.append((level, msg))
        except queue.Empty:
            break


if __name__ == "__main__":
    main()
