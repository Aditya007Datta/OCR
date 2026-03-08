"""
Streamlit UI for the Regulatory Expert Consultant.
Provides a chat interface, document upload, and real-time agent logs.
"""

import json
import logging
import os
import queue
import sys
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse, parse_qs

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
    .main-header p  { color: #d0e8f7; margin: 5px 0 0; font-size: 1rem; }

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
    .log-entry   { margin: 2px 0; line-height: 1.4; }
    .log-info    { color: #58a6ff; }
    .log-warn    { color: #f8b500; }
    .log-error   { color: #f85149; }
    .log-success { color: #3fb950; }

    .chat-user  { background: #e8f4fd; border-radius: 12px 12px 4px 12px; padding: 10px 14px; margin: 8px 0; }
    .chat-agent { background: #f0fff0; border-radius: 12px 12px 12px 4px; padding: 10px 14px; margin: 8px 0; border-left: 3px solid #2E75B6; }

    div[data-testid="stSidebar"] { background: #f8fafc; }
    .stButton > button { border-radius: 8px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ─── Logging Setup ──────────────────────────────────────────────────────────

class StreamlitLogHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put((record.levelname, self.format(record)))


def setup_logging(log_queue: queue.Queue):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, StreamlitLogHandler) for h in root.handlers):
        h = StreamlitLogHandler(log_queue)
        h.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(h)


# ─── Session State ────────────────────────────────────────────────────────────

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
        "full_uri": "",
        "step": "welcome",
        "selected_frameworks": [],
        "workbook_path": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ─── URI Parser Helper ────────────────────────────────────────────────────────

def parse_uri_preview(full_uri: str) -> dict:
    """Parse URI and return components for display."""
    try:
        uri = full_uri.strip()
        if not uri.startswith("http"):
            uri = "https://" + uri
        parsed = urlparse(uri)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        deployment = "?"
        if "/deployments/" in parsed.path:
            deployment = parsed.path.split("/deployments/")[1].split("/")[0]
        api_version = parse_qs(parsed.query).get("api-version", ["?"])[0]
        return {"base_url": base_url, "deployment": deployment, "api_version": api_version, "valid": True}
    except Exception:
        return {"valid": False}


# ─── Agent Factory ────────────────────────────────────────────────────────────

def get_agent(api_key: str, full_uri: str):
    """Create or return cached agent instance."""
    cache_key = f"{api_key}::{full_uri}"
    if (
        st.session_state.agent is None
        or st.session_state.get("_agent_cache_key") != cache_key
    ):
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from app.agent.dspy_agent import RegulatoryAgent

        log_queue = st.session_state.log_queue

        def log_callback(msg: str):
            log_queue.put(("INFO", msg))

        st.session_state.agent = RegulatoryAgent(
            api_key=api_key,
            full_uri=full_uri,
            log_callback=log_callback,
        )
        st.session_state["_agent_cache_key"] = cache_key
        setup_logging(log_queue)

    return st.session_state.agent


def test_connection(api_key: str, full_uri: str) -> str:
    """Test Azure connection directly before initializing DSPy."""
    from openai import AzureOpenAI
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from app.agent.dspy_agent import parse_azure_uri

    try:
        azure = parse_azure_uri(full_uri)
        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure["base_url"],
            api_version=azure["api_version"],
        )
        response = client.chat.completions.create(
            model=azure["deployment"],
            messages=[{"role": "user", "content": "Say: connection successful"}],
            max_tokens=10,
        )
        reply = response.choices[0].message.content or ""
        return f"✅ Connected to **{azure['deployment']}** @ {azure['base_url']} — {reply}"
    except Exception as e:
        return f"❌ Connection failed: {e}"


# ─── UI Components ────────────────────────────────────────────────────────────

def render_log_panel():
    lq = st.session_state.log_queue
    while not lq.empty():
        try:
            level, msg = lq.get_nowait()
            st.session_state.logs.append((level, msg))
        except queue.Empty:
            break

    logs = st.session_state.logs[-100:]
    if not logs:
        return

    parts = ['<div class="log-panel">']
    for level, msg in logs:
        cls = "log-info"
        if any(x in msg for x in ["✅", "complete", "saved", "success"]):
            cls = "log-success"
        elif any(x in msg for x in ["⚠️", "warning", "warn"]):
            cls = "log-warn"
        elif any(x in msg for x in ["❌", "error", "fail", "Error"]):
            cls = "log-error"
        safe = msg.replace("<", "&lt;").replace(">", "&gt;")
        parts.append(f'<div class="log-entry {cls}">{safe}</div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_frameworks(frameworks: list[dict]):
    if not frameworks:
        st.warning("No frameworks identified yet.")
        return

    cols = st.columns(4)
    cols[0].metric("Frameworks Found", len(frameworks))
    authorities = set(f.get("authority", "") for f in frameworks)
    cols[1].metric("Regulatory Bodies", len(authorities))
    years = [f.get("year", 0) for f in frameworks if f.get("year")]
    cols[2].metric("Newest Standard", max(years) if years else "N/A")
    industries_all = []
    for f in frameworks:
        ind = f.get("industries", [])
        if isinstance(ind, list):
            industries_all.extend(ind)
    cols[3].metric("Industries Covered", len(set(industries_all)))

    st.divider()

    df = pd.DataFrame([
        {
            "Framework": f.get("name", ""),
            "Year": f.get("year", ""),
            "Authority": f.get("authority", ""),
            "Industries": ", ".join(f.get("industries", [])) if isinstance(f.get("industries"), list) else "",
            "Summary": (f.get("summary", "")[:120] + "...") if len(f.get("summary", "")) > 120 else f.get("summary", ""),
        }
        for f in frameworks
    ])
    st.dataframe(df, use_container_width=True, height=300)


def add_chat_message(role: str, content: str):
    st.session_state.chat_history.append({"role": role, "content": content})


def render_chat_history():
    for msg in st.session_state.chat_history[-20:]:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user">👤 <strong>You:</strong> {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-agent">⚖️ <strong>Consultant:</strong> {msg["content"]}</div>',
                unsafe_allow_html=True,
            )


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    init_session_state()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚖️ Regulatory Expert Consultant")
        st.caption("AI-powered compliance analysis")
        st.divider()

        # API Key
        api_key = st.text_input(
            "🔑 Azure API Key",
            type="password",
            value=st.session_state.api_key,
            placeholder="Your Azure OpenAI key",
        )
        if api_key:
            st.session_state.api_key = api_key

        # Full URI
        full_uri = st.text_input(
            "🌐 Azure Deployment URI",
            value=st.session_state.full_uri,
            placeholder="https://xyz.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-02-02-preview",
            help="Paste your full Azure OpenAI deployment URI including ?api-version=",
        )
        if full_uri:
            st.session_state.full_uri = full_uri

        # Show parsed components
        if full_uri:
            info = parse_uri_preview(full_uri)
            if info["valid"]:
                st.caption(
                    f"📌 Deployment: `{info['deployment']}`  \n"
                    f"🔖 API Version: `{info['api_version']}`  \n"
                    f"🏠 Host: `{info['base_url']}`"
                )
            else:
                st.warning("⚠️ Could not parse URI")

        # Connection test
        if api_key and full_uri:
            if st.button("🔌 Test Connection", use_container_width=True):
                with st.spinner("Testing..."):
                    result = test_connection(api_key, full_uri)
                st.markdown(result)

        st.divider()

        # Industry
        st.markdown("### 🏭 Industry / Domain")
        industry_options = [
            "Fintech", "Healthcare", "Cloud SaaS", "Manufacturing",
            "Insurance", "Government", "Education", "Retail", "Custom",
        ]
        selected_industry = st.selectbox("Select industry", industry_options)

        if selected_industry == "Custom":
            custom = st.text_input("Enter industry", placeholder="e.g., Aerospace")
            industry = custom if custom else "General"
        else:
            industry = selected_industry

        st.divider()

        # Document Upload
        st.markdown("### 📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Drag & drop PDFs here",
            type=["pdf"],
            accept_multiple_files=True,
        )

        st.divider()

        # Agent Memory
        if api_key and full_uri:
            try:
                sys.path.insert(0, str(Path(__file__).parent.parent.parent))
                from app.agent.reflection import AgentMemory
                stats = AgentMemory().get_stats()
                st.markdown("### 📈 Agent Memory")
                st.metric("Sessions", stats["total_sessions"])
                st.metric("Documents Processed", stats["total_documents"])
                st.metric("Controls Generated", stats["total_controls"])
            except Exception:
                pass

        if st.button("🔄 New Session", use_container_width=True):
            for key in ["frameworks", "analysis_results", "chat_history",
                        "logs", "step", "workbook_path", "selected_frameworks", "agent"]:
                st.session_state.pop(key, None)
            st.rerun()

    # ── Main Panel ────────────────────────────────────────────────────────────

    st.markdown("""
    <div class="main-header">
        <h1>⚖️ Regulatory Expert Consultant</h1>
        <p>AI-powered compliance analysis • DSPy Agent • Local RAG • Real-time insights</p>
    </div>
    """, unsafe_allow_html=True)

    # Guard: require both key and URI
    if not st.session_state.api_key or not st.session_state.full_uri:
        st.info("👈 Enter your **Azure API Key** and **Deployment URI** in the sidebar to get started.")
        st.markdown("""
        ### What this system does:
        1. **Discovers** regulatory frameworks for your industry
        2. **Downloads** and **analyzes** official regulatory documents
        3. **Extracts** structured compliance requirements using AI
        4. **Models** themes and topics (BERTopic)
        5. **Builds** a normalized control library
        6. **Generates** a professional Excel compliance workbook
        7. **Answers** compliance questions via RAG Q&A
        """)
        return

    # ── Tabs ──────────────────────────────────────────────────────────────────

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Framework Discovery",
        "📄 Document Analysis",
        "💬 Q&A Assistant",
        "⚙️ Agent Logs",
    ])

    # ── TAB 1: Framework Discovery ────────────────────────────────────────────
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
                    agent = get_agent(st.session_state.api_key, st.session_state.full_uri)
                    frameworks = agent.identify_frameworks(industry)
                    st.session_state.frameworks = frameworks
                    st.session_state.industry = industry
                    add_chat_message(
                        "agent",
                        f"I've identified **{len(frameworks)}** regulatory frameworks for the "
                        f"**{industry}** industry. Review the table below and select which ones "
                        "to analyze in depth."
                    )
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
                if st.button("🔬 Full Document Analysis", type="primary", use_container_width=True):
                    st.session_state.step = "full_analysis"
            with col_c:
                if st.button("💬 Ask Questions (RAG)", use_container_width=True):
                    st.session_state.step = "qa"

    # ── TAB 2: Document Analysis ──────────────────────────────────────────────
    with tab2:
        st.markdown("## 📄 Document Analysis & Processing")

        # Uploaded PDFs
        if uploaded_files:
            st.markdown("### 📎 Uploaded Documents")
            for uf in uploaded_files:
                st.caption(f"• {uf.name} ({uf.size:,} bytes)")

            if st.button("🔬 Analyze Uploaded Documents", type="primary"):
                agent = get_agent(st.session_state.api_key, st.session_state.full_uri)
                all_records = []
                progress = st.progress(0)
                status = st.empty()

                for i, uf in enumerate(uploaded_files):
                    status.text(f"Processing {uf.name}...")
                    try:
                        local_path = agent.pdf_processor.process_uploaded_file(uf.read(), uf.name)
                        fw_name = Path(uf.name).stem.replace("_", " ").replace("-", " ").title()
                        result = agent.process_document(local_path, fw_name)
                        all_records.extend(result["records"])
                    except Exception as e:
                        st.error(f"Error processing {uf.name}: {e}")
                    progress.progress((i + 1) / len(uploaded_files))

                if all_records:
                    agent.current_records = all_records
                    st.session_state.analysis_results = {"records": all_records}

                    status.text("Running topic modeling...")
                    topics = agent.run_topic_modeling(all_records)
                    agent.current_topics = topics

                    status.text("Building control library...")
                    controls = agent.build_control_library(all_records, topics)
                    agent.current_controls = controls

                    status.text("Indexing for Q&A...")
                    agent.index_documents(all_records)

                    status.empty()
                    progress.empty()
                    st.success(f"✅ Processed {len(all_records)} requirements from {len(uploaded_files)} documents")

                    st.markdown("### 📋 Extracted Requirements (Preview)")
                    df = pd.DataFrame(all_records[:20])
                    if not df.empty:
                        display_cols = [c for c in ["source", "section_number", "topic", "requirement_summary"] if c in df.columns]
                        st.dataframe(df[display_cols] if display_cols else df, use_container_width=True)

        # Full analysis trigger
        if st.session_state.step == "full_analysis":
            st.markdown("### 🔬 Full Analysis Pipeline")
            st.info(
                f"Running full analysis for: **{st.session_state.industry}** | "
                f"Frameworks: {', '.join(st.session_state.selected_frameworks)}"
            )
            with st.spinner("Running full analysis pipeline... This may take several minutes."):
                try:
                    agent = get_agent(st.session_state.api_key, st.session_state.full_uri)
                    results = agent.run_full_analysis(
                        industry=st.session_state.industry,
                        framework_names=st.session_state.selected_frameworks or None,
                    )
                    st.session_state.analysis_results = results
                    st.session_state.workbook_path = results.get("workbook_path")
                    st.session_state.step = "done"
                    st.success("✅ Full analysis complete!")

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Requirements", len(results.get("records", [])))
                    c2.metric("Topics Found", len(results.get("topics", [])))
                    c3.metric("Controls", len(results.get("controls", [])))
                    c4.metric("Frameworks", len(results.get("frameworks", [])))

                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.exception(e)

        # Workbook-only trigger
        if st.session_state.step == "workbook_only":
            if st.session_state.frameworks:
                with st.spinner("Generating compliance workbook..."):
                    try:
                        agent = get_agent(st.session_state.api_key, st.session_state.full_uri)
                        path = agent.generate_workbook(
                            frameworks=st.session_state.frameworks,
                            output_name=f"{st.session_state.industry.lower()}_frameworks",
                        )
                        st.session_state.workbook_path = path
                        st.session_state.step = ""
                        st.success(f"✅ Workbook generated!")
                    except Exception as e:
                        st.error(f"Workbook generation failed: {e}")
            else:
                st.warning("Discover frameworks first.")

        # Workbook download
        if st.session_state.workbook_path and Path(st.session_state.workbook_path).exists():
            st.divider()
            st.markdown("### 📥 Download Workbook")
            with open(st.session_state.workbook_path, "rb") as f:
                wb_bytes = f.read()
            st.download_button(
                label="⬇️ Download Compliance Workbook (.xlsx)",
                data=wb_bytes,
                file_name=Path(st.session_state.workbook_path).name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        # Analysis results
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
                        "Requirement": (c.get("control_requirement", "")[:80] + "..."),
                        "Framework": c.get("framework_source", ""),
                    }
                    for c in results["controls"][:20]
                ])
                st.dataframe(controls_df, use_container_width=True)

    # ── TAB 3: Q&A Assistant ──────────────────────────────────────────────────
    with tab3:
        st.markdown("## 💬 Compliance Q&A Assistant")
        st.caption("Ask questions about regulatory requirements. Powered by RAG over indexed documents.")

        render_chat_history()

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

        question = st.chat_input("Ask a compliance question...")

        if "pending_question" in st.session_state:
            question = st.session_state.pop("pending_question")

        if question:
            add_chat_message("user", question)
            try:
                agent = get_agent(st.session_state.api_key, st.session_state.full_uri)
                if agent.embedding_store.document_count == 0:
                    if st.session_state.frameworks:
                        fw_context = "\n".join([
                            f"{f['name']}: {f.get('summary', '')}"
                            for f in st.session_state.frameworks
                        ])
                        answer = agent.qa_engine(
                            question=question,
                            retrieved_context=f"Available framework summaries:\n{fw_context}",
                        )
                    else:
                        answer = ("I don't have any documents indexed yet. Please run a document "
                                  "analysis first, or upload regulatory PDFs.")
                else:
                    answer = agent.answer_question(question)

                add_chat_message("agent", answer)
                st.rerun()

            except Exception as e:
                add_chat_message("agent", f"Error: {e}. Please check your connection and try again.")
                st.rerun()

    # ── TAB 4: Agent Logs ─────────────────────────────────────────────────────
    with tab4:
        st.markdown("## ⚙️ Real-Time Agent Logs")
        st.caption("Live view of all agent operations and DSPy reasoning steps.")

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

            with st.expander("📜 Full log history"):
                log_text = "\n".join(f"[{level}] {msg}" for level, msg in st.session_state.logs)
                st.text_area("", value=log_text, height=400)
                st.download_button(
                    "⬇️ Download Logs",
                    data=log_text,
                    file_name="agent_logs.txt",
                    mime="text/plain",
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
