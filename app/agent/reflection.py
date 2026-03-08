"""
Reflection and experiential learning memory for the DSPy agent.
Stores successful extraction patterns and quality metrics across sessions.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MEMORY_FILE = Path("data/agent_memory.json")


class AgentMemory:
    """
    Local persistent memory for the agent's experiential learning.
    Stores quality metrics, successful patterns, and extraction history.
    """

    def __init__(self, memory_path: Path = MEMORY_FILE):
        self.memory_path = memory_path
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if self.memory_path.exists():
            try:
                with open(self.memory_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning("Could not load agent memory, starting fresh")
        return {
            "sessions": [],
            "framework_quality": {},
            "extraction_patterns": {},
            "total_documents_processed": 0,
            "total_controls_generated": 0,
        }

    def save(self):
        with open(self.memory_path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    def record_session(self, industry: str, frameworks: list[str], session_id: str):
        """Record a new analysis session."""
        self._data["sessions"].append({
            "id": session_id,
            "industry": industry,
            "frameworks": frameworks,
            "timestamp": datetime.now().isoformat(),
        })
        self.save()

    def record_extraction_quality(self, framework: str, score: float, issues: list[str]):
        """Track extraction quality per framework."""
        if framework not in self._data["framework_quality"]:
            self._data["framework_quality"][framework] = {
                "scores": [],
                "common_issues": [],
                "avg_score": 0,
            }
        fw_data = self._data["framework_quality"][framework]
        fw_data["scores"].append(score)
        fw_data["avg_score"] = sum(fw_data["scores"]) / len(fw_data["scores"])
        fw_data["common_issues"].extend(issues)
        # Keep only unique issues
        fw_data["common_issues"] = list(set(fw_data["common_issues"]))[-20:]
        self.save()

    def record_document_processed(self, framework: str, doc_url: str, record_count: int):
        """Track processed documents and record counts."""
        self._data["total_documents_processed"] += 1
        if framework not in self._data["extraction_patterns"]:
            self._data["extraction_patterns"][framework] = []
        self._data["extraction_patterns"][framework].append({
            "url": doc_url,
            "records": record_count,
            "timestamp": datetime.now().isoformat(),
        })
        self.save()

    def record_controls_generated(self, count: int):
        """Track total controls generated."""
        self._data["total_controls_generated"] += count
        self.save()

    def get_framework_context(self, framework: str) -> dict[str, Any]:
        """Get historical context for a framework to improve future extractions."""
        return self._data["framework_quality"].get(framework, {})

    def get_stats(self) -> dict[str, Any]:
        """Get overall agent statistics."""
        return {
            "total_sessions": len(self._data["sessions"]),
            "total_documents": self._data["total_documents_processed"],
            "total_controls": self._data["total_controls_generated"],
            "frameworks_analyzed": list(self._data["framework_quality"].keys()),
        }

    def get_recent_sessions(self, n: int = 5) -> list[dict]:
        """Get the most recent sessions."""
        return self._data["sessions"][-n:]


class ReflectionLogger:
    """
    Logs reflection events and maintains a history of agent self-improvement.
    """

    def __init__(self, log_path: Path = Path("data/reflection_log.json")):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self._load()

    def _load(self) -> list:
        if self.log_path.exists():
            try:
                with open(self.log_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def save(self):
        with open(self.log_path, "w") as f:
            json.dump(self._log, f, indent=2, default=str)

    def log_reflection(
        self,
        step: str,
        original_score: float,
        final_score: float,
        iterations: int,
        improved: bool,
        issues: list[str],
    ):
        """Log a reflection event."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "original_score": original_score,
            "final_score": final_score,
            "iterations": iterations,
            "improved": improved,
            "issues": issues,
        }
        self._log.append(entry)
        self.save()
        logger.info(f"Reflection logged: {step} | Score: {original_score} -> {final_score}")

    def get_improvement_rate(self) -> float:
        """Calculate overall improvement rate from reflection."""
        if not self._log:
            return 0.0
        improved = sum(1 for e in self._log if e["improved"])
        return improved / len(self._log)
