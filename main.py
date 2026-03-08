"""
Main entry point for the Regulatory Expert Consultant.
Run: streamlit run main.py
"""

import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Create required directories
for dir_path in [
    "data/raw_documents",
    "data/processed_documents",
    "indexes",
    "outputs/workbooks",
    "config",
]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def main():
    """Launch the Streamlit application."""
    app_path = project_root / "app" / "ui" / "streamlit_app.py"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.port",
            "8501",
            "--server.headless",
            "false",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=str(project_root),
    )


if __name__ == "__main__":
    main()
