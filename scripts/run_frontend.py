"""Launch the Gradio frontend for report generation."""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import gradio as gr
from rag_project.agent.frontend.app import create_app

if __name__ == "__main__":
    app = create_app()

    app.launch(
        server_port=7860,
        inbrowser=True,
    )
