"""
Hugging Face Space entry point for CogniMed.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "ui"))

from ui.app import build_ui

demo = build_ui()

if __name__ == "__main__":
    demo.queue().launch()
