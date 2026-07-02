from pathlib import Path
import os
import runpy
import sys


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ROOT_DIR / "Resume-Screening-RAG-Pipeline-main"
DEMO_DIR = PROJECT_DIR / "demo"

os.chdir(PROJECT_DIR)
sys.path.insert(0, str(DEMO_DIR))

runpy.run_path(str(DEMO_DIR / "interface.py"), run_name="__main__")
