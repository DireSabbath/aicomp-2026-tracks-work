from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = ROOT / "02_训练代码"
sys.path.insert(0, str(SCRIPT_DIR))
runpy.run_path(str(SCRIPT_DIR / "find_new_designs.py"), run_name="__main__")
