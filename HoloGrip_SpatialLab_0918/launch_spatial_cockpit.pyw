from __future__ import annotations
import runpy
import sys
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import messagebox

HERE = Path(__file__).resolve().parent
TARGET = HERE / "spatial_cockpit_ui.py"
LOG_DIR = HERE / "logs"
LOG_DIR.mkdir(exist_ok=True)

try:
    sys.argv = [str(TARGET)]
    runpy.run_path(str(TARGET), run_name="__main__")
except Exception:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = LOG_DIR / f"gui_error_{stamp}.log"
    log.write_text(traceback.format_exc(), encoding="utf-8")
    try:
        messagebox.showerror("HoloGrip 空間診斷啟動失敗", f"錯誤紀錄：\n{log}")
    except Exception:
        pass
