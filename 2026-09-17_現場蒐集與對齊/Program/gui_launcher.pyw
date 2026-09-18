from __future__ import annotations
import os, runpy, sys, traceback
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / 'Data' / 'Logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

def fail(target: str, exc: BaseException) -> None:
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log = LOG_DIR / f'gui_error_{stamp}.log'
    text = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log.write_text(f'Target: {target}\n\n{text}', encoding='utf-8')
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror('HoloGrip 啟動失敗', f'程式啟動失敗。\n\n{exc}\n\n完整錯誤已保存：\n{log}')
        root.destroy()
    except Exception:
        pass

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: gui_launcher.pyw <python-script>')
    target = str(Path(sys.argv[1]).resolve())
    os.chdir(str(ROOT))
    sys.path.insert(0, str(Path(target).parent))
    sys.argv = [target]
    try:
        runpy.run_path(target, run_name='__main__')
    except SystemExit:
        raise
    except BaseException as exc:
        fail(target, exc)
        raise SystemExit(1)
