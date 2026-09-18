"""現場 MIDI + Raw CSV 對齊 GUI。

拖入一個 MIDI 與一個 Raw CSV 後，呼叫既有 midi_label_pipeline 找候選偏移，
再產生一份已嵌入新資料的前端檢查頁。候選結果仍須由人員在前端確認。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import threading
import time
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import midi_label_pipeline as pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parent
FRONTEND_TEMPLATE = APP_DIR / "HoloGrip_MIDI_CSV對齊檢查_0807_ZH_TW.html"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "Data" / "Derived" / "Song_Collection_COM"
DEFAULT_BPM = 110.0
DEFAULT_WINDOW_MS = 80
DISPLAY_BIN_MS = 100

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    HAS_DND = True
except ImportError:
    DND_FILES = None
    TkinterDnD = None
    HAS_DND = False


def signed_ms(value: float | int) -> str:
    number = float(value)
    return f"{number:+,.0f} ms"


def read_event_hints(path: Path) -> dict[int, dict[str, Any]]:
    """Read the pipeline event CSV when available to seed frontend tooltips."""
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        hints: dict[int, dict[str, Any]] = {}
        for row in rows:
            try:
                event_id = int(row["event_id"])
            except (KeyError, TypeError, ValueError):
                continue
            hints[event_id] = {
                "hand": row.get("hand_candidate", "?"),
                "conf": float(row.get("hand_confidence", 0) or 0),
            }
        return hints


def build_frontend_data(
    raw_csv: Path,
    midi_path: Path,
    bpm: float,
    offset_ms: int,
    event_csv: Path | None = None,
) -> dict[str, Any]:
    """Build the DATA object expected by the existing frontend page."""
    samples, raw_info = pipeline.read_raw_csv(raw_csv)
    arrays = pipeline._build_energy(
        samples,
        raw_info["duration_ms"],
    )
    energy = [
        {"t": index * pipeline.BIN_MS, "l": arrays["L"][index], "r": arrays["R"][index]}
        for index in range(len(arrays["L"]))
    ]
    display_buckets: dict[int, list[float]] = {}
    for point in energy:
        start = (int(point["t"]) // DISPLAY_BIN_MS) * DISPLAY_BIN_MS
        bucket = display_buckets.setdefault(start, [0.0, 0.0, 0.0])
        bucket[0] += float(point["l"])
        bucket[1] += float(point["r"])
        bucket[2] += 1.0
    display_energy = [
        {"t": start, "l": round(values[0] / values[2], 4), "r": round(values[1] / values[2], 4)}
        for start, values in sorted(display_buckets.items())
        if values[2]
    ]
    parsed = pipeline.parse_midi(midi_path, bpm)
    hints = read_event_hints(event_csv) if event_csv else {}
    events = []
    for event in parsed["events"]:
        hint = hints.get(event.event_id, {})
        events.append({
            "id": event.event_id,
            "mt": event.time_ms,
            "note": event.note,
            "vel": event.velocity,
            "zone": event.zone_id,
            "name": event.zone_name,
            "hand": hint.get("hand", "?"),
            "conf": hint.get("conf", 0),
        })
    return {
        "events": events,
        "energy": energy,
        "displayEnergy": display_energy,
        "duration": raw_info["duration_ms"],
        "bpm": bpm,
        "offset": offset_ms,
        "status": "pending_manual_alignment_review",
        "source": raw_csv.stem,
        "midiSource": midi_path.name,
        "csvSource": raw_csv.name,
        "energyBinMs": pipeline.BIN_MS,
        "displayEnergyBinMs": DISPLAY_BIN_MS,
        "timeBase": raw_info["time_base"],
        "timeBaseDescription": raw_info["time_base_description"],
        "sensorOriginMs": raw_info["sensor_origin_ms"],
        "sensorOriginsByHandMs": raw_info["sensor_origins_by_hand_ms"],
        "sensorOriginGapMs": raw_info["sensor_origin_gap_ms"],
        "songDurationMs": raw_info["song_duration_ms"],
    }


def render_frontend_html(
    template_text: str,
    data: dict[str, Any],
    auto_play: bool,
    window_radius_ms: int = DEFAULT_WINDOW_MS,
    media_defaults: bool = False,
) -> str:
    marker = "const DATA="
    data_start = template_text.find(marker)
    if data_start < 0:
        raise ValueError("前端模板找不到 DATA 資料區")
    data_start += len(marker)
    data_end = template_text.find(";const MAPPING", data_start)
    if data_end < 0:
        data_end = template_text.find(";\nconst MAPPING", data_start)
    if data_end < 0:
        raise ValueError("前端模板找不到 MAPPING 資料區")
    compact_data = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    rendered = template_text[:data_start] + compact_data + template_text[data_end:]
    if not media_defaults:
        rendered = rendered.replace(
            'data-default-src="../../Data/External/FlowAudio_20260805/IMG_6060.MOV"',
            'data-default-src=""',
        )
        rendered = rendered.replace(
            'data-default-src="../../Data/External/FlowAudio_20260805/Drum Audio_110BPM (0805).wav"',
            'data-default-src=""',
        )
        rendered = rendered.replace(
            "預設會嘗試載入 IMG_6060.MOV；若被 file:// 安全政策擋下，請在資料設定選取檔案。",
            "現場第一階段只驗證 MIDI＋CSV；需要影片時，請在資料設定手動選取檔案。",
        )
        rendered = rendered.replace(
            "預設會嘗試載入 Drum Audio WAV；若被安全政策擋下，請在資料設定選取檔案。",
            "現場第一階段不自動載入 WAV；需要聲音核對時，請在資料設定手動選取檔案。",
        )
    existing_bridge_marker = "GUI 已載入候選偏移"
    existing_bridge_position = rendered.find(existing_bridge_marker)
    if existing_bridge_position >= 0:
        existing_bridge_start = rendered.rfind("<script", 0, existing_bridge_position)
        existing_bridge_end = rendered.find("</script>", existing_bridge_position)
        if existing_bridge_start >= 0 and existing_bridge_end >= 0:
            rendered = rendered[:existing_bridge_start] + rendered[existing_bridge_end + len("</script>"):]
    auto_play_js = "true" if auto_play else "false"
    bridge = f"""
<script>
(() => {{
  const offset = Number(DATA.offset);
  const offsetInput = document.querySelector('[data-offset]');
  if (Number.isFinite(offset) && offsetInput) {{
    offsetInput.value = String(offset);
    offsetInput.dispatchEvent(new Event('input', {{bubbles: true}}));
  }}
  const radiusInput = document.querySelector('#sample-window-ms');
  if (radiusInput) {{
    radiusInput.value = '{window_radius_ms}';
    radiusInput.dispatchEvent(new Event('input', {{bubbles: true}}));
  }}
  const status = document.querySelector('[data-loader-status]');
  if (status) status.textContent = 'GUI 已載入候選偏移 ' + offset.toLocaleString() + ' ms；請用圖表確認。';
  const displayBin = Number(DATA.displayEnergyBinMs) || 100;
  const rawBin = Number(DATA.energyBinMs) || 10;
  const method = document.querySelector('[data-window-method]');
   if (method) method.textContent += ' 10 ms＝以共同 song_time_ms 建立的左右手原始最大 activity；20/50/100 ms＝10 ms 格的算術平均，只影響畫面，不參與對齊。';
  const sampleRow = document.querySelector('.sample-window-row');
  if (sampleRow && !document.querySelector('#plot-bin-ms')) {{
    const field = document.createElement('label');
    field.className = 'sample-window-field';
    field.textContent = '折線預覽解析度（只影響畫面）';
    const select = document.createElement('select');
    select.id = 'plot-bin-ms';
    select.setAttribute('aria-label', '折線預覽解析度');
    [10, 20, 50, 100].forEach(bin => {{
      const option = document.createElement('option');
      option.value = String(bin);
      option.textContent = bin + ' ms';
      select.appendChild(option);
    }});
    select.value = '10';
    select.addEventListener('change', () => {{ if (typeof render === 'function') render(); }});
    field.appendChild(select);
    sampleRow.appendChild(field);
  }}
  window.__holoPlotBinMs = () => Math.max(10, Number(document.querySelector('#plot-bin-ms')?.value) || 10);
  window.__holoPlotEnergy = data => {{
    const binMs = window.__holoPlotBinMs();
    const raw = Array.isArray(data?.energy) ? data.energy : [];
    if (binMs === 10) return raw;
    if (binMs === 100 && Array.isArray(data?.displayEnergy)) return data.displayEnergy;
    const buckets = new Map();
    for (const point of raw) {{
      const start = Math.floor((Number(point.t) || 0) / binMs) * binMs;
      const slot = buckets.get(start) || {{t: start, l: 0, r: 0, n: 0}};
      slot.l += Number(point.l) || 0;
      slot.r += Number(point.r) || 0;
      slot.n += 1;
      buckets.set(start, slot);
    }}
    return [...buckets.values()].map(slot => ({{
      t: slot.t,
      l: +(slot.l / Math.max(1, slot.n)).toFixed(4),
      r: +(slot.r / Math.max(1, slot.n)).toFixed(4),
    }}));
  }};
  const sourceNodes = document.querySelectorAll('[data-triad-video-file] ~ [data-default-path], [data-triad-audio-file] ~ [data-default-path], [data-triad-report-file] ~ [data-default-path]');
  sourceNodes.forEach(node => {{ node.textContent = '現場第一階段不自動載入；需要時請手動選取檔案。'; }});
  const sourceText = document.querySelector('[data-triad-video-file]')?.closest('.triad-file')?.parentElement?.querySelector('.triad-file:last-child');
  if (sourceText && DATA.midiSource && DATA.csvSource) sourceText.innerHTML = '<span class="triad-path">MIDI：' + DATA.midiSource + '</span><span class="triad-path">CSV：' + DATA.csvSource + '</span>';
  if (!{str(media_defaults).lower()}) {{
    const triadStatus = document.querySelector('[data-triad-status]');
    if (triadStatus) triadStatus.textContent = '目前為 MIDI＋CSV 第一階段；影片／WAV 尚未載入';
  }}
  if (typeof render === 'function') render();
  if ({auto_play_js}) window.setTimeout(() => document.querySelector('[data-play]')?.click(), 900);
}})();
</script>
"""
    closing = "</body></html>"
    if closing not in rendered:
        raise ValueError("前端模板缺少 body 結尾")
    return rendered.replace(closing, bridge + closing, 1)


def default_output_dir() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_ROOT / f"現場MIDI_CSV對齊_0809_ZH_TW_{stamp}"


class AlignmentGui:
    def __init__(self, root: tk.Misc) -> None:
        self.root = root
        self.root.title("HoloGrip 現場 MIDI + CSV 自動對齊工具")
        self.root.geometry("980x760")
        self.root.minsize(760, 620)
        self.queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.midi_path: Path | None = None
        self.csv_path: Path | None = None
        self.output_dir: Path | None = None
        self.frontend_path: Path | None = None
        self.report: dict[str, Any] | None = None
        self.midi_var = tk.StringVar(value="尚未選擇 MIDI")
        self.csv_var = tk.StringVar(value="尚未選擇 Raw CSV")
        self.output_var = tk.StringVar(value=str(default_output_dir()))
        self.bpm_var = tk.StringVar(value=str(DEFAULT_BPM))
        self.window_var = tk.StringVar(value=str(DEFAULT_WINDOW_MS))
        self.search_min_var = tk.StringVar(value="-5000")
        self.search_max_var = tk.StringVar(value="")
        self.auto_play_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="請拖入 MIDI 與 CSV，或用按鈕選擇檔案。")
        self._build_style()
        self._build_ui()
        self.root.after(120, self._poll_queue)

    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Heading.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Drop.TFrame", background="#f4f9fd")

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="HoloGrip 現場 MIDI + CSV 自動對齊", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            outer,
            text="演奏完成後拖入兩個檔案，估計 MIDI → CSV 候選偏移，再開啟已連動的對齊前端。",
        ).pack(anchor="w", pady=(4, 14))

        files = ttk.LabelFrame(outer, text="1. 放入現場檔案", padding=12)
        files.pack(fill="x")
        self._make_drop_zone(files, "MIDI 演奏檔（.mid / .midi）", self.midi_var, "midi")
        self._make_drop_zone(files, "手套 Raw CSV（.csv）", self.csv_var, "csv")
        dnd_note = "可直接拖曳檔案；" + ("已啟用拖放。" if HAS_DND else "目前未安裝拖放套件，請用選擇檔案按鈕。")
        ttk.Label(files, text=dnd_note, foreground="#66788a").pack(anchor="w", pady=(8, 0))

        params = ttk.LabelFrame(outer, text="2. 對齊參數", padding=12)
        params.pack(fill="x", pady=(14, 0))
        grid = ttk.Frame(params)
        grid.pack(fill="x")
        self._entry(grid, "BPM", self.bpm_var, 0, "110")
        self._entry(grid, "模型窗口半徑（ms）", self.window_var, 1, "80")
        self._entry(grid, "搜尋偏移最小值（ms）", self.search_min_var, 2, "-5000")
        self._entry(grid, "搜尋偏移最大值（ms，可空白）", self.search_max_var, 3, "自動依資料長度")
        output = ttk.Frame(params)
        output.pack(fill="x", pady=(10, 0))
        ttk.Label(output, text="輸出資料夾").pack(side="left")
        ttk.Entry(output, textvariable=self.output_var).pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(output, text="選擇資料夾", command=self._choose_output).pack(side="left")
        ttk.Checkbutton(
            params,
            text="開啟前端後，自動播放 MIDI + CSV 時間軸一次",
            variable=self.auto_play_var,
        ).pack(anchor="w", pady=(10, 0))

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(14, 0))
        self.find_button = ttk.Button(actions, text="尋找 MIDI → CSV 對齊", command=self._start_alignment)
        self.find_button.pack(side="left")
        self.open_frontend_button = ttk.Button(actions, text="開啟已連動前端", command=self._open_frontend, state="disabled")
        self.open_frontend_button.pack(side="left", padx=8)
        self.open_folder_button = ttk.Button(actions, text="開啟輸出資料夾", command=self._open_output, state="disabled")
        self.open_folder_button.pack(side="left")

        result = ttk.LabelFrame(outer, text="3. 對齊結果", padding=12)
        result.pack(fill="both", expand=True, pady=(14, 0))
        self.result_label = ttk.Label(result, text="尚未執行對齊。", style="Heading.TLabel")
        self.result_label.pack(anchor="w")
        self.result_text = tk.Text(result, height=12, wrap="word", state="disabled", font=("Consolas", 10))
        self.result_text.pack(fill="both", expand=True, pady=(8, 0))
        ttk.Label(outer, textvariable=self.status_var, foreground="#246b43").pack(anchor="w", pady=(10, 0))

    def _make_drop_zone(self, parent: ttk.Widget, title: str, variable: tk.StringVar, kind: str) -> None:
        frame = tk.Frame(parent, bg="#f4f9fd", highlightbackground="#c9d9e8", highlightthickness=1)
        frame.pack(fill="x", pady=4)
        left = tk.Frame(frame, bg="#f4f9fd")
        left.pack(side="left", fill="both", expand=True, padx=12, pady=10)
        tk.Label(left, text=title, bg="#f4f9fd", fg="#17324d", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(left, textvariable=variable, bg="#f4f9fd", fg="#66788a", anchor="w", justify="left", wraplength=700).pack(fill="x", pady=(3, 0))
        ttk.Button(frame, text="選擇檔案", command=lambda: self._choose_file(kind)).pack(side="right", padx=12)
        if HAS_DND:
            frame.drop_target_register(DND_FILES)
            frame.dnd_bind("<<Drop>>", lambda event, file_kind=kind: self._drop_files(event, file_kind))

    def _entry(self, parent: ttk.Widget, label: str, variable: tk.StringVar, column: int, placeholder: str) -> None:
        cell = ttk.Frame(parent)
        cell.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 8, 0))
        parent.columnconfigure(column, weight=1)
        ttk.Label(cell, text=label).pack(anchor="w")
        ttk.Entry(cell, textvariable=variable).pack(fill="x", pady=(4, 0))
        ttk.Label(cell, text=placeholder, foreground="#66788a").pack(anchor="w")

    def _choose_file(self, kind: str) -> None:
        if kind == "midi":
            path = filedialog.askopenfilename(filetypes=[("MIDI", "*.mid *.midi"), ("所有檔案", "*.*")])
        else:
            path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("所有檔案", "*.*")])
        if path:
            self._set_path(kind, Path(path))

    def _drop_files(self, event: Any, kind: str) -> None:
        for raw_path in self.root.tk.splitlist(event.data):
            path = Path(raw_path.strip("{}"))
            if path.is_file() and ((kind == "midi" and path.suffix.lower() in {".mid", ".midi"}) or (kind == "csv" and path.suffix.lower() == ".csv")):
                self._set_path(kind, path)
                return
        messagebox.showwarning("檔案格式不符", f"請拖入{kind.upper()} 檔案。")

    def _set_path(self, kind: str, path: Path) -> None:
        if kind == "midi":
            self.midi_path = path
            self.midi_var.set(str(path))
        else:
            self.csv_path = path
            self.csv_var.set(str(path))
        self.status_var.set("檔案已放入，請確認兩個檔案後開始尋找對齊。")

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(initialdir=str(DEFAULT_OUTPUT_ROOT))
        if path:
            self.output_var.set(path)

    def _number(self, variable: tk.StringVar, label: str, integer: bool = False) -> float | int:
        raw = variable.get().strip()
        if not raw:
            raise ValueError(f"{label}不能空白")
        value = float(raw)
        if integer:
            return int(value)
        return value

    def _start_alignment(self) -> None:
        if not self.midi_path or not self.midi_path.is_file():
            messagebox.showwarning("缺少 MIDI", "請先拖入或選擇 MIDI 演奏檔。")
            return
        if not self.csv_path or not self.csv_path.is_file():
            messagebox.showwarning("缺少 CSV", "請先拖入或選擇手套 Raw CSV。")
            return
        try:
            bpm = float(self.bpm_var.get())
            window = int(float(self.window_var.get()))
            search_min = int(float(self.search_min_var.get()))
            search_max = int(float(self.search_max_var.get())) if self.search_max_var.get().strip() else None
            if bpm <= 0 or window < 10 or search_max is not None and search_max <= search_min:
                raise ValueError("BPM、窗口或搜尋範圍不合理")
        except ValueError as error:
            messagebox.showwarning("參數錯誤", str(error))
            return
        output_dir = Path(self.output_var.get().strip()) if self.output_var.get().strip() else default_output_dir()
        midi_path = self.midi_path
        csv_path = self.csv_path
        auto_play = self.auto_play_var.get()
        self.output_dir = output_dir
        self.find_button.config(state="disabled")
        self.open_frontend_button.config(state="disabled")
        self.open_folder_button.config(state="disabled")
        self.status_var.set("正在讀取檔案並掃描候選偏移，請稍候。")
        self._write_result("正在分析中……\n\nPython 會掃描整體偏移，找 MIDI 鼓點與手套活動度最吻合的候選位置。")

        def worker() -> None:
            try:
                report = pipeline.build_outputs(
                    raw_csv=csv_path,
                    midi_path=midi_path,
                    output_dir=output_dir,
                    bpm=bpm,
                    offset_ms=None,
                    min_offset_ms=search_min,
                    max_offset_ms=search_max,
                    pre_ms=window,
                    post_ms=window,
                )
                report_path = output_dir / "MIDI_CSV對齊報告_0809_ZH_TW.json"
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                data = build_frontend_data(
                    csv_path,
                    midi_path,
                    bpm,
                    int(report["alignment"]["offset_ms"]),
                    output_dir / "midi_events.csv",
                )
                frontend_text = render_frontend_html(FRONTEND_TEMPLATE.read_text(encoding="utf-8"), data, auto_play, window, media_defaults=False)
                frontend_path = output_dir / "HoloGrip_MIDI_CSV對齊現場檢查_0809_ZH_TW.html"
                frontend_path.write_text(frontend_text, encoding="utf-8")
                manifest = {
                    "midi": str(midi_path.resolve()),
                    "raw_csv": str(csv_path.resolve()),
                    "output_dir": str(output_dir.resolve()),
                    "alignment_report": str(report_path.resolve()),
                    "frontend": str(frontend_path.resolve()),
                    "offset_ms": report["alignment"]["offset_ms"],
                    "window_radius_ms": window,
                    "bpm": bpm,
                    "auto_play": auto_play,
                    "status": report["status"],
                }
                (output_dir / "現場MIDI_CSV對齊工作區_0809_ZH_TW.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
                self.queue.put(("success", report, frontend_path, output_dir, report_path))
            except Exception as error:  # noqa: BLE001 - show every field/parse failure in the GUI.
                self.queue.put(("error", error))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_queue(self) -> None:
        try:
            item = self.queue.get_nowait()
        except queue.Empty:
            self.root.after(120, self._poll_queue)
            return
        self.find_button.config(state="normal")
        if item[0] == "error":
            error = item[1]
            self.status_var.set("分析失敗，請檢查檔案格式或 CSV 欄位。")
            self._write_result(f"分析失敗：{error}")
            messagebox.showerror("MIDI + CSV 對齊失敗", str(error))
        else:
            _, report, frontend_path, output_dir, report_path = item
            self.report = report
            self.frontend_path = frontend_path
            self.output_dir = output_dir
            self.open_frontend_button.config(state="normal")
            self.open_folder_button.config(state="normal")
            alignment = report["alignment"]
            midi = report["midi"]
            raw = report["raw_csv"]
            top = "\n".join(
                f"  {candidate['offset_ms']:+.0f} ms｜活動度分數 {candidate['mean_motion_score']:.4f}｜命中率 {candidate['active_ratio'] * 100:.1f}%"
                for candidate in alignment.get("top_candidates", [])[:5]
            )
            result = (
                f"候選偏移：{signed_ms(alignment['offset_ms'])}\n"
                f"模型窗口：±{alignment['pre_window_ms']} ms（前後分開固定切窗）\n"
                f"MIDI 事件：{midi['mapped_event_count']:,} 筆｜Raw CSV：{raw['row_count']:,} 列\n"
                f"狀態：候選對齊，仍需在前端看開頭／中段／結尾確認\n\n"
                f"前 5 個候選偏移：\n{top}\n\n"
                f"已輸出報告：{report_path}\n"
                f"已產生前端：{frontend_path}\n\n"
                "下一步：按「開啟已連動前端」，確認 MIDI 點是否落在 CSV 活動峰上。"
            )
            self.result_label.config(text=f"完成：候選偏移 {signed_ms(alignment['offset_ms'])}")
            self._write_result(result)
            self.status_var.set("對齊候選完成；請開啟前端做人工確認。")
        self.root.after(120, self._poll_queue)

    def _write_result(self, text: str) -> None:
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.config(state="disabled")

    def _open_frontend(self) -> None:
        if self.frontend_path and self.frontend_path.is_file():
            os.startfile(str(self.frontend_path))

    def _open_output(self) -> None:
        if self.output_dir and self.output_dir.is_dir():
            os.startfile(str(self.output_dir))


def create_root() -> tk.Misc:
    return TkinterDnD.Tk() if HAS_DND and TkinterDnD else tk.Tk()


def self_test() -> None:
    midi = PROJECT_ROOT / "Data" / "External" / "FlowAudio_20260805" / "Drum Midi_110BPM (0805).mid"
    csv_path = PROJECT_ROOT / "Data" / "Raw" / "Song_Collection_COM" / "S20260805_P01_song01_raw_100hz_20260805_161628.csv"
    data = build_frontend_data(csv_path, midi, DEFAULT_BPM, 16_030)
    if len(data["events"]) != 1_820:
        raise AssertionError(f"unexpected event count: {len(data['events'])}")
    if not data["energy"] or data["duration"] <= 0:
        raise AssertionError("frontend energy data is empty")
    if len(data["displayEnergy"]) >= len(data["energy"]):
        raise AssertionError("display energy was not downsampled")
    rendered = render_frontend_html(FRONTEND_TEMPLATE.read_text(encoding="utf-8"), data, True)
    if "const DATA=" not in rendered or "16030" not in rendered:
        raise AssertionError("frontend session bridge was not rendered")
    if 'data-default-src="../../Data/External/' in rendered:
        raise AssertionError("field frontend still has stale media defaults")
    if "displayEnergy" not in rendered:
        raise AssertionError("display energy was not embedded")
    print("midi_csv_alignment_gui self-test: OK")


def main() -> int:
    parser = argparse.ArgumentParser(description="HoloGrip 現場 MIDI + CSV 自動對齊 GUI")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    root = create_root()
    AlignmentGui(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
