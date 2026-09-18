"""Generate the 0812 Song 2 MIDI/CSV/video review page."""

from __future__ import annotations

import json
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parents[1]
sys.path.insert(0, str(APP_DIR))

from midi_csv_alignment_gui_0809_ZH_TW import (  # noqa: E402
    FRONTEND_TEMPLATE,
    build_frontend_data,
    render_frontend_html,
)
import midi_label_pipeline as pipeline  # noqa: E402


CSV_PATH = PROJECT_ROOT / "Data" / "Raw" / "Song_Collection_COM" / "S20260812_P01_song02_raw_100hz_20260812_112101.csv"
MIDI_PATH = Path(r"C:\Users\Typelin_Station\Downloads\0812T1127_去除雜訊T1213.mid")
VIDEO_PATH = Path(r"C:\Users\Typelin_Station\Downloads\0812T1127_去除雜訊T1213.MOV")
OUTPUT_DIR = PROJECT_ROOT / "Data" / "Derived" / "Song_Collection_COM" / "S20260812_P01_song02_T1127_cleaned"
PRELOADED_VIDEO_PATH = OUTPUT_DIR / "S20260812_P01_song02_T1127_video_h264.mp4"
PRELOADED_VIDEO_SRC = f"./{PRELOADED_VIDEO_PATH.name}"
EVENT_CSV = OUTPUT_DIR / "midi_events.csv"
AUDIO_REPORT = OUTPUT_DIR / "video_audio_alignment_0815" / "audio_alignment_report_0808.json"
OUTPUT_HTML = OUTPUT_DIR / "HoloGrip_Song2_影片_MIDI_CSV_預載對齊_0815_ZH_TW.html"

BPM = 110.0
WINDOW_RADIUS_MS = 80
CSV_OFFSET_MS = 6_027
VIDEO_OFFSET_MS = 4_960


def posix(path: Path) -> str:
    return path.resolve().as_posix()


def replace_context(html: str) -> str:
    replacements = {
        "MIDI ↔ 手套 CSV 對齊工具｜0805": "Song 2｜MIDI ↔ CSV ↔ 影片對齊工具",
        "這個工具使用 0805 實際資料：MIDI 的 1,820 個鼓點事件，對照雙手 100Hz CSV。": "本頁使用 0812 Song 2：311 筆清理版 MIDI，對照雙手 100Hz Raw CSV 與演奏影片。",
        "目前使用內嵌 0805 MIDI：Drum Midi_110BPM (0805).mid": "目前使用 Song 2 清理版 MIDI：0812T1127_去除雜訊T1213.mid",
        "目前使用內嵌 0805 CSV：S20260805_P01_song01_raw_100hz...": "目前使用 Song 2 CSV：S20260812_P01_song02_raw_100hz_20260812_112101.csv",
        "恢復內嵌 0805 範例資料": "恢復 Song 2 內嵌資料",
        "目前顯示內嵌 0805 範例資料。": "目前顯示 Song 2 清理版資料。",
        "0805 alignment review": "0812 Song 2 alignment review",
        "embedded-0805": "embedded-song2",
        "依 0805 實際 MIDI 311 筆與 Raw CSV 稽核結果": "依 0812 Song 2 清理版 MIDI 311 筆與 Raw CSV 稽核結果",
        "MIDI ↔ CSV alignment review｜來源：": "Song 2 MIDI ↔ CSV 對齊檢查｜來源：",
        "value=\"16000\" data-play-range": "value=\"10351\" data-play-range",
        "max=\"373125\" step=\"10\"": "max=\"80219\" step=\"10\"",
        "0:16.000 / 6:13.125": "0:10.351 / 1:20.219",
        "目前顯示整首歌 0:00 ～ 6:13，共 1,820 筆 MIDI 事件。": "目前顯示整首歌 0:00 ～ 1:29，共 311 筆 MIDI 事件。",
        "依 0805 實際 MIDI 1,820 筆與 Raw CSV 稽核結果": "依 0812 Song 2 清理版 MIDI 311 筆與 Raw CSV 稽核結果",
        "目前顯示整首歌 0:00 ～ 6:13，共 311 筆 MIDI 事件。": "目前顯示整首歌 0:00 ～ 1:29，共 311 筆 MIDI 事件。",
        "value=\"90\" inputmode=\"numeric\"": "value=\"80\" inputmode=\"numeric\"",
        "data-triad-play-radius>90": "data-triad-play-radius>80",
        "採樣窗口 ±90 ms": "採樣窗口 ±80 ms",
        "+16.000 秒": "+6.027 秒",
        "16,000 ms": "6,027 ms",
        'value="16000" data-offset': 'value="6027" data-offset',
        'value="14520" inputmode="numeric"': 'value="4960" inputmode="numeric"',
        "+14,520 ms": "+4,960 ms",
        "影片 14.5 秒來源": "影片 4.960 秒來源",
        "影片 +14,520 ms 是影片音訊、外部 WAV 與 MIDI 比較後的目前候選": "影片 +4,960 ms 是影片內建音訊與清理版 MIDI 比較後的目前候選",
        "目前 WAV 分析摘要": "影片內建音訊分析摘要",
        "2,754 個": "429 個",
        "1,632 / 1,820（89.7%）": "311 / 311（100.0%）",
        "77 / 148 對有兩個 onset": "0 / 0 對有兩個 onset",
        "快速 onset 稽核顯示 148 對同一 note 在 20 ms 內，其中 77 對有兩個音訊 onset 候選，仍需影片確認。": "清理版 MIDI 沒有同一 note 在 20 ms 內重複事件；影片音訊 311 / 311 命中。",
        "1,820": "311",
        "C:/Users/Typelin_Station/Desktop/HoloGrip/Data/External/FlowAudio_20260805/IMG_6060.MOV": posix(PRELOADED_VIDEO_PATH),
        "C:/Users/Typelin_Station/Desktop/HoloGrip/Data/External/FlowAudio_20260805/Drum Midi_110BPM (0805).mid": posix(MIDI_PATH),
        "C:/Users/Typelin_Station/Desktop/HoloGrip/Data/External/FlowAudio_20260805/Drum Audio_110BPM (0805).wav": "Song 2 WAV 未指定；需要聲音核對時請手動選取",
        "C:/Users/Typelin_Station/Desktop/HoloGrip/Data/Raw/Song_Collection_COM/S20260805_P01_song01_raw_100hz_20260805_161628.csv": posix(CSV_PATH),
        "C:/Users/Typelin_Station/Desktop/HoloGrip/Data/Derived/Song_Collection_COM/S20260805_P01_song01/audio_alignment_0808/audio_alignment_report_0808.json": posix(AUDIO_REPORT),
    }
    for old, new in replacements.items():
        html = html.replace(old, new)
    # Some legacy captions are transformed by the earlier 1,820 -> 311 replacement.
    # Normalize those final visible captions after all numeric substitutions.
    html = html.replace(
        "依 0805 實際 MIDI 311 筆與 Raw CSV 稽核結果",
        "依 0812 Song 2 清理版 MIDI 311 筆與 Raw CSV 稽核結果",
    )
    html = html.replace(
        "目前顯示整首歌 0:00 ～ 6:13，共 311 筆 MIDI 事件。",
        "目前顯示整首歌 0:00 ～ 1:29，共 311 筆 MIDI 事件。",
    )
    return html


def song2_bridge() -> str:
    config = {
        "csvOffsetMs": CSV_OFFSET_MS,
        "videoOffsetMs": VIDEO_OFFSET_MS,
        "windowRadiusMs": WINDOW_RADIUS_MS,
        "midi": posix(MIDI_PATH),
        "csv": posix(CSV_PATH),
        "video": posix(PRELOADED_VIDEO_PATH),
        "videoSource": posix(VIDEO_PATH),
        "videoRelativeSrc": PRELOADED_VIDEO_SRC,
        "audioReport": posix(AUDIO_REPORT),
    }
    return f"""
<script>
(() => {{
  window.__holoSong2Alignment = {json.dumps(config, ensure_ascii=False)};
  const setText = (selector, value) => {{
    const node = document.querySelector(selector);
    if (node) node.textContent = value;
  }};
  const csvOffset = document.querySelector('[data-offset]');
  if (csvOffset) {{
    csvOffset.value = '{CSV_OFFSET_MS}';
    csvOffset.dispatchEvent(new Event('input', {{bubbles: true}}));
  }}
  const videoOffset = document.querySelector('#triad-video-offset');
  if (videoOffset) {{
    videoOffset.value = '{VIDEO_OFFSET_MS}';
    videoOffset.dispatchEvent(new Event('input', {{bubbles: true}}));
  }}
  const radius = document.querySelector('#sample-window-ms');
  if (radius) {{
    radius.value = '{WINDOW_RADIUS_MS}';
    radius.dispatchEvent(new Event('input', {{bubbles: true}}));
  }}
  setText('[data-loader-status]', 'Song 2 清理版 MIDI 與 Raw CSV 已內嵌；同目錄 H.264 影片會自動載入。');
  setText('[data-triad-status]', 'Song 2 預載對齊：CSV +6,027 ms｜影片 +4,960 ms｜等待影片就緒');
  setText('[data-triad-stage-summary]', 'CSV 偏移 +6,027 ms｜影片偏移 +4,960 ms｜採樣窗口 ±80 ms');
  setText('[data-report-offset]', '+4,960 ms');
  setText('[data-report-onsets]', '429 個');
  setText('[data-report-match]', '311 / 311（100.0%）');
  setText('[data-report-dense]', '0 / 0 對同音重複');
  setText('[data-report-note]', '影片內建音訊容許誤差 ±35 ms；311 / 311 MIDI 事件命中。音訊只確認時間，不直接確認左右手。');
  setText('[data-triad-play-radius]', '{WINDOW_RADIUS_MS}');
  setText('[data-window-radius]', '總長度 {WINDOW_RADIUS_MS * 2} ms｜100Hz 約 {WINDOW_RADIUS_MS * 2 // 10 + 1} 點');
  const sourceCard = document.querySelector('[data-triad-video-file]')?.closest('.triad-file');
  const sourcePath = sourceCard?.querySelector('[data-default-path]');
  if (sourcePath) sourcePath.textContent = '同目錄預載影片：{PRELOADED_VIDEO_SRC}';
  const audioCard = document.querySelector('[data-triad-audio-file]')?.closest('.triad-file');
  const audioPath = audioCard?.querySelector('[data-default-path]');
  if (audioPath) audioPath.textContent = '本次 Song 2 未指定 WAV；需要聲音核對時請手動選取。';
  const reportCard = document.querySelector('[data-triad-report-file]')?.closest('.triad-file');
  const reportPath = reportCard?.querySelector('[data-default-path]');
  if (reportPath) reportPath.textContent = '分析報告位置：{posix(AUDIO_REPORT)}';
  const videoName = document.querySelector('[data-triad-video-name]');
  if (videoName) videoName.textContent = '正在自動載入：{PRELOADED_VIDEO_PATH.name}；若瀏覽器未顯示，仍可手動選取影片。';
  const rule = document.querySelector('.dense-review-rule');
  if (rule) rule.textContent += ' 「下一個異常區」到最後一區後會停止並停用，不會繞回第一區。';
  // The initial template render can run before the browser finishes sizing the
  // chart containers. Render again after layout so MIDI event marks are visible
  // without requiring the user to press previous/next.
  const renderAfterLayout = () => {{
    if (typeof render !== 'function') return;
    render();
    requestAnimationFrame(() => requestAnimationFrame(() => render()));
  }};
  if (document.readyState === 'complete') renderAfterLayout();
  else window.addEventListener('load', renderAfterLayout, {{once: true}});
  window.setTimeout(renderAfterLayout, 180);
}})();
</script>
"""


def main() -> int:
    required = [CSV_PATH, MIDI_PATH, VIDEO_PATH, PRELOADED_VIDEO_PATH, AUDIO_REPORT, FRONTEND_TEMPLATE]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input:\n" + "\n".join(missing))

    report = pipeline.build_outputs(
        raw_csv=CSV_PATH,
        midi_path=MIDI_PATH,
        output_dir=OUTPUT_DIR,
        bpm=BPM,
        offset_ms=None,
        min_offset_ms=-5_000,
        max_offset_ms=None,
        pre_ms=WINDOW_RADIUS_MS,
        post_ms=WINDOW_RADIUS_MS,
    )
    (OUTPUT_DIR / "alignment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    selected_offset_ms = int(report["alignment"]["offset_ms"])
    if selected_offset_ms != CSV_OFFSET_MS:
        raise RuntimeError(
            f"Song 2 auto alignment changed: expected {CSV_OFFSET_MS} ms, got {selected_offset_ms} ms"
        )
    data = build_frontend_data(
        CSV_PATH,
        MIDI_PATH,
        BPM,
        selected_offset_ms,
        EVENT_CSV,
    )
    html = render_frontend_html(
        FRONTEND_TEMPLATE.read_text(encoding="utf-8"),
        data,
        auto_play=False,
        window_radius_ms=WINDOW_RADIUS_MS,
        media_defaults=True,
    )
    html = html.replace(
        'data-default-src="../../Data/External/FlowAudio_20260805/IMG_6060.MOV"',
        f'data-default-src="{PRELOADED_VIDEO_SRC}"',
        1,
    )
    html = html.replace(
        'data-triad-video data-default-src="" controls preload="metadata"',
        f'data-triad-video data-default-src="{PRELOADED_VIDEO_SRC}" controls preload="auto"',
        1,
    )
    html = html.replace(
        'data-default-src="../../Data/External/FlowAudio_20260805/Drum Audio_110BPM (0805).wav"',
        'data-default-src=""',
        1,
    )
    html = html.replace(
        "預設會嘗試載入 IMG_6060.MOV；若被 file:// 安全政策擋下，請在資料設定選取檔案。",
        "頁面會自動載入同目錄的 H.264 MP4；不需要手動選取影片。",
        1,
    )
    html = html.replace(
        "現場第一階段只驗證 MIDI＋CSV；需要影片時，請在資料設定手動選取檔案。",
        "頁面會自動載入同目錄的 H.264 MP4；不需要手動選取影片。",
        1,
    )
    html = html.replace(
        "預設會嘗試載入 Drum Audio WAV；若被安全政策擋下，請在資料設定選取檔案。",
        "本次沒有額外預載 WAV；影片內建聲音已保留。",
        1,
    )
    html = html.replace('preload="metadata"', 'preload="auto"', 1)
    html = replace_context(html)
    html = html.replace("</body></html>", song2_bridge() + "</body></html>", 1)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"frontend={OUTPUT_HTML.resolve()}")
    print(f"events={len(data['events'])}")
    print(f"csv_offset_ms={CSV_OFFSET_MS}")
    print(f"video_offset_ms={VIDEO_OFFSET_MS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
