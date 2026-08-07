[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot

function Ensure-Directory {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Move-Exact {
    param(
        [string]$Source,
        [string]$Destination
    )
    if (-not (Test-Path -LiteralPath $Source)) {
        if (Test-Path -LiteralPath $Destination) {
            Write-Output "Already moved: $Destination"
            return
        }
        throw "Missing expected source: $Source"
    }
    if (Test-Path -LiteralPath $Destination) {
        throw "Destination already exists: $Destination"
    }
    Ensure-Directory (Split-Path -Parent $Destination)
    Move-Item -LiteralPath $Source -Destination $Destination
}

function Move-DirectoryContents {
    param(
        [string]$Source,
        [string]$Destination
    )
    if (-not (Test-Path -LiteralPath $Source)) {
        if (Test-Path -LiteralPath $Destination) {
            Write-Output "Already moved: $Destination"
            return
        }
        throw "Missing expected source: $Source"
    }
    Ensure-Directory $Destination
    if (Get-ChildItem -LiteralPath $Destination -Force | Select-Object -First 1) {
        throw "Destination must be empty before moving directory contents: $Destination"
    }
    Get-ChildItem -LiteralPath $Source -Force | ForEach-Object {
        Move-Item -LiteralPath $_.FullName -Destination $Destination
    }
    Remove-EmptyDirectory $Source
}

function Remove-EmptyDirectory {
    param([string]$Path)
    if ((Test-Path -LiteralPath $Path) -and -not (Get-ChildItem -LiteralPath $Path -Force | Select-Object -First 1)) {
        Remove-Item -LiteralPath $Path
    }
}

$directories = @(
    'Apps',
    'Firmware',
    'Data\\Raw\\Legacy',
    'Data\\Derived',
    'Data\\External',
    'Docs\\Current',
    'Docs\\Logs',
    'Docs\\Archive',
    'Presentations\\Current',
    'Presentations\\Archive',
    'Media\\Demo',
    'Tests',
    'Archive',
    'outputs',
    'Tools\\Battery'
)
foreach ($directory in $directories) {
    Ensure-Directory (Join-Path $ProjectRoot $directory)
}

# Current documents and co-located pipeline reports.
Move-Exact (Join-Path $ProjectRoot 'Docs\\HoloGrip_歌曲資料收集與AI標註指南.md') (Join-Path $ProjectRoot 'Docs\\Current\\HoloGrip_歌曲資料收集與AI標註指南_ZH_TW.md')
Move-Exact (Join-Path $ProjectRoot 'Docs\\HoloGrip_CSV_MIDI_training_flow_0806.svg') (Join-Path $ProjectRoot 'Docs\\Current\\HoloGrip_CSV_MIDI_training_flow_0806.svg')
Move-Exact (Join-Path $ProjectRoot 'Docs\\HoloGrip_MIDI_CSV_手別標記進度與整理計畫_0807_ZH_TW.md') (Join-Path $ProjectRoot 'Docs\\Current\\HoloGrip_MIDI_CSV_手別標記進度與整理計畫_0807_ZH_TW.md')
Move-Exact (Join-Path $ProjectRoot 'Docs\\HoloGrip_資料夾整理遷移表_0807_ZH_TW.md') (Join-Path $ProjectRoot 'Docs\\Current\\HoloGrip_資料夾整理遷移表_0807_ZH_TW.md')
Move-Exact (Join-Path $ProjectRoot 'HoloGrip_系統論文實驗計畫書_方案B_v1.docx') (Join-Path $ProjectRoot 'Docs\\Current\\HoloGrip_系統論文實驗計畫書_方案B_v1_ZH_TW.docx')
Move-Exact (Join-Path $ProjectRoot 'Song_Collection_COM\\ALIGNMENT_ALGORITHM_RECORD_0805.md') (Join-Path $ProjectRoot 'Docs\\Current\\HoloGrip_MIDI_CSV對齊演算法紀錄_0805_ZH_TW.md')
Move-Exact (Join-Path $ProjectRoot 'Song_Collection_COM\\MIDI_LABEL_PIPELINE.md') (Join-Path $ProjectRoot 'Docs\\Current\\HoloGrip_MIDI標記流程說明_ZH_TW.md')
Move-Exact (Join-Path $ProjectRoot 'Song_Collection_COM\\REPORT_0807_80MS_90MS_WINDOW_AND_TRAINING_ZH_TW.md') (Join-Path $ProjectRoot 'Docs\\Current\\HoloGrip_80MS_90MS窗口與訓練評估_0807_ZH_TW.md')
Move-Exact (Join-Path $ProjectRoot 'Song_Collection_COM\\REPORT_0807_90MS_TRAINING_FEASIBILITY.md') (Join-Path $ProjectRoot 'Docs\\Current\\HoloGrip_90MS訓練可行性報告_0807_ZH_TW.md')
Move-Exact (Join-Path $ProjectRoot 'Song_Collection_COM\\KEEPALIVE.md') (Join-Path $ProjectRoot 'outputs\\KEEPALIVE_MIDI_CSV對齊_0807_ZH_TW.md')

# Logs and historical documentation.
$logFiles = @(
    'battery_test_6_13.txt',
    'battery_test_7_10.txt',
    'battery_test_7_3.txt',
    'battery_test_raw_log_20260711.txt',
    'battery_test_raw_log_20260712.txt',
    'battery_test_report_20260711.txt',
    'battery_test_report_20260712.txt'
)
foreach ($file in $logFiles) {
    Move-Exact (Join-Path $ProjectRoot "Docs\\$file") (Join-Path $ProjectRoot "Docs\\Logs\\$file")
}
$archiveDocs = @(
    'data_format_specification.md',
    'data_sheet_table.csv',
    'HoloGrip_影片鼓譜CSV資料收集方案.md',
    'HoloGrip_數據集欄位規格說明書_ZH_TW.md',
    'research_data_specification.md',
    'slide_csv_example.csv',
    'slide_csv_example_old.csv',
    'slide_csv_example_original_5fields.csv'
)
foreach ($file in $archiveDocs) {
    Move-Exact (Join-Path $ProjectRoot "Docs\\$file") (Join-Path $ProjectRoot "Docs\\Archive\\$file")
}
Move-Exact (Join-Path $ProjectRoot '問題點.txt') (Join-Path $ProjectRoot 'Docs\\Archive\\問題點.txt')
Move-Exact (Join-Path $ProjectRoot 'temp_docx_text.txt') (Join-Path $ProjectRoot 'Docs\\Archive\\temp_docx_text.txt')
Move-Exact (Join-Path $ProjectRoot 'KEEPALIVE.md') (Join-Path $ProjectRoot 'outputs\\KEEPALIVE_專案歷史進度_0806.md')

# Source, firmware, raw data, derived data, and externally supplied session media.
Move-Exact (Join-Path $ProjectRoot 'Song_Collection_COM') (Join-Path $ProjectRoot 'Apps\\Song_Collection_COM')
Move-Exact (Join-Path $ProjectRoot 'UDP_version_release') (Join-Path $ProjectRoot 'Apps\\UDP_Collection')
Move-Exact (Join-Path $ProjectRoot 'Gloves_Firmware_INO_COM') (Join-Path $ProjectRoot 'Firmware\\COM')
Move-Exact (Join-Path $ProjectRoot 'Gloves_Firmware_INO') (Join-Path $ProjectRoot 'Firmware\\UDP_Legacy')
Move-Exact (Join-Path $ProjectRoot 'CSV_Data\\Song_Collection_COM') (Join-Path $ProjectRoot 'Data\\Raw\\Song_Collection_COM')
Move-Exact (Join-Path $ProjectRoot 'CSV_Data\\Song_Collections') (Join-Path $ProjectRoot 'Data\\Raw\\Song_Collections')
Move-Exact (Join-Path $ProjectRoot 'CSV_Data\\UDP_Collections') (Join-Path $ProjectRoot 'Data\\Raw\\UDP_Collections')
Move-Exact (Join-Path $ProjectRoot 'CSV_Data\\.gitkeep') (Join-Path $ProjectRoot 'Data\\Raw\\.gitkeep')
Remove-EmptyDirectory (Join-Path $ProjectRoot 'CSV_Data')
Move-Exact (Join-Path $ProjectRoot 'CSV_Data_Legacy') (Join-Path $ProjectRoot 'Data\\Raw\\Legacy\\CSV_Data_Legacy')
Move-Exact (Join-Path $ProjectRoot 'Derived_Data\\Song_Collection_COM') (Join-Path $ProjectRoot 'Data\\Derived\\Song_Collection_COM')
Remove-EmptyDirectory (Join-Path $ProjectRoot 'Derived_Data')
Move-Exact (Join-Path $ProjectRoot '流音給') (Join-Path $ProjectRoot 'Data\\External\\FlowAudio_20260805')

# Test code, legacy code, demo media, and presentation history.
Move-Exact (Join-Path $ProjectRoot 'Latency_Test') (Join-Path $ProjectRoot 'Tests\\Latency')
Move-Exact (Join-Path $ProjectRoot 'old') (Join-Path $ProjectRoot 'Archive\\Legacy_Code')
Move-DirectoryContents (Join-Path $ProjectRoot 'vids') (Join-Path $ProjectRoot 'Media\\Demo')
Move-Exact (Join-Path $ProjectRoot '簡報1') (Join-Path $ProjectRoot 'Presentations\\Archive\\簡報1')
$currentPresentationFiles = @(
    'HoloGrip_7_27.pptx',
    'HoloGrip_7鼓點資料收集流程_操作版_720.png',
    'HoloGrip_7鼓點資料收集流程_操作版.png',
    'HoloGrip_7鼓點資料收集流程_邏輯版_720.png',
    'HoloGrip_7鼓點資料收集流程_邏輯版.png',
    'HoloGrip_7鼓點資料收集流程.png',
    'HoloGrip_當前版本進度整理.pptx',
    'HoloGrip_資料收集操作流程_PPT.html',
    'HoloGrip_資料收集操作流程_PPT.png',
    'HoloGrip_歌曲資料收集_COM_UI_PPT.png',
    'HoloGrip_collection_UI_COM_preview.png',
    'HoloGrip_collection_UI_UDP_preview.png',
    'HoloGrip_song_collection_COM_UI_PPT.png',
    'README_流程圖狀態.md'
)
foreach ($file in $currentPresentationFiles) {
    Move-Exact (Join-Path $ProjectRoot "Presentations\\$file") (Join-Path $ProjectRoot "Presentations\\Current\\$file")
}
$archivePresentationFiles = @(
    '簡報1.pptx',
    'HoloGrip_6_15.pptx',
    'HoloGrip_6_29_no_vids.pptx',
    'HoloGrip_6_29.pptx',
    'HoloGrip_7_1.pptx',
    'HoloGrip_7_13.pptx',
    'HoloGrip_7_16.pptx',
    'HoloGrip_7_1會議、及會議記錄整理.pptx',
    'HoloGrip_7_20_BAD.pptx',
    'HoloGrip_7_22.pptx',
    'HoloGrip_7_6.pptx'
)
foreach ($file in $archivePresentationFiles) {
    Move-Exact (Join-Path $ProjectRoot "Presentations\\$file") (Join-Path $ProjectRoot "Presentations\\Archive\\$file")
}

Move-Exact (Join-Path $ProjectRoot 'battery_test_timer.py') (Join-Path $ProjectRoot 'Tools\\Battery\\battery_test_timer.py')
Write-Output 'HoloGrip folder reorganization completed. Run the verification commands before using collection tools.'
