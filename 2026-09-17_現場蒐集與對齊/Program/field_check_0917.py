from __future__ import annotations
import hashlib,importlib,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
EXPECTED='c2f9ce16750887011219a6f46fee5095c2e4f50c8df3f94548b3c5d806b2f660'
REQUIRED=[
 ROOT/'Firmware'/'Gloves_Left_L'/'Gloves_Left_L.ino',ROOT/'Firmware'/'Gloves_Right_R'/'Gloves_Right_R.ino',
 ROOT/'Model'/'hologrip_song2_七鼓點模型_真正驗證版_0826.joblib',
 ROOT/'Program'/'formal_validation_0917.py',ROOT/'Program'/'validation_capture_0917.py',ROOT/'Program'/'analyze_validation_0917.py',ROOT/'Program'/'analyze_validation_gui_0917.py',
 ROOT/'Program'/'song_collection_com.py',ROOT/'Program'/'midi_csv_alignment_gui_0917_ZH_TW.py',ROOT/'Program'/'no_drum_diagnostic_0918_ZH_TW.py',ROOT/'Program'/'live_3d_coordinate_diagnostic_0918_ZH_TW.py',ROOT/'Diagnostic'/'training_reference_0826.json',
 ROOT/'01_測試上次模型_並保存Raw_ZH_TW.bat',ROOT/'02_用MIDI計算上次模型成績_ZH_TW.bat',ROOT/'03_收集本次Raw_100Hz_ZH_TW.bat',ROOT/'04_對齊本次MIDI與CSV_ZH_TW.bat',ROOT/'05_開啟資料夾_ZH_TW.bat',ROOT/'06_無鼓自測診斷_不是正式成績_ZH_TW.bat',ROOT/'07_即時3D_IMU鼓位空間診斷_ZH_TW.bat',ROOT/'07_即時3D底層座標診斷_ZH_TW.bat',ROOT/'README_下次只看這份.md']
def main():
    print('=== HoloGrip 9/17 正式現場包檢查 ===');failed=False
    for n in ('customtkinter','serial','joblib','sklearn','numpy','tkinterdnd2'):
        try:m=importlib.import_module(n);print(f'[OK] Python: {n} {getattr(m,"__version__","")}')
        except Exception as e:failed=True;print(f'[FAIL] Python: {n}: {e}')
    for p in REQUIRED:
        if p.is_file():print(f'[OK] {p.relative_to(ROOT)}')
        else:failed=True;print(f'[FAIL] 缺少: {p.relative_to(ROOT)}')
    model=ROOT/'Model'/'hologrip_song2_七鼓點模型_真正驗證版_0826.joblib'
    if model.is_file():
        h=hashlib.sha256(model.read_bytes()).hexdigest();print(f'[OK] Model SHA-256: {h}')
        if h!=EXPECTED:failed=True;print('[FAIL] 不是 0826 Frozen model')
        try:
            joblib=importlib.import_module('joblib');clf=joblib.load(model);prob=clf.predict_proba(np.zeros((1,17)))[0]
            if clf.n_features_in_!=17 or list(clf.classes_)!=list(range(7)) or len(prob)!=7:raise ValueError('model contract mismatch')
            print('[OK] Model contract: 17 features -> 7 drums')
        except Exception as e:failed=True;print(f'[FAIL] Model inference: {e}')
    for fw,mark in ((ROOT/'Firmware'/'Gloves_Left_L'/'Gloves_Left_L.ino',"#define HAND_ID 'L'"),(ROOT/'Firmware'/'Gloves_Right_R'/'Gloves_Right_R.ino',"#define HAND_ID 'R'")):
        if fw.is_file() and mark in fw.read_text(encoding='utf-8'):print(f'[OK] HAND_ID: {fw.relative_to(ROOT)}')
        else:failed=True;print(f'[FAIL] HAND_ID: {fw.relative_to(ROOT)}')
    for d in (ROOT/'Data'/'Validation',ROOT/'Data'/'Validation_MIDI',ROOT/'Data'/'MIDI',ROOT/'Data'/'CSV',ROOT/'Data'/'Alignment'):
        d.mkdir(parents=True,exist_ok=True);print(f'[OK] Data dir: {d.relative_to(ROOT)}')
    try:
        from serial.tools import list_ports
        ports=[p.device for p in list_ports.comports()];print('[INFO] COM:',', '.join(ports) if ports else '未偵測到')
        if len(ports)<2:print('[WARN] 現場測上次模型或收本次 Raw 時，都必須看到兩個不同 COM。')
    except Exception as e:failed=True;print(f'[FAIL] COM scan: {e}')
    proc=subprocess.run([sys.executable,str(ROOT/'Program'/'smoke_test_entrypoints.py'),'formal'],capture_output=True,text=True)
    if proc.returncode==0:print('[OK] Test-last-model entry: Raw-only + Frozen Model')
    else:failed=True;print('[FAIL] Formal entry');print(proc.stdout);print(proc.stderr)
    print('\n結果：'+('有缺項，先處理。' if failed else '工作包就緒。現場先開 01_測試上次模型_並保存Raw_ZH_TW.bat；測完關掉後，再開 03_收集本次Raw_100Hz_ZH_TW.bat'))
    return 1 if failed else 0
if __name__=='__main__':raise SystemExit(main())


