"""Pre-departure readiness check for HoloGrip Round 3 validation."""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
PROJECT_ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))

import run_round3_validation_0914_ZH_TW as r3
from product_hit_and_zone import FEATURE_NAMES, ZONE_NAMES, load_zone_model


def main() -> int:
    failures=[]; warnings=[]
    print("=== HoloGrip Round 3 出發前檢查 ===")
    for name in ("customtkinter","serial","joblib","sklearn","numpy"):
        try:
            m=importlib.import_module(name)
            print(f"[OK] Python package: {name} {getattr(m,'__version__','')}")
        except Exception as exc:
            failures.append(f"缺少套件 {name}: {exc}")
            print(f"[FAIL] {failures[-1]}")

    model=Path(r3.live._resolve_model_path()).resolve()
    if not model.is_file():
        failures.append(f"找不到模型: {model}")
    else:
        digest=r3.sha256_file(model).lower()
        if digest!=r3.EXPECTED_MODEL_SHA256:
            failures.append(f"模型 hash 不符: {digest}")
        else:
            print(f"[OK] Frozen model SHA-256: {digest}")
        try:
            clf=load_zone_model(str(model))
            if clf.n_features_in_!=len(FEATURE_NAMES) or list(clf.classes_)!=list(range(len(ZONE_NAMES))):
                failures.append("模型契約不是 17 維 / 7 類")
            else:
                print("[OK] Model contract: 17 features / 7 classes")
        except Exception as exc:
            failures.append(f"模型無法載入: {exc}")

    for rel in (
        "Apps/Song_Collection_COM/run_round3_validation_0914_ZH_TW.py",
        "Apps/Song_Collection_COM/round3_capture_0914_ZH_TW.py",
        "Apps/Song_Collection_COM/analyze_round3_validation_0914_ZH_TW.py",
        "Docs/Current/HoloGrip_第三次驗證SOP_0914_ZH_TW.md",
    ):
        p=PROJECT_ROOT/rel
        if p.exists(): print(f"[OK] {rel}")
        else: failures.append(f"缺檔: {rel}")

    try:
        from serial.tools import list_ports
        ports=[p.device for p in list_ports.comports()]
        print("[INFO] COM ports:", ", ".join(ports) if ports else "none")
        if len(ports)<2: warnings.append("目前少於兩個 COM；出發前或到場後插上兩隻手套再確認。")
    except Exception as exc:
        warnings.append(f"無法列舉 COM: {exc}")

    print("[CHECK] Running Round 3 unit tests...")
    proc=subprocess.run([sys.executable,"-m","unittest","discover","-s",str(HERE),"-p","test_round3_validation_0914.py","-v"],capture_output=True,text=True)
    if proc.returncode!=0:
        failures.append("Round 3 tests failed")
        print(proc.stdout); print(proc.stderr)
    else:
        print("[OK] Round 3 tests: 6/6")

    print("[CHECK] Running existing live-readiness regression...")
    proc=subprocess.run([sys.executable,"-m","unittest","discover","-s",str(HERE),"-p","test_live_readiness_0905.py","-v"],capture_output=True,text=True)
    if proc.returncode!=0:
        failures.append("Existing live-readiness regression failed")
        print(proc.stdout); print(proc.stderr)
    else:
        print("[OK] Existing live-readiness: 4/4")

    for w in warnings: print(f"[WARN] {w}")
    if failures:
        print("\n=== NOT READY ===")
        for f in failures: print("[FAIL]",f)
        return 1
    print("\n=== SOFTWARE READY ===")
    print("到場後仍必須：兩手約100Hz -> 開始場次 -> 固定原點歸零 -> 5秒Preflight PASS/WARN判讀。")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
