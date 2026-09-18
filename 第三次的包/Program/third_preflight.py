from __future__ import annotations
import argparse, math, sys, time
from collections import Counter
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROGRAM = Path(__file__).resolve().parent
sys.path.insert(0, str(PROGRAM))

def parse_line(s: str):
    f=s.strip().split(",")
    if len(f)<10 or f[0]!="D" or f[1] not in ("L","R"): return None
    try:
        return dict(hand=f[1], ax=float(f[2]), ay=float(f[3]), az=float(f[4]),
                    yaw=float(f[5]), pitch=float(f[6]), roll=float(f[7]),
                    packet_id=int(f[8]), sensor_ms=int(f[9]))
    except Exception:
        return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--seconds",type=float,default=3.0)
    args=ap.parse_args()

    print("=== HoloGrip 第三次現場預檢 ===")
    failures=[]
    warnings=[]

    try:
        import joblib, numpy as np
        from serial.tools import list_ports
        import serial
        from production_core import FrameValidator
        print("[PASS] Python 套件可載入")
    except Exception as e:
        print("[FAIL] Python 套件:",repr(e)); return 2

    model_dir=PACKAGE_ROOT/"Model"/"TimingRobustAug20"
    for h in ("L","R"):
        p=model_dir/f"hologrip_timing_robust_{h}.joblib"
        if not p.exists():
            failures.append(f"缺模型 {p.name}"); continue
        try:
            m=joblib.load(p)
            scaler=m.named_steps["s"]; nn=m.named_steps["m"]
            ok=(int(scaler.n_features_in_)==31 and tuple(nn.hidden_layer_sizes)==(64,32) and len(nn.classes_)==7)
            print(f"[{'PASS' if ok else 'FAIL'}] {h} 模型: features={scaler.n_features_in_}, hidden={nn.hidden_layer_sizes}, classes={len(nn.classes_)}")
            if not ok: failures.append(f"{h} 模型契約不符")
        except Exception as e:
            failures.append(f"{h} 模型載入失敗 {e!r}")

    for d in ("上次模型測試","上次模型測試_MIDI","第三次Raw","第三次MIDI","對齊與異常報告","Logs"):
        p=PACKAGE_ROOT/"Data"/d
        p.mkdir(parents=True,exist_ok=True)
    print("[PASS] Data 資料夾")

    fw=PACKAGE_ROOT/"Firmware_第三次候選"
    for hand,name in (("L","Gloves_Left_L"),("R","Gloves_Right_R")):
        p=fw/name/f"{name}.ino"
        if not p.exists(): failures.append(f"缺第三次韌體 {hand}: {p}")
    if not failures: print("[PASS] 第三次候選韌體存在")

    ports=sorted([p for p in list_ports.comports() if "303A:1001" in (p.hwid or "").upper() or "VID_303A&PID_1001" in (p.hwid or "").upper()],
                 key=lambda p:p.device)
    print("XIAO ports:",[(p.device,p.hwid) for p in ports])
    if len(ports)!=2:
        warnings.append(f"目前找到 {len(ports)} 顆 XIAO，現場需要 2 顆")
    if args.smoke:
        print("SMOKE_TEST_PASS" if not failures else "SMOKE_TEST_FAIL")
        return 0 if not failures else 2

    if len(ports)!=2:
        for w in warnings: print("[WARN]",w)
        return 1 if not failures else 2

    opened=[]
    try:
        for p in ports:
            opened.append((p.device,serial.Serial(p.device,460800,timeout=.08)))
        time.sleep(.3)
        for _,s in opened:
            try:s.reset_input_buffer()
            except Exception:pass

        stats={dev:{"n":0,"hands":Counter(),"bad":Counter(),"ids":[]} for dev,_ in opened}
        vals={dev:FrameValidator() for dev,_ in opened}
        end=time.monotonic()+args.seconds
        while time.monotonic()<end:
            for dev,s in opened:
                b=s.readline()
                if not b: continue
                p=parse_line(b.decode("utf-8","ignore"))
                if p is None: continue
                st=stats[dev];st["n"]+=1;st["hands"][p["hand"]]+=1;st["ids"].append(p["packet_id"])
                ok,reason=vals[dev].check(p)
                if not ok:st["bad"][reason]+=1
        for dev,_ in opened:
            st=stats[dev]; hz=st["n"]/args.seconds
            gaps=sum(max(0,b-a-1) for a,b in zip(st["ids"],st["ids"][1:]) if b>=a)
            hand=st["hands"].most_common(1)[0][0] if st["hands"] else "?"
            ok=(90<=hz<=110 and sum(st["bad"].values())==0 and gaps==0 and hand in ("L","R"))
            print(f"[{'PASS' if ok else 'FAIL'}] {dev}: hand={hand}, {hz:.1f} Hz, bad={dict(st['bad'])}, missing_packet={gaps}")
            if not ok:failures.append(f"{dev} 資料健康檢查失敗")
        hands=set()
        for dev,_ in opened:
            hands.update(stats[dev]["hands"].keys())
        if hands!={"L","R"}:
            failures.append(f"雙手 hand ID 不完整: {sorted(hands)}")
    except Exception as e:
        failures.append(f"COM 開啟/讀取失敗: {e!r}")
    finally:
        for _,s in opened:
            try:s.close()
            except Exception:pass

    for w in warnings: print("[WARN]",w)
    if failures:
        for x in failures: print("[FAIL]",x)
        print("RESULT=FAIL")
        return 2
    print("RESULT=PASS")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
