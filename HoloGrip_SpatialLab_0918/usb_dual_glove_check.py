from __future__ import annotations
import threading
import time

import serial
from serial.tools import list_ports

PORTS = []
for p in list_ports.comports():
    hwid = (p.hwid or "").upper()
    if "VID:PID=303A:1001" in hwid or "VID_303A&PID_1001" in hwid:
        PORTS.append(p.device)

print("偵測到 XIAO ESP32-C6：", ", ".join(PORTS) if PORTS else "無")

if len(PORTS) < 2:
    print("FAIL：目前少於兩個 XIAO COM，請檢查 USB/線材/插頭。")
    raise SystemExit(2)

results = {}

def worker(port: str):
    out = {"err": None, "parsed": 0, "hands": set()}
    try:
        s = serial.Serial(port, 460800, timeout=0.08)
        time.sleep(0.8)
        s.reset_input_buffer()
        t0 = time.monotonic()
        while time.monotonic() - t0 < 3.0:
            b = s.readline()
            if not b:
                continue
            line = b.decode("utf-8", "replace").strip()
            f = line.split(",")
            if len(f) >= 10 and f[0] == "D" and f[1] in ("L", "R"):
                out["parsed"] += 1
                out["hands"].add(f[1])
        out["elapsed"] = time.monotonic() - t0
        s.close()
    except Exception as e:
        out["err"] = repr(e)
    results[port] = out

threads = [threading.Thread(target=worker, args=(p,)) for p in PORTS[:2]]
for t in threads:
    t.start()
for t in threads:
    t.join()

ok = True
seen_hands = set()
for p in PORTS[:2]:
    o = results[p]
    print(f"\n{p}:")
    if o["err"]:
        print("  FAIL：", o["err"])
        ok = False
        continue
    hz = o["parsed"] / o["elapsed"] if o.get("elapsed") else 0
    hands = sorted(o["hands"])
    seen_hands.update(hands)
    print(f"  HAND_ID = {','.join(hands) if hands else '無'}")
    print(f"  parsed = {o['parsed']}")
    print(f"  Hz = {hz:.1f}")
    if not (90 <= hz <= 110) or len(hands) != 1:
        ok = False

if seen_hands != {"L", "R"}:
    print("\nFAIL：兩個 COM 沒有形成一左一右 HAND_ID。")
    ok = False

print("\nRESULT:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
