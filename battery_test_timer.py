import socket
import time
import os
from datetime import datetime

# ================= 設定區 =================
UDP_PORT = 8888
TIMEOUT_LIMIT = 15  # 超過 15 秒沒收到封包，判定為設備已斷電關機
# ==========================================

def format_duration(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d} 小時 {m:02d} 分鐘 {s:02d} 秒"

def write_log(file_path, content, mode='a'):
    try:
        # 確保 Docs 目錄存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, mode, encoding='utf-8') as f:
            f.write(content + "\n")
    except Exception as e:
        print(f"❌ 寫入日誌失敗: {e}")

def main():
    # 建立 UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("", UDP_PORT))
    except Exception as e:
        print(f"❌ 無法綁定連接埠 {UDP_PORT}: {e}")
        return

    print("==================================================")
    print(f"🔋 HoloGrip 設備電池續航力自動測試腳本 啟動")
    print(f"📡 正在連接埠 {UDP_PORT} 監聽手套 UDP 封包...")
    print(f"📝 實時日誌將在接收首筆資料後以時間戳格式自動命名並建立")
    print("==================================================")

    start_time = None
    last_receive_time = None
    packet_count = 0
    last_file_update = 0  # 用於定時寫入檔案的時間標記
    log_file = None       # 將在接收首筆資料時動態生成

    # 設定 socket 為非阻塞，以便進行超時判斷
    sock.setblocking(False)

    while True:
        try:
            data, addr = sock.recvfrom(1024)
            packet_count += 1
            current_time = time.time()

            # 第一次接收到封包，啟動計時並動態命名檔案
            if start_time is None:
                start_time = current_time
                now_dt = datetime.now()
                start_str = now_dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # 自動產生唯一的檔名，例如 battery_test_log_20260712_120530.txt
                log_filename = f"battery_test_log_{now_dt.strftime('%Y%m%d_%H%M%S')}.txt"
                log_file = os.path.join("Docs", log_filename)
                
                print(f"⚡ [開機偵測] 已接收到首筆封包！")
                print(f"   * 開始時間: {start_str}")
                print(f"   * 實時日誌: {os.path.abspath(log_file)}")
                
                header = (
                    "==================================================\n"
                    f"HoloGrip 設備續航力自動測試日誌\n"
                    "==================================================\n"
                    f"測試開始時間: {start_str}\n"
                    f"監聽連接埠  : {UDP_PORT}\n"
                    "--------------------------------------------------"
                )
                write_log(log_file, header, mode='w')

            last_receive_time = current_time

            # 每 10 秒在終端機與檔案中更新一次進度，避免硬碟頻繁讀寫
            if current_time - last_file_update >= 10:
                elapsed = current_time - start_time
                elapsed_str = format_duration(elapsed)
                current_time_str = datetime.now().strftime('%H:%M:%S')
                
                status_text = f"[{current_time_str}] 設備運作中 | 已累計時間: {elapsed_str} | 已接收封包: {packet_count} 筆"
                print(status_text)
                write_log(log_file, status_text, mode='a')
                last_file_update = current_time

        except BlockingIOError:
            # 沒有接收到封包時，檢查是否超時
            if start_time is not None:
                inactive_time = time.time() - last_receive_time
                if inactive_time >= TIMEOUT_LIMIT:
                    # 判定為設備已沒電關機
                    end_time = last_receive_time
                    total_elapsed = end_time - start_time
                    
                    start_str = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
                    end_str = datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')
                    duration_str = format_duration(total_elapsed)

                    report = (
                        "\n==================================================\n"
                        "🏁 測試結束：偵測到手套停止發送訊號 (已斷電關機)\n"
                        "==================================================\n"
                        f"📊 最終續航統計分析：\n"
                        f"   * 設備開機時間: {start_str}\n"
                        f"   * 設備關機時間: {end_str}\n"
                        f"   * 總續航時間  : {duration_str}\n"
                        f"   * 總接收封包數: {packet_count} 筆\n"
                        "=================================================="
                    )
                    
                    print(report)
                    write_log(log_file, report, mode='a')
                    break

            time.sleep(0.01)  # 降低 CPU 使用率

if __name__ == "__main__":
    main()
