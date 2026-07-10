"""
Air Drum Studio — 單手控制修復、可拉動佈局、防呆機制、短測試模式、訓練暫停與選單標題全巨大化版 (UDP版)
"""

import customtkinter as ctk
import tkinter as tk
import socket
import threading
import math
import time
import queue
import csv
import os
from datetime import datetime
from collections import deque
from sklearn.neighbors import KNeighborsClassifier

ctk.set_appearance_mode("dark")
BG, PANEL, IDLE, DBGBG, DIM = "#0D0F1A", "#1A1B2E", "#252738", "#090A10", "#45475A"
FONT_BOLD = ("Microsoft JhengHei UI", 14, "bold")
FONT_BIG  = ("Microsoft JhengHei UI", 22, "bold")
FONT_HUGE = ("Microsoft JhengHei UI", 26, "bold")  # 主數值使用
FONT_BIG_LABEL = ("Consolas", 16, "bold")        # 差值使用
FONT_PANEL_TITLE = ("Microsoft JhengHei UI", 18, "bold") # 專為選單大標題設計的巨大粗體

ZONES_INFO = [
    {"id": 0, "name": "碎音鈸"}, {"id": 1, "name": "小鼓"}, {"id": 2, "name": "高音中鼓"},
    {"id": 3, "name": "低音中鼓"}, {"id": 4, "name": "落地鼓"}, {"id": 5, "name": "碎音鈸2"}, {"id": 6, "name": "疊音拔"}
]

UDP_PORT = 8888
FLASH_DURATION_MS = 150  # 打擊方格閃爍動畫持續時間（毫秒，150ms = 0.15秒）

class HandProcessor:
    def __init__(self, name, color_theme):
        self.name = name
        self.color = color_theme
        self.yaw_off = 0.0
        self.pitch_off = 0.0
        self.buffer = deque(maxlen=80) 
        self.last_hit_t = 0.0
        
        # 動態重力基準向量 (初始化為向下 Z)
        self.gx = 0.0
        self.gy = 0.0
        self.gz = 1.0
        
        self.X, self.y = [], []
        self.raw_hits = [] 
        
        self.clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
        self.is_trained = False
        self.is_training = False
        self.current_zone = 0
        self.hits_count = 0
        self.target_hits = 200 
        self.ui_refs = {} 
        self.ui_status_lbl = None
        self.btn_train_ref = None 
        self.btn_short_ref = None
        self.last_feat = None 
        
        # 新增實戰 Demo 報告記錄屬性
        self.demo_hits = []
        self.demo_start_t = 0.0
        self.last_actual_hit_t = 0

class AirDrumDualStudio(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Air Drum Studio — 相對揮幅與防抬手優化版 (UDP)")
        self.geometry("1600x950")
        self.minsize(1280, 720) 
        self.configure(fg_color=BG)

        self.hands = {
            'R': HandProcessor("右手", "#89B4FA"),
            'L': HandProcessor("左手", "#A6E3A1")
        }
        
        self._hit_queue = queue.Queue()
        self.is_paused = False 

        self._build_ui()
        self._start_udp()
        self._poll_hit_queue()
        self._print_local_ip()

    def _build_ui(self):
        self.lbl_status = ctk.CTkLabel(self, text="📡 雙手 UDP 伺服器運作中...", font=FONT_BOLD, fg_color=PANEL, height=40)
        self.lbl_status.pack(fill="x")

        main_frame = ctk.CTkFrame(self, fg_color=BG)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # ==================== 左側控制面板 ====================
        ctrl = ctk.CTkFrame(main_frame, width=320, fg_color=PANEL) 
        ctrl.pack(side="left", fill="y", padx=(0, 10))
        ctrl.pack_propagate(False) 

        # 「【全局控制】」字體調整為巨大的 FONT_PANEL_TITLE
        ctk.CTkLabel(ctrl, text="【全局控制】", font=FONT_PANEL_TITLE, text_color="#FFFFFF").pack(pady=(15, 6))
        
        self.btn_cal_all = ctk.CTkButton(ctrl, text="🎯 雙手統一歸零", font=FONT_BOLD, height=35, fg_color=DIM, text_color="#FFF",
                             command=self._calibrate_both)
        self.btn_cal_all.pack(pady=4, padx=20, fill="x")
        
        self.btn_pause_toggle = ctk.CTkButton(ctrl, text="⏸️ 暫停接收打擊 (換電池/手套)", font=FONT_BOLD, height=35, fg_color="#A6ADC8", text_color="#000",
                             command=self._toggle_pause)
        self.btn_pause_toggle.pack(pady=4, padx=20, fill="x")
        
        self.btn_short_all = ctk.CTkButton(ctrl, text="⚡ 啟動短測試模式 (20筆)", font=FONT_BOLD, height=35, fg_color="#89B4FA", text_color="#000",
                             command=self._start_short_test_both)
        self.btn_short_all.pack(pady=4, padx=20, fill="x")
                             
        self.btn_train_all = ctk.CTkButton(ctrl, text="🚀 啟動標準訓練 (200筆)", font=FONT_BOLD, height=35, fg_color="#F9E2AF", text_color="#000",
                             command=self._start_train_both)
        self.btn_train_all.pack(pady=4, padx=20, fill="x")
                             
        self.btn_cancel_all = ctk.CTkButton(ctrl, text="❌ 雙手統一取消訓練", font=FONT_BOLD, height=35, fg_color="#F38BA8", text_color="#000",
                             command=self._cancel_train_both)
        self.btn_cancel_all.pack(pady=4, padx=20, fill="x")
        self.btn_save_all = ctk.CTkButton(ctrl, text="💾 產生 CSV (未完成也可存)", font=FONT_BOLD, height=35, fg_color="#A6E3A1", text_color="#000",
                             command=self._save_csv_both)
        self.btn_save_all.pack(pady=4, padx=20, fill="x")
        
        self.btn_load_csv = ctk.CTkButton(ctrl, text="📁 載入訓練好的 CSV (直接進 Demo)", font=FONT_BOLD, height=35, fg_color="#CBA6F7", text_color="#000",
                             command=self._load_csv)
        self.btn_load_csv.pack(pady=4, padx=20, fill="x")
        
        ctk.CTkFrame(ctrl, height=2, fg_color=DIM).pack(fill="x", padx=20, pady=8) 

        # ==================== 獨立控制區 ====================
        for h_key in ['R', 'L']:
            h = self.hands[h_key]
            # 獨立控制標題（例如：【右手獨立控制】）字體調整為巨大的 FONT_PANEL_TITLE
            ctk.CTkLabel(ctrl, text=f"【{h.name}獨立控制】", font=FONT_PANEL_TITLE, text_color=h.color).pack(pady=(6, 2))
            
            h.ui_status_lbl = ctk.CTkLabel(ctrl, text="⚫ 待機中", font=("Microsoft JhengHei UI", 16, "bold"), text_color=DIM)
            h.ui_status_lbl.pack(pady=(2, 6))

            btn_frame_top = ctk.CTkFrame(ctrl, fg_color="transparent")
            btn_frame_top.pack(fill="x", padx=20, pady=(0, 4))
            ctk.CTkButton(btn_frame_top, text="🎯 歸零", fg_color=DIM, command=lambda k=h_key: self._calibrate(k)).pack(side="left", fill="x", expand=True, padx=(0, 3))
            ctk.CTkButton(btn_frame_top, text="❌ 取消", fg_color="#F38BA8", text_color="#000", command=lambda k=h_key: self._cancel_train_single(k)).pack(side="left", fill="x", expand=True, padx=(3, 0))

            btn_frame_bot = ctk.CTkFrame(ctrl, fg_color="transparent")
            btn_frame_bot.pack(fill="x", padx=20, pady=(0, 10))
            
            h.btn_short_ref = ctk.CTkButton(btn_frame_bot, text="⚡ 20筆測試", fg_color="#89B4FA", text_color="#000", height=32, command=lambda k=h_key: self._start_short_test_single(k))
            h.btn_short_ref.pack(side="left", fill="x", expand=True, padx=(0, 3))
            
            h.btn_train_ref = ctk.CTkButton(btn_frame_bot, text="▶ 200筆訓練", fg_color=h.color, text_color="#000", height=32, command=lambda k=h_key: self._start_train_single(k))
            h.btn_train_ref.pack(side="left", fill="x", expand=True, padx=(3, 0))

        # ==================== 左下角數據顯示區：標題同步放大 ====================
        self.monitor_area = ctk.CTkFrame(ctrl, fg_color="transparent")
        self.monitor_area.pack(fill="both", expand=True, padx=15, pady=(5, 5), side="bottom")
        
        self.monitor_area.rowconfigure(0, weight=1)
        self.monitor_area.rowconfigure(1, weight=1)
        self.monitor_area.columnconfigure(0, weight=1)

        self.monitors = {'R': {}, 'L': {}}
        hand_configs = [('R', "🎯 右手打擊特徵值", 0), ('L', "🎯 左手打擊特徵值", 1)]

        for h_key, title_text, row_idx in hand_configs:
            h = self.hands[h_key]
            h_panel = ctk.CTkFrame(self.monitor_area, fg_color=DBGBG, corner_radius=10)
            h_panel.grid(row=row_idx, column=0, pady=4, sticky="nsew")
            
            # 將打擊特徵值標題字體調整為 FONT_BOLD 放大版，微調至 14級粗體
            lbl_title = ctk.CTkLabel(h_panel, text=title_text, font=FONT_BOLD, text_color=h.color)
            lbl_title.pack(pady=(6, 2))

            data_row = ctk.CTkFrame(h_panel, fg_color="transparent")
            data_row.pack(fill="both", expand=True, padx=5, pady=2)
            data_row.rowconfigure(0, weight=1) # 讓列擴充填滿垂直空間，避免內容被擠壓
            for c in range(3): data_row.columnconfigure(c, weight=1)

            features_labels = [("Yaw", "YAW"), ("Pitch", "PIT"), ("SwingDepth", "DEL")]
            for idx, (title, f_key) in enumerate(features_labels):
                box = ctk.CTkFrame(data_row, fg_color=PANEL, corner_radius=6)
                box.grid(row=0, column=idx, padx=3, sticky="nsew")
                
                # 上方差值 (16級粗體，設定 height 避免佔用多餘空間)
                lbl_diff = ctk.CTkLabel(box, text=" -- ", font=FONT_BIG_LABEL, text_color=DIM, height=18)
                lbl_diff.pack(pady=(3, 0))
                
                # 當前讀值 (巨大26級粗體)
                lbl_val = ctk.CTkLabel(box, text="0.0", font=FONT_HUGE, text_color="#CDD6F4", height=30)
                lbl_val.pack(pady=(0, 0))
                
                # 底部特徵名稱文字 (設定 height 防止字型被底部邊框切除)
                lbl_name = ctk.CTkLabel(box, text=title, font=("Microsoft JhengHei UI", 9), text_color=DIM, height=14)
                lbl_name.pack(pady=(0, 2))
                
                self.monitors[h_key][f_key] = {"val": lbl_val, "diff": lbl_diff}

        # ==================== 右側顯示區域 (可拉動) ====================
        right_panel = ctk.CTkFrame(main_frame, fg_color=BG)
        right_panel.pack(side="right", fill="both", expand=True)

        self.paned_window = tk.PanedWindow(right_panel, orient="vertical", bg=BG, bd=0, sashwidth=6, sashrelief=tk.RAISED)
        self.paned_window.pack(fill="both", expand=True)

        display = ctk.CTkFrame(self.paned_window, fg_color=BG)
        
        for ridx, h_key in enumerate(['R', 'L']):
            h = self.hands[h_key]
            row_frame = ctk.CTkFrame(display, fg_color=IDLE, corner_radius=15)
            row_frame.pack(fill="both", expand=True, pady=8) 
            
            lbl_title = ctk.CTkLabel(row_frame, text=h.name, font=("Microsoft JhengHei UI", 24, "bold"), text_color=h.color, width=80)
            lbl_title.pack(side="left", fill="y", padx=15)
            
            grid = ctk.CTkFrame(row_frame, fg_color="transparent")
            grid.pack(side="left", fill="both", expand=True, padx=10, pady=15)
            grid.rowconfigure(0, weight=1) 
            for c in range(7): grid.columnconfigure(c, weight=1) 
            
            for z in ZONES_INFO:
                f = ctk.CTkFrame(grid, fg_color=DBGBG, corner_radius=12)
                f.grid(row=0, column=z["id"], padx=8, pady=0, sticky="nsew") 
                
                inner = ctk.CTkFrame(f, fg_color="transparent")
                inner.place(relx=0.5, rely=0.5, anchor="center")
                
                l = ctk.CTkLabel(inner, text=z["name"], font=FONT_BIG, text_color=DIM)
                l.pack(pady=(0, 5))
                
                c_lbl = ctk.CTkLabel(inner, text="", font=FONT_HUGE, text_color=DIM)
                c_lbl.pack()
                
                h.ui_refs[z["id"]] = {"frame": f, "label": l, "count_lbl": c_lbl}

        log_frame = ctk.CTkFrame(self.paned_window, fg_color=BG)
        self.txt_log = ctk.CTkTextbox(log_frame, fg_color=DBGBG, font=("Consolas", 13), text_color="#CDD6F4")
        self.txt_log.pack(fill="both", expand=True, pady=(10, 0))

        self.paned_window.add(display, minsize=300)
        self.paned_window.add(log_frame, minsize=100)

    def log(self, msg):
        self.txt_log.insert("end", f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.txt_log.see("end")

    def _print_local_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            self.log(f"💻 本機當前區域網路 IP: {local_ip}")
            self.log(f"📌 請確保 ESP32 的 targetIP 填寫此位址，且 Port 為 {UDP_PORT}")
        except Exception:
            self.log("💻 本機當前 IP: 無法獲取網路狀態")

    def _toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause_toggle.configure(text="▶️ 恢復接收打擊", fg_color="#F38BA8")
            self.log("⏸️ 系統已進入【暫停接收】狀態。")
        else:
            self.btn_pause_toggle.configure(text="⏸️ 暫停接收打擊 (換電池/手套)", fg_color="#A6ADC8")
            self.log("▶️ 系統已【恢復接收】打擊訊號。")

    # =============== CSV 資料儲存與載入邏輯 ===============
    def _save_csv_both(self):
        self.log("👆 點擊【產生 CSV】按鈕 -> 觸發雙手資料匯出")
        self._save_csv('R')
        self._save_csv('L')

    def _save_csv(self, hid):
        h = self.hands[hid]
        
        # 情況 A：如果手套處於訓練中，或者未訓練完 (即 raw_hits 有資料，且 is_trained 為 False) -> 匯出訓練資料 CSV (14欄位)
        if not h.is_trained:
            total_records = len(h.raw_hits)
            if total_records == 0: return 
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{hid}_Train_{timestamp}_{total_records}.csv"
            filepath = os.path.abspath(filename)
            
            try:
                with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    # 寫入包含 session_id 與 trial_id 的 14 欄位標頭
                    writer.writerow([
                        "Zone_ID", "Zone_Name", "Feature_Yaw", "Feature_Pitch", "Feature_SwingDepth",
                        "hardware_version", "participant_id", "session_id", "trial_id", 
                        "hit_id", "hand", "timestamp_ms", "actual_hit_time_ms", "inter_hit_interval_ms"
                    ])
                    for hit in h.raw_hits:
                        writer.writerow([
                            hit['zone_id'], 
                            hit['zone_name'], 
                            round(hit['yaw'], 2), 
                            round(hit['pitch'], 2), 
                            round(hit['swing_depth'], 2),
                            hit.get('hardware_version', 'TypeCSeparated_1.0'),
                            hit.get('participant_id', 'P01'),
                            hit.get('session_id', ''),
                            hit.get('trial_id', ''),
                            hit.get('hit_id', 1),
                            hit.get('hand', hid),
                            hit.get('timestamp_ms', int(time.time() * 1000)),
                            hit.get('actual_hit_time_ms', 0),
                            hit.get('inter_hit_interval_ms', 0)
                        ])
                self.log(f"💾 [{h.name}] 訓練資料 CSV 匯出成功 ({total_records} 筆)！")
                self.log(f"📁 存放路徑: {filepath}")
            except Exception as e:
                self.log(f"❌ [{h.name}] 儲存訓練 CSV 失敗: {e}")
                
        # 情況 B：如果手套處於實戰 Demo 模式 (is_trained 為 True) -> 匯出打擊報告 CSV (多出 predicted_zone 與 confidence 欄位，共 16 欄)
        else:
            total_records = len(h.demo_hits)
            if total_records == 0:
                self.log(f"⚠️ [{h.name}] 當前無實戰打擊數據，無法產生報告。")
                return 
                
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{hid}_Report_{timestamp}_{total_records}.csv"
            filepath = os.path.abspath(filename)
            
            try:
                with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Zone_ID", "Zone_Name", "Feature_Yaw", "Feature_Pitch", "Feature_SwingDepth",
                        "hardware_version", "participant_id", "session_id", "trial_id", 
                        "hit_id", "hand", "timestamp_ms", "actual_hit_time_ms", "inter_hit_interval_ms",
                        "predicted_zone", "confidence"
                    ])
                    for hit in h.demo_hits:
                        writer.writerow([
                            hit['zone_id'], 
                            hit['zone_name'], 
                            round(hit['yaw'], 2), 
                            round(hit['pitch'], 2), 
                            round(hit['swing_depth'], 2),
                            hit.get('hardware_version', 'TypeCSeparated_1.0'),
                            hit.get('participant_id', 'P01'),
                            hit.get('session_id', ''),
                            hit.get('trial_id', ''),
                            hit.get('hit_id', 1),
                            hit.get('hand', hid),
                            hit.get('timestamp_ms', int(time.time() * 1000)),
                            hit.get('actual_hit_time_ms', 0),
                            hit.get('inter_hit_interval_ms', 0),
                            hit.get('predicted_zone', -1),
                            hit.get('confidence', 1.0)
                        ])
                self.log(f"💾 [{h.name}] 打擊報告 CSV 匯出成功 ({total_records} 筆)！")
                self.log(f"📁 存放路徑: {filepath}")
                
                # 計算實戰 Demo 統計數據並輸出至控制台與 UI Log
                ihi_values = [hit['inter_hit_interval_ms'] for hit in h.demo_hits if hit['inter_hit_interval_ms'] > 0]
                avg_ihi = sum(ihi_values) / len(ihi_values) if ihi_values else 0.0
                
                conf_values = [hit['confidence'] for hit in h.demo_hits]
                avg_conf = sum(conf_values) / len(conf_values) if conf_values else 1.0
                
                demo_duration = time.time() - h.demo_start_t
                avg_rate = total_records / demo_duration if demo_duration > 0.1 else 100.0
                
                self.log(f"📊 === [{h.name}] 實戰性能統計分析 ===")
                self.log(f"   * 實戰打擊總數 : {total_records} 次")
                self.log(f"   * 實戰平均更新率: {avg_rate:.2f} Hz")
                self.log(f"   * AI 預測平均置信度: {avg_conf * 100.0:.2f}%")
                self.log(f"   * 平均擊鼓間隔 : {avg_ihi:.1f} ms")
                self.log(f"====================================")
                
            except Exception as e:
                self.log(f"❌ [{h.name}] 儲存打擊報告 CSV 失敗: {e}")

    def _load_csv(self):
        from tkinter import filedialog
        filepaths = filedialog.askopenfilenames(
            title="選擇訓練 CSV 檔案 (可多選 L/R)",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not filepaths:
            return
            
        for fp in filepaths:
            basename = os.path.basename(fp)
            # 判斷手向：檔名以 L/l 開頭為左手，R/r 開頭為右手
            if basename.lower().startswith('l'):
                hid = 'L'
            elif basename.lower().startswith('r'):
                hid = 'R'
            else:
                self.log(f"⚠️ 無法辨識檔名手向：{basename}，請確保檔名以 L_ 或 R_ 開頭。")
                continue
                
            h = self.hands[hid]
            try:
                X, y = [], []
                raw_hits = []
                with open(fp, 'r', encoding='utf-8-sig') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)  # 讀取標頭
                    for row in reader:
                        if len(row) < 5: continue
                        zone_id = int(row[0])
                        zone_name = row[1]
                        yaw = float(row[2])
                        pitch = float(row[3])
                        swing_depth = float(row[4])
                        
                        X.append([yaw, pitch, swing_depth])
                        y.append(zone_id)
                        raw_hits.append({
                            'zone_id': zone_id,
                            'zone_name': zone_name,
                            'yaw': yaw,
                            'pitch': pitch,
                            'swing_depth': swing_depth
                        })
                
                if len(X) < 5:
                    self.log(f"❌ 檔案 {basename} 筆數不足 ({len(X)} 筆)，無法訓練模型。")
                    continue
                    
                h.X = X
                h.y = y
                h.raw_hits = raw_hits
                
                h.X = X
                h.y = y
                h.raw_hits = raw_hits
                
                # 訓練 KNN 模型
                h.clf.fit(h.X, h.y)
                h.is_trained = True
                h.is_training = False
                h.current_zone = 0
                h.hits_count = 0
                h.demo_hits = [] # 清空實戰報告
                h.demo_start_t = time.time() # 記錄實戰開始時間
                
                self.log(f"🎉 [{h.name}] 成功載入 CSV {basename} ({len(X)} 筆資料)！已自動進入【實戰 Demo 模式】。")
                self._update_ui_state()
            except Exception as e:
                self.log(f"❌ 載入 {basename} 失敗: {e}")



    # =============== UDP 通訊區 ===============
    def _start_udp(self):
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_sock.bind(("0.0.0.0", UDP_PORT))
        self.active_hands = set()
        self.last_packet_time = {'R': 0.0, 'L': 0.0}
        
        threading.Thread(target=self._handle_udp, daemon=True).start()
        threading.Thread(target=self._check_udp_timeouts, daemon=True).start()
        self.log(f"UDP 伺服器啟動於 Port {UDP_PORT}")

    def _handle_udp(self):
        import select
        self.server_sock.setblocking(False)
        last_recv_t = {'R': 0.0, 'L': 0.0}
        
        while True:
            try:
                # 等待 Socket 可讀，逾時 0.5 秒
                r, _, _ = select.select([self.server_sock], [], [], 0.5)
                if not r:
                    continue
                
                # 進入緊密循環，一次清空作業系統內緩衝的所有 UDP 封包 (保證完整讀取，不丟棄數據)
                while True:
                    try:
                        data, addr = self.server_sock.recvfrom(8192)
                        if not data:
                            break
                        
                        chunk = data.decode('utf-8', errors='ignore')
                        lines = chunk.split('\n')
                        
                        for line in lines:
                            line = line.strip()
                            if not line.startswith("D,"): continue
                            p = line.split(",")
                            if len(p) < 8: continue
                            
                            hid = p[1]
                            if hid not in self.hands: continue
                            h = self.hands[hid]

                            # 連線偵測與紀錄
                            now_time = time.time()
                            self.last_packet_time[hid] = now_time
                            if hid not in self.active_hands:
                                self.active_hands.add(hid)
                                self.log(f"🔗 {h.name} UDP 連線成功！來源: {addr[0]}:{addr[1]}")
                                self.after(0, self._update_ui_state)
                                
                            # 檢測本地封包接收間隔 (原本正常應為 10ms，若突發間隔大於 50ms 代表網路存在卡頓/抖動)
                            if last_recv_t[hid] > 0.0:
                                gap = now_time - last_recv_t[hid]
                                if gap > 0.05:  # 大於 50 毫秒 (相當於延遲抖動)
                                    self.log(f"⚠️ [{h.name}] 網路延遲抖動！封包間隔 {gap*1000:.1f} ms")
                            last_recv_t[hid] = now_time

                            try:
                                ax, ay, az = float(p[2]), float(p[3]), float(p[4])
                                y_ang, p_ang = float(p[5]), float(p[6])
                                r_ang = float(p[7]) if len(p) > 7 else 0.0
                            except (ValueError, IndexError):
                                continue 
                            
                            cal_y = (y_ang - h.yaw_off + 180) % 360 - 180
                            cal_p = p_ang - h.pitch_off
                            # 讀取發送端高精度時間戳，不破壞前 7 個欄位索引
                            esp_t = int(p[9]) if len(p) >= 10 else int(time.time() * 1000)
                            h.buffer.append([ax, ay, az, cal_y, cal_p, p_ang, r_ang, esp_t])

                            mag = math.sqrt(ax*ax + ay*ay + az*az)
                            
                            # 【動態重力向量擷取】
                            if 0.8 < mag < 1.2:
                                alpha = 0.05
                                h.gx = h.gx * (1 - alpha) + ax * alpha
                                h.gy = h.gy * (1 - alpha) + ay * alpha
                                h.gz = h.gz * (1 - alpha) + az * alpha

                            if self.is_paused:
                                continue

                            now = time.monotonic()
                            now_t = time.time()
                            
                            # 【防抬手與快擊優化 - 局部峰值回落判定法】
                            buf_len = len(h.buffer)
                            if buf_len >= 45:
                                # target 封包 (30ms 之前，以讀取未來 30ms 封包確認 Peak 回落)
                                t_ax, t_ay, t_az = h.buffer[-4][0], h.buffer[-4][1], h.buffer[-4][2]
                                mag_target = math.sqrt(t_ax*t_ax + t_ay*t_ay + t_az*t_az)
                                
                                debounce = getattr(h, 'debounce_time', 0.15)
                                if mag_target > 1.7 and (now - h.last_hit_t) > debounce:
                                    # 比對 target 前後鄰近封包，確認 target 是否為局部最大值 (Peak)
                                    mags = []
                                    for idx in range(-6, 0):
                                        a_x, a_y, a_z = h.buffer[idx][0], h.buffer[idx][1], h.buffer[idx][2]
                                        mags.append(math.sqrt(a_x*a_x + a_y*a_y + a_z*a_z))
                                    
                                    # target 對應 mags[2] (即倒數第 4 個)
                                    if mags[2] == max(mags):
                                        snap = [h.buffer[buf_len - 4 - i] for i in range(40)]
                                        mag = mag_target  # 覆蓋原本 mag 變數，使後續 log 與判定無縫使用 Peak 加速度
                                        
                                        g_mag = math.sqrt(h.gx*h.gx + h.gy*h.gy + h.gz*h.gz)
                                        if g_mag > 0.1:
                                            ux, uy, uz = h.gx / g_mag, h.gy / g_mag, h.gz / g_mag
                                        else:
                                            ux, uy, uz = 0.0, 0.0, 1.0
                                            
                                        v_score = 0.0      # 短視窗 (3~15) 垂直動態加速度積分
                                        v_score_long = 0.0 # 長視窗 (3~30) 垂直動態加速度積分
                                        sum_dyn = 0.0
                                        sum_vert = 0.0
                                        
                                        for i in range(3, 30):
                                            f = snap[i]
                                            s_ax, s_ay, s_az = f[0], f[1], f[2]
                                            
                                            az_world = s_ax * ux + s_ay * uy + s_az * uz
                                            az_motion = az_world - 1.0
                                            
                                            v_score_long += az_motion
                                            
                                            if i < 15:
                                                v_score += az_motion
                                                
                                                dyn_x = s_ax - ux
                                                dyn_y = s_ay - uy
                                                dyn_z = s_az - uz
                                                dyn_mag = math.sqrt(dyn_x*dyn_x + dyn_y*dyn_y + dyn_z*dyn_z)
                                                
                                                sum_dyn += dyn_mag
                                                sum_vert += abs(az_motion)
                                            
                                        r_vert = sum_vert / sum_dyn if sum_dyn > 0.1 else 1.0
                                        
                                        # 計算打擊前 50ms 到 150ms 之間的相對 Pitch 俯仰角變化趨勢 (避開最新 5 個封包以防止撞擊瞬間 MEMS 姿態解算彈跳失真)
                                        pitch_diff = snap[5][4] - snap[15][4]
                                        
                                        # 結合「短視窗積分」、「長視窗衝量抵消」與「避開撞擊彈跳的 Pitch 變化趨勢」三維判定抬手動作
                                        is_heavy = (v_score < -10.0) or (mag >= 2.5)
                                        is_raise = (v_score > -0.25) or (
                                            (not is_heavy) and (
                                                (v_score_long > -1.5 and pitch_diff < -0.5) or (pitch_diff < -4.0)
                                            )
                                        )
                                        is_horizontal = (r_vert < 0.32) or ((v_score >= -8.0) and (r_vert < 0.45))
                                        
                                        if is_raise or is_horizontal:
                                            reason = "抬手動作" if is_raise else "左右/水平揮動"
                                            if is_raise and is_horizontal:
                                                reason = "抬手且偏向水平揮動"
                                            
                                            if not hasattr(h, 'last_ignore_log_t'):
                                                h.last_ignore_log_t = 0.0
                                            if now_t - h.last_ignore_log_t > 0.15:
                                                self.log(f"🚫 [{h.name}] 忽略動作 ({reason}) | V_Score={v_score:.2f}, V_Long={v_score_long:.2f}, P_Diff={pitch_diff:.2f}, R_Vert={r_vert:.2f}, Mag={mag:.2f}")
                                                h.last_ignore_log_t = now_t
                                            
                                            h.last_hit_t = now - 0.10
                                            continue
                                    
                                        # 動態防彈跳：重擊保持 150ms 靈敏冷卻；輕擊給予 250ms 冷卻以防止餘震連發
                                        h.debounce_time = 0.15 if is_heavy else 0.25
                                        h.last_hit_t = now
                                        self.log(f"🥁 [{h.name}] 有效敲擊! Mag={mag:.2f}, V_Score={v_score:.2f}, V_Long={v_score_long:.2f}, P_Diff={pitch_diff:.2f}, R_Vert={r_vert:.2f}")
                                        
                                        feat_yaw = snap[1][3]    
                                        feat_pitch = snap[1][4]  
                                        
                                        pitch_values = [f[4] for f in snap]
                                        swing_depth = max(pitch_values) - min(pitch_values)
                                        
                                        # 獲取打擊點手套發送時間戳
                                        esp_hit_t = snap[1][7] if len(snap[1]) > 7 else int(time.time() * 1000)
                                        self._hit_queue.put((hid, [feat_yaw, feat_pitch, swing_depth, esp_hit_t]))
                    except BlockingIOError:
                        # 緩衝區已空，跳出緊密循環，重新進入 select 等待
                        break
            except Exception as e:
                import traceback
                error_msg = f"❌ 系統錯誤: {e}\n{traceback.format_exc()}"
                self.log(f"❌ 系統錯誤: {e}")
                print(error_msg)

    def _check_udp_timeouts(self):
        while True:
            time.sleep(1.0)
            now = time.time()
            for hid in list(self.active_hands):
                if now - self.last_packet_time[hid] > 3.0:
                    self.active_hands.remove(hid)
                    self.log(f"❌ {self.hands[hid].name} UDP 斷線")
                    self.after(0, self._update_ui_state)

    def _poll_hit_queue(self):
        try:
            while True:
                hid, feat = self._hit_queue.get_nowait()
                self._process_hit(hid, feat)
        except queue.Empty: pass
        self.after(16, self._poll_hit_queue)

    # =============== 打擊處理與 UI 邏輯 ===============
    def _process_hit(self, hid, feat):
        h = self.hands[hid]
        
        diff_texts = ["--", "--", "--"]
        diff_colors = [DIM, DIM, DIM]
        
        if h.last_feat is not None:
            for i, key in enumerate(["YAW", "PIT", "DEL"]):
                diff_val = feat[i] - h.last_feat[i]
                if diff_val > 0:
                    diff_texts[i] = f"+{round(diff_val, 2)}"
                    diff_colors[i] = "#A6E3A1" 
                elif diff_val < 0:
                    diff_texts[i] = f"{round(diff_val, 2)}"
                    diff_colors[i] = "#F38BA8" 
                else:
                    diff_texts[i] = "0.0"
                    diff_colors[i] = "#FFF"
                    
        self.monitors[hid]["YAW"]["val"].configure(text=f"{round(feat[0], 1)}")
        self.monitors[hid]["YAW"]["diff"].configure(text=diff_texts[0], text_color=diff_colors[0])
        
        self.monitors[hid]["PIT"]["val"].configure(text=f"{round(feat[1], 1)}")
        self.monitors[hid]["PIT"]["diff"].configure(text=diff_texts[1], text_color=diff_colors[1])
        
        self.monitors[hid]["DEL"]["val"].configure(text=f"{round(feat[2], 1)}")
        self.monitors[hid]["DEL"]["diff"].configure(text=diff_texts[2], text_color=diff_colors[2])
        
        h.last_feat = feat

        # ====== 原有狀態機訓練與實戰邏輯 ======
        if h.is_training:
            # 提取時間戳與計算間隔
            actual_hit_t = feat[3] if len(feat) > 3 else int(time.time() * 1000)
            if not hasattr(h, 'last_actual_hit_t'):
                h.last_actual_hit_t = 0
            inter_hit_interval = actual_hit_t - h.last_actual_hit_t if h.last_actual_hit_t > 0 else 0
            h.last_actual_hit_t = actual_hit_t

            h.X.append(feat[:3]) # 防護：僅將前 3 維度 Yaw, Pitch, SwingDepth 用於模型訓練
            h.y.append(h.current_zone)
            session_id = f"S{datetime.now().strftime('%Y%m%d')}_P01_TypeCSeparated"
            trial_id = f"T01_{hid}_{ZONES_INFO[h.current_zone]['name']}_{h.target_hits}hits"
            h.raw_hits.append({
                'zone_id': h.current_zone,
                'zone_name': ZONES_INFO[h.current_zone]['name'],
                'yaw': feat[0], 'pitch': feat[1], 'swing_depth': feat[2],
                'hardware_version': 'TypeCSeparated_1.0',
                'participant_id': 'P01',
                'session_id': session_id,
                'trial_id': trial_id,
                'hit_id': len(h.raw_hits) + 1,
                'hand': hid,
                'timestamp_ms': int(time.time() * 1000),
                'actual_hit_time_ms': actual_hit_t,
                'inter_hit_interval_ms': inter_hit_interval
            })
            h.hits_count += 1
            
            self._flash(hid, h.current_zone, "#FFFFFF")
            
            prog_text = f"{h.hits_count}/{h.target_hits}"
            h.ui_refs[h.current_zone]["count_lbl"].configure(text=prog_text, text_color="#F9E2AF")
            
            if h.hits_count >= h.target_hits:
                self.log(f"✅ {h.name} - {ZONES_INFO[h.current_zone]['name']} 訓練完畢")
                h.current_zone += 1
                h.hits_count = 0
                
                if h.current_zone >= 7:
                    self.log(f"⏳ 正在為 [{h.name}] 計算並配對 KNN 模型...")
                    h.clf.fit(h.X, h.y)
                    h.is_trained = True
                    h.is_training = False
                    h.demo_hits = [] # 清空實戰報告
                    h.demo_start_t = time.time() # 記錄實戰開始時間
                    self.log(f"🎉 🎉 [{h.name}] 模型訓練完成！已自動進入【實戰 Demo 模式】。")
                    self._save_csv(hid)
                    
                    any_training = any(hand.is_training for hand in self.hands.values())
                    if not any_training:
                        self._unlock_buttons()
                    
                self._update_ui_state()
        
        elif h.is_trained:
            # 1. 預測分類
            pred = int(h.clf.predict([feat[:3]])[0]) # 防護：僅傳入前 3 維度進行模型預測
            
            # 2. 計算 KNN 置信度
            try:
                probs = h.clf.predict_proba([feat[:3]])[0]
                confidence = float(max(probs))
            except Exception:
                confidence = 1.0 # KNN 未能計算概率時防呆
                
            actual_hit_t = feat[3] if len(feat) > 3 else int(time.time() * 1000)
            if not hasattr(h, 'last_actual_hit_t'):
                h.last_actual_hit_t = 0
            inter_hit_interval = actual_hit_t - h.last_actual_hit_t if h.last_actual_hit_t > 0 else 0
            h.last_actual_hit_t = actual_hit_t
            
            # 3. 記錄實戰 Demo 打擊資料 (多出 predicted_zone 與 confidence 欄位)
            session_id = f"S{datetime.now().strftime('%Y%m%d')}_P01_TypeCSeparated"
            trial_id = f"T_Demo_{hid}_Live"
            h.demo_hits.append({
                'zone_id': -1, # 實戰無目標鼓 ID
                'zone_name': '實戰Demo',
                'yaw': feat[0], 'pitch': feat[1], 'swing_depth': feat[2],
                'hardware_version': 'TypeCSeparated_1.0',
                'participant_id': 'P01',
                'session_id': session_id,
                'trial_id': trial_id,
                'hit_id': len(h.demo_hits) + 1,
                'hand': hid,
                'timestamp_ms': int(time.time() * 1000),
                'actual_hit_time_ms': actual_hit_t,
                'inter_hit_interval_ms': inter_hit_interval,
                'predicted_zone': pred,
                'confidence': confidence
            })
            
            self._flash(hid, pred, h.color)

    # =============== 操作邏輯與防呆鎖定 ===============
    def _lock_buttons(self):
        self.btn_train_all.configure(state="disabled", fg_color=DIM)
        self.btn_short_all.configure(state="disabled", fg_color=DIM)
        self.btn_load_csv.configure(state="disabled", fg_color=DIM)
        for h in self.hands.values():
            if h.btn_train_ref: h.btn_train_ref.configure(state="disabled", fg_color=DIM)
            if h.btn_short_ref: h.btn_short_ref.configure(state="disabled", fg_color=DIM)

    def _unlock_buttons(self):
        self.btn_train_all.configure(state="normal", fg_color="#F9E2AF")
        self.btn_short_all.configure(state="normal", fg_color="#89B4FA")
        self.btn_load_csv.configure(state="normal", fg_color="#CBA6F7")
        for h in self.hands.values():
            if h.btn_train_ref: h.btn_train_ref.configure(state="normal", fg_color=h.color)
            if h.btn_short_ref: h.btn_short_ref.configure(state="normal", fg_color="#89B4FA")

    def _calibrate_both(self):
        self.log("👆 點擊【雙手統一歸零】")
        self._calibrate('R'); self._calibrate('L')

    def _start_short_test_both(self):
        self.log("🚀 啟動雙手【短測試訓練模式】(每音鼓 20 筆)")
        self._lock_buttons()
        self._start_train('R', target_hits=20)
        self._start_train('L', target_hits=20)

    def _start_train_both(self):
        self.log("🚀 啟動雙手【標準訓練模式】(每音鼓 200 筆)")
        self._lock_buttons()
        self._start_train('R', target_hits=200)
        self._start_train('L', target_hits=200)
        
    def _start_short_test_single(self, hid):
        self.log(f"🚀 啟動 {self.hands[hid].name}【獨立短測試訓練】")
        self._lock_buttons()
        self._start_train(hid, target_hits=20)

    def _start_train_single(self, hid):
        self.log(f"🚀 啟動 {self.hands[hid].name}【獨立標準訓練】")
        self._lock_buttons()
        self._start_train(hid, target_hits=200)

    def _cancel_train_both(self):
        self.log("❌ 雙手統一取消訓練，回歸待機。")
        self._cancel_train('R'); self._cancel_train('L')

    def _cancel_train_single(self, hid):
        self.log(f"❌ 取消 {self.hands[hid].name} 訓練。")
        self._cancel_train(hid)

    def _start_train(self, hid, target_hits=200):
        h = self.hands[hid]
        h.X, h.y, h.raw_hits = [], [], [] 
        h.is_training, h.is_trained = True, False
        h.current_zone, h.hits_count = 0, 0
        h.target_hits = target_hits
        h.last_feat = None 
        self._update_ui_state()

    def _cancel_train(self, hid):
        h = self.hands[hid]
        h.X, h.y, h.raw_hits = [], [], []
        h.is_training, h.is_trained = False, False
        h.current_zone, h.hits_count = 0, 0
        h.last_feat = None
        
        any_training = any(hand.is_training for hand in self.hands.values())
        if not any_training:
            self._unlock_buttons()
            
        self._update_ui_state()

    def _calibrate(self, hid):
        h = self.hands[hid]
        if h.buffer:
            h.yaw_off = (h.yaw_off + h.buffer[-1][3]) % 360
            h.pitch_off += h.buffer[-1][4]
            self.log(f"🎯 [{h.name}] 已成功校準歸零基準點")

    def _update_ui_state(self):
        for hid in ['R', 'L']:
            h = self.hands[hid]
            for zid, ui in h.ui_refs.items():
                ui["frame"].configure(border_width=0, fg_color=DBGBG)
                ui["label"].configure(text_color=DIM)
                ui["count_lbl"].configure(text="") 
                
                if h.is_training and zid == h.current_zone:
                    ui["frame"].configure(border_width=3, border_color="#F9E2AF")
                    ui["label"].configure(text_color="#F9E2AF")
                    ui["count_lbl"].configure(text=f"{h.hits_count}/{h.target_hits}", text_color="#F9E2AF")
            
            is_connected = hid in self.active_hands
            
            if h.is_training:
                if is_connected:
                    h.ui_status_lbl.configure(text=f"🟡 訓練中 ({h.hits_count}/{h.target_hits})", text_color="#F9E2AF")
                else:
                    h.ui_status_lbl.configure(text=f"⚫ 訓練中 ({h.hits_count}/{h.target_hits}) (等待連線)", text_color=DIM)
            elif h.is_trained:
                if is_connected:
                    h.ui_status_lbl.configure(text="🟢 實戰 Demo 模式", text_color="#A6E3A1")
                else:
                    h.ui_status_lbl.configure(text="⚫ 實戰 Demo 模式 (等待連線)", text_color=DIM)
            else:
                if is_connected:
                    h.ui_status_lbl.configure(text="🟢 已連線", text_color="#A6E3A1")
                else:
                    h.ui_status_lbl.configure(text="⚫ 等待連線", text_color=DIM)
                    
        # 動態變更雙手統一產生/儲存 CSV 按鈕的文字與顏色
        any_trained = any(hand.is_trained for hand in self.hands.values())
        any_training = any(hand.is_training for hand in self.hands.values())
        
        if any_trained and not any_training:
            self.btn_save_all.configure(text="💾 儲存當前打擊報告CSV", fg_color="#89B4FA")
        else:
            self.btn_save_all.configure(text="💾 產生訓練 CSV (未完成也可存)", fg_color="#A6E3A1")

    def _flash(self, hid, zid, color):
        ui = self.hands[hid].ui_refs[zid]
        ui["frame"].configure(fg_color=color)
        ui["label"].configure(text_color="#000")
        self.after(FLASH_DURATION_MS, lambda: (ui["frame"].configure(fg_color=DBGBG), ui["label"].configure(text_color=DIM)))

if __name__ == "__main__":
    app = AirDrumDualStudio()
    app.mainloop()
