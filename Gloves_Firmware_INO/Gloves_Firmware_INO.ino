/**
 * Air Drum Firmware v4.7 — 最新 UDP 10ms 極致低延遲傳輸版 (帶時間戳與序號)
 * 
 * 說明：
 * 1. 本檔案是 HoloGrip 系統的最新實戰手套韌體。
 * 2. 相比最原始的版本，本版本在 UDP 封包最後面新增了「封包序號」與「開機時間戳」兩個欄位。
 * 3. 配合電腦端 UDP正式版 (server.py) 使用，能即時算出高精度打擊時間差與傳輸穩定度。
 */

#include <Wire.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <math.h>
#include "esp_wifi.h"

// ================= 設定區 =================
const char* ssid = "Realme";          // 填寫您的 Wi-Fi 熱點名稱
const char* password = "RealmeZZZ";    // 填寫您的 Wi-Fi 密碼

// ⚠️ 請填寫您電腦目前的區域網路 IP (行動熱點網關通常為 192.168.137.1)
const char* targetIP = "192.168.137.1"; 
const int targetPort = 8888;
const int localPort = 8889;           // 本地 UDP 埠口

// ⚠️ 重要：右手燒錄 'R'，左手燒錄 'L'
#define HAND_ID 'R' 
// ==========================================

WiFiUDP udp;

#define JY_ADDR     0x50
#define REG_ACC     0x34  // 加速度起始暫存器 (0x34~0x36)
#define REG_ANG     0x3D  // 角度起始暫存器 (0x3D~0x3F)

static const uint32_t INTERVAL_MS = 10; 
static uint32_t lastTx = 0;

// 自增封包序號，用於在電腦端檢測丟包率與包間隔
static uint32_t packet_id = 0; 

void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000); // 確保 I2C 處於 400kHz 高速模式
  delay(1000);

  Serial.println("\n==================================");
  Serial.print("  Air Drum [最新 10欄位 UDP 版] - ");
  Serial.print((char)HAND_ID);
  Serial.println(" 手");
  Serial.println("==================================");
  
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) { 
    delay(500); 
    Serial.print("."); 
  }
  Serial.println("\n✅ WiFi 連線成功!");
  
  // 限制 Wi-Fi 最大發射功率以省電，適用於近距離傳輸
  esp_wifi_set_max_tx_power(34); // 8.5dBm * 4 = 34
  WiFi.setTxPower(WIFI_POWER_8_5dBm);

  // 初始化 UDP
  udp.begin(localPort);
  Serial.printf("📡 UDP 啟圖，目標 IP: %s:%d\n", targetIP, targetPort);
}

void loop() {
  uint32_t now = millis();
  if (now - lastTx < INTERVAL_MS) return;
  lastTx = now;

  float ax = 0, ay = 0, az = 0;
  float roll = 0, pitch = 0, yaw = 0;

  // 1. 讀取 3 個加速度暫存器 (共 6 位元組)
  Wire.beginTransmission(JY_ADDR);
  Wire.write(REG_ACC);
  if (Wire.endTransmission(false) == 0) {
    Wire.requestFrom((uint16_t)JY_ADDR, (uint8_t)6);
    if (Wire.available() >= 6) {
      int16_t ax_raw = Wire.read() | (Wire.read() << 8);
      int16_t ay_raw = Wire.read() | (Wire.read() << 8);
      int16_t az_raw = Wire.read() | (Wire.read() << 8);
      ax = ax_raw / 32768.0f * 16.0f;
      ay = ay_raw / 32768.0f * 16.0f;
      az = az_raw / 32768.0f * 16.0f;
    }
  }

  // 2. 讀取 3 個角度暫存器 (共 6 位元組)
  Wire.beginTransmission(JY_ADDR);
  Wire.write(REG_ANG);
  if (Wire.endTransmission(false) == 0) {
    Wire.requestFrom((uint16_t)JY_ADDR, (uint8_t)6);
    if (Wire.available() >= 6) {
      int16_t roll_raw  = Wire.read() | (Wire.read() << 8);
      int16_t pitch_raw = Wire.read() | (Wire.read() << 8);
      int16_t yaw_raw   = Wire.read() | (Wire.read() << 8);
      roll  = roll_raw  / 32768.0f * 180.0f;
      pitch = pitch_raw / 32768.0f * 180.0f;
      yaw   = yaw_raw   / 32768.0f * 180.0f;
    }
  }

  // 3. 發送最新 10 欄位數據 (末尾新增 packet_id 與 millis() 函數讀取)
  udp.beginPacket(targetIP, targetPort);
  udp.printf("D,%c,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%lu,%lu\n", 
             HAND_ID, ax, ay, az, yaw, pitch, roll, packet_id++, millis());
  udp.endPacket();
}
