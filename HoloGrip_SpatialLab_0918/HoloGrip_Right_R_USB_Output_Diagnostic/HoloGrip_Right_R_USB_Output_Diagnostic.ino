/**
 * HoloGrip RIGHT glove - USB Serial / JY901S diagnostic firmware
 *
 * 目的：
 * 1. 開機一定輸出 BOOT 訊息。
 * 2. 每 10 ms 一定輸出一行 D,R,...，即使 JY901S / I2C 讀取失敗也照樣輸出。
 * 3. 每秒輸出一次 #STAT，顯示 ACC/ANGLE 讀取成功率。
 *
 * HoloGrip UI 只會解析 D,R,... 行；#BOOT / #STAT 只給 Serial Monitor 看，
 * 不會影響既有 parser。
 *
 * Serial: 460800 baud
 * I2C: 400 kHz
 * Hand ID: R
 */

#include <Arduino.h>
#include <Wire.h>
#include <math.h>

#if !defined(CONFIG_IDF_TARGET_ESP32C6)
#error "請在 Arduino IDE 選擇 XIAO_ESP32C6；此診斷韌體只允許 ESP32-C6 編譯。"
#endif

#define HAND_ID 'R'

static const int I2C_SDA_PIN = 22;  // XIAO ESP32C6 D4
static const int I2C_SCL_PIN = 23;  // XIAO ESP32C6 D5

static const uint8_t JY_ADDR = 0x50;
static const uint8_t REG_ACC = 0x34;
static const uint8_t REG_ANG = 0x3D;

static const uint32_t SERIAL_BAUD = 460800;
static const uint32_t I2C_CLOCK = 400000;
static const uint32_t INTERVAL_US = 10000;   // 100 Hz

static uint32_t next_tx_us = 0;
static uint32_t packet_id = 0;

static uint32_t stat_last_ms = 0;
static uint32_t stat_packets = 0;
static uint32_t stat_acc_ok = 0;
static uint32_t stat_ang_ok = 0;
static uint32_t stat_acc_fail = 0;
static uint32_t stat_ang_fail = 0;

// 最近一次成功讀到的值。
// 若 I2C 暫時失敗，D 封包仍會送出；但 #STAT 可看出失敗。
// 為避免誤以為失敗時的舊值是新資料，我們在失敗時把對應欄位設成 0。
static float ax = 0.0f;
static float ay = 0.0f;
static float az = 0.0f;
static float roll_deg = 0.0f;
static float pitch_deg = 0.0f;
static float yaw_deg = 0.0f;

static bool read6(uint8_t reg, uint8_t *buf) {
  Wire.beginTransmission(JY_ADDR);
  Wire.write(reg);

  uint8_t err = Wire.endTransmission(false);
  if (err != 0) {
    return false;
  }

  uint8_t got = Wire.requestFrom((uint16_t)JY_ADDR, (uint8_t)6);
  if (got < 6 || Wire.available() < 6) {
    while (Wire.available()) {
      (void)Wire.read();
    }
    return false;
  }

  for (int i = 0; i < 6; ++i) {
    buf[i] = Wire.read();
  }
  return true;
}

static int16_t le16(const uint8_t *p) {
  return (int16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static bool readAccel() {
  uint8_t b[6];

  if (!read6(REG_ACC, b)) {
    ax = 0.0f;
    ay = 0.0f;
    az = 0.0f;
    return false;
  }

  int16_t rx = le16(&b[0]);
  int16_t ry = le16(&b[2]);
  int16_t rz = le16(&b[4]);

  ax = (float)rx / 32768.0f * 16.0f;
  ay = (float)ry / 32768.0f * 16.0f;
  az = (float)rz / 32768.0f * 16.0f;

  return true;
}

static bool readAngles() {
  uint8_t b[6];

  if (!read6(REG_ANG, b)) {
    roll_deg = 0.0f;
    pitch_deg = 0.0f;
    yaw_deg = 0.0f;
    return false;
  }

  int16_t rroll  = le16(&b[0]);
  int16_t rpitch = le16(&b[2]);
  int16_t ryaw   = le16(&b[4]);

  roll_deg  = (float)rroll  / 32768.0f * 180.0f;
  pitch_deg = (float)rpitch / 32768.0f * 180.0f;
  yaw_deg   = (float)ryaw   / 32768.0f * 180.0f;

  return true;
}

static void printBootBanner() {
  Serial.println();
  Serial.println("#BOOT,HoloGrip_XIAO_ESP32C6_DIAGNOSTIC_R");
  Serial.printf("#BOOT,chip=%s,baud=%lu,i2c=%lu,target_hz=100,hand=%c\n",
                ESP.getChipModel(),
                (unsigned long)SERIAL_BAUD,
                (unsigned long)I2C_CLOCK,
                HAND_ID);
  Serial.printf("#BOOT,i2c_sda=GPIO%d(D4),i2c_scl=GPIO%d(D5)\n",
                I2C_SDA_PIN, I2C_SCL_PIN);
  Serial.printf("#BOOT,jy_addr=0x%02X,acc_reg=0x%02X,ang_reg=0x%02X\n",
                JY_ADDR, REG_ACC, REG_ANG);
  Serial.println("#BOOT,If_you_can_read_this_USB_SERIAL_is_alive");
}

void setup() {
  Serial.begin(SERIAL_BAUD);

  // 給 Windows / USB CDC 一點時間完成枚舉。
  delay(1200);

  printBootBanner();

  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  Wire.setClock(I2C_CLOCK);

  delay(100);

  // 做一次最基本的 I2C ACK 測試。
  Wire.beginTransmission(JY_ADDR);
  uint8_t err = Wire.endTransmission();

  if (err == 0) {
    Serial.println("#I2C,JY901S_ACK_OK,addr=0x50");
  } else {
    Serial.printf("#I2C,JY901S_ACK_FAIL,addr=0x50,err=%u\n", err);
  }

  next_tx_us = micros();
  stat_last_ms = millis();

  Serial.println("#READY,stream_start");
}

void loop() {
  uint32_t now_us = micros();

  // 用有符號差值避免 micros() rollover 造成問題。
  if ((int32_t)(now_us - next_tx_us) < 0) {
    return;
  }

  // 固定 100 Hz 排程。
  next_tx_us += INTERVAL_US;

  // 若因某些原因落後太多，不要一次補印幾百包。
  if ((int32_t)(now_us - next_tx_us) > 50000) {
    next_tx_us = now_us + INTERVAL_US;
  }

  bool acc_ok = readAccel();
  bool ang_ok = readAngles();

  stat_packets++;

  if (acc_ok) {
    stat_acc_ok++;
  } else {
    stat_acc_fail++;
  }

  if (ang_ok) {
    stat_ang_ok++;
  } else {
    stat_ang_fail++;
  }

  // 這行格式就是 HoloGrip Python parser 需要的格式。
  // 無論 I2C 成不成功都一定輸出。
  Serial.printf(
    "D,%c,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%lu,%lu\n",
    HAND_ID,
    ax, ay, az,
    yaw_deg, pitch_deg, roll_deg,
    (unsigned long)packet_id++,
    (unsigned long)millis()
  );

  uint32_t now_ms = millis();

  if (now_ms - stat_last_ms >= 1000) {
    float hz = 0.0f;
    uint32_t dt = now_ms - stat_last_ms;

    if (dt > 0) {
      hz = (float)stat_packets * 1000.0f / (float)dt;
    }

    Serial.printf(
      "#STAT,hz=%.1f,packets=%lu,acc_ok=%lu,acc_fail=%lu,ang_ok=%lu,ang_fail=%lu\n",
      hz,
      (unsigned long)stat_packets,
      (unsigned long)stat_acc_ok,
      (unsigned long)stat_acc_fail,
      (unsigned long)stat_ang_ok,
      (unsigned long)stat_ang_fail
    );

    stat_last_ms = now_ms;
    stat_packets = 0;
    stat_acc_ok = 0;
    stat_ang_ok = 0;
    stat_acc_fail = 0;
    stat_ang_fail = 0;
  }
}
