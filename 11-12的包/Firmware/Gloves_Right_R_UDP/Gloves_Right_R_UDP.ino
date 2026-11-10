/**
 * HoloGrip wireless firmware - RIGHT glove (UDP over a phone hotspot).
 *
 * Same sensor reading and the same text line as the wired firmware
 * (Firmware/COM/Gloves_Firmware_INO_COM.ino):
 *     D,HAND,ax,ay,az,yaw,pitch,roll,packet_id,millis
 * at 100 Hz. The line is also printed on USB serial, so a glove on a cable
 * still works with the wired programs.
 *
 * Wireless:
 *   - Each datagram carries the newest line plus the previous REDUNDANT-1
 *     lines; the PC drops duplicates by packet_id, so one lost datagram
 *     loses no sample.
 *   - Until the PC is found the glove broadcasts to port 8888. The PC
 *     program broadcasts "HG_HELLO" to port 8889 every second; once heard,
 *     the glove sends unicast to that PC (unicast gets Wi-Fi retries).
 *     No HELLO for HELLO_TIMEOUT_MS -> back to broadcast.
 *   - Wi-Fi power saving is off (it adds 100+ ms bursts).
 *   - Wi-Fi loss never stops sampling; reconnection is retried in the
 *     background and packet_id keeps counting, so the PC sees the gap.
 *
 * Before flashing:
 *   1. HAND_ID is fixed to R in this sketch.
 *   2. Copy wifi_secrets.example.h to wifi_secrets.h and fill in the hotspot.
 */

#include <Wire.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <math.h>
#include "wifi_secrets.h"

// RIGHT glove (fixed).
#define HAND_ID 'R'

#define JY_ADDR 0x50
#define REG_ACC 0x34
#define REG_ANG 0x3D

static const uint16_t DATA_PORT = 8888;    // PC listens here
static const uint16_t HELLO_PORT = 8889;   // glove listens here for HG_HELLO
static const uint32_t INTERVAL_MS = 10;
static const uint32_t HELLO_TIMEOUT_MS = 5000;
static const uint32_t RECONNECT_MS = 2000;
static const int REDUNDANT = 3;
static const int LINE_MAX = 96;

static WiFiUDP udp;
static bool udp_started = false;
static IPAddress pc_ip;
static bool pc_known = false;
static uint32_t last_hello = 0;
static uint32_t last_reconnect = 0;

static uint32_t next_tx = 0;
static uint32_t packet_id = 0;
static char lines[REDUNDANT][LINE_MAX];
static int line_len[REDUNDANT];
static int newest = -1;
static int stored = 0;

static void read_sensor(float &ax, float &ay, float &az, float &yaw, float &pitch, float &roll) {
  ax = ay = az = yaw = pitch = roll = 0;
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
  Wire.beginTransmission(JY_ADDR);
  Wire.write(REG_ANG);
  if (Wire.endTransmission(false) == 0) {
    Wire.requestFrom((uint16_t)JY_ADDR, (uint8_t)6);
    if (Wire.available() >= 6) {
      int16_t roll_raw = Wire.read() | (Wire.read() << 8);
      int16_t pitch_raw = Wire.read() | (Wire.read() << 8);
      int16_t yaw_raw = Wire.read() | (Wire.read() << 8);
      roll = roll_raw / 32768.0f * 180.0f;
      pitch = pitch_raw / 32768.0f * 180.0f;
      yaw = yaw_raw / 32768.0f * 180.0f;
    }
  }
}

static void keep_wifi() {
  uint32_t now = millis();
  if (WiFi.status() == WL_CONNECTED) {
    if (!udp_started) {
      udp.begin(HELLO_PORT);
      udp_started = true;
      pc_known = false;
    }
    return;
  }
  if (udp_started) {
    udp.stop();
    udp_started = false;
    pc_known = false;
  }
  if (now - last_reconnect >= RECONNECT_MS) {
    last_reconnect = now;
    WiFi.disconnect();
    WiFi.begin(WIFI_SSID, WIFI_PASS);   // returns at once; connection completes in the background
  }
}

static void check_hello() {
  if (!udp_started) return;
  int n;
  while ((n = udp.parsePacket()) > 0) {
    char buf[16];
    int got = udp.read(buf, sizeof(buf) - 1);
    if (got == 8 && memcmp(buf, "HG_HELLO", 8) == 0) {
      pc_ip = udp.remoteIP();
      pc_known = true;
      last_hello = millis();
    }
  }
  if (pc_known && millis() - last_hello > HELLO_TIMEOUT_MS) pc_known = false;
}

static void send_datagram() {
  if (!udp_started || stored == 0) return;
  IPAddress target = pc_known ? pc_ip : WiFi.broadcastIP();
  if (!udp.beginPacket(target, DATA_PORT)) return;
  for (int k = stored - 1; k >= 0; --k) {              // oldest first
    int i = (newest - k + REDUNDANT) % REDUNDANT;
    udp.write((const uint8_t *)lines[i], line_len[i]);
  }
  udp.endPacket();
}

void setup() {
  Serial.begin(460800);
  Wire.begin();
  Wire.setClock(400000);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setTxPower(WIFI_POWER_8_5dBm);   // same as the July UDP firmware (short range, stable on the XIAO ESP32C6)
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  last_reconnect = millis();
  delay(300);
  next_tx = millis();
}

void loop() {
  keep_wifi();
  check_hello();

  uint32_t now = millis();
  if ((int32_t)(now - next_tx) < 0) return;
  next_tx += INTERVAL_MS;
  if ((int32_t)(now - next_tx) > (int32_t)(5 * INTERVAL_MS)) next_tx = now + INTERVAL_MS;   // fell far behind: resync, no burst

  float ax, ay, az, yaw, pitch, roll;
  read_sensor(ax, ay, az, yaw, pitch, roll);

  newest = (newest + 1) % REDUNDANT;
  int len = snprintf(lines[newest], LINE_MAX, "D,%c,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%lu,%lu\n",
                     HAND_ID, ax, ay, az, yaw, pitch, roll, (unsigned long)packet_id++, (unsigned long)millis());
  line_len[newest] = (len > 0 && len < LINE_MAX) ? len : LINE_MAX - 1;
  if (stored < REDUNDANT) stored++;

  Serial.write((const uint8_t *)lines[newest], line_len[newest]);
  send_datagram();
}
