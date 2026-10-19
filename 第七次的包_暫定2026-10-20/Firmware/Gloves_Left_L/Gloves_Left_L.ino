/**
 * HoloGrip wired firmware - LEFT glove.
 * 100 Hz USB serial, 460800 baud, HAND_ID fixed to L.
 */

#include <Wire.h>
#include <math.h>

#define HAND_ID 'L'

#define JY_ADDR 0x50
#define REG_ACC 0x34
#define REG_ANG 0x3D

static const uint32_t INTERVAL_MS = 10;
static uint32_t last_tx = 0;
static uint32_t packet_id = 0;

void setup() {
  Serial.begin(460800);
  Wire.begin();
  Wire.setClock(400000);
  delay(300);
}

void loop() {
  uint32_t now = millis();
  if (now - last_tx < INTERVAL_MS) return;
  last_tx = now;

  float ax = 0, ay = 0, az = 0;
  float roll = 0, pitch = 0, yaw = 0;

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

  Serial.printf("D,%c,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%lu,%lu\n",
                HAND_ID, ax, ay, az, yaw, pitch, roll, packet_id++, millis());
}
