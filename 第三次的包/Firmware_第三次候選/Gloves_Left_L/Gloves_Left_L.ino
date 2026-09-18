/**
 * HoloGrip wired firmware - LEFT glove.
 * Third-field candidate.
 *
 * 100 Hz USB serial, 460800 baud.
 * IMPORTANT: failed I2C reads NEVER emit a normal D packet.
 * Error lines use E,<hand>,I2C,<packet_id>,<millis> and are ignored by host parsers.
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
static uint16_t consecutive_i2c_fail = 0;

static void initI2C() {
  Wire.begin();
  Wire.setClock(400000);
}

static bool readReg6(uint8_t reg, int16_t &v0, int16_t &v1, int16_t &v2) {
  for (uint8_t attempt = 0; attempt < 2; ++attempt) {
    while (Wire.available()) Wire.read();

    Wire.beginTransmission(JY_ADDR);
    Wire.write(reg);
    uint8_t tx = Wire.endTransmission(false);
    if (tx == 0) {
      uint8_t got = Wire.requestFrom((uint16_t)JY_ADDR, (uint8_t)6);
      if (got >= 6 && Wire.available() >= 6) {
        v0 = (int16_t)(Wire.read() | (Wire.read() << 8));
        v1 = (int16_t)(Wire.read() | (Wire.read() << 8));
        v2 = (int16_t)(Wire.read() | (Wire.read() << 8));
        return true;
      }
    }
    delayMicroseconds(250);
  }
  return false;
}

void setup() {
  Serial.begin(460800);
  initI2C();
  delay(300);
}

void loop() {
  uint32_t now = millis();
  if (now - last_tx < INTERVAL_MS) return;
  last_tx = now;

  const uint32_t this_packet_id = packet_id++;

  int16_t ax_raw = 0, ay_raw = 0, az_raw = 0;
  int16_t roll_raw = 0, pitch_raw = 0, yaw_raw = 0;

  bool acc_ok = readReg6(REG_ACC, ax_raw, ay_raw, az_raw);
  bool ang_ok = readReg6(REG_ANG, roll_raw, pitch_raw, yaw_raw);

  if (!acc_ok || !ang_ok) {
    consecutive_i2c_fail++;
    Serial.printf("E,%c,I2C,%lu,%lu\n",
                  HAND_ID, this_packet_id, millis());

    if (consecutive_i2c_fail >= 3) {
      Wire.end();
      delay(2);
      initI2C();
      consecutive_i2c_fail = 0;
    }
    return;
  }

  consecutive_i2c_fail = 0;

  float ax = ax_raw / 32768.0f * 16.0f;
  float ay = ay_raw / 32768.0f * 16.0f;
  float az = az_raw / 32768.0f * 16.0f;

  float roll = roll_raw / 32768.0f * 180.0f;
  float pitch = pitch_raw / 32768.0f * 180.0f;
  float yaw = yaw_raw / 32768.0f * 180.0f;

  Serial.printf("D,%c,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%lu,%lu\n",
                HAND_ID, ax, ay, az, yaw, pitch, roll,
                this_packet_id, millis());
}
