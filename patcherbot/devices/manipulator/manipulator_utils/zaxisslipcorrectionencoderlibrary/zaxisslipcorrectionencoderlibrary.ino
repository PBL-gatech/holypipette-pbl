#include <Encoder.h>
constexpr uint8_t PIN_A = 2, PIN_B = 3;
Encoder enc(PIN_A, PIN_B);

void setup() {
  Serial.begin(115200);
  enc.write(0);
}

void loop() {
  static uint32_t last = 0;
  uint32_t now = micros();
  if (now - last >= 5000) { // 5ms = 200Hz
    last = now;
    Serial.println(enc.read());
  }
}


