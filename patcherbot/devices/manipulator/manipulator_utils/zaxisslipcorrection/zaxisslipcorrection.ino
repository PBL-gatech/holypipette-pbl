// AMT132S-V quadrature reader + serial streamer for ScientificaSerialEncoder (UNO R4 WiFi)
//
// Wiring (AMT13 connector pins per datasheet):
//   +5V (pin 6) -> UNO 5V
//   GND (pin 4) -> UNO GND
//   A+  (pin 8) -> UNO D2
//   B+  (pin 10)-> UNO D3
//
// Serial protocol (matches your Python):
//   Each sample is TWO lines:
//     ENC
//     <integer_count>
//
// Your calibration observation:
//   1000 µm travel ~ 459 counts  => 2.178649 µm/count (in Python scaling)

constexpr uint8_t PIN_A = 2;   // Encoder A+ (datasheet pin 8)
constexpr uint8_t PIN_B = 3;   // Encoder B+ (datasheet pin 10)

constexpr uint32_t STREAM_HZ = 10;
constexpr uint32_t STREAM_PERIOD_MS = 1000 / STREAM_HZ;

volatile long positionCount = 0;
volatile uint8_t lastAB = 0;

static inline uint8_t readAB()
{
  uint8_t a = (uint8_t)digitalRead(PIN_A);
  uint8_t b = (uint8_t)digitalRead(PIN_B);
  return (uint8_t)((a << 1) | b);  // A in bit1, B in bit0
}

void isrAB()
{
  uint8_t newAB = readAB();
  uint8_t t = (uint8_t)((lastAB << 2) | newAB);

  // Valid quadrature transitions:
  // +1 : 00->01->11->10->00
  // -1 : 00->10->11->01->00
  switch (t) {
    case 0b0001:
    case 0b0111:
    case 0b1110:
    case 0b1000:
      positionCount++;
      break;

    case 0b0010:
    case 0b1011:
    case 0b1101:
    case 0b0100:
      positionCount--;
      break;

    default:
      // illegal transition (noise/glitch); ignore
      break;
  }

  lastAB = newAB;
}

void setup()
{
  Serial.begin(115200);
  while (!Serial) {} // helpful on some USB-serial setups

  // AMT132S-V outputs are push-pull CMOS -> INPUT (no pullups required)
  pinMode(PIN_A, INPUT);
  pinMode(PIN_B, INPUT);

  lastAB = readAB();

  // UNO R4 WiFi external interrupts on D2/D3
  attachInterrupt(digitalPinToInterrupt(PIN_A), isrAB, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_B), isrAB, CHANGE);

  // Optional: one-time header for humans (comment out if your Python dislikes it)
  // Serial.println("ENCODER_READY");
}

void loop()
{
  static uint32_t lastMs = 0;
  uint32_t now = millis();
  if (now - lastMs < STREAM_PERIOD_MS) return;
  lastMs = now;

  noInterrupts();
  long p = positionCount;
  interrupts();

  // Two-line packet to match your Python (discard one line, parse next line)
  Serial.println("ENC");
  Serial.println(p);
}
