// Lines 1–92: AMT132S-V quadrature reader (A/B only) for UNO R4 WiFi

constexpr uint8_t PIN_A = 2;   // Encoder A+ (datasheet pin 8)
constexpr uint8_t PIN_B = 3;   // Encoder B+ (datasheet pin 10)

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
  while (!Serial) {} // helps on some USB-serial setups

  // AMT132S-V is push-pull CMOS -> use INPUT (no pullups needed)
  pinMode(PIN_A, INPUT);
  pinMode(PIN_B, INPUT);

  lastAB = readAB();

  // UNO R4 WiFi supports external interrupts on D2 and D3
  attachInterrupt(digitalPinToInterrupt(PIN_A), isrAB, CHANGE);
  attachInterrupt(digitalPinToInterrupt(PIN_B), isrAB, CHANGE);

  Serial.println("AMT132S-V A/B quadrature started. Rotate to see counts.");
}

void loop()
{
  static long lastPrint = 0;

  noInterrupts();
  long p = positionCount;
  interrupts();

  if (p != lastPrint) {
    Serial.println(p);
    lastPrint = p;
  }
}
