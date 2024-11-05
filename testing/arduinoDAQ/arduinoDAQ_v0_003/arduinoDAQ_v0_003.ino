// Define the baud rate
const unsigned long BAUD_RATE = 115200;
// Set up analog DAC out
const int DAC_PIN = A0;
// Set up analog ADC in
const int ADC_PIN1 = A1;

// Define a struct to hold parsed command values
struct CommandParameters {
  int signalDurationMicros;
  int waveFrequencyMicros;
  int waveAmplitude;
  int sampleIntervalMicros;
  int dutyCycle;
};

// Global struct instance to hold parsed values
volatile CommandParameters commandParams = {0, 0, 0, 0, 0};

// Global variable for readData
int* readData;

void setup() {
  analogWriteResolution(10); // Set analog out resolution to max, 10-bits (if supported by board)
  analogReadResolution(12);  // Set analog input resolution to max, 12-bits (if supported by board)

  Serial.begin(BAUD_RATE);
  Serial.println("Arduino is ready and listening for commands...");
}

void loop() {
  if (Serial.available() > 0) {
    char command[50]; // Buffer to hold incoming command
    int len = Serial.readBytesUntil('\n', command, sizeof(command) - 1);
    command[len] = '\0'; // Null-terminate the string

    parseCommand(command); // Parse the command
    generateData(const_cast<CommandParameters&>(commandParams)); // Generate wave and capture readings

    // Print the captured response data
    Serial.println("start");
    int numSamples = commandParams.signalDurationMicros / commandParams.sampleIntervalMicros;
    for (int i = 0; i < numSamples; i++) {
      Serial.println(readData[i]);
    }
    Serial.println("end"); // Fixed missing semicolon

    // Free readData after use
    free(readData);
    readData = nullptr; // Optional: Set pointer to nullptr to avoid accidental access
  }
}


void generateData(CommandParameters& params) {
  // Calculate the required parameters
  int numSamples = params.signalDurationMicros / params.sampleIntervalMicros;
  int period = params.waveFrequencyMicros / params.sampleIntervalMicros;
  int onTime = (period * params.dutyCycle) / 100;

  // Allocate memory for wave and read data arrays
  int* waveData = (int*)malloc(numSamples * sizeof(int));
  readData = (int*)malloc(numSamples * sizeof(int)); // Allocate readData

  if (waveData == nullptr || readData == nullptr) {
    Serial.println("Error: Memory allocation failed.");
    return; // Exit if allocation fails
  }

  // Generate square wave pattern in waveData array
  for (int i = 0; i < numSamples; i += period) {
    // Set 'on' values for the duty cycle period
    for (int j = 0; j < onTime && (i + j) < numSamples; ++j) {
      waveData[i + j] = params.waveAmplitude;
    }
    // Set 'off' values for the remainder of the period
    for (int j = onTime; j < period && (i + j) < numSamples; ++j) {
      waveData[i + j] = 0;
    }
  }

  // Write wave to DAC and read response to ADC
  for (int i = 0; i < numSamples; i++) {
    analogWrite(DAC_PIN, waveData[i]);           // Write each sample to analog pin
    readData[i] = analogRead(ADC_PIN1);           // Read response from analog input
    delayMicroseconds(params.sampleIntervalMicros); // Wait for sample interval
  }

  // Free dynamically allocated memory
  free(waveData);
  // No free for readData here, as it’s used outside of generateData
}
