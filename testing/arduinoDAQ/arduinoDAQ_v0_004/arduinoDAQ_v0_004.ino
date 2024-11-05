// Define the baud rate
const unsigned long BAUD_RATE = 500000; 
// Set up analog DAC out
const int DAC_PIN = A0;
// Set up analog ADC in
const int ADC_PIN1 = A1;
const int ADC_PIN2 = A2;

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

// Global variables for readData and respData
int* readData;
int* respData;

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
      Serial.print(readData[i]);
      Serial.print(",");
      Serial.println(respData[i]);
    }
    Serial.println("end"); // Fixed missing semicolon

    // Free readData and respData after use
    free(readData);
    readData = nullptr; // Optional: Set pointer to nullptr to avoid accidental access
    free(respData);
    respData = nullptr;
  }
}

void parseCommand(const char* command) {
  // Parse values into struct fields using sscanf
  if (sscanf(command, "a %d %d %d %d %d", &commandParams.signalDurationMicros, &commandParams.waveFrequencyMicros, 
             &commandParams.waveAmplitude, &commandParams.sampleIntervalMicros, &commandParams.dutyCycle) != 5) {
    Serial.println("Error: Incorrect command format.");
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
  respData = (int*)malloc(numSamples * sizeof(int)); // Allocate respData

  if (waveData == nullptr || readData == nullptr || respData == nullptr) {
    Serial.println("Error: Memory allocation failed.");
    free(waveData); // Ensure all allocated memory is freed if allocation fails
    free(readData);
    free(respData);
    return;
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
    readData[i] = analogRead(ADC_PIN1);          // Read command from analog input 1
    respData[i] = analogRead(ADC_PIN2);          // Read response from analog input 2
    delayMicroseconds(params.sampleIntervalMicros); // Wait for sample interval
  }

  // Free dynamically allocated memory for waveData
  free(waveData);
}
