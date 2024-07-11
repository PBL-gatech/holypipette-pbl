const int numReadings = 25;  // number of readings to average
float readings[numReadings];   // the readings from the analog input
int readIndex = 0;           // the index of the current reading
float total = 0;               // the running total
float average = 0;             // the average

// So how do we calculate M and B? Victor plotted set values vs the truth (digital nanometer) vs the sensor value, and fit a line
// Nanometer: 0.9559x + 38.237
// Sensor: 0.3942x + 523.01
// M = 1 / 0.3942 = 2.53678336

// float M = 5.824;
float M = 1;
// float M = 2.559;

void setup() {
  Serial.begin(9600);  // initialize serial communication
  for (int i = 0; i < numReadings; i++) {
    readings[i] = 0;  // initialize all readings to 0
  }
}

void loop() {
  if (Serial.available() > 0) {  // check if data is available to read
    char command = Serial.read();  // read the incoming command
    if (command == 'R') {  // if the command is 'R'
      // read the input on analog pin 0:
      int sensorValue = analogRead(A10);

      // subtract the last reading:
      total = total - readings[readIndex];
      // read from the sensor:
      readings[readIndex] = sensorValue;
      // add the reading to the total:
      total = total + readings[readIndex];
      // advance to the next position in the array:
      readIndex = readIndex + 1;

      // if we're at the end of the array:
      if (readIndex >= numReadings) {
        // wrap around to the beginning:
        readIndex = 0;
      }

      // calculate the average:
      average = total / numReadings;
      
      // compute the adjusted value
      // float Y = M * (average - 513.62);
      float Y = M * (average - 0);
      
      // send the result
      Serial.print("S");
      Serial.print(Y);
      Serial.println("E");
    }
  }
}
