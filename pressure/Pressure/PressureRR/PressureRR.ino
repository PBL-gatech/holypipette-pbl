const int numReadings = 25;  // number of readings to average
float readings[numReadings];   // the readings from the analog input
int readIndex = 0;           // the index of the current reading
float total = 0;               // the running total
float average = 0;             // the average
float M = 2.559;

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
      int sensorValue = analogRead(A9);

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
      float Y = M * (average - 512);
      
      // send the result
      Serial.print("S");
      Serial.print(Y);
      Serial.println("E");
    }
  }
}
