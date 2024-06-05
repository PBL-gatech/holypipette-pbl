/*
  AnalogReadSerial

  Reads an analog input on pin 0, prints the result to the Serial Monitor.
  Graphical representation is available using Serial Plotter (Tools > Serial Plotter menu).
  Attach the center pin of a potentiometer to pin A0, and the outside pins to +5V and ground.

  This example code is in the public domain.

  https://www.arduino.cc/en/Tutorial/BuiltInExamples/AnalogReadSerial
*/

// ! NOTE: Sending air to the BOTTON valve DECREASES analog pressure reading
// ! NOTE: Sending air to the TOP valve INCREASES analog pressure reading

const int numReadings = 100;  // number of readings to average
float readings[numReadings];   // the readings from the analog input
int readIndex = 0;           // the index of the current reading
float total = 0;               // the running total
float average = 0;             // the average

// the setup routine runs once when you press reset:
void setup() {
  // initialize serial communication at 9600 bits per second:
  Serial.begin(9600);
  
  // initialize all the readings to 0:
  for (int i = 0; i < numReadings; i++) {
    readings[i] = 0;
  }
}

float M = 2.559;
// the loop routine runs over and over again forever:
void loop() {
  // read the input on analog pin 0:
  int sensorValue = analogRead(A0);

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
  
  // print out the value you read:
  // Serial.println(Y);
  Serial.print("S");
  Serial.print(Y);
  Serial.println("E");

  // THIS DELAY IS CRUCIAL, ANYTHING LESS THAN 33ms WILL CAUSE THE GRAPH TO NOT UPDATE
  // FAST ENOUGH. Think about trying to compress 100fps video in 60fps. You can't.
  // this is because of what is in graph.py, updateDt (set to 33ms)
  delay(16);  // delay in between reads for stability
}
