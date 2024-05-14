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

// the setup routine runs once when you press reset:
void setup() {
  // initialize serial communication at 9600 bits per second:
  Serial.begin(9600);
}

// the loop routine runs over and over again forever:
void loop() {
  // read the input on analog pin 0:
  float M = 2.559;
  int sensorValue = analogRead(A0);
  // print out the value you read:
  // Serial.println(sensorValue - 512);
  float Y = M * (sensorValue - 512);
  Serial.println(Y);
  // float outputV = ((0.8 * 5) / (1034*2)) * (sensorValue + 1034) + 0.1 * 5;
  // Serial.println(outputV);
  delay(10);  // delay in between reads for stability
}
