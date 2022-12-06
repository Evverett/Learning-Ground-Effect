#include "HX711.h"

HX711 scale;

uint8_t dataPin = 5;
uint8_t clockPin = 4;

uint32_t start, stop;
volatile float f;
#include <Servo.h>

Servo ESC;     // create servo object to control the ESC


void setup()
{
  Serial.begin(115200);
  // Serial.println(__FILE__);
  // Serial.print("LIBRARY VERSION: ");
  // Serial.println(HX711_LIB_VERSION);
  // Serial.println();

  ESC.attach(20,1000,2000); // (pin, min pulse width, max pulse width in microseconds) 
  ESC.write(0);
  delay(5000);
  
  scale.begin(dataPin, clockPin);

  scale.set_scale(500);       // TODO you need to calibrate this yourself.
  scale.tare();
}
int power = 0;
float Powers[] = {0,0,0,0,0,0,0,0,0,0}
const long intervl = 10000;

void loop()
{
  // continuous scale 4x per second
  // Serial.println(power);
  scale.tare();
  ESC.write(Powers[power]);    // Send the signal to the ESC

  unsigned long previousMillis = millis();
  unsigned long currentMillis = millis();
  Serial.println("power");
  Serial.println(Powers[power]);
  while (currentMillis - previousMillis < intervl) {
    currentMillis = millis();
    f = scale.get_units(5);
    Serial.println(f);
  }
  Serial.println("clear");
  ESC.write(0);
  power += 1;
  unsigned long previousMillis2 = millis();
  unsigned long currentMillis2 = millis();
  while (currentMillis2 - previousMillis2 < intervl) {
    currentMillis2 = millis();
    f = scale.get_units(5);
    Serial.println(f);
  }
}
