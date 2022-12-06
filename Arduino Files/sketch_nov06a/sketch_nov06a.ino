/*
        Arduino Brushless Motor Control
     by Dejan, https://howtomechatronics.com
*/

#include <Servo.h>

Servo ESC;     // create servo object to control the ESC

void setup() {
  // Attach the ESC on pin 9
  ESC.attach(4,1000,2000); // (pin, min pulse width, max pulse width in microseconds) 
  ESC.write(0);
  delay(5000);
}
int power =90;
void loop() {
  Serial.print(power);
  ESC.write(power);    // Send the signal to the ESC
  power -= 10;
  delay(10000);
  ESC.write(0);
  delay(1000);
}
