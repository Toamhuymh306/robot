/*
 * TicTacToe Robot - Arduino Firmware
 * Controls 3-DOF robot arm with serial communication
 * 
 * Servo Configuration:
 *   - Servo 1 (Pin 9):  Base rotation
 *   - Servo 2 (Pin 10): Shoulder
 *   - Servo 3 (Pin 11): Elbow
 *   - Servo 4 (Pin 6):  Gripper (optional)
 * 
 * Serial Commands:
 *   M<angle1>,<angle2>,<angle3>  - Move to angles
 *   G0                           - Open gripper
 *   G1                           - Close gripper
 *   H                            - Go to home position
 *   S                            - Get current status
 * 
 * Author: AI & Robotics Research
 * Version: 1.0
 */

#include <Servo.h>

// Servo objects
Servo servo1;  // Base
Servo servo2;  // Shoulder
Servo servo3;  // Elbow
Servo servo4;  // Gripper

// Servo pins
const int SERVO1_PIN = 9;
const int SERVO2_PIN = 10;
const int SERVO3_PIN = 11;
const int SERVO4_PIN = 6;

// Home position (degrees)
const int HOME_ANGLE1 = 90;
const int HOME_ANGLE2 = 90;
const int HOME_ANGLE3 = 90;
const int GRIPPER_OPEN = 90;
const int GRIPPER_CLOSE = 45;

// Current angles
int currentAngle1 = HOME_ANGLE1;
int currentAngle2 = HOME_ANGLE2;
int currentAngle3 = HOME_ANGLE3;
int gripperAngle = GRIPPER_OPEN;

// Movement speed (delay in ms between each degree)
const int MOVE_DELAY = 15;

// Serial buffer
String inputString = "";
bool stringComplete = false;

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  Serial.println("TicTacToe Robot Initialized");
  
  // Attach servos
  servo1.attach(SERVO1_PIN);
  servo2.attach(SERVO2_PIN);
  servo3.attach(SERVO3_PIN);
  servo4.attach(SERVO4_PIN);
  
  // Go to home position
  goHome();
  
  Serial.println("Ready for commands");
}

void loop() {
  // Process serial commands when complete
  if (stringComplete) {
    processCommand(inputString);
    inputString = "";
    stringComplete = false;
  }
}

// Serial event handler
void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    
    if (inChar == '\n') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }
}

// Process incoming command
void processCommand(String cmd) {
  cmd.trim();
  
  if (cmd.startsWith("M")) {
    // Move command: M<angle1>,<angle2>,<angle3>
    processMoveCommand(cmd.substring(1));
  }
  else if (cmd.startsWith("G")) {
    // Gripper command: G0 (open) or G1 (close)
    processGripperCommand(cmd.substring(1));
  }
  else if (cmd == "H") {
    // Home command
    goHome();
    Serial.println("OK:HOME");
  }
  else if (cmd == "S") {
    // Status command
    printStatus();
  }
  else {
    Serial.println("ERR:UNKNOWN_CMD");
  }
}

// Process move command
void processMoveCommand(String params) {
  // Parse angles from comma-separated string
  int comma1 = params.indexOf(',');
  int comma2 = params.indexOf(',', comma1 + 1);
  
  if (comma1 == -1 || comma2 == -1) {
    Serial.println("ERR:INVALID_PARAMS");
    return;
  }
  
  int angle1 = params.substring(0, comma1).toInt();
  int angle2 = params.substring(comma1 + 1, comma2).toInt();
  int angle3 = params.substring(comma2 + 1).toInt();
  
  // Validate angles
  angle1 = constrain(angle1, 0, 180);
  angle2 = constrain(angle2, 0, 180);
  angle3 = constrain(angle3, 0, 180);
  
  // Move smoothly to target angles
  moveToAngles(angle1, angle2, angle3);
  
  Serial.print("OK:MOVE:");
  Serial.print(currentAngle1);
  Serial.print(",");
  Serial.print(currentAngle2);
  Serial.print(",");
  Serial.println(currentAngle3);
}

// Move all servos smoothly to target angles
void moveToAngles(int target1, int target2, int target3) {
  // Calculate number of steps needed
  int diff1 = abs(target1 - currentAngle1);
  int diff2 = abs(target2 - currentAngle2);
  int diff3 = abs(target3 - currentAngle3);
  int maxDiff = max(max(diff1, diff2), diff3);
  
  if (maxDiff == 0) return;
  
  // Calculate increments
  float inc1 = (float)(target1 - currentAngle1) / maxDiff;
  float inc2 = (float)(target2 - currentAngle2) / maxDiff;
  float inc3 = (float)(target3 - currentAngle3) / maxDiff;
  
  // Move incrementally
  for (int step = 0; step < maxDiff; step++) {
    int newAngle1 = currentAngle1 + round(inc1 * (step + 1));
    int newAngle2 = currentAngle2 + round(inc2 * (step + 1));
    int newAngle3 = currentAngle3 + round(inc3 * (step + 1));
    
    servo1.write(newAngle1);
    servo2.write(newAngle2);
    servo3.write(newAngle3);
    
    delay(MOVE_DELAY);
  }
  
  // Ensure final positions
  servo1.write(target1);
  servo2.write(target2);
  servo3.write(target3);
  
  currentAngle1 = target1;
  currentAngle2 = target2;
  currentAngle3 = target3;
}

// Process gripper command
void processGripperCommand(String param) {
  int state = param.toInt();
  
  if (state == 0) {
    // Open gripper
    moveGripper(GRIPPER_OPEN);
    Serial.println("OK:GRIP_OPEN");
  }
  else if (state == 1) {
    // Close gripper
    moveGripper(GRIPPER_CLOSE);
    Serial.println("OK:GRIP_CLOSE");
  }
  else {
    Serial.println("ERR:INVALID_GRIP");
  }
}

// Move gripper smoothly
void moveGripper(int targetAngle) {
  int steps = abs(targetAngle - gripperAngle);
  int dir = (targetAngle > gripperAngle) ? 1 : -1;
  
  for (int i = 0; i < steps; i++) {
    gripperAngle += dir;
    servo4.write(gripperAngle);
    delay(10);
  }
  
  gripperAngle = targetAngle;
}

// Go to home position
void goHome() {
  moveToAngles(HOME_ANGLE1, HOME_ANGLE2, HOME_ANGLE3);
  moveGripper(GRIPPER_OPEN);
}

// Print current status
void printStatus() {
  Serial.print("STATUS:");
  Serial.print(currentAngle1);
  Serial.print(",");
  Serial.print(currentAngle2);
  Serial.print(",");
  Serial.print(currentAngle3);
  Serial.print(",GRIP:");
  Serial.println(gripperAngle);
}

/*
 * WIRING DIAGRAM:
 * 
 * Arduino UNO/Nano:
 * 
 *   +5V ────────────────────────────┬─────────────────┐
 *                                   │                 │
 *   GND ────┬───────────┬───────────┼─────────────────┼──────┐
 *           │           │           │                 │      │
 *      ┌────┴───┐  ┌────┴───┐  ┌────┴───┐       ┌─────┴────┐ │
 *      │ SERVO1 │  │ SERVO2 │  │ SERVO3 │       │ SERVO4   │ │
 *      │ (Base) │  │(Shoulder)│ │(Elbow)│       │(Gripper) │ │
 *      └───┬────┘  └───┬────┘  └───┬────┘       └────┬─────┘ │
 *          │           │           │                 │       │
 *   PIN 9 ─┘    PIN 10 ┘    PIN 11 ┘          PIN 6 ─┘       │
 *                                                            │
 *   External 5V Power Supply (recommended) ──────────────────┘
 *   
 * NOTES:
 * - Use external 5V power supply for servos (Arduino can't supply enough current)
 * - Connect GND of external supply to Arduino GND
 * - Servos: SG90 (light duty) or MG996R (heavy duty)
 * 
 */
