/*
 * TicTacToe Robot - ESP32 Firmware
 * Controls 3-DOF robot arm with serial communication
 * 
 * ESP32 Servo Configuration:
 *   - Servo 1 (GPIO 13): Base rotation
 *   - Servo 2 (GPIO 12): Shoulder
 *   - Servo 3 (GPIO 14): Elbow
 *   - Servo 4 (GPIO 27): Gripper (optional)
 * 
 * Serial Commands:
 *   M<angle1>,<angle2>,<angle3>  - Move to angles
 *   G0                           - Open gripper
 *   G1                           - Close gripper
 *   H                            - Go to home position
 *   S                            - Get current status
 * 
 * Author: AI & Robotics Research
 * Version: 1.0 (ESP32)
 */

#include <ESP32Servo.h>

// Servo objects
Servo servo1;  // Base
Servo servo2;  // Shoulder
Servo servo3;  // Elbow
Servo servo4;  // Gripper

// ESP32 GPIO Pins (PWM capable)
const int SERVO1_PIN = 13;   // Base
const int SERVO2_PIN = 12;   // Shoulder
const int SERVO3_PIN = 14;   // Elbow
const int SERVO4_PIN = 27;   // Gripper

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
  Serial.begin(115200);  // ESP32 typically uses faster baud rate
  Serial.println("TicTacToe Robot ESP32 Initialized");
  
  // Allow allocation of all timers for servos
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);
  
  // Standard 50hz servo frequency
  servo1.setPeriodHertz(50);
  servo2.setPeriodHertz(50);
  servo3.setPeriodHertz(50);
  servo4.setPeriodHertz(50);
  
  // Attach servos (min/max pulse width in microseconds)
  servo1.attach(SERVO1_PIN, 500, 2400);
  servo2.attach(SERVO2_PIN, 500, 2400);
  servo3.attach(SERVO3_PIN, 500, 2400);
  servo4.attach(SERVO4_PIN, 500, 2400);
  
  // Go to home position
  goHome();
  
  Serial.println("Ready for commands");
}

void loop() {
  // Check for serial data
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    
    if (inChar == '\n') {
      stringComplete = true;
    } else if (inChar != '\r') {
      inputString += inChar;
    }
  }
  
  // Process serial commands when complete
  if (stringComplete) {
    processCommand(inputString);
    inputString = "";
    stringComplete = false;
  }
  
  delay(10);  // Small delay for stability
}

// Process incoming command
void processCommand(String cmd) {
  cmd.trim();
  
  if (cmd.length() == 0) return;
  
  Serial.print("Received: ");
  Serial.println(cmd);
  
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
  else if (cmd == "T") {
    // Test command - move through test positions
    testMovement();
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

// Test movement sequence
void testMovement() {
  Serial.println("Starting test sequence...");
  
  // Test each servo individually
  Serial.println("Testing Servo 1 (Base)...");
  moveToAngles(45, currentAngle2, currentAngle3);
  delay(500);
  moveToAngles(135, currentAngle2, currentAngle3);
  delay(500);
  moveToAngles(90, currentAngle2, currentAngle3);
  
  Serial.println("Testing Servo 2 (Shoulder)...");
  moveToAngles(currentAngle1, 45, currentAngle3);
  delay(500);
  moveToAngles(currentAngle1, 135, currentAngle3);
  delay(500);
  moveToAngles(currentAngle1, 90, currentAngle3);
  
  Serial.println("Testing Servo 3 (Elbow)...");
  moveToAngles(currentAngle1, currentAngle2, 45);
  delay(500);
  moveToAngles(currentAngle1, currentAngle2, 135);
  delay(500);
  moveToAngles(currentAngle1, currentAngle2, 90);
  
  Serial.println("Testing Gripper...");
  moveGripper(GRIPPER_CLOSE);
  delay(500);
  moveGripper(GRIPPER_OPEN);
  
  Serial.println("Test complete!");
  printStatus();
}

/*
 * ESP32 WIRING DIAGRAM:
 * 
 * ESP32 DevKit:
 * 
 *   3.3V (NOT for servos!)
 *   
 *   VIN (5V) ───────────────────────┬─────────────────┐
 *                                   │                 │
 *   GND ────┬───────────┬───────────┼─────────────────┼──────┐
 *           │           │           │                 │      │
 *      ┌────┴───┐  ┌────┴───┐  ┌────┴───┐       ┌─────┴────┐ │
 *      │ SERVO1 │  │ SERVO2 │  │ SERVO3 │       │ SERVO4   │ │
 *      │ (Base) │  │(Shoulder)│ │(Elbow)│       │(Gripper) │ │
 *      └───┬────┘  └───┬────┘  └───┬────┘       └────┬─────┘ │
 *          │           │           │                 │       │
 *  GPIO 13 ┘   GPIO 12 ┘   GPIO 14 ┘        GPIO 27 ─┘       │
 *                                                            │
 *   External 5V Power Supply (REQUIRED!) ────────────────────┘
 *   
 * IMPORTANT NOTES:
 * 1. ESP32 GPIO pins output 3.3V, but servo signal works fine
 * 2. NEVER power servos from ESP32's 3.3V pin!
 * 3. Use external 5V 2A+ power supply for servos
 * 4. Always connect GND of external supply to ESP32 GND
 * 5. Default baud rate is 115200 (faster than Arduino)
 * 
 * INSTALLING ESP32Servo LIBRARY:
 * Arduino IDE -> Sketch -> Include Library -> Manage Libraries
 * Search for "ESP32Servo" by Kevin Harrington
 * 
 * ESP32 BOARD SETUP:
 * Arduino IDE -> File -> Preferences -> Additional Board URLs:
 * https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
 * 
 * Then: Tools -> Board -> Board Manager -> Search "esp32" -> Install
 * Select: "ESP32 Dev Module" or your specific board
 * 
 */
