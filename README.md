# Smart Robotic Arm Playing Tic-Tac-Toe

## Based on Unsupervised Learning Using K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)

## 📋 Overview

This project implements a complete system for a robotic arm that can play tic-tac-toe against a human opponent. The system uses:

- **Computer Vision** to detect the game board and pieces
- **K-Means Clustering** (unsupervised learning) to classify cell states
- **Minimax Algorithm** with alpha-beta pruning to find optimal moves
- **3-DOF Robot Arm** control with inverse kinematics

## 🎯 Features

- Real-time board detection and tracking
- Automatic piece classification using K-means clustering
- Optimal move calculation using Minimax AI
- Robot arm control with smooth movements
- Support for simulation mode (no hardware required)
- Comprehensive debugging and visualization tools

## 📁 Project Structure

```
TicTacToe-Robot-KMeans/
├── modules/
│   ├── camera_input.py        # Module 1: Camera capture
│   ├── image_preprocessing.py # Module 2: Image processing
│   ├── board_detection.py     # Module 3: Board detection
│   ├── feature_extraction.py  # Module 4: Feature extraction
│   ├── kmeans_classifier.py   # Module 5: K-means classification
│   ├── board_state.py         # Module 6: Game state management
│   ├── game_ai.py             # Module 7: Minimax AI
│   ├── coordinate_mapping.py  # Module 8: Coordinate mapping
│   └── robot_control.py       # Module 9: Robot control
├── docs/
│   └── SYSTEM_ARCHITECTURE.md # System documentation
├── test_data/                # Collected test samples
├── config.py                 # Configuration parameters
├── main.py                   # Main integration script
├── demo.py                   # Demo and testing script
├── test_camera.py            # Camera test script
├── test_paper_board.py       # Paper board detection (with trackbars)
├── test_accuracy.py          # K-means accuracy testing
├── test_esp32.py             # ESP32 connection test
├── requirements.txt          # Python dependencies
├── arduino_firmware.ino      # Arduino UNO/Nano firmware
├── esp32_firmware.ino        # ESP32 firmware
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam (for real-time detection)
- ESP32/Arduino + 3-DOF robot arm (optional, simulation available)

### Installation

1. Navigate to project folder:

```bash
cd d:/robot/model1/TicTacToe-Robot-KMeans
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Test camera first:

```bash
python test_camera.py
```

4. Run the demo:

```bash
python demo.py
```

### Running the Full System

```bash
python main.py
```

## 🎮 How to Use

### Demo Mode

```bash
# Run all demos
python demo.py

# Run specific demo
python demo.py board    # Board detection demo
python demo.py kmeans   # K-means classification demo
python demo.py ai       # Game AI demo
python demo.py robot    # Robot simulation demo
python demo.py game     # Full game simulation
python demo.py camera   # Live camera demo
```

### Test Scripts

```bash
# Test camera cơ bản
python test_camera.py

# Test nhận diện bàn cờ giấy (với trackbars điều chỉnh)
python test_paper_board.py

# Test độ chính xác K-means
python test_accuracy.py
# Chọn 1: Thu thập data (chụp ảnh + gán nhãn)
# Chọn 2: Test accuracy trên data đã thu thập
# Chọn 3: Test realtime

# Test kết nối ESP32
python test_esp32.py
```

### Game Controls

- **'n'** - Confirm your move (after placing your piece)
- **'r'** - Reset game
- **'q'** - Quit

## 🔧 Configuration

Edit `config.py` to adjust:

```python
# Camera settings
CAMERA_ID = 0              # Camera index
CAMERA_WIDTH = 640         # Frame width
CAMERA_HEIGHT = 480        # Frame height

# K-means settings
N_CLUSTERS = 3             # Number of clusters
FEATURE_TYPE = 'hsv_mean'  # Feature extraction method

# Robot settings
SIMULATION_MODE = True     # Run without real robot
SERIAL_PORT = 'COM3'       # Serial port (check Device Manager)
SERIAL_BAUDRATE = 115200   # ESP32: 115200, Arduino: 9600
```

## 🧠 System Architecture

```
┌──────────┐    ┌─────────────────┐    ┌─────────────────┐
│  CAMERA  │───▶│ IMAGE           │───▶│ BOARD           │
│  INPUT   │    │ PREPROCESSING   │    │ DETECTION       │
└──────────┘    └─────────────────┘    └────────┬────────┘
                                                │
                                                ▼
┌──────────┐    ┌─────────────────┐    ┌─────────────────┐
│  ROBOT   │◀───│ COORDINATE      │◀───│ FEATURE         │
│  CONTROL │    │ MAPPING         │    │ EXTRACTION      │
└──────────┘    └─────────────────┘    └────────┬────────┘
     ▲                                          │
     │          ┌─────────────────┐    ┌────────▼────────┐
     │          │ BOARD STATE     │◀───│ K-MEANS         │
     │          │ MATRIX          │    │ CLUSTERING      │
     │          └────────┬────────┘    └─────────────────┘
     │                   │
     │                   ▼
     │          ┌─────────────────┐
     └──────────│ GAME AI         │
                │ (MINIMAX)       │
                └─────────────────┘
```

## 📊 K-Means Classification

The system uses K-means clustering to classify board cells:

| Cluster | Meaning      | Color (default) |
| ------- | ------------ | --------------- |
| 0       | Empty        | Gray            |
| 1       | Robot piece  | Blue/Red        |
| -1      | Player piece | Green           |

Features used for classification:

- HSV mean values (default)
- RGB mean values
- Color histograms

## 🤖 Robot Specifications

### 3-DOF Arm Configuration

- **Joint 1 (Base)**: Rotation around vertical axis
- **Joint 2 (Shoulder)**: Controls arm elevation
- **Joint 3 (Elbow)**: Controls forearm angle

### Inverse Kinematics

```python
# Calculate joint angles for target position
θ1 = atan2(y, x)
θ3 = acos((d² - L2² - L3²) / (2·L2·L3))
θ2 = atan2(z', r) + acos((L2² + d² - L3²) / (2·L2·d))
```

## 🔌 Hardware Setup

### Components Required

1. **ESP32** (recommended) or Arduino Uno/Nano
2. 3x Servo motors (SG90 or MG996R)
3. Webcam (USB) or Laptop camera
4. Power supply (5V 2A for servos)
5. 3D printed/assembled robot arm
6. Paper board 15-20cm with 3x3 grid (for testing)

### ESP32 Setup (Recommended)

1. Install ESP32 board in Arduino IDE:
   - File → Preferences → Additional Board URLs:

   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```

   - Tools → Board Manager → Search "esp32" → Install

2. Install ESP32Servo library:
   - Sketch → Include Library → Manage Libraries → Search "ESP32Servo"

3. Upload `esp32_firmware.ino` to ESP32

4. Test connection:
   ```bash
   python test_esp32.py
   ```

### ESP32 Wiring

```
ESP32          Servo
-----          -----
GPIO 13  →     Servo 1 (Base)
GPIO 12  →     Servo 2 (Shoulder)
GPIO 14  →     Servo 3 (Elbow)
GPIO 27  →     Servo 4 (Gripper)
GND      →     GND chung

5V Power Supply (riêng) → VCC các Servo
```

⚠️ **Quan trọng**: Dùng nguồn 5V riêng cho servo, KHÔNG lấy từ ESP32!

### Arduino Wiring (Alternative)

```
Arduino:
  - Pin 9  → Servo 1 (Base)
  - Pin 10 → Servo 2 (Shoulder)
  - Pin 11 → Servo 3 (Elbow)
  - Pin 6  → Servo 4 (Gripper)
  - 5V/GND → External power for servos
```

Upload `arduino_firmware.ino` (Baudrate: 9600)

## 📝 Making a Test Board

1. Cut white paper into square (15-20cm)
2. Draw 3x3 grid with black marker
3. Use colored objects as pieces:
   - **X**: Red caps, red paper
   - **O**: Blue/Green caps, colored paper

```
┌───┬───┬───┐
│   │   │   │
├───┼───┼───┤
│   │   │   │  ← Kẻ đậm ~2mm
├───┼───┼───┤
│   │   │   │
└───┴───┴───┘
```

**Tips for better detection:**

- Dark background (wooden table, black cloth)
- Even lighting, no shadows
- Camera straight-on or slight angle

## 📊 Testing Model Accuracy

```bash
python test_accuracy.py
```

1. **Collect data** (Option 1): Capture images with labeled states
2. **Test accuracy** (Option 2): Run K-means on collected data
3. **Realtime test** (Option 3): Compare predictions with actual state

Expected results:
| Accuracy | Status |
|----------|--------|
| < 70% | Need better pieces/lighting |
| 70-85% | Acceptable |
| 85-95% | Good |
| > 95% | Excellent |

## 📱 Deployment

### Raspberry Pi

```bash
# Install on Raspberry Pi
sudo apt-get update
sudo apt-get install python3-opencv python3-numpy
pip3 install scikit-learn pyserial
```

### Jetson Nano

```bash
# Install on Jetson Nano (CUDA support)
pip3 install opencv-python numpy scikit-learn pyserial
```

## 🔮 Future Improvements

1. **Deep Learning**: Replace K-means with CNN for better piece detection
2. **Reinforcement Learning**: Train AI through self-play
3. **Voice Control**: Add speech recognition
4. **6-DOF Support**: Support more complex robot arms
5. **Online Learning**: Adapt to changing lighting conditions

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📞 Contact

For questions or support, please open an issue on GitHub.

---

_Built with ❤️ for AI and Robotics Research_
