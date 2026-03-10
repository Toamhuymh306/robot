# Smart Robotic Arm Playing Tic-Tac-Toe

## Based on Unsupervised Learning Using K-means Clustering

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SYSTEM ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │  CAMERA  │───▶│ IMAGE           │───▶│ BOARD           │                │
│  │  INPUT   │    │ PREPROCESSING   │    │ DETECTION       │                │
│  └──────────┘    └─────────────────┘    └────────┬────────┘                │
│                                                   │                         │
│                                                   ▼                         │
│  ┌──────────┐    ┌─────────────────┐    ┌─────────────────┐                │
│  │  ROBOT   │◀───│ COORDINATE      │◀───│ FEATURE         │                │
│  │  CONTROL │    │ MAPPING         │    │ EXTRACTION      │                │
│  └──────────┘    └─────────────────┘    └────────┬────────┘                │
│       ▲                                          │                         │
│       │                                          ▼                         │
│       │          ┌─────────────────┐    ┌─────────────────┐                │
│       │          │ BOARD STATE     │◀───│ K-MEANS         │                │
│       │          │ MATRIX          │    │ CLUSTERING      │                │
│       │          └────────┬────────┘    └─────────────────┘                │
│       │                   │                                                 │
│       │                   ▼                                                 │
│       │          ┌─────────────────┐                                       │
│       └──────────│ GAME AI         │                                       │
│                  │ (MINIMAX)       │                                       │
│                  └─────────────────┘                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT STAGE                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Camera Frame (640x480 RGB) ──▶ Grayscale ──▶ Blur ──▶ Edge Detect  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                        │                                     │
│                                        ▼                                     │
│  PROCESSING STAGE                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Perspective Transform ──▶ 9 Cell Extraction ──▶ Feature Vectors    │    │
│  │                                                   (RGB/HSV mean)    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                        │                                     │
│                                        ▼                                     │
│  CLASSIFICATION STAGE                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ K-Means (K=3) ──▶ Cluster Labels ──▶ Board Matrix 3x3              │    │
│  │ Clusters: EMPTY(0), PLAYER(-1), ROBOT(+1)                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                        │                                     │
│                                        ▼                                     │
│  DECISION STAGE                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Minimax Algorithm ──▶ Optimal Move (i,j) ──▶ Robot Coordinates     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                        │                                     │
│                                        ▼                                     │
│  OUTPUT STAGE                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Inverse Kinematics ──▶ Servo Angles (θ1, θ2, θ3) ──▶ Robot Arm     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Module Architecture

### Module 1: Camera Input (`camera_input.py`)

- Capture video from webcam using OpenCV
- Support multiple camera indices
- Frame rate control and buffering

### Module 2: Image Preprocessing (`image_preprocessing.py`)

- Grayscale conversion
- Gaussian blur for noise reduction
- Canny edge detection
- Perspective transformation (bird's eye view)

### Module 3: Board Detection (`board_detection.py`)

- Detect board boundaries using contour analysis
- Identify the 3x3 grid using corner detection
- Extract individual cell regions

### Module 4: Feature Extraction (`feature_extraction.py`)

- Extract color features from each cell
- Support RGB and HSV color spaces
- Calculate mean, std, histogram features

### Module 5: K-means Clustering (`kmeans_classifier.py`)

- Implement K-means with K=3
- Classify cells into: empty, player, robot
- Adaptive threshold for classification

### Module 6: Board State Matrix (`board_state.py`)

- Convert cluster labels to board matrix
- Track game state changes
- Detect new moves by comparing states

### Module 7: Game AI (`game_ai.py`)

- Minimax algorithm with alpha-beta pruning
- Evaluate board positions
- Find optimal move for robot

### Module 8: Coordinate Mapping (`coordinate_mapping.py`)

- Map cell index (i,j) to physical coordinates (x,y,z)
- Calibration support for different setups
- Workspace boundary checking

### Module 9: Robot Control (`robot_control.py`)

- 3-DOF inverse kinematics
- Servo angle calculation
- Serial communication with Arduino
- Simulation mode for testing

### Module 10: Integration (`main.py`)

- Full pipeline orchestration
- Game loop management
- User interface and display

---

## 4. Class Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLASS DIAGRAM                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────┐         ┌─────────────────────┐                   │
│  │    CameraInput      │         │  ImagePreprocessor  │                   │
│  ├─────────────────────┤         ├─────────────────────┤                   │
│  │ - camera_id: int    │         │ - blur_kernel: int  │                   │
│  │ - cap: VideoCapture │         │ - canny_thresh: tuple│                  │
│  ├─────────────────────┤         ├─────────────────────┤                   │
│  │ + get_frame()       │────────▶│ + preprocess()      │                   │
│  │ + release()         │         │ + perspective_warp()│                   │
│  └─────────────────────┘         └──────────┬──────────┘                   │
│                                              │                              │
│                                              ▼                              │
│  ┌─────────────────────┐         ┌─────────────────────┐                   │
│  │   BoardDetector     │◀────────│  FeatureExtractor   │                   │
│  ├─────────────────────┤         ├─────────────────────┤                   │
│  │ - board_corners     │         │ - color_space: str  │                   │
│  │ - cell_size         │         │ - feature_type: str │                   │
│  ├─────────────────────┤         ├─────────────────────┤                   │
│  │ + detect_board()    │         │ + extract()         │                   │
│  │ + get_cells()       │         │ + get_feature_vector│                   │
│  └─────────────────────┘         └──────────┬──────────┘                   │
│                                              │                              │
│                                              ▼                              │
│  ┌─────────────────────┐         ┌─────────────────────┐                   │
│  │  KMeansClassifier   │◀────────│   BoardState        │                   │
│  ├─────────────────────┤         ├─────────────────────┤                   │
│  │ - n_clusters: int   │         │ - matrix: 3x3 array │                   │
│  │ - centroids         │         │ - history: list     │                   │
│  ├─────────────────────┤         ├─────────────────────┤                   │
│  │ + fit()             │         │ + update()          │                   │
│  │ + predict()         │         │ + detect_move()     │                   │
│  └─────────────────────┘         └──────────┬──────────┘                   │
│                                              │                              │
│                                              ▼                              │
│  ┌─────────────────────┐         ┌─────────────────────┐                   │
│  │      GameAI         │────────▶│ CoordinateMapper    │                   │
│  ├─────────────────────┤         ├─────────────────────┤                   │
│  │ - depth: int        │         │ - workspace_origin  │                   │
│  │ - player: int       │         │ - cell_spacing      │                   │
│  ├─────────────────────┤         ├─────────────────────┤                   │
│  │ + minimax()         │         │ + index_to_coords() │                   │
│  │ + get_best_move()   │         │ + calibrate()       │                   │
│  └─────────────────────┘         └──────────┬──────────┘                   │
│                                              │                              │
│                                              ▼                              │
│                                  ┌─────────────────────┐                   │
│                                  │   RobotController   │                   │
│                                  ├─────────────────────┤                   │
│                                  │ - link_lengths      │                   │
│                                  │ - serial_port       │                   │
│                                  ├─────────────────────┤                   │
│                                  │ + move_to()         │                   │
│                                  │ + inverse_kinematics│                   │
│                                  │ + pick_and_place()  │                   │
│                                  └─────────────────────┘                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Pseudo-code for Main Algorithms

### 5.1 K-means Clustering Algorithm

```
ALGORITHM KMeansClustering(data, K, max_iterations):
    INPUT: data - feature vectors for each cell
           K - number of clusters (3: empty, player, robot)
           max_iterations - maximum iterations
    OUTPUT: cluster_labels for each cell

    BEGIN
        // Initialize centroids randomly
        centroids = RANDOM_SELECT(data, K)

        FOR iteration = 1 TO max_iterations:
            // Assignment step
            FOR each point in data:
                distances = COMPUTE_DISTANCE(point, centroids)
                label[point] = ARGMIN(distances)
            END FOR

            // Update step
            FOR each cluster k:
                centroids[k] = MEAN(points where label == k)
            END FOR

            // Check convergence
            IF centroids_change < tolerance:
                BREAK
            END IF
        END FOR

        RETURN labels
    END
```

### 5.2 Minimax Algorithm with Alpha-Beta Pruning

```
ALGORITHM Minimax(board, depth, isMaximizing, alpha, beta):
    INPUT: board - current game state
           depth - search depth
           isMaximizing - true if maximizing player
           alpha, beta - pruning bounds
    OUTPUT: best score and move

    BEGIN
        IF depth == 0 OR game_over(board):
            RETURN evaluate(board)
        END IF

        IF isMaximizing:
            maxEval = -INFINITY
            bestMove = NULL

            FOR each move in get_valid_moves(board):
                new_board = make_move(board, move, ROBOT)
                eval = Minimax(new_board, depth-1, FALSE, alpha, beta)

                IF eval > maxEval:
                    maxEval = eval
                    bestMove = move
                END IF

                alpha = MAX(alpha, eval)
                IF beta <= alpha:
                    BREAK  // Prune
                END IF
            END FOR

            RETURN maxEval, bestMove
        ELSE:
            minEval = +INFINITY

            FOR each move in get_valid_moves(board):
                new_board = make_move(board, move, PLAYER)
                eval = Minimax(new_board, depth-1, TRUE, alpha, beta)
                minEval = MIN(minEval, eval)

                beta = MIN(beta, eval)
                IF beta <= alpha:
                    BREAK  // Prune
                END IF
            END FOR

            RETURN minEval
        END IF
    END
```

### 5.3 3-DOF Inverse Kinematics

```
ALGORITHM InverseKinematics3DOF(x, y, z, L1, L2, L3):
    INPUT: x, y, z - target position in Cartesian coordinates
           L1, L2, L3 - link lengths of robot arm
    OUTPUT: theta1, theta2, theta3 - joint angles

    BEGIN
        // Base rotation (joint 1)
        theta1 = ATAN2(y, x)

        // Project to 2D plane
        r = SQRT(x² + y²)
        z_prime = z - L1  // Adjust for base height

        // Calculate distance to target in 2D plane
        d = SQRT(r² + z_prime²)

        // Check if target is reachable
        IF d > L2 + L3 OR d < ABS(L2 - L3):
            RETURN ERROR("Target unreachable")
        END IF

        // Calculate joint 3 (elbow) using cosine law
        cos_theta3 = (d² - L2² - L3²) / (2 * L2 * L3)
        theta3 = ACOS(cos_theta3)

        // Calculate joint 2 (shoulder)
        alpha = ATAN2(z_prime, r)
        beta = ACOS((L2² + d² - L3²) / (2 * L2 * d))
        theta2 = alpha + beta

        RETURN theta1, theta2, theta3
    END
```

---

## 6. Hardware Connections

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          HARDWARE CONNECTIONS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌────────────┐      USB       ┌────────────┐                              │
│   │   WEBCAM   │◀──────────────▶│  COMPUTER  │                              │
│   └────────────┘                └─────┬──────┘                              │
│                                       │ USB/Serial                          │
│                                       ▼                                      │
│                                ┌────────────┐                                │
│                                │  ARDUINO   │                                │
│                                │   UNO/NANO │                                │
│                                └─────┬──────┘                                │
│                                      │                                       │
│                    ┌─────────────────┼─────────────────┐                    │
│                    │                 │                 │                    │
│                    ▼                 ▼                 ▼                    │
│              ┌──────────┐      ┌──────────┐      ┌──────────┐              │
│              │  SERVO 1 │      │  SERVO 2 │      │  SERVO 3 │              │
│              │  (Base)  │      │ (Shoulder)│     │  (Elbow) │              │
│              │  Pin D9  │      │  Pin D10 │      │  Pin D11 │              │
│              └──────────┘      └──────────┘      └──────────┘              │
│                   │                 │                 │                     │
│                   └─────────────────┼─────────────────┘                     │
│                                     ▼                                       │
│                              ┌────────────┐                                 │
│                              │ 3-DOF ARM  │                                 │
│                              │  (GRIPPER) │                                 │
│                              └────────────┘                                 │
│                                                                              │
│   Power Supply: 5V 2A for servos (separate from Arduino power)              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Directory Structure

```
TicTacToe-Robot-KMeans/
├── docs/
│   ├── SYSTEM_ARCHITECTURE.md
│   └── DEPLOYMENT_GUIDE.md
├── modules/
│   ├── __init__.py
│   ├── camera_input.py
│   ├── image_preprocessing.py
│   ├── board_detection.py
│   ├── feature_extraction.py
│   ├── kmeans_classifier.py
│   ├── board_state.py
│   ├── game_ai.py
│   ├── coordinate_mapping.py
│   └── robot_control.py
├── images/
│   ├── inputs/
│   └── outputs/
├── config.py
├── main.py
├── demo.py
├── requirements.txt
├── arduino_firmware.ino
└── README.md
```

---

## 8. Future Improvements

1. **Deep Learning Integration**: Replace K-means with CNN for more robust piece detection
2. **Reinforcement Learning**: Train robot to learn optimal strategies through self-play
3. **Multi-camera System**: Use multiple cameras for better 3D perception
4. **Voice Control**: Add speech recognition for hands-free interaction
5. **Gesture Recognition**: Detect human moves through hand gestures
6. **Online Learning**: Adapt K-means clusters in real-time
7. **6-DOF Arm**: Support for more complex robot arms
8. **Vision-based Calibration**: Automatic camera-robot calibration using markers

---

## 9. Deployment Options

### Raspberry Pi 4

- CPU: Quad-core ARM Cortex-A72
- RAM: 4GB+ recommended
- Storage: 32GB SD card
- OS: Raspberry Pi OS (64-bit)
- Additional: Camera module or USB webcam

### Jetson Nano

- GPU: 128-core NVIDIA Maxwell
- CPU: Quad-core ARM A57
- RAM: 4GB
- Better for real-time ML inference
- CUDA support for faster K-means

### Performance Comparison

| Feature           | Raspberry Pi 4 | Jetson Nano |
| ----------------- | -------------- | ----------- |
| K-means inference | ~50ms          | ~10ms       |
| Image processing  | ~30ms          | ~15ms       |
| Total latency     | ~150ms         | ~80ms       |
| Power consumption | 3W             | 5-10W       |
| Cost              | $35-55         | $99-129     |

---

_Document Version: 1.0_
_Last Updated: March 2026_
