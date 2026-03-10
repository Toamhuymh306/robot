"""
Configuration file for TicTacToe Robot System
All system parameters can be adjusted here
"""

# =============================================================================
# CAMERA SETTINGS
# =============================================================================
CAMERA_ID = 0                    # Camera index (0 for default webcam)
CAMERA_WIDTH = 640               # Frame width
CAMERA_HEIGHT = 480              # Frame height
CAMERA_FPS = 30                  # Frames per second
CAMERA_WARMUP_FRAMES = 30        # Frames to skip for camera warmup

# =============================================================================
# IMAGE PREPROCESSING SETTINGS
# =============================================================================
BLUR_KERNEL_SIZE = 5             # Gaussian blur kernel size
CANNY_THRESHOLD1 = 50            # Canny edge detection lower threshold
CANNY_THRESHOLD2 = 150           # Canny edge detection upper threshold
DILATE_ITERATIONS = 2            # Dilation iterations for edge enhancement
ERODE_ITERATIONS = 1             # Erosion iterations for noise removal

# =============================================================================
# BOARD DETECTION SETTINGS
# =============================================================================
MIN_BOARD_AREA = 10000           # Minimum area to consider as board (pixels^2)
MAX_BOARD_AREA = 200000          # Maximum area to consider as board (pixels^2)
BOARD_ASPECT_RATIO_MIN = 0.8     # Minimum aspect ratio for square detection
BOARD_ASPECT_RATIO_MAX = 1.2     # Maximum aspect ratio for square detection
CELL_PADDING = 10                # Padding inside each cell (pixels)

# =============================================================================
# K-MEANS CLUSTERING SETTINGS
# =============================================================================
N_CLUSTERS = 3                   # Number of clusters (empty, player, robot)
KMEANS_MAX_ITERATIONS = 100      # Maximum iterations for convergence
KMEANS_TOLERANCE = 1e-4          # Convergence tolerance
KMEANS_RANDOM_STATE = 42         # Random seed for reproducibility
FEATURE_TYPE = 'hsv_mean'        # Feature type: 'rgb_mean', 'hsv_mean', 'histogram'

# Cluster labels
EMPTY = 0
PLAYER = -1
ROBOT = 1

# =============================================================================
# GAME AI SETTINGS
# =============================================================================
MINIMAX_DEPTH = 9                # Maximum search depth for minimax
AI_FIRST = False                 # Whether AI plays first

# Win scores
WIN_SCORE = 100
LOSE_SCORE = -100
DRAW_SCORE = 0

# =============================================================================
# COORDINATE MAPPING SETTINGS (in mm)
# =============================================================================
# Board physical dimensions
BOARD_SIZE_MM = 150              # Physical board size (mm)
CELL_SIZE_MM = 50                # Physical cell size (mm)

# Board position relative to robot base
BOARD_ORIGIN_X = 100             # X offset from robot base (mm)
BOARD_ORIGIN_Y = 0               # Y offset from robot base (mm)
BOARD_ORIGIN_Z = 0               # Z offset (board height) (mm)

# Cell center spacing
CELL_SPACING_X = 50              # X spacing between cell centers (mm)
CELL_SPACING_Y = 50              # Y spacing between cell centers (mm)

# =============================================================================
# ROBOT ARM SETTINGS (3-DOF)
# =============================================================================
# Link lengths (mm)
LINK_1_LENGTH = 50               # Base to shoulder (vertical)
LINK_2_LENGTH = 100              # Shoulder to elbow
LINK_3_LENGTH = 100              # Elbow to end effector

# Servo angle limits (degrees)
SERVO_1_MIN = 0                  # Base rotation min
SERVO_1_MAX = 180                # Base rotation max
SERVO_2_MIN = 0                  # Shoulder min
SERVO_2_MAX = 180                # Shoulder max
SERVO_3_MIN = 0                  # Elbow min
SERVO_3_MAX = 180                # Elbow max

# Default positions
HOME_POSITION = (90, 90, 90)     # Home position (servo angles)
PICK_HEIGHT = 30                 # Height to pick pieces (mm)
PLACE_HEIGHT = 10                # Height to place pieces (mm)

# =============================================================================
# SERIAL COMMUNICATION SETTINGS
# =============================================================================
# ESP32 Settings (default)
SERIAL_PORT = 'COM3'             # Serial port (kiểm tra Device Manager)
# SERIAL_PORT = '/dev/ttyUSB0'   # Serial port for Linux
SERIAL_BAUDRATE = 115200         # ESP32 baud rate (115200)
# SERIAL_BAUDRATE = 9600         # Arduino baud rate (9600)
SERIAL_TIMEOUT = 1               # Timeout in seconds

# Cách tìm COM port:
# Windows: Device Manager -> Ports (COM & LPT) -> USB Serial Device
# Linux: ls /dev/tty* hoặc dmesg | grep tty

# =============================================================================
# SIMULATION MODE
# =============================================================================
SIMULATION_MODE = True           # True for simulation (no real robot)
SIMULATION_DELAY = 0.5           # Delay between simulated movements (seconds)

# =============================================================================
# DISPLAY SETTINGS
# =============================================================================
WINDOW_NAME = 'TicTacToe Robot'  # Main window name
SHOW_DEBUG_WINDOWS = True        # Show intermediate processing windows
DISPLAY_SCALE = 1.0              # Scale factor for display windows

# Colors for visualization (BGR format)
COLOR_EMPTY = (200, 200, 200)    # Gray for empty cells
COLOR_PLAYER = (0, 0, 255)       # Red for player
COLOR_ROBOT = (255, 0, 0)        # Blue for robot
COLOR_BOARD = (0, 255, 0)        # Green for board outline
COLOR_TEXT = (255, 255, 255)     # White for text

# =============================================================================
# PIECE STORAGE POSITIONS (for robot to pick pieces)
# =============================================================================
PIECE_STORAGE = [
    (150, 100, 0),               # Position 1
    (150, 120, 0),               # Position 2
    (150, 140, 0),               # Position 3
    (150, 160, 0),               # Position 4
    (150, 180, 0),               # Position 5
]

# =============================================================================
# HSV COLOR RANGES (for color-based detection backup)
# =============================================================================
# Green pieces (player)
HSV_GREEN_LOWER = (40, 100, 20)
HSV_GREEN_UPPER = (70, 255, 150)

# Red pieces (robot)
HSV_RED_LOWER1 = (0, 165, 20)
HSV_RED_UPPER1 = (12, 255, 255)
HSV_RED_LOWER2 = (160, 165, 20)
HSV_RED_UPPER2 = (185, 255, 255)

# Blue pieces (alternative)
HSV_BLUE_LOWER = (100, 165, 20)
HSV_BLUE_UPPER = (130, 255, 255)
