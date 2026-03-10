"""
TicTacToe Robot - Main Integration
Full system integration for smart robotic arm playing tic-tac-toe
Based on unsupervised learning using K-means clustering

Author: AI & Robotics Research
Version: 1.0.0
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple

# Import configuration
import config

# Import all modules
from modules.camera_input import CameraInput
from modules.image_preprocessing import ImagePreprocessor
from modules.board_detection import BoardDetector
from modules.feature_extraction import FeatureExtractor
from modules.kmeans_classifier import KMeansClassifier
from modules.board_state import BoardState
from modules.game_ai import GameAI
from modules.coordinate_mapping import CoordinateMapper
from modules.robot_control import RobotController


class TicTacToeRobot:
    """
    Main class integrating all components for the tic-tac-toe robot system.
    
    Pipeline:
    Camera → Vision → K-Means → Minimax → Robot Move
    """
    
    def __init__(self):
        """Initialize all system components."""
        print("=" * 60)
        print("  TicTacToe Robot System v1.0")
        print("  Based on Unsupervised Learning with K-Means Clustering")
        print("=" * 60)
        
        # Initialize components
        print("\n[INIT] Initializing components...")
        
        self.camera = CameraInput()
        self.preprocessor = ImagePreprocessor()
        self.board_detector = BoardDetector()
        self.feature_extractor = FeatureExtractor()
        self.classifier = KMeansClassifier()
        self.board_state = BoardState()
        self.game_ai = GameAI()
        self.coord_mapper = CoordinateMapper()
        self.robot = RobotController(simulation=config.SIMULATION_MODE)
        
        # Game state
        self.is_running = False
        self.game_count = 0
        self.robot_wins = 0
        self.player_wins = 0
        self.draws = 0
        
        # Piece tracking
        self.next_piece_idx = 0
        
        # Calibration data
        self.board_detected = False
        self.calibration_complete = False
        
        print("[INIT] All components initialized")
    
    def initialize(self) -> bool:
        """
        Initialize hardware and calibrate system.
        
        Returns:
            True if initialization successful
        """
        print("\n[INIT] Starting system initialization...")
        
        # Initialize camera
        if not self.camera.initialize():
            print("[ERROR] Failed to initialize camera")
            return False
        
        # Connect robot
        if not self.robot.connect():
            print("[WARNING] Robot connection failed, running in simulation")
        
        # Move robot to home position
        self.robot.go_home()
        
        print("[INIT] System initialization complete")
        return True
    
    def calibrate_board(self, max_attempts: int = 50) -> bool:
        """
        Calibrate board detection by finding the board in camera view.
        
        Args:
            max_attempts: Maximum frame attempts for detection
            
        Returns:
            True if calibration successful
        """
        print("\n[CALIB] Starting board calibration...")
        print("[CALIB] Please position the board in view of the camera")
        
        for attempt in range(max_attempts):
            ret, frame = self.camera.get_frame()
            if not ret:
                continue
            
            success, corners = self.board_detector.detect_board(frame)
            
            # Show detection progress
            if config.SHOW_DEBUG_WINDOWS:
                display = frame.copy()
                if success:
                    display = self.board_detector.draw_board_overlay(display, corners)
                    cv2.putText(display, "Board Detected! Press 'c' to confirm",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display, f"Searching for board... ({attempt}/{max_attempts})",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow("Calibration", display)
                key = cv2.waitKey(100)
                
                if key == ord('c') and success:
                    self.board_detected = True
                    self.calibration_complete = True
                    print("[CALIB] Board calibration complete!")
                    cv2.destroyWindow("Calibration")
                    return True
                elif key == ord('q'):
                    print("[CALIB] Calibration cancelled")
                    cv2.destroyWindow("Calibration")
                    return False
        
        print("[CALIB] Failed to detect board - using default calibration")
        return False
    
    def capture_and_process(self) -> Optional[np.ndarray]:
        """
        Capture frame and process for board state detection.
        
        Returns:
            Array of 9 cell labels, None if processing failed
        """
        # Capture frame
        ret, frame = self.camera.get_frame()
        if not ret:
            return None
        
        # Detect board
        success, corners = self.board_detector.detect_board(frame)
        if not success:
            print("[WARNING] Board not detected in frame")
            return None
        
        # Get warped board view
        warped = self.board_detector.get_warped_board(frame, corners)
        if warped is None:
            return None
        
        # Extract cells
        cells = self.board_detector.get_cells(warped)
        if len(cells) != 9:
            print(f"[WARNING] Expected 9 cells, got {len(cells)}")
            return None
        
        # Extract features
        features = self.feature_extractor.extract_all_cells(cells)
        
        # Classify using K-means
        labels, _ = self.classifier.fit_and_classify(features)
        
        # Display debug info
        if config.SHOW_DEBUG_WINDOWS:
            self._show_debug_display(frame, warped, cells, labels)
        
        return labels
    
    def _show_debug_display(self, frame: np.ndarray, warped: np.ndarray,
                            cells: list, labels: np.ndarray):
        """Show debug visualization windows."""
        # Original frame with board overlay
        display = self.board_detector.draw_board_overlay(frame)
        cv2.imshow("Camera View", display)
        
        # Warped board with grid
        warped_display = self.board_detector.draw_grid_overlay(warped)
        
        # Add cell labels
        label_names = {config.EMPTY: 'E', config.PLAYER: 'P', config.ROBOT: 'R'}
        height, width = warped_display.shape[:2]
        cell_h, cell_w = height // 3, width // 3
        
        for i in range(9):
            row, col = i // 3, i % 3
            cx = col * cell_w + cell_w // 2
            cy = row * cell_h + cell_h // 2
            label = label_names.get(labels[i], '?')
            color = {config.EMPTY: (200, 200, 200), 
                    config.PLAYER: (0, 255, 0), 
                    config.ROBOT: (0, 0, 255)}.get(labels[i], (255, 255, 255))
            cv2.putText(warped_display, label, (cx - 10, cy + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow("Board Analysis", warped_display)
        cv2.waitKey(1)
    
    def process_player_turn(self) -> bool:
        """
        Wait for and process player's move.
        
        Returns:
            True if valid move detected
        """
        print("\n[GAME] Waiting for player's move...")
        print("[GAME] Press 'n' when you've placed your piece")
        
        previous_state = self.board_state.matrix.copy()
        
        while True:
            labels = self.capture_and_process()
            
            if labels is not None:
                # Check for new move
                new_move = self.board_state.detect_new_move(labels)
                
                if new_move is not None:
                    row, col, player = new_move
                    if player == config.PLAYER:
                        self.board_state.update_from_labels(labels)
                        print(f"[GAME] Player moved to ({row}, {col})")
                        return True
            
            key = cv2.waitKey(100)
            if key == ord('n'):
                # Manual confirmation
                if labels is not None:
                    self.board_state.update_from_labels(labels)
                    print("[GAME] Move confirmed manually")
                    return True
            elif key == ord('q'):
                return False
    
    def execute_robot_turn(self) -> bool:
        """
        Calculate and execute robot's move.
        
        Returns:
            True if move executed successfully
        """
        print("\n[GAME] Robot's turn...")
        
        # Get best move from AI
        move = self.game_ai.get_best_move(self.board_state)
        
        if move is None:
            print("[GAME] No valid moves available")
            return False
        
        row, col = move
        print(f"[GAME] AI chose move: ({row}, {col})")
        
        # Execute pick and place
        success = self.robot.pick_and_place(row, col, self.next_piece_idx)
        
        if success:
            # Update board state
            self.board_state.set_cell(row, col, config.ROBOT)
            self.next_piece_idx += 1
            print(f"[GAME] Robot placed piece at ({row}, {col})")
            return True
        else:
            print("[ERROR] Robot failed to place piece")
            return False
    
    def check_game_end(self) -> Tuple[bool, Optional[int]]:
        """
        Check if game has ended.
        
        Returns:
            Tuple of (is_over, winner)
        """
        return self.board_state.is_game_over()
    
    def play_game(self) -> Optional[int]:
        """
        Play a complete game of tic-tac-toe.
        
        Returns:
            Winner (ROBOT, PLAYER, or None for draw)
        """
        print("\n" + "=" * 50)
        print(f"  GAME {self.game_count + 1}")
        print("=" * 50)
        
        # Reset for new game
        self.board_state.reset()
        self.next_piece_idx = 0
        
        # Determine who goes first
        robot_turn = config.AI_FIRST
        
        while True:
            # Print current state
            print("\n" + self.board_state.render())
            
            # Check for game end
            is_over, winner = self.check_game_end()
            if is_over:
                return winner
            
            if robot_turn:
                # Robot's turn
                if not self.execute_robot_turn():
                    print("[ERROR] Robot turn failed")
                    break
            else:
                # Player's turn
                if not self.process_player_turn():
                    print("[INFO] Player quit")
                    break
            
            # Switch turns
            robot_turn = not robot_turn
            
            # Check for game end after move
            is_over, winner = self.check_game_end()
            if is_over:
                return winner
        
        return None
    
    def run(self):
        """
        Main game loop.
        """
        print("\n[SYSTEM] Starting TicTacToe Robot...")
        
        # Initialize
        if not self.initialize():
            print("[ERROR] Initialization failed")
            return
        
        # Calibrate
        self.calibrate_board()
        
        self.is_running = True
        
        print("\n" + "=" * 50)
        print("  WELCOME TO TIC-TAC-TOE ROBOT!")
        print("  You play as X, Robot plays as O")
        print("=" * 50)
        print("\nControls:")
        print("  'n' - Confirm your move")
        print("  'r' - Reset game")
        print("  'q' - Quit")
        
        while self.is_running:
            # Play a game
            winner = self.play_game()
            
            # Display result
            print("\n" + "=" * 50)
            if winner == config.ROBOT:
                print("  ROBOT WINS!")
                self.robot_wins += 1
            elif winner == config.PLAYER:
                print("  PLAYER WINS!")
                self.player_wins += 1
            elif winner is None:
                print("  IT'S A DRAW!")
                self.draws += 1
            print("=" * 50)
            
            self.game_count += 1
            
            # Final board state
            print("\nFinal Board:")
            print(self.board_state.render())
            
            # Statistics
            print(f"\nScore: Robot {self.robot_wins} - Player {self.player_wins} - Draws {self.draws}")
            
            # Ask to play again
            print("\nPlay again? (y/n)")
            while True:
                key = cv2.waitKey(0)
                if key == ord('y'):
                    break
                elif key == ord('n') or key == ord('q'):
                    self.is_running = False
                    break
        
        self.shutdown()
    
    def shutdown(self):
        """Shutdown system cleanly."""
        print("\n[SYSTEM] Shutting down...")
        
        self.robot.go_home()
        self.robot.disconnect()
        self.camera.release()
        cv2.destroyAllWindows()
        
        print("[SYSTEM] Goodbye!")


def main():
    """Main entry point."""
    try:
        robot = TicTacToeRobot()
        robot.run()
    except KeyboardInterrupt:
        print("\n[SYSTEM] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
