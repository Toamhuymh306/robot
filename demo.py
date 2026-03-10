"""
TicTacToe Robot - Demo and Testing Script
Demonstrates all system components without requiring actual hardware

Run with: python demo.py
"""

import cv2
import numpy as np
import time
import sys

# Import configuration
import config

# Import modules
from modules.camera_input import CameraInput
from modules.image_preprocessing import ImagePreprocessor
from modules.board_detection import BoardDetector
from modules.feature_extraction import FeatureExtractor
from modules.kmeans_classifier import KMeansClassifier
from modules.board_state import BoardState
from modules.game_ai import GameAI
from modules.coordinate_mapping import CoordinateMapper
from modules.robot_control import RobotController


def create_test_board(board_state: np.ndarray) -> np.ndarray:
    """
    Create a synthetic board image for testing.
    
    Args:
        board_state: 3x3 array with state values
        
    Returns:
        Synthetic board image (300x300 BGR)
    """
    img = np.full((300, 300, 3), (200, 200, 200), dtype=np.uint8)
    
    # Draw grid
    for i in range(4):
        cv2.line(img, (i * 100, 0), (i * 100, 300), (0, 0, 0), 2)
        cv2.line(img, (0, i * 100), (300, i * 100), (0, 0, 0), 2)
    
    # Draw pieces
    for row in range(3):
        for col in range(3):
            cx, cy = col * 100 + 50, row * 100 + 50
            
            if board_state[row, col] == config.PLAYER:
                # Draw X (green)
                cv2.line(img, (cx - 30, cy - 30), (cx + 30, cy + 30), (0, 180, 0), 4)
                cv2.line(img, (cx + 30, cy - 30), (cx - 30, cy + 30), (0, 180, 0), 4)
            elif board_state[row, col] == config.ROBOT:
                # Draw O (blue)
                cv2.circle(img, (cx, cy), 35, (200, 50, 0), 4)
    
    return img


def demo_board_detection():
    """Demonstrate board detection module."""
    print("\n" + "=" * 60)
    print("DEMO: Board Detection")
    print("=" * 60)
    
    # Create test image with perspective board
    img = np.full((480, 640, 3), (100, 100, 100), dtype=np.uint8)
    
    # Draw perspective quadrilateral (board)
    pts = np.array([[150, 100], [490, 80], [520, 380], [120, 400]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (200, 200, 200))
    cv2.polylines(img, [pts], True, (0, 0, 0), 3)
    
    # Draw grid lines on board
    for i in range(1, 3):
        # Interpolate grid lines
        t = i / 3
        p1 = pts[0] + t * (pts[3] - pts[0])
        p2 = pts[1] + t * (pts[2] - pts[1])
        cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), (0, 0, 0), 2)
        
        p1 = pts[0] + t * (pts[1] - pts[0])
        p2 = pts[3] + t * (pts[2] - pts[3])
        cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), (0, 0, 0), 2)
    
    detector = BoardDetector()
    success, corners = detector.detect_board(img)
    
    print(f"Board detected: {success}")
    if success:
        print(f"Corners: {corners}")
        
        # Get warped view
        warped = detector.get_warped_board(img, corners, 300)
        cells = detector.get_cells(warped)
        
        print(f"Extracted {len(cells)} cells")
        
        # Show results
        display = detector.draw_board_overlay(img, corners)
        cv2.imshow("Board Detection - Original", display)
        cv2.imshow("Board Detection - Warped", detector.draw_grid_overlay(warped))
        
        print("\nPress any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def demo_kmeans_classification():
    """Demonstrate K-means classification."""
    print("\n" + "=" * 60)
    print("DEMO: K-Means Classification")
    print("=" * 60)
    
    # Create test cells with different states
    cells = []
    expected_labels = []
    
    # Empty cells (gray)
    for _ in range(5):
        cell = np.full((80, 80, 3), (150, 150, 150), dtype=np.uint8)
        cell += np.random.randint(-10, 10, cell.shape, dtype=np.int16).astype(np.uint8)
        cells.append(cell)
        expected_labels.append(config.EMPTY)
    
    # Player cells (green)
    for _ in range(2):
        cell = np.full((80, 80, 3), (50, 180, 50), dtype=np.uint8)
        # Draw X pattern
        cv2.line(cell, (10, 10), (70, 70), (0, 200, 0), 3)
        cv2.line(cell, (70, 10), (10, 70), (0, 200, 0), 3)
        cells.append(cell)
        expected_labels.append(config.PLAYER)
    
    # Robot cells (blue)
    for _ in range(2):
        cell = np.full((80, 80, 3), (180, 50, 50), dtype=np.uint8)
        # Draw O pattern
        cv2.circle(cell, (40, 40), 30, (200, 0, 0), 3)
        cells.append(cell)
        expected_labels.append(config.ROBOT)
    
    # Extract features
    extractor = FeatureExtractor(feature_type='hsv_mean')
    features = extractor.extract_all_cells(cells)
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Features:\n{features}")
    
    # Classify with K-means
    classifier = KMeansClassifier()
    labels, cluster_info = classifier.fit_and_classify(features)
    
    print(f"\nCluster mapping: {cluster_info['cluster_to_label']}")
    print(f"Cluster counts: {cluster_info['cluster_counts']}")
    
    # Compare results
    print("\nClassification Results:")
    correct = 0
    label_names = {config.EMPTY: 'Empty', config.PLAYER: 'Player', config.ROBOT: 'Robot'}
    
    for i, (pred, exp) in enumerate(zip(labels, expected_labels)):
        status = "✓" if pred == exp else "✗"
        if pred == exp:
            correct += 1
        print(f"  Cell {i}: Predicted={label_names.get(pred, '?'):6s}, "
              f"Expected={label_names.get(exp, '?'):6s} {status}")
    
    accuracy = correct / len(labels) * 100
    print(f"\nAccuracy: {accuracy:.1f}%")
    
    # Visualize
    combined_cells = np.hstack(cells[:3])
    cv2.imshow("Sample Cells (Empty, Player, Robot)", combined_cells)
    
    vis = classifier.get_cluster_visualization(features, classifier.kmeans.labels_)
    cv2.imshow("K-Means Clusters", vis)
    
    print("\nPress any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def demo_game_ai():
    """Demonstrate game AI (Minimax)."""
    print("\n" + "=" * 60)
    print("DEMO: Game AI (Minimax)")
    print("=" * 60)
    
    board = BoardState()
    ai = GameAI()
    
    # Scenario 1: AI should win
    print("\nScenario 1: AI should find winning move")
    board.reset()
    board.set_cell(0, 0, config.ROBOT)
    board.set_cell(0, 1, config.ROBOT)
    board.set_cell(1, 0, config.PLAYER)
    board.set_cell(1, 1, config.PLAYER)
    
    print(board.render())
    move = ai.get_best_move(board)
    print(f"AI move: {move} (should be (0, 2) to win)")
    
    # Scenario 2: AI should block
    print("\nScenario 2: AI should block player")
    board.reset()
    board.set_cell(0, 0, config.PLAYER)
    board.set_cell(0, 1, config.PLAYER)
    board.set_cell(1, 1, config.ROBOT)
    
    print(board.render())
    move = ai.get_best_move(board)
    print(f"AI move: {move} (should be (0, 2) to block)")
    
    # Scenario 3: Opening move
    print("\nScenario 3: Opening move (empty board)")
    board.reset()
    
    print(board.render())
    move = ai.get_best_move(board)
    print(f"AI move: {move} (should be (1, 1) - center)")


def demo_robot_simulation():
    """Demonstrate robot control simulation."""
    print("\n" + "=" * 60)
    print("DEMO: Robot Control Simulation")
    print("=" * 60)
    
    robot = RobotController(simulation=True)
    robot.connect()
    
    print("\n1. Moving to home position...")
    robot.go_home()
    
    print("\n2. Testing inverse kinematics...")
    test_positions = [(100, 0, 50), (100, 50, 30), (150, 0, 0)]
    for pos in test_positions:
        angles = robot.inverse_kinematics(*pos)
        if angles:
            print(f"   Position {pos} -> Angles: {angles}")
        else:
            print(f"   Position {pos} -> UNREACHABLE")
    
    print("\n3. Simulating pick and place for cell (1, 1)...")
    robot.pick_and_place(1, 1, piece_idx=0)
    
    robot.disconnect()


def demo_full_game():
    """Demonstrate a full game simulation."""
    print("\n" + "=" * 60)
    print("DEMO: Full Game Simulation")
    print("=" * 60)
    
    board = BoardState()
    ai = GameAI()
    robot = RobotController(simulation=True)
    robot.connect()
    
    print("\nSimulating a game where Human plays randomly...")
    print("Robot (O) plays optimally using Minimax\n")
    
    move_num = 0
    robot_turn = config.AI_FIRST
    
    while True:
        print(f"\n--- Move {move_num + 1} ---")
        print(board.render())
        
        # Check game end
        is_over, winner = board.is_game_over()
        if is_over:
            break
        
        if robot_turn:
            # Robot's turn
            move = ai.get_best_move(board)
            if move:
                row, col = move
                print(f"Robot plays: ({row}, {col})")
                robot.pick_and_place(row, col, move_num // 2)
                board.set_cell(row, col, config.ROBOT)
        else:
            # Simulate human (random valid move)
            empty = board.get_empty_cells()
            if empty:
                import random
                row, col = random.choice(empty)
                print(f"Human plays: ({row}, {col})")
                board.set_cell(row, col, config.PLAYER)
        
        robot_turn = not robot_turn
        move_num += 1
        time.sleep(0.5)
    
    # Final result
    print("\n" + "=" * 40)
    print("FINAL RESULT")
    print("=" * 40)
    print(board.render())
    
    if winner == config.ROBOT:
        print("\n🤖 ROBOT WINS!")
    elif winner == config.PLAYER:
        print("\n👤 PLAYER WINS!")
    else:
        print("\n🤝 IT'S A DRAW!")
    
    robot.disconnect()


def demo_live_camera():
    """Demonstrate live camera processing (requires webcam)."""
    print("\n" + "=" * 60)
    print("DEMO: Live Camera Processing")
    print("=" * 60)
    
    print("\nAttempting to access camera...")
    
    camera = CameraInput()
    if not camera.initialize():
        print("Camera not available. Skipping live demo.")
        return
    
    board_detector = BoardDetector()
    feature_extractor = FeatureExtractor()
    classifier = KMeansClassifier()
    board_state = BoardState()
    
    print("\nCamera active. Controls:")
    print("  'c' - Capture and classify board")
    print("  'q' - Quit")
    
    while True:
        ret, frame = camera.get_frame()
        if not ret:
            break
        
        # Try to detect board
        success, corners = board_detector.detect_board(frame)
        
        display = frame.copy()
        if success:
            display = board_detector.draw_board_overlay(display, corners)
            cv2.putText(display, "Board Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display, "Searching for board...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Live Camera", display)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c') and success:
            # Capture and classify
            warped = board_detector.get_warped_board(frame, corners)
            cells = board_detector.get_cells(warped)
            features = feature_extractor.extract_all_cells(cells)
            labels, _ = classifier.fit_and_classify(features)
            
            board_state.update_from_labels(labels)
            print("\nCaptured Board State:")
            print(board_state.render())
            
            # Show warped board
            cv2.imshow("Captured Board", 
                      board_detector.draw_grid_overlay(warped))
    
    camera.release()


def run_all_demos():
    """Run all demonstration functions."""
    print("\n" + "=" * 60)
    print("  TICTACTOE ROBOT - DEMONSTRATION SUITE")
    print("=" * 60)
    
    demos = [
        ("Board Detection", demo_board_detection),
        ("K-Means Classification", demo_kmeans_classification),
        ("Game AI (Minimax)", demo_game_ai),
        ("Robot Simulation", demo_robot_simulation),
        ("Full Game Simulation", demo_full_game),
    ]
    
    print("\nAvailable demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print(f"  {len(demos) + 1}. Live Camera (requires webcam)")
    print(f"  0. Run all demos")
    print(f"  q. Quit")
    
    while True:
        choice = input("\nSelect demo (0-6, q): ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == '0':
            for name, func in demos:
                func()
                input("\nPress Enter to continue to next demo...")
            print("\n[INFO] All demos completed!")
        elif choice == str(len(demos) + 1):
            demo_live_camera()
        elif choice.isdigit() and 1 <= int(choice) <= len(demos):
            idx = int(choice) - 1
            demos[idx][1]()
        else:
            print("Invalid choice. Please try again.")
    
    print("\n[INFO] Demo session ended. Goodbye!")


def quick_test():
    """Quick test to verify all modules work."""
    print("\n" + "=" * 60)
    print("QUICK TEST: Verifying all modules...")
    print("=" * 60)
    
    try:
        print("✓ config")
        from modules.camera_input import CameraInput
        print("✓ camera_input")
        from modules.image_preprocessing import ImagePreprocessor
        print("✓ image_preprocessing")
        from modules.board_detection import BoardDetector
        print("✓ board_detection")
        from modules.feature_extraction import FeatureExtractor
        print("✓ feature_extraction")
        from modules.kmeans_classifier import KMeansClassifier
        print("✓ kmeans_classifier")
        from modules.board_state import BoardState
        print("✓ board_state")
        from modules.game_ai import GameAI
        print("✓ game_ai")
        from modules.coordinate_mapping import CoordinateMapper
        print("✓ coordinate_mapping")
        from modules.robot_control import RobotController
        print("✓ robot_control")
        
        print("\n[SUCCESS] All modules imported successfully!")
        
    except ImportError as e:
        print(f"\n[ERROR] Import failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "test":
            quick_test()
        elif arg == "board":
            demo_board_detection()
        elif arg == "kmeans":
            demo_kmeans_classification()
        elif arg == "ai":
            demo_game_ai()
        elif arg == "robot":
            demo_robot_simulation()
        elif arg == "game":
            demo_full_game()
        elif arg == "camera":
            demo_live_camera()
        else:
            print(f"Unknown argument: {arg}")
            print("Usage: python demo.py [test|board|kmeans|ai|robot|game|camera]")
    else:
        if quick_test():
            run_all_demos()
