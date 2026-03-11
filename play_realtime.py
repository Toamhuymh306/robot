"""
TicTacToe Realtime - Camera + K-Means + Minimax AI
Chơi cờ với AI bằng bàn cờ thật + camera!

Chạy: python play_realtime.py

Cách chơi:
  1. Đặt bàn cờ trước camera (nền tối, đường kẻ đậm)
  2. Chờ camera nhận diện (khung xanh bao quanh bàn cờ)
  3. Nhấn 'c' để calibrate (xác nhận bàn cờ trống)
  4. Đặt quân X lên bàn cờ → nhấn 'n' để xác nhận nước đi
  5. AI sẽ tính nước đi tối ưu và hiển thị trên màn hình
  6. Bạn đặt quân O lên vị trí AI chỉ → nhấn 'n' tiếp
  7. Lặp lại đến khi có người thắng hoặc hòa

Phím tắt:
  'c' - Calibrate (bàn cờ trống)
  'n' - Xác nhận nước đi mới
  'r' - Reset game
  'q' - Thoát
"""

import cv2
import numpy as np
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from modules.camera_input import CameraInput
from modules.image_preprocessing import ImagePreprocessor
from modules.board_detection import BoardDetector
from modules.feature_extraction import FeatureExtractor
from modules.kmeans_classifier import KMeansClassifier
from modules.board_state import BoardState
from modules.game_ai import GameAI


class RealtimeGame:
    """Chơi cờ realtime qua camera."""
    
    def __init__(self):
        # Modules
        self.camera = CameraInput()
        self.preprocessor = ImagePreprocessor()
        self.detector = BoardDetector()
        self.extractor = FeatureExtractor(feature_type=config.FEATURE_TYPE)
        self.classifier = KMeansClassifier()
        self.board = BoardState()
        self.ai = GameAI()
        
        # Game state
        self.calibrated = False
        self.calibration_features = None
        self.game_active = False
        self.player_turn = True
        self.ai_move = None  # Nước đi AI gợi ý
        self.move_count = 0
        self.game_message = ""
        self.message_time = 0
        
        # Tracking
        self.last_board_corners = None
        self.stable_count = 0
        
    def initialize(self):
        """Khởi tạo camera."""
        print("=" * 50)
        print("  TICTACTOE REALTIME")
        print("  Camera + K-Means + Minimax AI")
        print("=" * 50)
        
        if not self.camera.initialize():
            print("ERROR: Không mở được camera!")
            print("Kiểm tra camera có kết nối không.")
            return False
        
        print("\n✅ Camera đã sẵn sàng!")
        print("\nHướng dẫn:")
        print("  1. Đặt bàn cờ trước camera")
        print("  2. Chờ khung XANH bao quanh bàn cờ")
        print("  3. Nhấn 'c' để calibrate bàn cờ trống")
        print("  4. Đặt quân lên bàn → nhấn 'n'")
        print("  5. AI sẽ hiện ô cần đánh bằng ô ĐỎ")
        print("  6. Bạn đặt quân AI lên ô đó → nhấn 'n'")
        return True
    
    def detect_and_classify(self, frame):
        """Pipeline: Detect board → Extract cells → K-Means classify."""
        # Detect board
        success, corners = self.detector.detect_board(frame)
        if not success:
            return None, None, None, None
        
        self.last_board_corners = corners
        
        # Warp board
        warped = self.detector.get_warped_board(frame, corners, 300)
        if warped is None:
            return None, None, None, None
        
        # Extract cells
        cells = self.detector.get_cells(warped)
        if len(cells) != 9:
            return None, None, None, None
        
        # Extract features
        features = self.extractor.extract_all_cells(cells)
        
        # K-Means classify
        labels, info = self.classifier.fit_and_classify(features)
        
        return warped, cells, labels, info
    
    def calibrate(self, frame):
        """Calibrate: chụp bàn cờ trống để làm baseline."""
        warped, cells, labels, info = self.detect_and_classify(frame)
        
        if warped is None:
            self.set_message("❌ Không thấy bàn cờ! Đặt lại.", (0, 0, 255))
            return False
        
        # Lưu features của bàn cờ trống
        self.calibration_features = self.extractor.extract_all_cells(cells)
        self.calibrated = True
        self.game_active = True
        self.board.reset()
        self.player_turn = not config.AI_FIRST
        self.ai_move = None
        self.move_count = 0
        
        self.set_message("✅ Calibrate OK! Đặt quân X rồi nhấn 'n'", (0, 255, 0))
        print("\n✅ Calibrate thành công!")
        print("   Bàn cờ trống đã ghi nhận.")
        print("   Đặt quân X lên bàn → nhấn 'n'")
        
        # Nếu AI đi trước
        if config.AI_FIRST:
            self.player_turn = False
            move = self.ai.get_best_move(self.board)
            if move:
                self.ai_move = move
                row, col = move
                self.set_message(f"🤖 AI chọn ô ({row},{col})! Đặt quân O lên đó → 'n'", (255, 100, 0))
                print(f"\n🤖 AI đi trước! Đặt quân O vào ô ({row},{col})")
        
        return True
    
    def process_move(self, frame):
        """Xử lý khi người chơi nhấn 'n' (xác nhận nước đi)."""
        if not self.game_active:
            self.set_message("Chưa calibrate! Nhấn 'c' trước.", (0, 0, 255))
            return
        
        warped, cells, labels, info = self.detect_and_classify(frame)
        
        if warped is None:
            self.set_message("❌ Không thấy bàn cờ!", (0, 0, 255))
            return
        
        # Chuyển labels thành ma trận 3x3
        new_matrix = np.array(labels).reshape(3, 3)
        
        # Phát hiện ô thay đổi so với trạng thái trước
        old_matrix = self.board.matrix.copy()
        changed_cells = []
        
        for r in range(3):
            for c in range(3):
                if old_matrix[r, c] == config.EMPTY and new_matrix[r, c] != config.EMPTY:
                    changed_cells.append((r, c))
        
        if len(changed_cells) == 0:
            # Nếu đang chờ AI move được đặt
            if self.ai_move is not None and not self.player_turn:
                # Người chơi đã đặt quân AI lên bàn
                r, c = self.ai_move
                self.board.set_cell(r, c, config.ROBOT)
                self.move_count += 1
                self.ai_move = None
                self.player_turn = True
                
                # Kiểm tra game over
                is_over, winner = self.board.is_game_over()
                if is_over:
                    self.end_game(winner)
                    return
                
                self.set_message("👤 Lượt bạn! Đặt quân X → 'n'", (0, 255, 0))
                print(f"\n👤 Lượt bạn! Đặt quân X lên bàn rồi nhấn 'n'")
            else:
                self.set_message("Không thấy nước đi mới. Đặt quân rồi thử lại.", (0, 165, 255))
            return
        
        if self.player_turn:
            # Lượt người chơi
            if len(changed_cells) >= 1:
                r, c = changed_cells[0]  # Lấy ô thay đổi đầu tiên
                self.board.set_cell(r, c, config.PLAYER)
                self.move_count += 1
                
                print(f"\n👤 Bạn đánh ô ({r},{c})")
                print(self.board.render())
                
                # Kiểm tra game over
                is_over, winner = self.board.is_game_over()
                if is_over:
                    self.end_game(winner)
                    return
                
                # AI tính nước đi
                self.player_turn = False
                ai_move = self.ai.get_best_move(self.board)
                
                if ai_move:
                    self.ai_move = ai_move
                    r, c = ai_move
                    idx = r * 3 + c + 1
                    self.set_message(f"🤖 AI chọn ô {idx} ({r},{c})! Đặt quân O vào đó → 'n'", (255, 100, 0))
                    print(f"🤖 AI chọn ô {idx} ({r},{c}) — đặt quân O vào đó rồi nhấn 'n'")
                    print(f"   Nodes đánh giá: {self.ai.nodes_evaluated}")
        else:
            # Đang chờ người đặt quân AI
            r, c = self.ai_move
            self.board.set_cell(r, c, config.ROBOT)
            self.move_count += 1
            self.ai_move = None
            self.player_turn = True
            
            # Kiểm tra game over
            is_over, winner = self.board.is_game_over()
            if is_over:
                self.end_game(winner)
                return
            
            self.set_message("👤 Lượt bạn! Đặt quân X → 'n'", (0, 255, 0))
            print(f"\n👤 Lượt bạn! Đặt quân X lên bàn rồi nhấn 'n'")
    
    def end_game(self, winner):
        """Xử lý kết thúc game."""
        self.game_active = False
        
        if winner == config.ROBOT:
            self.set_message("🤖 AI THẮNG! Nhấn 'r' để chơi lại", (255, 0, 0))
            print("\n" + "=" * 40)
            print("  🤖 AI THẮNG!")
            print("=" * 40)
        elif winner == config.PLAYER:
            self.set_message("🎉 BẠN THẮNG! Nhấn 'r' để chơi lại", (0, 255, 0))
            print("\n" + "=" * 40)
            print("  🎉 BẠN THẮNG!")
            print("=" * 40)
        else:
            self.set_message("🤝 HÒA! Nhấn 'r' để chơi lại", (255, 255, 0))
            print("\n" + "=" * 40)
            print("  🤝 HÒA!")
            print("=" * 40)
        
        print(self.board.render())
        print("Nhấn 'r' để chơi lại, 'q' để thoát")
    
    def set_message(self, text, color=(255, 255, 255)):
        """Set thông báo hiển thị trên màn hình."""
        self.game_message = text
        self.message_color = color
        self.message_time = time.time()
    
    def draw_ui(self, frame, warped=None):
        """Vẽ giao diện lên frame."""
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Vẽ khung bàn cờ
        if self.last_board_corners is not None:
            pts = self.last_board_corners.astype(np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 0), 3)
        
        # Panel thông tin bên trái trên
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 45), (0, 0, 0), -1)
        display = cv2.addWeighted(overlay, 0.6, display, 0.4, 0)
        
        # Status text
        if not self.calibrated:
            status = "Chua calibrate - Nhan 'c'"
            status_color = (0, 165, 255)
        elif not self.game_active:
            status = "Game Over! Nhan 'r' de choi lai"
            status_color = (0, 255, 255)
        elif self.player_turn:
            status = f"Luot ban (X) | Nuoc di #{self.move_count + 1}"
            status_color = (0, 255, 0)
        else:
            status = f"AI da chon! Dat quan O len ban → 'n'"
            status_color = (255, 100, 0)
        
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, status_color, 2)
        
        # Message bar phía dưới
        if self.game_message and (time.time() - self.message_time < 10):
            overlay2 = display.copy()
            cv2.rectangle(overlay2, (0, h - 50), (w, h), (0, 0, 0), -1)
            display = cv2.addWeighted(overlay2, 0.6, display, 0.4, 0)
            cv2.putText(display, self.game_message, (10, h - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.message_color, 2)
        
        # Vẽ ma trận bàn cờ (góc phải trên)
        self.draw_board_state(display, w - 160, 10)
        
        # Vẽ AI move suggestion lên bàn cờ
        if self.ai_move is not None and self.last_board_corners is not None:
            self.draw_ai_move(display)
        
        return display
    
    def draw_board_state(self, frame, x_start, y_start):
        """Vẽ ma trận trạng thái bàn cờ nhỏ góc phải."""
        cell_size = 45
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start - 5, y_start - 5), 
                     (x_start + cell_size * 3 + 10, y_start + cell_size * 3 + 30),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "Board State", (x_start, y_start + cell_size * 3 + 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        for r in range(3):
            for c in range(3):
                x = x_start + c * cell_size
                y = y_start + r * cell_size
                
                # Grid
                cv2.rectangle(frame, (x, y), (x + cell_size, y + cell_size), 
                            (100, 100, 100), 1)
                
                val = self.board.matrix[r, c]
                cx, cy = x + cell_size // 2, y + cell_size // 2
                
                if val == config.PLAYER:
                    # X — xanh lá
                    d = 14
                    cv2.line(frame, (cx-d, cy-d), (cx+d, cy+d), (0, 255, 0), 3)
                    cv2.line(frame, (cx+d, cy-d), (cx-d, cy+d), (0, 255, 0), 3)
                elif val == config.ROBOT:
                    # O — xanh dương
                    cv2.circle(frame, (cx, cy), 15, (255, 100, 0), 3)
                else:
                    # Số ô
                    idx = r * 3 + c + 1
                    cv2.putText(frame, str(idx), (cx - 6, cy + 6),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
                
                # Highlight AI move
                if self.ai_move == (r, c):
                    cv2.rectangle(frame, (x+2, y+2), (x + cell_size-2, y + cell_size-2),
                                (0, 0, 255), 3)
                    cv2.circle(frame, (cx, cy), 15, (0, 0, 255), 3)
    
    def draw_ai_move(self, frame):
        """Vẽ gợi ý nước đi AI trực tiếp lên bàn cờ camera."""
        if self.ai_move is None or self.last_board_corners is None:
            return
        
        row, col = self.ai_move
        corners = self.last_board_corners.astype(np.float32)
        
        # Tính vị trí ô trên frame gốc bằng inverse perspective
        # Tạo điểm trung tâm ô trên bàn cờ warped (300x300)
        cx_warped = col * 100 + 50
        cy_warped = row * 100 + 50
        
        # Tính ma trận perspective ngược
        dst = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
        M = cv2.getPerspectiveTransform(dst, corners)
        
        # Chuyển đổi tọa độ
        pt = np.float32([[[cx_warped, cy_warped]]])
        pt_transformed = cv2.perspectiveTransform(pt, M)
        cx_frame = int(pt_transformed[0][0][0])
        cy_frame = int(pt_transformed[0][0][1])
        
        # Vẽ hình tròn đỏ nhấp nháy (AI move)
        blink = int(time.time() * 3) % 2
        if blink:
            cv2.circle(frame, (cx_frame, cy_frame), 25, (0, 0, 255), 4)
            cv2.circle(frame, (cx_frame, cy_frame), 30, (0, 0, 255), 2)
        else:
            cv2.circle(frame, (cx_frame, cy_frame), 25, (0, 100, 255), 4)
        
        # Label
        idx = row * 3 + col + 1
        cv2.putText(frame, f"AI: {idx}", (cx_frame - 20, cy_frame - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    def run(self):
        """Main game loop."""
        if not self.initialize():
            return
        
        print("\n🎮 Camera đang chạy... Đặt bàn cờ trước camera.")
        
        while True:
            ret, frame = self.camera.get_frame()
            if not ret:
                print("ERROR: Không đọc được frame từ camera!")
                break
            
            # Detect board mỗi frame
            warped = None
            success, corners = self.detector.detect_board(frame)
            if success:
                self.last_board_corners = corners
                warped = self.detector.get_warped_board(frame, corners, 300)
            
            # Vẽ UI
            display = self.draw_ui(frame, warped)
            
            # Hiển thị
            cv2.imshow("TicTacToe Realtime - Camera + K-Means + AI", display)
            
            # Hiển thị bàn cờ warped (nếu có)
            if warped is not None:
                warped_display = self.detector.draw_grid_overlay(warped.copy())
                
                # Vẽ trạng thái lên warped
                for r in range(3):
                    for c in range(3):
                        cx, cy = c * 100 + 50, r * 100 + 50
                        val = self.board.matrix[r, c]
                        if val == config.PLAYER:
                            cv2.putText(warped_display, "X", (cx-15, cy+15),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                        elif val == config.ROBOT:
                            cv2.putText(warped_display, "O", (cx-15, cy+15),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 100, 0), 3)
                        
                        # AI move highlight
                        if self.ai_move == (r, c):
                            cv2.rectangle(warped_display, (c*100+5, r*100+5),
                                        ((c+1)*100-5, (r+1)*100-5), (0, 0, 255), 3)
                
                cv2.imshow("Board View", warped_display)
            
            # Xử lý phím
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n👋 Đã thoát!")
                break
            
            elif key == ord('c'):
                # Calibrate
                print("\n📐 Đang calibrate...")
                self.calibrate(frame)
            
            elif key == ord('n'):
                # Xác nhận nước đi
                self.process_move(frame)
            
            elif key == ord('r'):
                # Reset game
                self.board.reset()
                self.ai_move = None
                self.game_active = True
                self.player_turn = not config.AI_FIRST
                self.move_count = 0
                self.set_message("🔄 Reset! Calibrate lai - nhan 'c'", (0, 255, 255))
                self.calibrated = False
                print("\n🔄 Game đã reset! Nhấn 'c' để calibrate lại.")
        
        # Cleanup
        self.camera.release()
        cv2.destroyAllWindows()
        print("Đã đóng camera.")


def main():
    print("╔══════════════════════════════════════════════╗")
    print("║  🤖 TICTACTOE REALTIME                      ║")
    print("║  Camera + K-Means + Minimax AI               ║")
    print("║                                              ║")
    print("║  Chơi cờ với AI bằng bàn cờ thật!            ║")
    print("╚══════════════════════════════════════════════╝")
    print()
    print("  Cần: Webcam + Bàn cờ 3x3 + Quân cờ 2 màu")
    print("  Không cần: Robot arm")
    print()
    
    game = RealtimeGame()
    game.run()


if __name__ == "__main__":
    main()
