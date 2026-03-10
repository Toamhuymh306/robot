"""
Test nhận diện bàn cờ giấy đơn giản
Hỗ trợ: tờ giấy trắng hình vuông với lưới 3x3

Chạy: python test_paper_board.py
"""

import cv2
import numpy as np

# Trackbar callback
def nothing(x):
    pass

def create_trackbars():
    """Tạo cửa sổ điều chỉnh tham số"""
    cv2.namedWindow('Settings', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Settings', 400, 300)
    
    # Canny edge detection
    cv2.createTrackbar('Canny Low', 'Settings', 30, 255, nothing)
    cv2.createTrackbar('Canny High', 'Settings', 100, 255, nothing)
    
    # Board detection
    cv2.createTrackbar('Min Area', 'Settings', 50, 500, nothing)  # x100
    cv2.createTrackbar('Max Area', 'Settings', 300, 1000, nothing)  # x100
    
    # Blur
    cv2.createTrackbar('Blur', 'Settings', 2, 10, nothing)  # x2+1
    
    # Adaptive threshold (thay vì Canny)
    cv2.createTrackbar('Use Adaptive', 'Settings', 0, 1, nothing)
    cv2.createTrackbar('Block Size', 'Settings', 5, 25, nothing)  # x2+3


def get_trackbar_values():
    """Lấy giá trị từ trackbars"""
    canny_low = cv2.getTrackbarPos('Canny Low', 'Settings')
    canny_high = cv2.getTrackbarPos('Canny High', 'Settings')
    min_area = cv2.getTrackbarPos('Min Area', 'Settings') * 100
    max_area = cv2.getTrackbarPos('Max Area', 'Settings') * 100
    blur = cv2.getTrackbarPos('Blur', 'Settings') * 2 + 1
    use_adaptive = cv2.getTrackbarPos('Use Adaptive', 'Settings')
    block_size = cv2.getTrackbarPos('Block Size', 'Settings') * 2 + 3
    
    return canny_low, canny_high, min_area, max_area, blur, use_adaptive, block_size


def order_points(pts):
    """Sắp xếp 4 điểm: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def detect_board(frame, canny_low, canny_high, min_area, max_area, blur, use_adaptive, block_size):
    """Detect bàn cờ hình vuông"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Blur
    if blur > 1:
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    
    # Edge detection
    if use_adaptive:
        # Adaptive threshold - tốt cho giấy trắng
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, block_size, 2)
        edges = thresh
    else:
        # Canny edge
        edges = cv2.Canny(gray, canny_low, canny_high)
    
    # Dilate để nối các đường
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Tìm contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    board_contour = None
    board_pts = None
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if min_area < area < max_area:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:
                # Kiểm tra là hình vuông
                x, y, w, h = cv2.boundingRect(approx)
                aspect = float(w) / h if h > 0 else 0
                
                if 0.7 < aspect < 1.3:  # Cho phép sai số
                    board_contour = approx
                    board_pts = order_points(approx.reshape(4, 2))
                    break
    
    return edges, board_contour, board_pts


def extract_cells(frame, board_pts):
    """Trích xuất 9 ô từ bàn cờ"""
    if board_pts is None:
        return None, []
    
    # Perspective transform
    pts = np.float32(board_pts)
    dst = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
    
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(frame, M, (300, 300))
    
    # Chia thành 9 ô
    cells = []
    cell_size = 100
    padding = 15  # Bỏ viền
    
    for row in range(3):
        for col in range(3):
            x1 = col * cell_size + padding
            y1 = row * cell_size + padding
            x2 = (col + 1) * cell_size - padding
            y2 = (row + 1) * cell_size - padding
            
            cell = warped[y1:y2, x1:x2]
            cells.append(cell)
    
    return warped, cells


def classify_cells_kmeans(cells):
    """Phân loại ô bằng K-means"""
    if not cells:
        return [0] * 9
    
    # Trích xuất features
    features = []
    for cell in cells:
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
        mean_h = np.mean(hsv[:, :, 0])
        mean_s = np.mean(hsv[:, :, 1])
        mean_v = np.mean(hsv[:, :, 2])
        features.append([mean_h, mean_s, mean_v])
    
    features = np.array(features, dtype=np.float32)
    
    # K-means với OpenCV (không cần sklearn)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(features, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    labels = labels.flatten()
    
    # Xác định cluster nào là empty (nhiều nhất hoặc V cao nhất)
    counts = [np.sum(labels == i) for i in range(3)]
    mean_v = [centers[i][2] for i in range(3)]  # Value channel
    
    # Empty thường có V cao (trắng) và nhiều ô
    empty_scores = [counts[i] * 0.5 + mean_v[i] * 0.5 for i in range(3)]
    empty_cluster = np.argmax(empty_scores)
    
    # Map labels: 0=empty, 1=player1, 2=player2
    result = []
    for label in labels:
        if label == empty_cluster:
            result.append(0)  # Empty
        else:
            result.append(label + 1)  # Player 1 or 2
    
    return result


def main():
    print("=" * 60)
    print("   NHẬN DIỆN BÀN CỜ GIẤY - TicTacToe Robot")
    print("=" * 60)
    print("\nHướng dẫn:")
    print("  - Đặt tờ giấy vuông có kẻ lưới 3x3 trước camera")
    print("  - Điều chỉnh trackbars để detect tốt hơn")
    print("  - Nhấn 'a' để phân tích K-means")
    print("  - Nhấn 's' để lưu ảnh")
    print("  - Nhấn 'q' để thoát")
    print()
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Không mở được camera!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    create_trackbars()
    
    frame_count = 0
    last_board_state = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Lấy tham số
        canny_low, canny_high, min_area, max_area, blur, use_adaptive, block_size = get_trackbar_values()
        
        # Detect bàn cờ
        edges, board_contour, board_pts = detect_board(
            frame, canny_low, canny_high, min_area, max_area, blur, use_adaptive, block_size
        )
        
        # Vẽ kết quả
        display = frame.copy()
        
        if board_contour is not None:
            cv2.drawContours(display, [board_contour], -1, (0, 255, 0), 3)
            cv2.putText(display, "BOARD FOUND!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Vẽ 4 góc
            for i, pt in enumerate(board_pts):
                cv2.circle(display, tuple(pt.astype(int)), 8, (0, 0, 255), -1)
                cv2.putText(display, str(i), tuple(pt.astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Extract và hiển thị warped
            warped, cells = extract_cells(frame, board_pts)
            if warped is not None:
                # Vẽ lưới
                for i in range(1, 3):
                    cv2.line(warped, (i*100, 0), (i*100, 300), (0, 255, 0), 2)
                    cv2.line(warped, (0, i*100), (300, i*100), (0, 255, 0), 2)
                
                cv2.imshow('Warped Board', warped)
        else:
            cv2.putText(display, "Looking for board...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display, "Adjust trackbars if needed", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Hiển thị tham số
        params_text = f"Canny:{canny_low}-{canny_high} Area:{min_area}-{max_area}"
        cv2.putText(display, params_text, (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow('Camera', display)
        cv2.imshow('Edges', edges)
        
        # Xử lý phím
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('s'):
            cv2.imwrite(f'frame_{frame_count}.jpg', frame)
            print(f"Saved: frame_{frame_count}.jpg")
        
        elif key == ord('a') and board_pts is not None:
            # Phân tích K-means
            warped, cells = extract_cells(frame, board_pts)
            if cells:
                board_state = classify_cells_kmeans(cells)
                last_board_state = board_state
                
                print("\n=== KẾT QUẢ K-MEANS ===")
                symbols = ['.', 'X', 'O']
                print(f" {symbols[board_state[0]]} | {symbols[board_state[1]]} | {symbols[board_state[2]]}")
                print("-----------")
                print(f" {symbols[board_state[3]]} | {symbols[board_state[4]]} | {symbols[board_state[5]]}")
                print("-----------")
                print(f" {symbols[board_state[6]]} | {symbols[board_state[7]]} | {symbols[board_state[8]]}")
                
                # Hiển thị kết quả trên ảnh
                result_img = warped.copy()
                colors = [(200, 200, 200), (0, 0, 255), (255, 0, 0)]
                
                for idx in range(9):
                    row, col = idx // 3, idx % 3
                    cx = col * 100 + 50
                    cy = row * 100 + 50
                    state = board_state[idx]
                    cv2.putText(result_img, symbols[state], (cx - 15, cy + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[state], 3)
                
                cv2.imshow('K-means Result', result_img)
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
