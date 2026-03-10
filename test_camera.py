"""
Camera Test Script - Test với camera laptop trước
Chạy: python test_camera.py
"""

import cv2
import numpy as np
import sys

def test_camera_basic():
    """Test camera cơ bản - chỉ hiển thị hình ảnh"""
    print("=" * 50)
    print("TEST CAMERA CƠ BẢN")
    print("Nhấn 'q' để thoát, 's' để chụp ảnh")
    print("=" * 50)
    
    # Mở camera laptop (index 0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERRO: Không thể mở camera!")
        print("Hãy thử:")
        print("  - Kiểm tra camera có được kết nối")
        print("  - Thử đổi index: cv2.VideoCapture(1)")
        return False
    
    # Cấu hình camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print(f"Camera opened: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Không đọc được frame!")
                break
            
            frame_count += 1
            
            # Vẽ thông tin lên frame
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Camera Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'captured_frame_{frame_count}.jpg'
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
    except KeyboardInterrupt:
        print("\nĐã dừng.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return True


def test_board_detection():
    """Test nhận dạng bàn cờ từ camera"""
    print("=" * 50)
    print("TEST NHẬN DẠNG BÀN CỜ")
    print("Đặt bàn cờ Tic-Tac-Toe trước camera")
    print("Nhấn 'q' để thoát")
    print("=" * 50)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Không thể mở camera!")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Copy để vẽ
        display = frame.copy()
        
        # Chuyển sang grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Blur và Canny
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Tìm contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        board_found = False
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Lọc theo diện tích
            if 10000 < area < 200000:
                # Xấp xỉ polygon
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # Nếu là hình vuông/chữ nhật (4 đỉnh)
                if len(approx) == 4:
                    # Kiểm tra tỷ lệ
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect = float(w) / h
                    
                    if 0.8 < aspect < 1.2:
                        cv2.drawContours(display, [approx], -1, (0, 255, 0), 3)
                        cv2.putText(display, "BOARD DETECTED!", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        board_found = True
                        
                        # Vẽ lưới 3x3
                        pts = approx.reshape(4, 2)
                        pts = order_points(pts)
                        
                        # Chia thành 9 ô
                        draw_grid(display, pts)
        
        if not board_found:
            cv2.putText(display, "Looking for board...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Hiển thị
        cv2.imshow('Board Detection', display)
        cv2.imshow('Edges', edges)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return True


def order_points(pts):
    """Sắp xếp 4 điểm theo thứ tự: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    
    return rect


def draw_grid(image, pts):
    """Vẽ lưới 3x3 trên bàn cờ"""
    pts = pts.astype(np.int32)
    
    # Tính điểm trên các cạnh
    for i in range(1, 3):
        # Cạnh trên
        t1 = pts[0] + (pts[1] - pts[0]) * i // 3
        # Cạnh dưới
        t2 = pts[3] + (pts[2] - pts[3]) * i // 3
        cv2.line(image, tuple(t1), tuple(t2), (255, 0, 0), 2)
        
        # Cạnh trái
        l1 = pts[0] + (pts[3] - pts[0]) * i // 3
        # Cạnh phải
        l2 = pts[1] + (pts[2] - pts[1]) * i // 3
        cv2.line(image, tuple(l1), tuple(l2), (255, 0, 0), 2)


def test_kmeans_segmentation():
    """Test K-means clustering trên hình ảnh"""
    print("=" * 50)
    print("TEST K-MEANS SEGMENTATION")
    print("Nhấn 'c' để chạy K-means clustering")
    print("Nhấn 'q' để thoát")
    print("=" * 50)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Không thể mở camera!")
        return False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            display = frame.copy()
            cv2.putText(display, "Press 'c' for K-means", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Original', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Chạy K-means clustering
                print("Running K-means clustering...")
                
                # Reshape thành danh sách pixel
                Z = frame.reshape((-1, 3))
                Z = np.float32(Z)
                
                # K-means với 3 clusters
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                K = 3
                _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                
                # Chuyển về uint8
                centers = np.uint8(centers)
                segmented = centers[labels.flatten()]
                segmented = segmented.reshape(frame.shape)
                
                # Hiển thị kết quả
                cv2.imshow('K-means Segmented', segmented)
                print(f"Cluster centers (BGR):")
                for i, center in enumerate(centers):
                    print(f"  Cluster {i}: {center}")
    except KeyboardInterrupt:
        print("\nĐã dừng.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return True


def test_full_pipeline():
    """Test full pipeline - nhận dạng bàn cờ và phân loại ô"""
    print("=" * 50)
    print("TEST FULL PIPELINE")
    print("Đặt bàn cờ có quân cờ trước camera")
    print("Nhấn 'a' để phân tích, 'q' để thoát")
    print("=" * 50)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Không thể mở camera!")
        return False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            display = frame.copy()
            
            # Tìm bàn cờ
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            board_pts = None
            for contour in contours:
                area = cv2.contourArea(contour)
                if 10000 < area < 200000:
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                    
                    if len(approx) == 4:
                        x, y, w, h = cv2.boundingRect(approx)
                        aspect = float(w) / h
                        if 0.8 < aspect < 1.2:
                            board_pts = order_points(approx.reshape(4, 2))
                            cv2.drawContours(display, [approx], -1, (0, 255, 0), 2)
                            break
            
            cv2.putText(display, "Press 'a' to analyze board", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('TicTacToe Detection', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('a') and board_pts is not None:
                # Phân tích bàn cờ
                analyze_board(frame, board_pts)
    except KeyboardInterrupt:
        print("\nĐã dừng.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return True


def analyze_board(frame, board_pts):
    """Phân tích bàn cờ và xác định trạng thái các ô"""
    # Perspective transform
    board_pts = np.float32(board_pts)
    dst_pts = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
    
    M = cv2.getPerspectiveTransform(board_pts, dst_pts)
    warped = cv2.warpPerspective(frame, M, (300, 300))
    
    print("\n=== PHÂN TÍCH BÀN CỜ ===")
    
    # Chia thành 9 ô
    cell_size = 100
    cells = []
    features = []
    
    for row in range(3):
        for col in range(3):
            x1, y1 = col * cell_size + 10, row * cell_size + 10
            x2, y2 = (col + 1) * cell_size - 10, (row + 1) * cell_size - 10
            cell = warped[y1:y2, x1:x2]
            cells.append(cell)
            
            # Trích xuất features (HSV mean)
            hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
            mean_hsv = np.mean(hsv, axis=(0, 1))
            features.append(mean_hsv)
    
    features = np.array(features)
    
    # K-means clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(features)
    
    # Hiển thị kết quả
    result_img = warped.copy()
    
    label_names = ['', '', '']
    # Xác định cluster nào là empty (thường có nhiều ô nhất)
    counts = [np.sum(labels == i) for i in range(3)]
    empty_cluster = np.argmax(counts)
    label_names[empty_cluster] = 'E'
    
    # Còn lại là X và O (dựa vào màu)
    for i in range(3):
        if i != empty_cluster:
            center = kmeans.cluster_centers_[i]
            if center[0] < 20 or center[0] > 160:  # Red hue
                label_names[i] = 'X'
            else:
                label_names[i] = 'O'
    
    # Vẽ kết quả
    board_state = []
    for idx, (row, col) in enumerate([(r, c) for r in range(3) for c in range(3)]):
        label = labels[idx]
        name = label_names[label]
        board_state.append(name)
        
        cx = col * cell_size + cell_size // 2
        cy = row * cell_size + cell_size // 2
        
        color = (200, 200, 200) if name == 'E' else (0, 0, 255) if name == 'X' else (255, 0, 0)
        cv2.putText(result_img, name if name != 'E' else '.', (cx - 20, cy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    
    # Hiển thị bàn cờ text
    print("\nTrạng thái bàn cờ:")
    print(f" {board_state[0]} | {board_state[1]} | {board_state[2]} ")
    print("-----------")
    print(f" {board_state[3]} | {board_state[4]} | {board_state[5]} ")
    print("-----------")
    print(f" {board_state[6]} | {board_state[7]} | {board_state[8]} ")
    
    cv2.imshow('Warped Board', warped)
    cv2.imshow('Analysis Result', result_img)
    cv2.waitKey(0)


def main():
    print("=" * 60)
    print("     CAMERA TEST - TicTacToe Robot K-Means")
    print("=" * 60)
    print("\nChọn test:")
    print("  1. Test camera cơ bản")
    print("  2. Test nhận dạng bàn cờ")
    print("  3. Test K-means segmentation")
    print("  4. Test full pipeline")
    print("  5. Chạy tất cả")
    print("  q. Thoát")
    
    while True:
        choice = input("\nNhập lựa chọn: ").strip()
        
        if choice == '1':
            test_camera_basic()
        elif choice == '2':
            test_board_detection()
        elif choice == '3':
            test_kmeans_segmentation()
        elif choice == '4':
            test_full_pipeline()
        elif choice == '5':
            test_camera_basic()
            test_board_detection()
            test_kmeans_segmentation()
            test_full_pipeline()
        elif choice.lower() == 'q':
            break
        else:
            print("Lựa chọn không hợp lệ!")
    
    print("\nĐã thoát.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nĐã thoát.")
        cv2.destroyAllWindows()
