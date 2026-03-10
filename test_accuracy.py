"""
Test độ chính xác của K-means Model
Chạy: python test_accuracy.py

Các bước test:
1. Thu thập dữ liệu: chụp ảnh bàn cờ + ghi nhãn thực tế
2. Chạy K-means prediction
3. So sánh và tính accuracy
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime

# Thư mục lưu dữ liệu test
DATA_DIR = "test_data"
RESULTS_FILE = "accuracy_results.json"


def create_data_dir():
    """Tạo thư mục dữ liệu"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Đã tạo thư mục: {DATA_DIR}")


def order_points(pts):
    """Sắp xếp 4 điểm"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def detect_board(frame):
    """Detect bàn cờ"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Thử adaptive threshold
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if 5000 < area < 200000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect = float(w) / h
                if 0.7 < aspect < 1.3:
                    return order_points(approx.reshape(4, 2))
    
    return None


def extract_cells(frame, board_pts):
    """Trích xuất 9 ô"""
    if board_pts is None:
        return None, []
    
    pts = np.float32(board_pts)
    dst = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(frame, M, (300, 300))
    
    cells = []
    cell_size = 100
    padding = 15
    
    for row in range(3):
        for col in range(3):
            x1, y1 = col * cell_size + padding, row * cell_size + padding
            x2, y2 = (col + 1) * cell_size - padding, (row + 1) * cell_size - padding
            cells.append(warped[y1:y2, x1:x2])
    
    return warped, cells


def extract_features(cells, feature_type='hsv'):
    """Trích xuất features từ cells"""
    features = []
    
    for cell in cells:
        if feature_type == 'hsv':
            hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
            mean_h = np.mean(hsv[:, :, 0])
            mean_s = np.mean(hsv[:, :, 1])
            mean_v = np.mean(hsv[:, :, 2])
            features.append([mean_h, mean_s, mean_v])
        
        elif feature_type == 'rgb':
            mean_b = np.mean(cell[:, :, 0])
            mean_g = np.mean(cell[:, :, 1])
            mean_r = np.mean(cell[:, :, 2])
            features.append([mean_b, mean_g, mean_r])
        
        elif feature_type == 'combined':
            hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            
            features.append([
                np.mean(hsv[:, :, 0]),  # H
                np.mean(hsv[:, :, 1]),  # S
                np.mean(hsv[:, :, 2]),  # V
                np.mean(cell[:, :, 0]),  # B
                np.mean(cell[:, :, 1]),  # G
                np.mean(cell[:, :, 2]),  # R
                np.std(gray),            # Texture
            ])
    
    return np.array(features, dtype=np.float32)


def kmeans_predict(features, n_clusters=3):
    """K-means prediction"""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(features, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.flatten()
    
    # Map clusters to labels
    counts = [np.sum(labels == i) for i in range(n_clusters)]
    mean_v = [centers[i][2] if len(centers[i]) > 2 else centers[i][0] for i in range(n_clusters)]
    
    # Empty = highest count or highest brightness
    scores = [counts[i] * 0.3 + mean_v[i] * 0.7 for i in range(n_clusters)]
    empty_cluster = np.argmax(scores)
    
    # Convert to 0 (empty), 1, 2
    result = []
    mapping = {empty_cluster: 0}
    next_label = 1
    for i in range(n_clusters):
        if i != empty_cluster:
            mapping[i] = next_label
            next_label += 1
    
    for label in labels:
        result.append(mapping[label])
    
    return result, centers


def calculate_accuracy(predictions, ground_truth):
    """Tính accuracy"""
    if len(predictions) != len(ground_truth):
        return 0.0
    
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    accuracy = correct / len(predictions)
    
    return accuracy


def calculate_metrics(all_predictions, all_ground_truths):
    """Tính các metrics chi tiết"""
    total_cells = 0
    correct_cells = 0
    
    # Confusion matrix: [actual][predicted]
    confusion = [[0, 0, 0] for _ in range(3)]  # 0=empty, 1=X, 2=O
    
    for preds, truths in zip(all_predictions, all_ground_truths):
        for p, g in zip(preds, truths):
            total_cells += 1
            if p == g:
                correct_cells += 1
            confusion[g][p] += 1
    
    accuracy = correct_cells / total_cells if total_cells > 0 else 0
    
    # Per-class metrics
    class_metrics = {}
    labels = ['Empty', 'X', 'O']
    
    for i in range(3):
        tp = confusion[i][i]
        fp = sum(confusion[j][i] for j in range(3)) - tp
        fn = sum(confusion[i][j] for j in range(3)) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[labels[i]] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return {
        'accuracy': accuracy,
        'total_cells': total_cells,
        'correct_cells': correct_cells,
        'confusion_matrix': confusion,
        'class_metrics': class_metrics
    }


def collect_data_mode():
    """Chế độ thu thập dữ liệu training/test"""
    print("\n" + "=" * 60)
    print("   CHẾ ĐỘ THU THẬP DỮ LIỆU")
    print("=" * 60)
    print("\nHướng dẫn:")
    print("  1. Đặt bàn cờ với trạng thái cụ thể")
    print("  2. Nhấn phím số (0-8) để đánh dấu ô")
    print("  3. Mỗi ô nhấn: . → X → O → . (vòng tròn)")
    print("  4. Nhấn 's' để lưu sample")
    print("  5. Nhấn 'q' để thoát (hoặc Ctrl+C)")
    print()
    
    create_data_dir()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Không mở được camera!")
        return
    
    try:
        # Trạng thái bàn cờ hiện tại (0=empty, 1=X, 2=O)
        current_state = [0] * 9
        sample_count = len([f for f in os.listdir(DATA_DIR) if f.endswith('.jpg')])
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            display = frame.copy()
            board_pts = detect_board(frame)
            
            if board_pts is not None:
                # Vẽ board
                pts_int = board_pts.astype(np.int32)
                cv2.polylines(display, [pts_int], True, (0, 255, 0), 2)
                
                # Hiển thị warped với labels
                warped, cells = extract_cells(frame, board_pts)
                if warped is not None:
                    symbols = ['.', 'X', 'O']
                    colors = [(200, 200, 200), (0, 0, 255), (255, 0, 0)]
                    
                    for idx in range(9):
                        row, col = idx // 3, idx % 3
                        cx, cy = col * 100 + 50, row * 100 + 50
                        state = current_state[idx]
                        
                        # Vẽ số ô và trạng thái
                        cv2.putText(warped, f"{idx}", (col * 100 + 5, row * 100 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                        cv2.putText(warped, symbols[state], (cx - 15, cy + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, colors[state], 2)
                    
                    # Vẽ lưới
                    for i in range(1, 3):
                        cv2.line(warped, (i * 100, 0), (i * 100, 300), (0, 255, 0), 1)
                        cv2.line(warped, (0, i * 100), (300, i * 100), (0, 255, 0), 1)
                    
                    cv2.imshow('Label Board (press 0-8)', warped)
            
            # Hiển thị trạng thái
            state_text = f"State: {current_state}"
            cv2.putText(display, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, f"Samples: {sample_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display, "Press 0-8 to label, 's' to save, 'q' to quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('Camera', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            # Phím số 0-8 để đánh dấu ô
            elif ord('0') <= key <= ord('8'):
                idx = key - ord('0')
                current_state[idx] = (current_state[idx] + 1) % 3
                symbols = ['.', 'X', 'O']
                print(f"Ô {idx}: {symbols[current_state[idx]]}")
            
            # Lưu sample
            elif key == ord('s') and board_pts is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_file = os.path.join(DATA_DIR, f"sample_{timestamp}.jpg")
                label_file = os.path.join(DATA_DIR, f"sample_{timestamp}.json")
                
                cv2.imwrite(img_file, frame)
                with open(label_file, 'w') as f:
                    json.dump({'state': current_state.copy(), 'board_pts': board_pts.tolist()}, f)
                
                sample_count += 1
                print(f"✓ Saved: {img_file}")
                print(f"  State: {current_state}")
            
            # Reset state
            elif key == ord('r'):
                current_state = [0] * 9
                print("Reset state")
    
    except KeyboardInterrupt:
        print("\n\nĐã dừng bởi Ctrl+C")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Đã đóng camera.")


def test_accuracy_mode():
    """Chế độ test accuracy trên dữ liệu đã thu thập"""
    print("\n" + "=" * 60)
    print("   TEST ĐỘ CHÍNH XÁC K-MEANS")
    print("=" * 60)
    
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Không tìm thấy thư mục {DATA_DIR}")
        print("Hãy chạy chế độ thu thập dữ liệu trước!")
        return
    
    # Load tất cả samples
    samples = []
    for f in os.listdir(DATA_DIR):
        if f.endswith('.json'):
            json_path = os.path.join(DATA_DIR, f)
            img_path = json_path.replace('.json', '.jpg')
            
            if os.path.exists(img_path):
                with open(json_path, 'r') as jf:
                    data = json.load(jf)
                samples.append({
                    'image': img_path,
                    'state': data['state'],
                    'board_pts': np.array(data['board_pts'], dtype=np.float32)
                })
    
    if not samples:
        print("Không có samples nào!")
        return
    
    print(f"\nĐã load {len(samples)} samples")
    
    # Test với các feature types
    feature_types = ['hsv', 'rgb', 'combined']
    all_results = {}
    
    for feat_type in feature_types:
        print(f"\n--- Testing feature type: {feat_type} ---")
        
        all_preds = []
        all_truths = []
        
        for sample in samples:
            frame = cv2.imread(sample['image'])
            board_pts = sample['board_pts']
            ground_truth = sample['state']
            
            # Extract cells
            warped, cells = extract_cells(frame, board_pts)
            if not cells:
                continue
            
            # Extract features & predict
            features = extract_features(cells, feat_type)
            predictions, _ = kmeans_predict(features)
            
            all_preds.append(predictions)
            all_truths.append(ground_truth)
        
        # Calculate metrics
        metrics = calculate_metrics(all_preds, all_truths)
        all_results[feat_type] = metrics
        
        print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  Correct: {metrics['correct_cells']}/{metrics['total_cells']}")
    
    # Hiển thị kết quả chi tiết
    print("\n" + "=" * 60)
    print("   KẾT QUẢ CHI TIẾT")
    print("=" * 60)
    
    best_type = max(all_results.keys(), key=lambda k: all_results[k]['accuracy'])
    
    for feat_type, metrics in all_results.items():
        marker = " ★ BEST" if feat_type == best_type else ""
        print(f"\n[{feat_type.upper()}]{marker}")
        print(f"  Overall Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  Confusion Matrix:")
        print(f"              Predicted")
        print(f"              Empty   X      O")
        labels = ['Empty', 'X', 'O']
        for i, row in enumerate(metrics['confusion_matrix']):
            print(f"    {labels[i]:6} [{row[0]:4d}  {row[1]:4d}  {row[2]:4d}]")
        
        print(f"\n  Per-class metrics:")
        for cls, m in metrics['class_metrics'].items():
            print(f"    {cls}: Precision={m['precision']:.2f}, Recall={m['recall']:.2f}, F1={m['f1']:.2f}")
    
    # Lưu kết quả
    with open(RESULTS_FILE, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(samples),
            'results': {k: {
                'accuracy': v['accuracy'],
                'total_cells': v['total_cells'],
                'correct_cells': v['correct_cells']
            } for k, v in all_results.items()}
        }, f, indent=2)
    
    print(f"\n✓ Đã lưu kết quả vào: {RESULTS_FILE}")
    
    return all_results


def realtime_test_mode():
    """Test realtime với camera"""
    print("\n" + "=" * 60)
    print("   TEST REALTIME")
    print("=" * 60)
    print("\nNhập trạng thái thực tế để so sánh")
    print("Nhấn 'q' để thoát, 't' để nhập ground truth")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Không mở được camera!")
        return
    
    ground_truth = None
    total_tests = 0
    correct_tests = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display = frame.copy()
        board_pts = detect_board(frame)
        
        prediction = None
        
        if board_pts is not None:
            pts_int = board_pts.astype(np.int32)
            cv2.polylines(display, [pts_int], True, (0, 255, 0), 2)
            
            warped, cells = extract_cells(frame, board_pts)
            if cells:
                features = extract_features(cells, 'hsv')
                prediction, _ = kmeans_predict(features)
                
                # Hiển thị prediction
                symbols = ['.', 'X', 'O']
                colors = [(200, 200, 200), (0, 0, 255), (255, 0, 0)]
                
                for idx in range(9):
                    row, col = idx // 3, idx % 3
                    cx, cy = col * 100 + 50, row * 100 + 50
                    cv2.putText(warped, symbols[prediction[idx]], (cx - 15, cy + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, colors[prediction[idx]], 2)
                
                cv2.imshow('Prediction', warped)
        
        # Hiển thị stats
        if total_tests > 0:
            acc = correct_tests / total_tests * 100
            cv2.putText(display, f"Accuracy: {acc:.1f}% ({correct_tests}/{total_tests})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if ground_truth:
            cv2.putText(display, f"Ground Truth: {ground_truth}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        cv2.putText(display, "Press 't' to enter ground truth", (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow('Camera', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('t') and prediction is not None:
            # Nhập ground truth
            print("\nNhập trạng thái thực tế (9 số, 0=empty, 1=X, 2=O)")
            print("Ví dụ: 101020100")
            gt_str = input("Ground truth: ").strip()
            
            if len(gt_str) == 9 and all(c in '012' for c in gt_str):
                ground_truth = [int(c) for c in gt_str]
                
                # So sánh
                accuracy = calculate_accuracy(prediction, ground_truth)
                total_tests += 1
                if accuracy == 1.0:
                    correct_tests += 1
                
                print(f"Prediction:   {prediction}")
                print(f"Ground Truth: {ground_truth}")
                print(f"Sample Accuracy: {accuracy*100:.0f}%")
                print(f"Overall: {correct_tests}/{total_tests}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if total_tests > 0:
        print(f"\n=== KẾT QUẢ CUỐI CÙNG ===")
        print(f"Total tests: {total_tests}")
        print(f"Correct: {correct_tests}")
        print(f"Accuracy: {correct_tests/total_tests*100:.1f}%")


def main():
    print("=" * 60)
    print("   TEST ĐỘ CHÍNH XÁC K-MEANS MODEL")
    print("   TicTacToe Robot")
    print("=" * 60)
    
    while True:
        print("\n--- MENU ---")
        print("  1. Thu thập dữ liệu (label samples)")
        print("  2. Test accuracy trên dữ liệu đã thu thập")
        print("  3. Test realtime với camera")
        print("  q. Thoát")
        
        choice = input("\nChọn: ").strip()
        
        if choice == '1':
            collect_data_mode()
        elif choice == '2':
            test_accuracy_mode()
        elif choice == '3':
            realtime_test_mode()
        elif choice.lower() == 'q':
            break
        else:
            print("Lựa chọn không hợp lệ!")
    
    print("\nĐã thoát.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nĐã thoát bởi Ctrl+C")
        cv2.destroyAllWindows()
