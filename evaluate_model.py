"""
Đánh giá toàn diện mô hình TicTacToe Robot
Chạy: python evaluate_model.py

Đánh giá 3 thành phần chính:
  1. K-Means Clustering: Accuracy, Precision, Recall, F1-Score
  2. Game AI (Minimax): Tỷ lệ thắng/hòa/thua
  3. Pipeline tổng hợp: Tốc độ xử lý, độ ổn định

Không cần camera hay robot - tất cả chạy trên dữ liệu tổng hợp.
"""

import sys
import os
import time
import numpy as np
import cv2
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from modules.feature_extraction import FeatureExtractor
from modules.kmeans_classifier import KMeansClassifier
from modules.board_state import BoardState
from modules.game_ai import GameAI, SimpleAI
from modules.image_preprocessing import ImagePreprocessor
from modules.board_detection import BoardDetector
from modules.coordinate_mapping import CoordinateMapper
from modules.robot_control import RobotController


# ============================================================================
# PHẦN 1: ĐÁNH GIÁ K-MEANS CLUSTERING
# ============================================================================

def generate_synthetic_cells(n_empty=5, n_player=2, n_robot=2, noise_level=10):
    """
    Tạo ảnh ô cờ giả lập cho test.
    
    Args:
        n_empty: Số ô trống
        n_player: Số ô người chơi (xanh)
        n_robot: Số ô robot (đỏ)
        noise_level: Mức nhiễu (0-50)
    
    Returns:
        cells: danh sách ảnh ô
        labels: nhãn thực tế
    """
    cells = []
    labels = []
    
    for _ in range(n_empty):
        cell = np.full((80, 80, 3), (150, 150, 150), dtype=np.uint8)
        noise = np.random.randint(-noise_level, noise_level, cell.shape, dtype=np.int16)
        cell = np.clip(cell.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cells.append(cell)
        labels.append(config.EMPTY)
    
    for _ in range(n_player):
        cell = np.full((80, 80, 3), (50, 180, 50), dtype=np.uint8)
        cv2.line(cell, (10, 10), (70, 70), (0, 220, 0), 4)
        cv2.line(cell, (70, 10), (10, 70), (0, 220, 0), 4)
        noise = np.random.randint(-noise_level, noise_level, cell.shape, dtype=np.int16)
        cell = np.clip(cell.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cells.append(cell)
        labels.append(config.PLAYER)
    
    for _ in range(n_robot):
        cell = np.full((80, 80, 3), (50, 50, 200), dtype=np.uint8)
        cv2.circle(cell, (40, 40), 30, (0, 0, 240), 4)
        noise = np.random.randint(-noise_level, noise_level, cell.shape, dtype=np.int16)
        cell = np.clip(cell.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cells.append(cell)
        labels.append(config.ROBOT)
    
    return cells, labels


def evaluate_kmeans_accuracy(n_trials=50, verbose=True):
    """
    Đánh giá accuracy của K-Means trên nhiều bộ dữ liệu giả lập.
    
    Thay đổi: số lượng quân, mức nhiễu, loại feature.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("  ĐÁNH GIÁ 1: K-MEANS CLUSTERING ACCURACY")
        print("=" * 60)
    
    results = {}
    feature_types = ['hsv_mean', 'rgb_mean', 'combined', 'color_stats']
    
    # Các kịch bản game khác nhau
    scenarios = [
        {"name": "Đầu game (7T-1P-1R)", "empty": 7, "player": 1, "robot": 1},
        {"name": "Giữa game (5T-2P-2R)", "empty": 5, "player": 2, "robot": 2},
        {"name": "Cuối game (3T-3P-3R)", "empty": 3, "player": 3, "robot": 3},
        {"name": "Gần hết (1T-4P-4R)",   "empty": 1, "player": 4, "robot": 4},
    ]
    
    noise_levels = [5, 10, 20, 30]
    
    for feat_type in feature_types:
        results[feat_type] = {
            "overall_accuracy": 0,
            "scenario_results": {},
            "noise_results": {},
            "confusion_matrix": np.zeros((3, 3), dtype=int),
        }
        
        total_correct = 0
        total_samples = 0
        
        for scenario in scenarios:
            scenario_correct = 0
            scenario_total = 0
            
            for noise in noise_levels:
                trial_correct = 0
                trial_total = 0
                
                for _ in range(n_trials):
                    cells, true_labels = generate_synthetic_cells(
                        scenario["empty"], scenario["player"], scenario["robot"], noise
                    )
                    
                    extractor = FeatureExtractor(feature_type=feat_type)
                    classifier = KMeansClassifier()
                    
                    features = extractor.extract_all_cells(cells)
                    pred_labels, _ = classifier.fit_and_classify(features)
                    
                    for pred, true in zip(pred_labels, true_labels):
                        total_samples += 1
                        trial_total += 1
                        scenario_total += 1
                        
                        if pred == true:
                            total_correct += 1
                            trial_correct += 1
                            scenario_correct += 1
                        
                        # Confusion matrix (map: EMPTY=0, PLAYER→1, ROBOT→2)
                        true_idx = {config.EMPTY: 0, config.PLAYER: 1, config.ROBOT: 2}[true]
                        pred_idx = {config.EMPTY: 0, config.PLAYER: 1, config.ROBOT: 2}.get(pred, 0)
                        results[feat_type]["confusion_matrix"][true_idx][pred_idx] += 1
                
                noise_key = f"noise_{noise}"
                if noise_key not in results[feat_type]["noise_results"]:
                    results[feat_type]["noise_results"][noise_key] = {"correct": 0, "total": 0}
                results[feat_type]["noise_results"][noise_key]["correct"] += trial_correct
                results[feat_type]["noise_results"][noise_key]["total"] += trial_total
            
            results[feat_type]["scenario_results"][scenario["name"]] = {
                "accuracy": scenario_correct / scenario_total if scenario_total > 0 else 0,
                "correct": scenario_correct,
                "total": scenario_total,
            }
        
        results[feat_type]["overall_accuracy"] = total_correct / total_samples if total_samples > 0 else 0
    
    # === In kết quả ===
    if verbose:
        print("\n┌──────────────────────────────────────────────────────┐")
        print("│  TỔNG HỢP KẾT QUẢ K-MEANS                          │")
        print("├──────────────────────────────────────────────────────┤")
        
        best_type = max(results.keys(), key=lambda k: results[k]["overall_accuracy"])
        
        for feat_type, data in results.items():
            marker = " ⭐ TỐT NHẤT" if feat_type == best_type else ""
            acc = data["overall_accuracy"] * 100
            print(f"│  {feat_type:<15} : {acc:6.2f}%{marker:<20}│")
        
        print("├──────────────────────────────────────────────────────┤")
        
        # Chi tiết feature tốt nhất
        best_data = results[best_type]
        print(f"│  Feature tốt nhất: {best_type:<33}│")
        print("│                                                      │")
        print("│  Theo kịch bản:                                      │")
        for name, sr in best_data["scenario_results"].items():
            acc = sr["accuracy"] * 100
            bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
            print(f"│    {name:<22} {bar} {acc:.1f}%  │")
        
        print("│                                                      │")
        print("│  Theo mức nhiễu:                                     │")
        for noise_key, nr in best_data["noise_results"].items():
            noise_val = noise_key.split("_")[1]
            acc = nr["correct"] / nr["total"] * 100 if nr["total"] > 0 else 0
            bar = "█" * int(acc / 5) + "░" * (20 - int(acc / 5))
            print(f"│    Noise={noise_val:<3}              {bar} {acc:.1f}%  │")
        
        # Confusion Matrix
        print("│                                                      │")
        print("│  Confusion Matrix (feature tốt nhất):                │")
        cm = best_data["confusion_matrix"]
        print("│            Predicted: Empty  Player  Robot           │")
        names = ["Empty ", "Player", "Robot "]
        for i, name in enumerate(names):
            print(f"│    {name}          {cm[i][0]:5d}   {cm[i][1]:5d}  {cm[i][2]:5d}           │")
        
        # Per-class metrics
        print("│                                                      │")
        print("│  Per-class Metrics:                                  │")
        labels_str = ["Empty", "Player", "Robot"]
        for i, name in enumerate(labels_str):
            tp = cm[i][i]
            fp = sum(cm[j][i] for j in range(3)) - tp
            fn = sum(cm[i][j] for j in range(3)) - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            print(f"│    {name:<7}: P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}         │")
        
        print("└──────────────────────────────────────────────────────┘")
    
    return results


# ============================================================================
# PHẦN 2: ĐÁNH GIÁ GAME AI (MINIMAX)
# ============================================================================

def evaluate_ai_performance(n_games=100, verbose=True):
    """
    Đánh giá hiệu suất AI bằng cách cho đánh nhiều ván với các đối thủ khác nhau.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("  ĐÁNH GIÁ 2: GAME AI (MINIMAX) PERFORMANCE")
        print("=" * 60)
    
    import random
    ai = GameAI()
    
    results = {}
    
    # --- Test 1: AI vs Random ---
    stats = {"wins": 0, "losses": 0, "draws": 0, "total_nodes": 0}
    
    for game in range(n_games):
        board = BoardState()
        robot_turn = (game % 2 == 0)  # Xen kẽ ai đi trước
        
        while True:
            is_over, winner = board.is_game_over()
            if is_over:
                if winner == config.ROBOT:
                    stats["wins"] += 1
                elif winner == config.PLAYER:
                    stats["losses"] += 1
                else:
                    stats["draws"] += 1
                break
            
            if robot_turn:
                move = ai.get_best_move(board)
                if move:
                    board.set_cell(move[0], move[1], config.ROBOT)
                    stats["total_nodes"] += ai.nodes_evaluated
            else:
                empty = board.get_empty_cells()
                if empty:
                    row, col = random.choice(empty)
                    board.set_cell(row, col, config.PLAYER)
            
            robot_turn = not robot_turn
    
    results["vs_random"] = stats
    
    # --- Test 2: AI vs SimpleAI ---
    simple_ai = SimpleAI()
    stats2 = {"wins": 0, "losses": 0, "draws": 0}
    
    for game in range(n_games):
        board = BoardState()
        robot_turn = (game % 2 == 0)
        
        while True:
            is_over, winner = board.is_game_over()
            if is_over:
                if winner == config.ROBOT:
                    stats2["wins"] += 1
                elif winner == config.PLAYER:
                    stats2["losses"] += 1
                else:
                    stats2["draws"] += 1
                break
            
            if robot_turn:
                move = ai.get_best_move(board)
                if move:
                    board.set_cell(move[0], move[1], config.ROBOT)
            else:
                move = simple_ai.get_best_move(board)
                if move:
                    board.set_cell(move[0], move[1], config.PLAYER)
            
            robot_turn = not robot_turn
    
    results["vs_simple_ai"] = stats2
    
    # --- Test 3: AI vs AI (tự đánh) ---
    ai2 = GameAI()
    stats3 = {"wins": 0, "losses": 0, "draws": 0}
    
    for game in range(n_games):
        board = BoardState()
        first_turn = True
        
        while True:
            is_over, winner = board.is_game_over()
            if is_over:
                if winner == config.ROBOT:
                    stats3["wins"] += 1
                elif winner == config.PLAYER:
                    stats3["losses"] += 1
                else:
                    stats3["draws"] += 1
                break
            
            if first_turn:
                move = ai.get_best_move(board)
                if move:
                    board.set_cell(move[0], move[1], config.ROBOT)
            else:
                # AI2 đánh cho Player
                empty = board.get_empty_cells()
                best_move = None
                best_score = float('inf')
                
                for row, col in empty:
                    bc = board.copy()
                    bc.matrix[row, col] = config.PLAYER
                    score = ai2._minimax(bc, 8, True, float('-inf'), float('inf'))
                    if score < best_score:
                        best_score = score
                        best_move = (row, col)
                
                if best_move:
                    board.set_cell(best_move[0], best_move[1], config.PLAYER)
            
            first_turn = not first_turn
    
    results["vs_minimax"] = stats3
    
    # --- Test 4: Kiểm tra tình huống cụ thể ---
    tactical_tests = [
        {
            "name": "Thắng khi có thể",
            "setup": [(0,0,config.ROBOT), (0,1,config.ROBOT), (1,1,config.PLAYER), (2,2,config.PLAYER)],
            "expected_move": (0, 2),
        },
        {
            "name": "Chặn đối thủ",
            "setup": [(0,0,config.PLAYER), (0,1,config.PLAYER), (1,1,config.ROBOT)],
            "expected_move": (0, 2),
        },
        {
            "name": "Đi trung tâm khi trống",
            "setup": [],
            "expected_move": (1, 1),
        },
        {
            "name": "Ưu tiên góc",
            "setup": [(1,1,config.PLAYER)],
            "expected_corners": [(0,0), (0,2), (2,0), (2,2)],
        },
    ]
    
    tactical_pass = 0
    tactical_total = len(tactical_tests)
    
    for test in tactical_tests:
        board = BoardState()
        for r, c, p in test.get("setup", []):
            board.matrix[r, c] = p
        
        move = ai.get_best_move(board)
        
        if "expected_move" in test:
            passed = move == test["expected_move"]
        elif "expected_corners" in test:
            passed = move in test["expected_corners"]
        else:
            passed = move is not None
        
        if passed:
            tactical_pass += 1
        
        test["result"] = "✅ PASS" if passed else "❌ FAIL"
        test["actual_move"] = move
    
    results["tactical_tests"] = {
        "passed": tactical_pass,
        "total": tactical_total,
        "details": tactical_tests,
    }
    
    # --- Test 5: Tốc độ xử lý ---
    times = []
    for _ in range(20):
        board = BoardState()
        board.set_cell(0, 0, config.PLAYER)
        
        start = time.time()
        ai.get_best_move(board)
        elapsed = time.time() - start
        times.append(elapsed)
    
    results["speed"] = {
        "avg_ms": np.mean(times) * 1000,
        "max_ms": np.max(times) * 1000,
        "min_ms": np.min(times) * 1000,
    }
    
    # === In kết quả ===
    if verbose:
        print("\n┌──────────────────────────────────────────────────────┐")
        print("│  TỔNG HỢP KẾT QUẢ AI                                │")
        print("├──────────────────────────────────────────────────────┤")
        
        # vs Random
        s = results["vs_random"]
        wr = s["wins"] / n_games * 100
        print(f"│  vs Random ({n_games} ván):                              │")
        print(f"│    Thắng: {s['wins']:3d} ({wr:.1f}%)  "
              f"Hòa: {s['draws']:3d}  Thua: {s['losses']:3d}         │")
        avg_nodes = s["total_nodes"] / n_games
        print(f"│    Trung bình {avg_nodes:.0f} nodes/ván                       │")
        
        print("│                                                      │")
        
        # vs SimpleAI
        s2 = results["vs_simple_ai"]
        wr2 = s2["wins"] / n_games * 100
        print(f"│  vs SimpleAI ({n_games} ván):                            │")
        print(f"│    Thắng: {s2['wins']:3d} ({wr2:.1f}%)  "
              f"Hòa: {s2['draws']:3d}  Thua: {s2['losses']:3d}         │")
        
        print("│                                                      │")
        
        # vs Minimax
        s3 = results["vs_minimax"]
        print(f"│  vs Minimax (AI tự đánh, {n_games} ván):                 │")
        print(f"│    Thắng: {s3['wins']:3d}  Hòa: {s3['draws']:3d}  "
              f"Thua: {s3['losses']:3d}                  │")
        
        print("│                                                      │")
        
        # Tactical tests
        tt = results["tactical_tests"]
        print(f"│  Kiểm tra chiến thuật: {tt['passed']}/{tt['total']} PASS           │")
        for t in tt["details"]:
            print(f"│    {t['result']} {t['name']:<30}       │")
            if "expected_move" in t:
                print(f"│         Mong đợi: {t['expected_move']}, "
                      f"Thực tế: {t['actual_move']}              │")
        
        print("│                                                      │")
        
        # Speed
        sp = results["speed"]
        print(f"│  Tốc độ tính nước đi:                                │")
        print(f"│    Trung bình: {sp['avg_ms']:.1f}ms                            │")
        print(f"│    Nhanh nhất: {sp['min_ms']:.1f}ms                            │")
        print(f"│    Chậm nhất:  {sp['max_ms']:.1f}ms                            │")
        
        print("└──────────────────────────────────────────────────────┘")
    
    return results


# ============================================================================
# PHẦN 3: ĐÁNH GIÁ PIPELINE TỔNG HỢP
# ============================================================================

def evaluate_pipeline(verbose=True):
    """Đánh giá toàn bộ pipeline xử lý ảnh."""
    if verbose:
        print("\n" + "=" * 60)
        print("  ĐÁNH GIÁ 3: PIPELINE XỬ LÝ ẢNH")
        print("=" * 60)
    
    results = {}
    
    # --- Test Board Detection ---
    detection_success = 0
    detection_total = 20
    detection_times = []
    
    for trial in range(detection_total):
        # Tạo ảnh bàn cờ với góc nhìn ngẫu nhiên
        img = np.full((480, 640, 3), (80 + trial * 3, 80 + trial * 3, 80 + trial * 3), dtype=np.uint8)
        
        # Random perspective
        offset = np.random.randint(-20, 20, (4, 2))
        base_pts = np.array([[150, 100], [490, 80], [520, 380], [120, 400]])
        pts = np.clip(base_pts + offset, 0, [639, 479]).astype(np.int32)
        
        cv2.fillPoly(img, [pts], (200, 200, 200))
        cv2.polylines(img, [pts], True, (0, 0, 0), 3)
        
        detector = BoardDetector()
        
        start = time.time()
        success, corners = detector.detect_board(img)
        elapsed = time.time() - start
        detection_times.append(elapsed)
        
        if success:
            detection_success += 1
    
    results["board_detection"] = {
        "success_rate": detection_success / detection_total * 100,
        "avg_time_ms": np.mean(detection_times) * 1000,
    }
    
    # --- Test Feature Extraction Speed ---
    extractor = FeatureExtractor()
    cells = [np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8) for _ in range(9)]
    
    feat_times = {}
    for feat_type in ['hsv_mean', 'rgb_mean', 'combined', 'histogram']:
        ext = FeatureExtractor(feature_type=feat_type)
        times = []
        for _ in range(100):
            start = time.time()
            ext.extract_all_cells(cells)
            times.append(time.time() - start)
        feat_times[feat_type] = np.mean(times) * 1000
    
    results["feature_extraction"] = feat_times
    
    # --- Test K-Means Speed ---
    kmeans_times = []
    for _ in range(100):
        features = np.random.rand(9, 3).astype(np.float32)
        classifier = KMeansClassifier()
        start = time.time()
        classifier.fit_and_classify(features)
        kmeans_times.append(time.time() - start)
    
    results["kmeans_speed"] = {
        "avg_ms": np.mean(kmeans_times) * 1000,
        "max_ms": np.max(kmeans_times) * 1000,
    }
    
    # --- Test IK ---
    robot = RobotController(simulation=True)
    ik_success = 0
    ik_total = 0
    mapper = CoordinateMapper()
    
    for row in range(3):
        for col in range(3):
            coords = mapper.index_to_coords(row, col)
            if coords is not None:
                ik_total += 1
                angles = robot.inverse_kinematics(coords[0], coords[1], coords[2])
                if angles is not None:
                    ik_success += 1
    
    results["inverse_kinematics"] = {
        "success_rate": ik_success / ik_total * 100 if ik_total > 0 else 0,
        "reachable_cells": ik_success,
        "total_cells": ik_total,
    }
    
    # === In kết quả ===
    if verbose:
        print("\n┌──────────────────────────────────────────────────────┐")
        print("│  TỔNG HỢP PIPELINE                                  │")
        print("├──────────────────────────────────────────────────────┤")
        
        bd = results["board_detection"]
        print(f"│  Board Detection:                                    │")
        print(f"│    Tỷ lệ phát hiện: {bd['success_rate']:.1f}%                        │")
        print(f"│    Thời gian TB: {bd['avg_time_ms']:.1f}ms                           │")
        
        print("│                                                      │")
        print(f"│  Feature Extraction (ms/frame):                      │")
        for name, t in results["feature_extraction"].items():
            print(f"│    {name:<15}: {t:.2f}ms                           │")
        
        print("│                                                      │")
        km = results["kmeans_speed"]
        print(f"│  K-Means Classification:                             │")
        print(f"│    Thời gian TB: {km['avg_ms']:.1f}ms                            │")
        
        print("│                                                      │")
        ik = results["inverse_kinematics"]
        print(f"│  Inverse Kinematics:                                 │")
        print(f"│    Reachable: {ik['reachable_cells']}/{ik['total_cells']} cells ({ik['success_rate']:.0f}%)                      │")
        
        # Tổng thời gian pipeline
        total_pipeline_ms = (bd['avg_time_ms'] + 
                            results["feature_extraction"].get("hsv_mean", 0) + 
                            km['avg_ms'])
        fps = 1000 / total_pipeline_ms if total_pipeline_ms > 0 else 0
        
        print("│                                                      │")
        print(f"│  Tổng pipeline (ước tính):                           │")
        print(f"│    Thời gian: ~{total_pipeline_ms:.1f}ms/frame                       │")
        print(f"│    FPS lý thuyết: ~{fps:.0f} FPS                            │")
        
        print("└──────────────────────────────────────────────────────┘")
    
    return results


# ============================================================================
# PHẦN 4: BÁO CÁO TỔNG HỢP
# ============================================================================

def print_final_report(kmeans_results, ai_results, pipeline_results):
    """In báo cáo đánh giá tổng hợp."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║            📊 BÁO CÁO ĐÁNH GIÁ TỔNG HỢP              ║")
    print("║         TicTacToe Robot - K-Means + Minimax             ║")
    print("╠══════════════════════════════════════════════════════════╣")
    
    # Tìm best feature type
    best_feat = max(kmeans_results.keys(), key=lambda k: kmeans_results[k]["overall_accuracy"])
    best_acc = kmeans_results[best_feat]["overall_accuracy"] * 100
    
    # AI stats
    ai_vs_random = ai_results["vs_random"]
    ai_wr = ai_vs_random["wins"] / (ai_vs_random["wins"] + ai_vs_random["losses"] + ai_vs_random["draws"]) * 100
    
    ai_vs_simple = ai_results["vs_simple_ai"]
    ai_wr2 = ai_vs_simple["wins"] / (ai_vs_simple["wins"] + ai_vs_simple["losses"] + ai_vs_simple["draws"]) * 100
    
    tactical = ai_results["tactical_tests"]
    
    print("║                                                          ║")
    print("║  📌 K-MEANS CLUSTERING                                   ║")
    print(f"║     Feature tốt nhất: {best_feat:<20}              ║")
    print(f"║     Accuracy tổng: {best_acc:.1f}%                               ║")
    
    # Rating
    if best_acc >= 95:
        rating_km = "⭐⭐⭐⭐⭐ XUẤT SẮC"
    elif best_acc >= 85:
        rating_km = "⭐⭐⭐⭐  TỐT"
    elif best_acc >= 70:
        rating_km = "⭐⭐⭐   CHẤP NHẬN"
    else:
        rating_km = "⭐⭐     CẦN CẢI THIỆN"
    print(f"║     Đánh giá: {rating_km:<30}         ║")
    
    print("║                                                          ║")
    print("║  📌 GAME AI (MINIMAX)                                    ║")
    print(f"║     vs Random: {ai_wr:.0f}% thắng                              ║")
    print(f"║     vs SimpleAI: {ai_wr2:.0f}% thắng                            ║")
    print(f"║     Chiến thuật: {tactical['passed']}/{tactical['total']} PASS                            ║")
    print(f"║     Tốc độ: {ai_results['speed']['avg_ms']:.1f}ms/nước đi                        ║")
    
    # AI rating
    if ai_wr >= 90 and tactical['passed'] == tactical['total']:
        rating_ai = "⭐⭐⭐⭐⭐ XUẤT SẮC"
    elif ai_wr >= 80:
        rating_ai = "⭐⭐⭐⭐  TỐT"
    elif ai_wr >= 60:
        rating_ai = "⭐⭐⭐   TRUNG BÌNH"
    else:
        rating_ai = "⭐⭐     YẾU"
    print(f"║     Đánh giá: {rating_ai:<30}         ║")
    
    print("║                                                          ║")
    print("║  📌 PIPELINE                                             ║")
    bd = pipeline_results["board_detection"]
    print(f"║     Board Detection: {bd['success_rate']:.0f}%                           ║")
    
    ik = pipeline_results["inverse_kinematics"]
    print(f"║     IK Reachable: {ik['reachable_cells']}/{ik['total_cells']} cells                          ║")
    
    km_speed = pipeline_results["kmeans_speed"]["avg_ms"]
    total_ms = bd['avg_time_ms'] + pipeline_results["feature_extraction"].get("hsv_mean", 0) + km_speed
    print(f"║     Pipeline speed: ~{total_ms:.0f}ms/frame                       ║")
    
    print("║                                                          ║")
    print("╠══════════════════════════════════════════════════════════╣")
    
    # Overall score
    overall = (best_acc * 0.4 + ai_wr * 0.4 + 
               bd['success_rate'] * 0.1 + 
               ik['success_rate'] * 0.1)
    
    if overall >= 90:
        grade = "A+"
        comment = "Mô hình hoạt động xuất sắc!"
    elif overall >= 80:
        grade = "A"
        comment = "Mô hình hoạt động tốt."
    elif overall >= 70:
        grade = "B"
        comment = "Chấp nhận được, có thể cải thiện."
    elif overall >= 60:
        grade = "C"
        comment = "Cần cải thiện đáng kể."
    else:
        grade = "D"
        comment = "Cần xem lại thiết kế."
    
    print(f"║  🏆 ĐIỂM TỔNG: {overall:.1f}/100 — Grade: {grade:<25}   ║")
    print(f"║     {comment:<50}   ║")
    print("╚══════════════════════════════════════════════════════════╝")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        🔬 ĐÁNH GIÁ MÔ HÌNH TICTACTOE ROBOT            ║")
    print("║     K-Means Clustering + Minimax + Robot Control        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print("  Chọn chế độ đánh giá:")
    print("  1. 🧪 Đánh giá K-Means (accuracy, precision, recall)")
    print("  2. 🤖 Đánh giá Game AI (Minimax performance)")
    print("  3. ⚡ Đánh giá Pipeline (tốc độ, detection)")
    print("  4. 📊 Đánh giá TOÀN BỘ (khuyên dùng)")
    print("  5. 📷 Đánh giá K-Means từ camera (cần webcam + bàn cờ)")
    print("  q. Thoát")
    print()
    
    choice = input("  ➤ Chọn (1-5, q): ").strip()
    
    if choice == '1':
        evaluate_kmeans_accuracy(n_trials=30)
    elif choice == '2':
        evaluate_ai_performance(n_games=100)
    elif choice == '3':
        evaluate_pipeline()
    elif choice == '4':
        print("\n  ⏳ Đang chạy đánh giá toàn diện... (có thể mất 1-2 phút)\n")
        km_results = evaluate_kmeans_accuracy(n_trials=30)
        ai_results = evaluate_ai_performance(n_games=100)
        pipe_results = evaluate_pipeline()
        print_final_report(km_results, ai_results, pipe_results)
    elif choice == '5':
        print("\n  Chuyển sang test_accuracy.py với camera...\n")
        os.system("python test_accuracy.py")
    elif choice.lower() == 'q':
        print("  Tạm biệt!")
    else:
        print("  ❌ Lựa chọn không hợp lệ!")


if __name__ == "__main__":
    main()
