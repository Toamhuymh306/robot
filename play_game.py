"""
TicTacToe - Chơi với AI trên Terminal
Không cần robot, không cần camera - chỉ cần terminal!

Chạy: python play_game.py

Cách chơi:
  - Bạn là X (Người chơi), AI là O (Robot)
  - Nhập số 1-9 để đánh vào ô tương ứng
  - Bàn cờ được đánh số như sau:
      1 | 2 | 3
      ---------
      4 | 5 | 6
      ---------
      7 | 8 | 9
"""

import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from modules.board_state import BoardState
from modules.game_ai import GameAI, SimpleAI


def clear_screen():
    """Xóa màn hình terminal."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner():
    """In banner game."""
    print("╔══════════════════════════════════════════════╗")
    print("║     🤖 TIC-TAC-TOE: BẠN vs AI MINIMAX 🤖    ║")
    print("║  Unsupervised Learning + K-Means Clustering  ║")
    print("╚══════════════════════════════════════════════╝")


def print_board_with_numbers(board: BoardState):
    """In bàn cờ có đánh số ô."""
    symbols = {config.EMPTY: ' ', config.PLAYER: 'X', config.ROBOT: 'O'}
    colors = {
        config.EMPTY: '',
        config.PLAYER: '\033[92m',  # Xanh lá
        config.ROBOT: '\033[94m',   # Xanh dương
    }
    reset = '\033[0m'
    
    print()
    print("    Bàn cờ hiện tại:          Ô số:")
    print()
    
    for row in range(3):
        # Bàn cờ chính
        cells_main = []
        cells_num = []
        for col in range(3):
            val = board.matrix[row, col]
            sym = symbols.get(val, '?')
            color = colors.get(val, '')
            
            if val != config.EMPTY:
                cells_main.append(f" {color}{sym}{reset} ")
            else:
                cells_main.append("   ")
            
            idx = row * 3 + col + 1
            if val == config.EMPTY:
                cells_num.append(f" {idx} ")
            else:
                cells_num.append(f" {color}{sym}{reset} ")
        
        print(f"      {cells_main[0]}│{cells_main[1]}│{cells_main[2]}", end="")
        print(f"          {cells_num[0]}│{cells_num[1]}│{cells_num[2]}")
        
        if row < 2:
            print(f"      ───┼───┼───          ───┼───┼───")
    
    print()


def print_move_analysis(ai: GameAI, board: BoardState):
    """In phân tích nước đi của AI (để hiểu AI suy nghĩ thế nào)."""
    analysis = ai.get_move_analysis(board)
    
    print("  ┌─────────────────────────────────────────┐")
    print("  │  🧠 AI đang phân tích...                │")
    print("  ├─────────────────────────────────────────┤")
    
    moves = sorted(analysis['moves'], key=lambda m: m['score'], reverse=True)
    
    for m in moves:
        pos = m['position']
        score = m['score']
        idx = pos[0] * 3 + pos[1] + 1
        
        # Thanh điểm số
        bar_len = max(0, min(20, int((score + 110) / 11)))
        bar = '█' * bar_len + '░' * (20 - bar_len)
        
        flags = []
        if m['is_winning']:
            flags.append('🏆 THẮNG!')
        if m['blocks_player']:
            flags.append('🛡️ CHẶN')
        
        flag_str = ' '.join(flags)
        print(f"  │  Ô {idx} ({pos[0]},{pos[1]}): {bar} {score:+.0f} {flag_str}")
    
    best = analysis['best_move']
    best_idx = best[0] * 3 + best[1] + 1
    print(f"  │")
    print(f"  │  ➤ Chọn ô {best_idx} (điểm: {analysis['best_score']:+.0f})")
    print(f"  │  Nodes đánh giá: {ai.nodes_evaluated}")
    print(f"  └─────────────────────────────────────────┘")


def play_game(show_analysis: bool = True, ai_first: bool = False):
    """
    Chơi một ván cờ với AI.
    
    Args:
        show_analysis: Hiển thị phân tích nước đi của AI
        ai_first: AI đi trước
    """
    board = BoardState()
    ai = GameAI()
    
    robot_turn = ai_first
    move_num = 0
    
    clear_screen()
    print_banner()
    print()
    
    if ai_first:
        print("  AI đi trước! AI là O, bạn là X.")
    else:
        print("  Bạn đi trước! Bạn là X, AI là O.")
    
    print("  Nhập số 1-9 để đánh vào ô tương ứng.")
    print("  Nhập 'q' để thoát, 'r' để chơi lại.")
    
    while True:
        print_board_with_numbers(board)
        
        # Kiểm tra kết thúc game
        is_over, winner = board.is_game_over()
        if is_over:
            if winner == config.ROBOT:
                print("  ╔═══════════════════════════╗")
                print("  ║  🤖 AI THẮNG! Quá mạnh!  ║")
                print("  ╚═══════════════════════════╝")
            elif winner == config.PLAYER:
                print("  ╔═══════════════════════════════╗")
                print("  ║  🎉 BẠN THẮNG! Tuyệt vời!   ║")
                print("  ╚═══════════════════════════════╝")
            else:
                print("  ╔══════════════════════════════╗")
                print("  ║  🤝 HÒA! Cả hai đều giỏi!   ║")
                print("  ╚══════════════════════════════╝")
            return winner
        
        if robot_turn:
            # === LƯỢT AI ===
            print(f"  ⏳ Lượt AI (O) - Nước đi #{move_num + 1}...")
            print()
            
            if show_analysis:
                # Hiện phân tích chi tiết
                print_move_analysis(ai, board)
                print()
            
            move = ai.get_best_move(board)
            if move:
                row, col = move
                idx = row * 3 + col + 1
                board.set_cell(row, col, config.ROBOT)
                print(f"  ✅ AI đánh ô {idx} (hàng {row}, cột {col})")
                time.sleep(0.5)
        else:
            # === LƯỢT NGƯỜI CHƠI ===
            print(f"  👤 Lượt bạn (X) - Nước đi #{move_num + 1}")
            
            while True:
                try:
                    choice = input("  ➤ Nhập số ô (1-9): ").strip().lower()
                    
                    if choice == 'q':
                        print("\n  Đã thoát game.")
                        return None
                    elif choice == 'r':
                        return 'restart'
                    
                    num = int(choice)
                    if num < 1 or num > 9:
                        print("  ❌ Vui lòng nhập số từ 1 đến 9!")
                        continue
                    
                    row = (num - 1) // 3
                    col = (num - 1) % 3
                    
                    if board.matrix[row, col] != config.EMPTY:
                        print("  ❌ Ô này đã có quân rồi! Chọn ô khác.")
                        continue
                    
                    board.set_cell(row, col, config.PLAYER)
                    print(f"  ✅ Bạn đánh ô {num}")
                    break
                    
                except ValueError:
                    print("  ❌ Vui lòng nhập số hợp lệ!")
        
        robot_turn = not robot_turn
        move_num += 1
        print()
        print("  " + "─" * 45)


def demo_ai_vs_ai():
    """Cho 2 AI đánh với nhau để xem."""
    board = BoardState()
    ai1 = GameAI()       # AI Minimax (mạnh)
    ai2 = SimpleAI()     # AI đơn giản (yếu hơn)
    
    clear_screen()
    print("╔══════════════════════════════════════════════╗")
    print("║   🤖 vs 🤖  AI MINIMAX  vs  AI ĐƠN GIẢN    ║")
    print("╚══════════════════════════════════════════════╝")
    print()
    print("  Minimax (O) vs SimpleAI (X)")
    print("  Xem AI đánh với nhau...")
    print()
    
    robot_turn = True  # Minimax đi trước
    move_num = 0
    
    while True:
        print_board_with_numbers(board)
        
        is_over, winner = board.is_game_over()
        if is_over:
            if winner == config.ROBOT:
                print("  🏆 Minimax AI (O) THẮNG!")
            elif winner == config.PLAYER:
                print("  🏆 Simple AI (X) THẮNG!")
            else:
                print("  🤝 HÒA!")
            return
        
        if robot_turn:
            move = ai1.get_best_move(board)
            if move:
                row, col = move
                idx = row * 3 + col + 1
                board.set_cell(row, col, config.ROBOT)
                print(f"  Minimax (O) đánh ô {idx}")
        else:
            move = ai2.get_best_move(board)
            if move:
                row, col = move
                idx = row * 3 + col + 1
                board.set_cell(row, col, config.PLAYER)
                print(f"  SimpleAI (X) đánh ô {idx}")
        
        robot_turn = not robot_turn
        move_num += 1
        time.sleep(1)
        print()


def main():
    """Main menu."""
    while True:
        clear_screen()
        print_banner()
        print()
        print("  ┌────────────────────────────────────┐")
        print("  │  CHỌN CHẾ ĐỘ:                      │")
        print("  │                                     │")
        print("  │  1. 👤 Bạn vs AI (bạn đi trước)     │")
        print("  │  2. 🤖 AI vs Bạn (AI đi trước)      │")
        print("  │  3. 👤 Bạn vs AI (ẩn phân tích)      │")
        print("  │  4. 🤖 AI vs AI (xem AI đánh nhau)   │")
        print("  │  5. ❌ Thoát                         │")
        print("  │                                     │")
        print("  └────────────────────────────────────┘")
        print()
        
        choice = input("  ➤ Chọn (1-5): ").strip()
        
        if choice == '1':
            while True:
                result = play_game(show_analysis=True, ai_first=False)
                if result == 'restart':
                    continue
                input("\n  Nhấn Enter để tiếp tục...")
                break
        elif choice == '2':
            while True:
                result = play_game(show_analysis=True, ai_first=True)
                if result == 'restart':
                    continue
                input("\n  Nhấn Enter để tiếp tục...")
                break
        elif choice == '3':
            while True:
                result = play_game(show_analysis=False, ai_first=False)
                if result == 'restart':
                    continue
                input("\n  Nhấn Enter để tiếp tục...")
                break
        elif choice == '4':
            demo_ai_vs_ai()
            input("\n  Nhấn Enter để tiếp tục...")
        elif choice == '5' or choice.lower() == 'q':
            print("\n  👋 Tạm biệt!")
            break
        else:
            print("  ❌ Lựa chọn không hợp lệ!")
            time.sleep(1)


if __name__ == "__main__":
    main()
