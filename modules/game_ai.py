"""
Module 7: Game AI
Implements Minimax algorithm with Alpha-Beta pruning for optimal moves
"""

import numpy as np
from typing import Optional, Tuple, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.board_state import BoardState


class GameAI:
    """
    Game AI using Minimax algorithm with Alpha-Beta pruning.
    Finds the optimal move for the robot player.
    """
    
    def __init__(self, max_depth: int = config.MINIMAX_DEPTH):
        """
        Initialize Game AI.
        
        Args:
            max_depth: Maximum search depth for minimax
        """
        self.max_depth = max_depth
        self.nodes_evaluated = 0
        
        # Player values
        self.robot = config.ROBOT
        self.player = config.PLAYER
        self.empty = config.EMPTY
    
    def get_best_move(self, board: BoardState) -> Optional[Tuple[int, int]]:
        """
        Find the best move for the robot.
        
        Args:
            board: Current board state
            
        Returns:
            Tuple of (row, col) for best move, None if no moves available
        """
        empty_cells = board.get_empty_cells()
        
        if len(empty_cells) == 0:
            return None
        
        # If board is empty, play center
        if len(empty_cells) == 9:
            return (1, 1)
        
        self.nodes_evaluated = 0
        
        best_score = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        
        for row, col in empty_cells:
            # Make move
            board_copy = board.copy()
            board_copy.matrix[row, col] = self.robot
            
            # Evaluate move
            score = self._minimax(board_copy, self.max_depth - 1, 
                                 False, alpha, beta)
            
            if score > best_score:
                best_score = score
                best_move = (row, col)
            
            alpha = max(alpha, score)
        
        print(f"[AI] Evaluated {self.nodes_evaluated} nodes")
        print(f"[AI] Best move: {best_move} with score: {best_score}")
        
        return best_move
    
    def _minimax(self, board: BoardState, depth: int, 
                 is_maximizing: bool, alpha: float, beta: float) -> float:
        """
        Minimax algorithm with Alpha-Beta pruning.
        
        Args:
            board: Current board state
            depth: Remaining search depth
            is_maximizing: True if maximizing player (robot)
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            
        Returns:
            Best score for current position
        """
        self.nodes_evaluated += 1
        
        # Check terminal conditions
        winner = board.check_winner()
        
        if winner == self.robot:
            return config.WIN_SCORE + depth  # Prefer earlier wins
        elif winner == self.player:
            return config.LOSE_SCORE - depth  # Prefer later losses
        
        empty_cells = board.get_empty_cells()
        
        if len(empty_cells) == 0 or depth == 0:
            return self._evaluate(board)
        
        if is_maximizing:
            max_eval = float('-inf')
            
            for row, col in empty_cells:
                board_copy = board.copy()
                board_copy.matrix[row, col] = self.robot
                
                eval_score = self._minimax(board_copy, depth - 1, 
                                          False, alpha, beta)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                
                if beta <= alpha:
                    break  # Beta cutoff
            
            return max_eval
        else:
            min_eval = float('inf')
            
            for row, col in empty_cells:
                board_copy = board.copy()
                board_copy.matrix[row, col] = self.player
                
                eval_score = self._minimax(board_copy, depth - 1, 
                                          True, alpha, beta)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                
                if beta <= alpha:
                    break  # Alpha cutoff
            
            return min_eval
    
    def _evaluate(self, board: BoardState) -> float:
        """
        Heuristic evaluation of board position.
        
        Args:
            board: Board state to evaluate
            
        Returns:
            Evaluation score
        """
        score = 0
        
        # Evaluate all winning lines
        lines = self._get_all_lines(board.matrix)
        
        for line in lines:
            score += self._evaluate_line(line)
        
        # Add positional bonus (prefer center and corners)
        if board.matrix[1, 1] == self.robot:
            score += 3
        elif board.matrix[1, 1] == self.player:
            score -= 3
        
        for i, j in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            if board.matrix[i, j] == self.robot:
                score += 2
            elif board.matrix[i, j] == self.player:
                score -= 2
        
        return score
    
    def _get_all_lines(self, matrix: np.ndarray) -> List[List[int]]:
        """Get all possible winning lines from matrix."""
        lines = []
        
        # Rows
        for row in matrix:
            lines.append(list(row))
        
        # Columns
        for col in range(3):
            lines.append([matrix[row, col] for row in range(3)])
        
        # Diagonals
        lines.append([matrix[i, i] for i in range(3)])
        lines.append([matrix[i, 2-i] for i in range(3)])
        
        return lines
    
    def _evaluate_line(self, line: List[int]) -> float:
        """
        Evaluate a single line (row/column/diagonal).
        
        Args:
            line: List of 3 cell values
            
        Returns:
            Score for the line
        """
        robot_count = line.count(self.robot)
        player_count = line.count(self.player)
        empty_count = line.count(self.empty)
        
        # Win condition
        if robot_count == 3:
            return 100
        if player_count == 3:
            return -100
        
        # Potential win for robot
        if robot_count == 2 and empty_count == 1:
            return 10
        if robot_count == 1 and empty_count == 2:
            return 1
        
        # Potential win for player (block!)
        if player_count == 2 and empty_count == 1:
            return -10
        if player_count == 1 and empty_count == 2:
            return -1
        
        return 0
    
    def get_immediate_win_or_block(self, board: BoardState) -> Optional[Tuple[int, int]]:
        """
        Quick check for immediate win or block opportunity.
        
        Args:
            board: Current board state
            
        Returns:
            Position (row, col) if immediate action needed, None otherwise
        """
        empty_cells = board.get_empty_cells()
        
        # Check for winning move
        for row, col in empty_cells:
            board_copy = board.copy()
            board_copy.matrix[row, col] = self.robot
            
            if board_copy.check_winner() == self.robot:
                return (row, col)
        
        # Check for blocking move
        for row, col in empty_cells:
            board_copy = board.copy()
            board_copy.matrix[row, col] = self.player
            
            if board_copy.check_winner() == self.player:
                return (row, col)
        
        return None
    
    def get_move_analysis(self, board: BoardState) -> dict:
        """
        Get detailed analysis of all possible moves.
        
        Args:
            board: Current board state
            
        Returns:
            Dictionary with move analysis
        """
        empty_cells = board.get_empty_cells()
        analysis = {'moves': [], 'best_move': None, 'best_score': float('-inf')}
        
        for row, col in empty_cells:
            board_copy = board.copy()
            board_copy.matrix[row, col] = self.robot
            
            score = self._minimax(board_copy, self.max_depth - 1,
                                 False, float('-inf'), float('inf'))
            
            move_info = {
                'position': (row, col),
                'score': score,
                'is_winning': board_copy.check_winner() == self.robot,
                'blocks_player': self._blocks_player_win(board, row, col)
            }
            analysis['moves'].append(move_info)
            
            if score > analysis['best_score']:
                analysis['best_score'] = score
                analysis['best_move'] = (row, col)
        
        return analysis
    
    def _blocks_player_win(self, board: BoardState, row: int, col: int) -> bool:
        """Check if a move blocks player's winning move."""
        board_copy = board.copy()
        board_copy.matrix[row, col] = self.player
        return board_copy.check_winner() == self.player


class SimpleAI:
    """
    Simpler AI for faster response (less computational overhead).
    Uses rule-based approach with limited lookahead.
    """
    
    def __init__(self):
        self.robot = config.ROBOT
        self.player = config.PLAYER
        self.empty = config.EMPTY
    
    def get_best_move(self, board: BoardState) -> Optional[Tuple[int, int]]:
        """Get best move using simple heuristics."""
        empty_cells = board.get_empty_cells()
        
        if len(empty_cells) == 0:
            return None
        
        # 1. Try to win
        for row, col in empty_cells:
            board_copy = board.copy()
            board_copy.matrix[row, col] = self.robot
            if board_copy.check_winner() == self.robot:
                return (row, col)
        
        # 2. Block opponent's win
        for row, col in empty_cells:
            board_copy = board.copy()
            board_copy.matrix[row, col] = self.player
            if board_copy.check_winner() == self.player:
                return (row, col)
        
        # 3. Take center if available
        if (1, 1) in empty_cells:
            return (1, 1)
        
        # 4. Take a corner
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        for corner in corners:
            if corner in empty_cells:
                return corner
        
        # 5. Take any edge
        edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
        for edge in edges:
            if edge in empty_cells:
                return edge
        
        return empty_cells[0]


def test_game_ai():
    """
    Test function for Game AI module.
    """
    print("=" * 50)
    print("Testing Game AI Module")
    print("=" * 50)
    
    ai = GameAI()
    board = BoardState()
    
    # Test 1: Empty board - should play center
    print("\n[TEST] Empty board:")
    board.print_state()
    move = ai.get_best_move(board)
    print(f"AI suggests: {move}")
    
    # Test 2: Block opponent
    print("\n[TEST] Block opponent (Player has 2 in a row):")
    board.reset()
    board.set_cell(0, 0, config.PLAYER)
    board.set_cell(0, 1, config.PLAYER)
    board.print_state()
    move = ai.get_best_move(board)
    print(f"AI suggests: {move} (should be (0, 2) to block)")
    
    # Test 3: Win when possible
    print("\n[TEST] Win when possible (Robot has 2 in a row):")
    board.reset()
    board.set_cell(0, 0, config.ROBOT)
    board.set_cell(0, 1, config.ROBOT)
    board.set_cell(1, 1, config.PLAYER)
    board.print_state()
    move = ai.get_best_move(board)
    print(f"AI suggests: {move} (should be (0, 2) to win)")
    
    # Test 4: Complex position
    print("\n[TEST] Complex position:")
    board.reset()
    board.set_cell(0, 0, config.ROBOT)
    board.set_cell(1, 1, config.PLAYER)
    board.set_cell(0, 2, config.ROBOT)
    board.set_cell(2, 1, config.PLAYER)
    board.print_state()
    move = ai.get_best_move(board)
    print(f"AI suggests: {move}")
    
    # Test move analysis
    print("\n[TEST] Move analysis:")
    analysis = ai.get_move_analysis(board)
    for move_info in analysis['moves']:
        print(f"  Move {move_info['position']}: score={move_info['score']}")
    
    # Test SimpleAI
    print("\n[TEST] SimpleAI comparison:")
    simple_ai = SimpleAI()
    simple_move = simple_ai.get_best_move(board)
    print(f"SimpleAI suggests: {simple_move}")
    
    print("\n[INFO] Game AI test completed")


if __name__ == "__main__":
    test_game_ai()
