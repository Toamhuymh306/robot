"""
Module 6: Board State Matrix
Manages the game state matrix and tracks changes
"""

import numpy as np
from typing import Optional, Tuple, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class BoardState:
    """
    Manages the tic-tac-toe board state as a 3x3 matrix.
    Tracks game state changes and detects new moves.
    
    Board representation:
        0  = Empty
        1  = Robot piece
       -1  = Player piece
    """
    
    def __init__(self):
        """Initialize board state."""
        # 3x3 matrix representing the board
        self.matrix = np.zeros((3, 3), dtype=int)
        
        # History of states for tracking changes
        self.history = []
        
        # Move counter
        self.move_count = 0
        
        # Last detected move
        self.last_move = None
        
        # Current player (1 = Robot, -1 = Player)
        self.current_player = config.PLAYER if not config.AI_FIRST else config.ROBOT
    
    def reset(self):
        """Reset board to initial state."""
        self.matrix = np.zeros((3, 3), dtype=int)
        self.history = []
        self.move_count = 0
        self.last_move = None
        self.current_player = config.PLAYER if not config.AI_FIRST else config.ROBOT
    
    def update_from_labels(self, labels: np.ndarray) -> bool:
        """
        Update board state from classification labels.
        
        Args:
            labels: Array of 9 labels (row-major order)
            
        Returns:
            True if state changed
        """
        if len(labels) != 9:
            print(f"[WARNING] Expected 9 labels, got {len(labels)}")
            return False
        
        # Reshape to 3x3
        new_matrix = labels.reshape(3, 3)
        
        # Check if state changed
        if np.array_equal(new_matrix, self.matrix):
            return False
        
        # Store previous state
        self.history.append(self.matrix.copy())
        
        # Update matrix
        self.matrix = new_matrix.copy()
        
        return True
    
    def detect_new_move(self, new_labels: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Detect new move by comparing with previous state.
        
        Args:
            new_labels: Array of 9 labels
            
        Returns:
            Tuple of (row, col, player) if new move detected, None otherwise
        """
        if len(new_labels) != 9:
            return None
        
        new_matrix = new_labels.reshape(3, 3)
        
        # Find differences
        diff = new_matrix - self.matrix
        
        # Find cells that changed from empty to occupied
        new_moves = []
        for i in range(3):
            for j in range(3):
                if self.matrix[i, j] == config.EMPTY and new_matrix[i, j] != config.EMPTY:
                    new_moves.append((i, j, new_matrix[i, j]))
        
        if len(new_moves) == 1:
            row, col, player = new_moves[0]
            self.last_move = (row, col, player)
            self.move_count += 1
            return self.last_move
        elif len(new_moves) > 1:
            print(f"[WARNING] Multiple new moves detected: {new_moves}")
        
        return None
    
    def set_cell(self, row: int, col: int, player: int) -> bool:
        """
        Set a cell value.
        
        Args:
            row: Row index (0-2)
            col: Column index (0-2)
            player: Player value (EMPTY, PLAYER, or ROBOT)
            
        Returns:
            True if cell was set successfully
        """
        if not (0 <= row < 3 and 0 <= col < 3):
            print(f"[WARNING] Invalid cell position: ({row}, {col})")
            return False
        
        if self.matrix[row, col] != config.EMPTY:
            print(f"[WARNING] Cell ({row}, {col}) is not empty")
            return False
        
        # Store history
        self.history.append(self.matrix.copy())
        
        # Set cell
        self.matrix[row, col] = player
        self.last_move = (row, col, player)
        self.move_count += 1
        
        return True
    
    def get_cell(self, row: int, col: int) -> int:
        """Get cell value."""
        if 0 <= row < 3 and 0 <= col < 3:
            return self.matrix[row, col]
        return config.EMPTY
    
    def get_empty_cells(self) -> List[Tuple[int, int]]:
        """Get list of empty cell positions."""
        empty = []
        for i in range(3):
            for j in range(3):
                if self.matrix[i, j] == config.EMPTY:
                    empty.append((i, j))
        return empty
    
    def is_valid_move(self, row: int, col: int) -> bool:
        """Check if a move is valid."""
        return (0 <= row < 3 and 0 <= col < 3 and 
                self.matrix[row, col] == config.EMPTY)
    
    def check_winner(self) -> int:
        """
        Check if there's a winner.
        
        Returns:
            ROBOT (1) if robot wins
            PLAYER (-1) if player wins
            EMPTY (0) if no winner yet
        """
        # Check rows
        for row in self.matrix:
            if abs(sum(row)) == 3:
                return int(row[0])
        
        # Check columns
        for col in range(3):
            col_sum = sum(self.matrix[row, col] for row in range(3))
            if abs(col_sum) == 3:
                return int(self.matrix[0, col])
        
        # Check diagonals
        diag1 = sum(self.matrix[i, i] for i in range(3))
        if abs(diag1) == 3:
            return int(self.matrix[0, 0])
        
        diag2 = sum(self.matrix[i, 2-i] for i in range(3))
        if abs(diag2) == 3:
            return int(self.matrix[0, 2])
        
        return config.EMPTY
    
    def is_game_over(self) -> Tuple[bool, Optional[int]]:
        """
        Check if game is over.
        
        Returns:
            Tuple of (is_over, winner)
            winner is None for draw, ROBOT/PLAYER for win
        """
        winner = self.check_winner()
        
        if winner != config.EMPTY:
            return True, winner
        
        # Check for draw (no empty cells)
        if len(self.get_empty_cells()) == 0:
            return True, None
        
        return False, None
    
    def get_state_string(self) -> str:
        """Get string representation of board state."""
        symbols = {config.EMPTY: '.', config.PLAYER: 'X', config.ROBOT: 'O'}
        lines = []
        
        for row in self.matrix:
            line = ' '.join(symbols.get(cell, '?') for cell in row)
            lines.append(line)
        
        return '\n'.join(lines)
    
    def get_state_hash(self) -> str:
        """Get unique hash of current state."""
        return ''.join(str(c + 1) for c in self.matrix.flatten())
    
    def copy(self) -> 'BoardState':
        """Create a copy of current state."""
        new_state = BoardState()
        new_state.matrix = self.matrix.copy()
        new_state.move_count = self.move_count
        new_state.current_player = self.current_player
        return new_state
    
    def index_to_position(self, index: int) -> Tuple[int, int]:
        """
        Convert linear index (0-8) to (row, col).
        
        Args:
            index: Linear index
            
        Returns:
            Tuple of (row, col)
        """
        return index // 3, index % 3
    
    def position_to_index(self, row: int, col: int) -> int:
        """
        Convert (row, col) to linear index.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            Linear index (0-8)
        """
        return row * 3 + col
    
    def render(self) -> str:
        """Render board as ASCII art."""
        symbols = {config.EMPTY: ' ', config.PLAYER: 'X', config.ROBOT: 'O'}
        
        lines = []
        lines.append('┌───┬───┬───┐')
        
        for i, row in enumerate(self.matrix):
            cells = [symbols.get(cell, '?') for cell in row]
            lines.append(f'│ {cells[0]} │ {cells[1]} │ {cells[2]} │')
            
            if i < 2:
                lines.append('├───┼───┼───┤')
        
        lines.append('└───┴───┴───┘')
        
        return '\n'.join(lines)
    
    def print_state(self):
        """Print current board state."""
        print("\nCurrent Board State:")
        print(self.render())
        print(f"\nMove count: {self.move_count}")
        print(f"Last move: {self.last_move}")
        
        is_over, winner = self.is_game_over()
        if is_over:
            if winner == config.ROBOT:
                print("Game Over: Robot wins!")
            elif winner == config.PLAYER:
                print("Game Over: Player wins!")
            else:
                print("Game Over: Draw!")


def test_board_state():
    """
    Test function for board state module.
    """
    print("=" * 50)
    print("Testing Board State Module")
    print("=" * 50)
    
    board = BoardState()
    
    # Test initial state
    print("\n[TEST] Initial state:")
    board.print_state()
    
    # Test setting cells
    print("\n[TEST] Making moves...")
    board.set_cell(0, 0, config.ROBOT)    # Robot plays center
    board.set_cell(1, 1, config.PLAYER)   # Player plays corner
    board.set_cell(0, 1, config.ROBOT)    # Robot plays edge
    board.set_cell(2, 2, config.PLAYER)   # Player plays corner
    
    print("\nAfter moves:")
    board.print_state()
    
    # Test empty cells
    empty = board.get_empty_cells()
    print(f"\n[TEST] Empty cells: {empty}")
    
    # Test winner detection
    print("\n[TEST] Creating winning condition for Robot...")
    board.set_cell(0, 2, config.ROBOT)  # Robot wins with top row
    board.print_state()
    
    # Test from labels
    print("\n[TEST] Testing update from labels...")
    board.reset()
    labels = np.array([
        config.ROBOT, config.EMPTY, config.PLAYER,
        config.EMPTY, config.ROBOT, config.EMPTY,
        config.PLAYER, config.EMPTY, config.ROBOT
    ])
    board.update_from_labels(labels)
    board.print_state()
    
    # Check diagonal win
    is_over, winner = board.is_game_over()
    print(f"\n[TEST] Game over: {is_over}, Winner: {winner}")
    
    print("\n[INFO] Board state test completed")


if __name__ == "__main__":
    test_board_state()
