"""
Module 8: Coordinate Mapping
Maps board cell indices to robot arm coordinates
"""

import numpy as np
from typing import Tuple, Optional, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class CoordinateMapper:
    """
    Maps board cell positions (i, j) to robot arm coordinates (x, y, z).
    Handles calibration and workspace transformations.
    """
    
    def __init__(self,
                 board_origin: Tuple[float, float, float] = (
                     config.BOARD_ORIGIN_X,
                     config.BOARD_ORIGIN_Y,
                     config.BOARD_ORIGIN_Z
                 ),
                 cell_spacing: Tuple[float, float] = (
                     config.CELL_SPACING_X,
                     config.CELL_SPACING_Y
                 ),
                 board_size: float = config.BOARD_SIZE_MM):
        """
        Initialize coordinate mapper.
        
        Args:
            board_origin: (x, y, z) position of board origin in robot base frame (mm)
            cell_spacing: (dx, dy) spacing between cell centers (mm)
            board_size: Total board size (mm)
        """
        # Board position relative to robot base
        self.board_origin = np.array(board_origin, dtype=float)
        
        # Cell spacing
        self.cell_spacing = np.array(cell_spacing, dtype=float)
        
        # Board size
        self.board_size = board_size
        self.cell_size = board_size / 3
        
        # Calibration offset (adjustable)
        self.calibration_offset = np.array([0.0, 0.0, 0.0])
        
        # Workspace limits
        self.workspace_min = np.array([0, -200, 0])
        self.workspace_max = np.array([300, 200, 150])
        
        # Pre-calculate cell centers
        self._calculate_cell_centers()
    
    def _calculate_cell_centers(self):
        """Pre-calculate all cell center positions."""
        self.cell_centers = {}
        
        for row in range(3):
            for col in range(3):
                # Calculate position relative to board origin
                # Row 0 is at top (higher Y), Row 2 is at bottom (lower Y)
                x = self.board_origin[0] + (col - 1) * self.cell_spacing[0]
                y = self.board_origin[1] + (1 - row) * self.cell_spacing[1]
                z = self.board_origin[2]
                
                self.cell_centers[(row, col)] = np.array([x, y, z])
    
    def index_to_coords(self, row: int, col: int) -> Optional[np.ndarray]:
        """
        Convert cell index (row, col) to robot coordinates (x, y, z).
        
        Args:
            row: Row index (0-2)
            col: Column index (0-2)
            
        Returns:
            NumPy array [x, y, z] in mm, None if invalid
        """
        if not (0 <= row < 3 and 0 <= col < 3):
            print(f"[WARNING] Invalid cell index: ({row}, {col})")
            return None
        
        coords = self.cell_centers[(row, col)] + self.calibration_offset
        return coords
    
    def linear_index_to_coords(self, index: int) -> Optional[np.ndarray]:
        """
        Convert linear index (0-8) to robot coordinates.
        
        Args:
            index: Linear index (0-8, row-major order)
            
        Returns:
            NumPy array [x, y, z] in mm
        """
        if not 0 <= index < 9:
            return None
        
        row = index // 3
        col = index % 3
        return self.index_to_coords(row, col)
    
    def coords_to_index(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """
        Convert robot coordinates to nearest cell index.
        
        Args:
            x: X coordinate in mm
            y: Y coordinate in mm
            
        Returns:
            Tuple of (row, col) for nearest cell
        """
        # Find nearest cell
        min_dist = float('inf')
        nearest = None
        
        for (row, col), center in self.cell_centers.items():
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            if dist < min_dist:
                min_dist = dist
                nearest = (row, col)
        
        return nearest
    
    def is_in_workspace(self, coords: np.ndarray) -> bool:
        """
        Check if coordinates are within robot workspace.
        
        Args:
            coords: [x, y, z] coordinates
            
        Returns:
            True if within workspace
        """
        return (np.all(coords >= self.workspace_min) and 
                np.all(coords <= self.workspace_max))
    
    def clamp_to_workspace(self, coords: np.ndarray) -> np.ndarray:
        """
        Clamp coordinates to workspace limits.
        
        Args:
            coords: [x, y, z] coordinates
            
        Returns:
            Clamped coordinates
        """
        return np.clip(coords, self.workspace_min, self.workspace_max)
    
    def calibrate(self, 
                  actual_positions: List[Tuple[Tuple[int, int], np.ndarray]]):
        """
        Calibrate mapping using actual measured positions.
        
        Args:
            actual_positions: List of ((row, col), [x, y, z]) pairs
        """
        if len(actual_positions) < 3:
            print("[WARNING] Need at least 3 points for calibration")
            return
        
        # Calculate average offset
        offsets = []
        for (row, col), actual in actual_positions:
            expected = self.cell_centers[(row, col)]
            offset = actual - expected
            offsets.append(offset)
        
        self.calibration_offset = np.mean(offsets, axis=0)
        print(f"[INFO] Calibration offset: {self.calibration_offset}")
    
    def get_approach_position(self, row: int, col: int, 
                               height: float = 50) -> Optional[np.ndarray]:
        """
        Get approach position above a cell (for safe movement).
        
        Args:
            row: Row index
            col: Column index
            height: Height above board (mm)
            
        Returns:
            [x, y, z] approach position
        """
        coords = self.index_to_coords(row, col)
        if coords is None:
            return None
        
        coords[2] += height
        return coords
    
    def get_pick_position(self, storage_idx: int = 0) -> np.ndarray:
        """
        Get position to pick up a piece from storage.
        
        Args:
            storage_idx: Index of storage position
            
        Returns:
            [x, y, z] pick position
        """
        if 0 <= storage_idx < len(config.PIECE_STORAGE):
            return np.array(config.PIECE_STORAGE[storage_idx])
        return np.array(config.PIECE_STORAGE[0])
    
    def get_place_position(self, row: int, col: int) -> Optional[np.ndarray]:
        """
        Get position to place a piece on board.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            [x, y, z] place position
        """
        coords = self.index_to_coords(row, col)
        if coords is None:
            return None
        
        # Adjust Z for piece placement height
        coords[2] = config.PLACE_HEIGHT
        return coords
    
    def get_trajectory(self, 
                       start: np.ndarray, 
                       end: np.ndarray,
                       num_points: int = 10) -> List[np.ndarray]:
        """
        Generate linear trajectory between two points.
        
        Args:
            start: Start position [x, y, z]
            end: End position [x, y, z]
            num_points: Number of intermediate points
            
        Returns:
            List of positions along trajectory
        """
        trajectory = []
        for t in np.linspace(0, 1, num_points):
            point = start + t * (end - start)
            trajectory.append(point)
        return trajectory
    
    def get_pick_and_place_trajectory(self, 
                                       row: int, 
                                       col: int,
                                       piece_storage_idx: int = 0) -> List[dict]:
        """
        Generate complete pick-and-place trajectory.
        
        Args:
            row: Target row on board
            col: Target column on board
            piece_storage_idx: Index of piece storage
            
        Returns:
            List of waypoints with positions and actions
        """
        waypoints = []
        
        # Get positions
        pick_pos = self.get_pick_position(piece_storage_idx)
        place_pos = self.get_place_position(row, col)
        
        if place_pos is None:
            return waypoints
        
        # Approach pick position
        approach_pick = pick_pos.copy()
        approach_pick[2] += 40
        waypoints.append({'position': approach_pick, 'action': 'move'})
        
        # Move down to pick
        waypoints.append({'position': pick_pos.copy(), 'action': 'move'})
        
        # Close gripper
        waypoints.append({'position': pick_pos.copy(), 'action': 'grip_close'})
        
        # Lift up
        lift_pos = pick_pos.copy()
        lift_pos[2] += 60
        waypoints.append({'position': lift_pos, 'action': 'move'})
        
        # Approach place position
        approach_place = place_pos.copy()
        approach_place[2] += 40
        waypoints.append({'position': approach_place, 'action': 'move'})
        
        # Move down to place
        waypoints.append({'position': place_pos.copy(), 'action': 'move'})
        
        # Open gripper
        waypoints.append({'position': place_pos.copy(), 'action': 'grip_open'})
        
        # Lift up
        final_pos = place_pos.copy()
        final_pos[2] += 40
        waypoints.append({'position': final_pos, 'action': 'move'})
        
        return waypoints
    
    def visualize_board_coords(self) -> str:
        """Create text visualization of board coordinates."""
        lines = []
        lines.append("\nBoard Cell Coordinates (X, Y, Z) in mm:")
        lines.append("=" * 50)
        
        for row in range(3):
            row_coords = []
            for col in range(3):
                coords = self.index_to_coords(row, col)
                row_coords.append(f"({coords[0]:.0f},{coords[1]:.0f},{coords[2]:.0f})")
            lines.append(" | ".join(row_coords))
        
        return "\n".join(lines)


def test_coordinate_mapping():
    """
    Test function for coordinate mapping module.
    """
    print("=" * 50)
    print("Testing Coordinate Mapping Module")
    print("=" * 50)
    
    mapper = CoordinateMapper()
    
    # Test coordinate mapping
    print(mapper.visualize_board_coords())
    
    # Test specific cell
    print("\n[TEST] Cell (1, 1) coordinates:")
    coords = mapper.index_to_coords(1, 1)
    print(f"  Position: {coords}")
    
    # Test all cells
    print("\n[TEST] All cell coordinates:")
    for row in range(3):
        for col in range(3):
            coords = mapper.index_to_coords(row, col)
            in_ws = "✓" if mapper.is_in_workspace(coords) else "✗"
            print(f"  Cell ({row},{col}): {coords} {in_ws}")
    
    # Test linear index
    print("\n[TEST] Linear index mapping:")
    for idx in range(9):
        coords = mapper.linear_index_to_coords(idx)
        print(f"  Index {idx}: {coords}")
    
    # Test trajectory
    print("\n[TEST] Pick and place trajectory for cell (0, 2):")
    trajectory = mapper.get_pick_and_place_trajectory(0, 2)
    for i, wp in enumerate(trajectory):
        print(f"  Step {i}: {wp['action']} -> {wp['position']}")
    
    print("\n[INFO] Coordinate mapping test completed")


if __name__ == "__main__":
    test_coordinate_mapping()
