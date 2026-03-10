"""
Module 3: Board Detection
Detects tic-tac-toe board and extracts individual cells
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from modules.image_preprocessing import ImagePreprocessor


class BoardDetector:
    """
    Detects and extracts the tic-tac-toe board from camera images.
    Uses contour analysis to find the board boundaries and divides it into 9 cells.
    """
    
    def __init__(self,
                 min_area: int = config.MIN_BOARD_AREA,
                 max_area: int = config.MAX_BOARD_AREA,
                 aspect_ratio_min: float = config.BOARD_ASPECT_RATIO_MIN,
                 aspect_ratio_max: float = config.BOARD_ASPECT_RATIO_MAX,
                 cell_padding: int = config.CELL_PADDING):
        """
        Initialize board detector.
        
        Args:
            min_area: Minimum contour area to consider as board
            max_area: Maximum contour area to consider as board
            aspect_ratio_min: Minimum aspect ratio (for square detection)
            aspect_ratio_max: Maximum aspect ratio (for square detection)
            cell_padding: Padding inside each cell
        """
        self.min_area = min_area
        self.max_area = max_area
        self.aspect_ratio_min = aspect_ratio_min
        self.aspect_ratio_max = aspect_ratio_max
        self.cell_padding = cell_padding
        
        self.preprocessor = ImagePreprocessor()
        self.board_corners = None
        self.warped_board = None
        self.transform_matrix = None
        self.cell_regions = None
    
    def detect_board(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Detect the tic-tac-toe board in the image.
        
        Args:
            image: Input BGR image
            
        Returns:
            Tuple of (success, board_corners)
        """
        # Preprocess image
        results = self.preprocessor.preprocess(image)
        contours = results['contours']
        
        # Find largest quadrilateral that matches board criteria
        board_contour = None
        max_area = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Check area bounds
            if area < self.min_area or area > self.max_area:
                continue
            
            # Approximate contour to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Check if it's a quadrilateral
            if len(approx) == 4:
                # Check aspect ratio
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / float(h) if h > 0 else 0
                
                if self.aspect_ratio_min <= aspect_ratio <= self.aspect_ratio_max:
                    if area > max_area:
                        max_area = area
                        board_contour = approx
        
        if board_contour is None:
            return False, None
        
        self.board_corners = board_contour.reshape(4, 2)
        return True, self.board_corners
    
    def get_warped_board(self, image: np.ndarray, 
                         corners: np.ndarray = None,
                         output_size: int = 300) -> Optional[np.ndarray]:
        """
        Get bird's-eye view of the board.
        
        Args:
            image: Input BGR image
            corners: Board corners (uses stored corners if None)
            output_size: Size of output square image
            
        Returns:
            Warped board image
        """
        if corners is None:
            corners = self.board_corners
        
        if corners is None:
            print("[WARNING] No board corners available")
            return None
        
        self.warped_board, self.transform_matrix = \
            self.preprocessor.get_perspective_transform(
                image, corners, output_size
            )
        
        return self.warped_board
    
    def get_cells(self, warped_board: np.ndarray = None) -> List[np.ndarray]:
        """
        Divide the board into 9 cells.
        
        Args:
            warped_board: Warped board image (uses stored if None)
            
        Returns:
            List of 9 cell images (row-major order)
        """
        if warped_board is None:
            warped_board = self.warped_board
        
        if warped_board is None:
            print("[WARNING] No warped board available")
            return []
        
        height, width = warped_board.shape[:2]
        cell_h = height // 3
        cell_w = width // 3
        
        cells = []
        self.cell_regions = []
        
        for i in range(3):
            for j in range(3):
                # Calculate cell bounds with padding
                y1 = i * cell_h + self.cell_padding
                y2 = (i + 1) * cell_h - self.cell_padding
                x1 = j * cell_w + self.cell_padding
                x2 = (j + 1) * cell_w - self.cell_padding
                
                # Extract cell
                cell = warped_board[y1:y2, x1:x2].copy()
                cells.append(cell)
                
                # Store cell region for later use
                self.cell_regions.append({
                    'index': (i, j),
                    'bounds': (x1, y1, x2, y2),
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                })
        
        return cells
    
    def get_cell_centers_original(self) -> List[Tuple[int, int]]:
        """
        Get cell centers in original image coordinates.
        
        Returns:
            List of 9 (x, y) coordinates in original image space
        """
        if self.cell_regions is None or self.transform_matrix is None:
            return []
        
        centers = []
        for cell in self.cell_regions:
            # Transform warped center back to original
            warped_center = cell['center']
            original_center = self.preprocessor.inverse_perspective_transform(
                warped_center, self.transform_matrix
            )
            centers.append(original_center)
        
        return centers
    
    def draw_board_overlay(self, image: np.ndarray, 
                           corners: np.ndarray = None,
                           color: Tuple[int, int, int] = config.COLOR_BOARD,
                           thickness: int = 2) -> np.ndarray:
        """
        Draw board outline on image.
        
        Args:
            image: Input image
            corners: Board corners (uses stored if None)
            color: Line color (BGR)
            thickness: Line thickness
            
        Returns:
            Image with board overlay
        """
        output = image.copy()
        
        if corners is None:
            corners = self.board_corners
        
        if corners is None:
            return output
        
        # Draw board outline
        pts = corners.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(output, [pts], True, color, thickness)
        
        # Draw corner points
        for corner in corners:
            cv2.circle(output, tuple(corner.astype(int)), 5, (0, 0, 255), -1)
        
        return output
    
    def draw_grid_overlay(self, image: np.ndarray,
                           color: Tuple[int, int, int] = config.COLOR_BOARD,
                           thickness: int = 1) -> np.ndarray:
        """
        Draw 3x3 grid overlay on warped board image.
        
        Args:
            image: Warped board image
            color: Line color (BGR)
            thickness: Line thickness
            
        Returns:
            Image with grid overlay
        """
        output = image.copy()
        height, width = output.shape[:2]
        
        cell_h = height // 3
        cell_w = width // 3
        
        # Draw vertical lines
        for i in range(1, 3):
            cv2.line(output, (i * cell_w, 0), (i * cell_w, height), color, thickness)
        
        # Draw horizontal lines
        for i in range(1, 3):
            cv2.line(output, (0, i * cell_h), (width, i * cell_h), color, thickness)
        
        # Draw cell indices
        for idx, cell in enumerate(self.cell_regions or []):
            center = cell['center']
            cv2.putText(output, str(idx), 
                       (center[0] - 10, center[1] + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return output
    
    def detect_board_markers(self, image: np.ndarray, 
                              marker_type: str = 'white') -> Optional[np.ndarray]:
        """
        Detect board using color markers (alternative method).
        
        Args:
            image: Input BGR image
            marker_type: Color of board markers ('white', 'green', 'red')
            
        Returns:
            Board corners if found
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for different markers
        if marker_type == 'white':
            lower = np.array([0, 0, 200])
            upper = np.array([180, 50, 255])
        elif marker_type == 'green':
            lower = np.array([40, 100, 20])
            upper = np.array([70, 255, 150])
        elif marker_type == 'red':
            lower = np.array([0, 100, 100])
            upper = np.array([10, 255, 255])
        else:
            return None
        
        mask = cv2.inRange(hsv, lower, upper)
        
        # Find contours in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the 4 largest contours (corners)
        if len(contours) < 4:
            return None
        
        # Sort by area and get top 4
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]
        
        # Get centers of each contour
        centers = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append([cx, cy])
        
        if len(centers) != 4:
            return None
        
        self.board_corners = np.array(centers, dtype=np.float32)
        return self.board_corners


def test_board_detection():
    """
    Test function for board detection module.
    """
    print("=" * 50)
    print("Testing Board Detection Module")
    print("=" * 50)
    
    # Create a test image with a tic-tac-toe board
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (50, 50, 50)  # Gray background
    
    # Draw a board (perspective view)
    board_pts = np.array([[150, 100], [490, 80], [520, 380], [120, 400]], dtype=np.int32)
    
    # Fill board background
    cv2.fillPoly(test_image, [board_pts], (200, 200, 200))
    
    # Draw grid lines
    cv2.polylines(test_image, [board_pts], True, (0, 0, 0), 2)
    
    detector = BoardDetector()
    
    # Detect board
    success, corners = detector.detect_board(test_image)
    print(f"\n[INFO] Board detected: {success}")
    
    if success:
        print(f"[INFO] Board corners:\n{corners}")
        
        # Get warped board
        warped = detector.get_warped_board(test_image, corners)
        
        # Get cells
        cells = detector.get_cells(warped)
        print(f"[INFO] Extracted {len(cells)} cells")
        
        # Draw overlays
        overlay = detector.draw_board_overlay(test_image)
        
        if config.SHOW_DEBUG_WINDOWS:
            cv2.imshow("Original with Overlay", overlay)
            cv2.imshow("Warped Board", detector.draw_grid_overlay(warped))
            
            # Show individual cells
            for i, cell in enumerate(cells[:3]):  # Show first 3 cells
                cv2.imshow(f"Cell {i}", cell)
            
            print("\nPress any key to close windows...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    print("\n[INFO] Board detection test completed")


if __name__ == "__main__":
    test_board_detection()
