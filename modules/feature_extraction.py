"""
Module 4: Feature Extraction
Extracts color features from board cells for K-means classification
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class FeatureExtractor:
    """
    Extracts color-based features from board cell images for classification.
    Supports multiple feature types: RGB mean, HSV mean, color histogram.
    """
    
    def __init__(self, feature_type: str = config.FEATURE_TYPE):
        """
        Initialize feature extractor.
        
        Args:
            feature_type: Type of features to extract
                - 'rgb_mean': Mean RGB values (3 features)
                - 'hsv_mean': Mean HSV values (3 features)
                - 'histogram': Color histogram (configurable bins)
                - 'combined': RGB + HSV means (6 features)
        """
        self.feature_type = feature_type
        self.histogram_bins = 8  # Bins per channel for histogram
        
    def extract_rgb_mean(self, cell: np.ndarray) -> np.ndarray:
        """
        Extract mean RGB values from cell.
        
        Args:
            cell: Cell image (BGR)
            
        Returns:
            Array of [mean_B, mean_G, mean_R]
        """
        mean_vals = np.mean(cell, axis=(0, 1))
        return mean_vals
    
    def extract_hsv_mean(self, cell: np.ndarray) -> np.ndarray:
        """
        Extract mean HSV values from cell.
        
        Args:
            cell: Cell image (BGR)
            
        Returns:
            Array of [mean_H, mean_S, mean_V]
        """
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
        mean_vals = np.mean(hsv, axis=(0, 1))
        return mean_vals
    
    def extract_histogram(self, cell: np.ndarray) -> np.ndarray:
        """
        Extract color histogram features from cell.
        
        Args:
            cell: Cell image (BGR)
            
        Returns:
            Flattened histogram array
        """
        hist_features = []
        
        # Calculate histogram for each channel
        for i in range(3):  # B, G, R channels
            hist = cv2.calcHist([cell], [i], None, 
                               [self.histogram_bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.extend(hist)
        
        return np.array(hist_features)
    
    def extract_hsv_histogram(self, cell: np.ndarray) -> np.ndarray:
        """
        Extract HSV histogram features from cell.
        
        Args:
            cell: Cell image (BGR)
            
        Returns:
            Flattened HSV histogram array
        """
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
        hist_features = []
        
        # H channel: 0-180 in OpenCV
        h_hist = cv2.calcHist([hsv], [0], None, [self.histogram_bins], [0, 180])
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        hist_features.extend(h_hist)
        
        # S channel: 0-255
        s_hist = cv2.calcHist([hsv], [1], None, [self.histogram_bins], [0, 256])
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        hist_features.extend(s_hist)
        
        # V channel: 0-255
        v_hist = cv2.calcHist([hsv], [2], None, [self.histogram_bins], [0, 256])
        v_hist = cv2.normalize(v_hist, v_hist).flatten()
        hist_features.extend(v_hist)
        
        return np.array(hist_features)
    
    def extract_color_stats(self, cell: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive color statistics.
        
        Args:
            cell: Cell image (BGR)
            
        Returns:
            Array of color statistics [mean_rgb, std_rgb, mean_hsv]
        """
        # RGB statistics
        rgb_mean = np.mean(cell, axis=(0, 1))
        rgb_std = np.std(cell, axis=(0, 1))
        
        # HSV statistics
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
        hsv_mean = np.mean(hsv, axis=(0, 1))
        
        return np.concatenate([rgb_mean, rgb_std, hsv_mean])
    
    def extract_features(self, cell: np.ndarray) -> np.ndarray:
        """
        Extract features based on configured feature type.
        
        Args:
            cell: Cell image (BGR)
            
        Returns:
            Feature vector
        """
        if cell is None or cell.size == 0:
            return np.zeros(self._get_feature_size())
        
        if self.feature_type == 'rgb_mean':
            return self.extract_rgb_mean(cell)
        elif self.feature_type == 'hsv_mean':
            return self.extract_hsv_mean(cell)
        elif self.feature_type == 'histogram':
            return self.extract_histogram(cell)
        elif self.feature_type == 'hsv_histogram':
            return self.extract_hsv_histogram(cell)
        elif self.feature_type == 'combined':
            rgb = self.extract_rgb_mean(cell)
            hsv = self.extract_hsv_mean(cell)
            return np.concatenate([rgb, hsv])
        elif self.feature_type == 'color_stats':
            return self.extract_color_stats(cell)
        else:
            # Default to HSV mean
            return self.extract_hsv_mean(cell)
    
    def _get_feature_size(self) -> int:
        """Get the size of feature vector based on feature type."""
        sizes = {
            'rgb_mean': 3,
            'hsv_mean': 3,
            'histogram': self.histogram_bins * 3,
            'hsv_histogram': self.histogram_bins * 3,
            'combined': 6,
            'color_stats': 9
        }
        return sizes.get(self.feature_type, 3)
    
    def extract_all_cells(self, cells: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from all cells.
        
        Args:
            cells: List of 9 cell images
            
        Returns:
            Feature matrix of shape (9, n_features)
        """
        if len(cells) != 9:
            print(f"[WARNING] Expected 9 cells, got {len(cells)}")
        
        features = []
        for cell in cells:
            feature = self.extract_features(cell)
            features.append(feature)
        
        return np.array(features)
    
    def detect_piece_by_color(self, cell: np.ndarray) -> int:
        """
        Detect piece type using color thresholds (backup method).
        
        Args:
            cell: Cell image (BGR)
            
        Returns:
            0: empty, 1: robot piece, -1: player piece
        """
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)
        
        # Check for player color (green)
        lower_green = np.array(config.HSV_GREEN_LOWER)
        upper_green = np.array(config.HSV_GREEN_UPPER)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_pixels = np.sum(green_mask > 0)
        
        # Check for robot color (red)
        lower_red1 = np.array(config.HSV_RED_LOWER1)
        upper_red1 = np.array(config.HSV_RED_UPPER1)
        lower_red2 = np.array(config.HSV_RED_LOWER2)
        upper_red2 = np.array(config.HSV_RED_UPPER2)
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = red_mask1 | red_mask2
        red_pixels = np.sum(red_mask > 0)
        
        # Threshold for detection
        total_pixels = cell.shape[0] * cell.shape[1]
        threshold = 0.1 * total_pixels  # 10% of cell area
        
        if red_pixels > threshold:
            return config.ROBOT  # Robot piece
        elif green_pixels > threshold:
            return config.PLAYER  # Player piece
        else:
            return config.EMPTY  # Empty cell
    
    def visualize_features(self, cell: np.ndarray) -> np.ndarray:
        """
        Create visualization of extracted features.
        
        Args:
            cell: Cell image (BGR)
            
        Returns:
            Visualization image
        """
        features = self.extract_features(cell)
        
        # Create visualization
        vis = np.zeros((100, 200, 3), dtype=np.uint8)
        
        # Draw feature bars
        n_features = len(features)
        bar_width = 200 // n_features
        
        for i, val in enumerate(features):
            # Normalize to 0-100
            height = int(val) if self.feature_type in ['rgb_mean', 'hsv_mean'] else int(val * 100)
            height = min(height, 100)
            
            x1 = i * bar_width
            x2 = x1 + bar_width - 2
            y1 = 100 - height
            y2 = 100
            
            color = (0, int(255 * i / n_features), 255 - int(255 * i / n_features))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, -1)
        
        return vis


def test_feature_extraction():
    """
    Test function for feature extraction module.
    """
    print("=" * 50)
    print("Testing Feature Extraction Module")
    print("=" * 50)
    
    extractor = FeatureExtractor(feature_type='hsv_mean')
    
    # Create test cells
    # Empty cell (gray)
    empty_cell = np.full((80, 80, 3), (150, 150, 150), dtype=np.uint8)
    
    # Player cell (green)
    player_cell = np.full((80, 80, 3), (0, 200, 0), dtype=np.uint8)
    
    # Robot cell (red)
    robot_cell = np.full((80, 80, 3), (0, 0, 200), dtype=np.uint8)
    
    cells = [empty_cell, player_cell, robot_cell,
             empty_cell.copy(), empty_cell.copy(), player_cell.copy(),
             robot_cell.copy(), empty_cell.copy(), empty_cell.copy()]
    
    # Extract features
    features = extractor.extract_all_cells(cells)
    
    print(f"\n[INFO] Feature type: {extractor.feature_type}")
    print(f"[INFO] Feature shape: {features.shape}")
    print(f"\n[INFO] Features for each cell:")
    
    labels = ['Empty', 'Player', 'Robot', 'Empty', 'Empty', 
              'Player', 'Robot', 'Empty', 'Empty']
    for i, (feat, label) in enumerate(zip(features, labels)):
        print(f"  Cell {i} ({label}): {feat}")
    
    # Test color-based detection
    print("\n[INFO] Color-based detection:")
    for i, cell in enumerate(cells[:3]):
        result = extractor.detect_piece_by_color(cell)
        result_map = {config.EMPTY: 'Empty', config.PLAYER: 'Player', config.ROBOT: 'Robot'}
        print(f"  Cell {i}: Detected as {result_map.get(result, 'Unknown')}")
    
    if config.SHOW_DEBUG_WINDOWS:
        # Show cells
        combined = np.hstack([empty_cell, player_cell, robot_cell])
        cv2.imshow("Test Cells (Empty, Player, Robot)", combined)
        
        # Show feature visualizations
        vis_empty = extractor.visualize_features(empty_cell)
        vis_player = extractor.visualize_features(player_cell)
        vis_robot = extractor.visualize_features(robot_cell)
        vis_combined = np.hstack([vis_empty, vis_player, vis_robot])
        cv2.imshow("Feature Visualizations", vis_combined)
        
        print("\nPress any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("\n[INFO] Feature extraction test completed")


if __name__ == "__main__":
    test_feature_extraction()
