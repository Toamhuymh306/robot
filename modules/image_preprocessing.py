"""
Module 2: Image Preprocessing
Handles image preprocessing: grayscale, blur, edge detection, perspective transform
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ImagePreprocessor:
    """
    Image preprocessing pipeline for board detection and analysis.
    Supports grayscale conversion, Gaussian blur, edge detection,
    and perspective transformation.
    """
    
    def __init__(self,
                 blur_kernel: int = config.BLUR_KERNEL_SIZE,
                 canny_threshold1: int = config.CANNY_THRESHOLD1,
                 canny_threshold2: int = config.CANNY_THRESHOLD2,
                 dilate_iterations: int = config.DILATE_ITERATIONS,
                 erode_iterations: int = config.ERODE_ITERATIONS):
        """
        Initialize image preprocessor.
        
        Args:
            blur_kernel: Size of Gaussian blur kernel (must be odd)
            canny_threshold1: Lower threshold for Canny edge detection
            canny_threshold2: Upper threshold for Canny edge detection
            dilate_iterations: Number of dilation iterations
            erode_iterations: Number of erosion iterations
        """
        self.blur_kernel = blur_kernel
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
        self.dilate_iterations = dilate_iterations
        self.erode_iterations = erode_iterations
        
        # Morphological kernel
        self.morph_kernel = np.ones((3, 3), np.uint8)
    
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale.
        
        Args:
            image: Input BGR image
            
        Returns:
            Grayscale image
        """
        if len(image.shape) == 2:
            return image  # Already grayscale
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def apply_blur(self, image: np.ndarray, kernel_size: int = None) -> np.ndarray:
        """
        Apply Gaussian blur to reduce noise.
        
        Args:
            image: Input image
            kernel_size: Optional custom kernel size
            
        Returns:
            Blurred image
        """
        k = kernel_size if kernel_size else self.blur_kernel
        # Ensure kernel size is odd
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(image, (k, k), 0)
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Canny edge detection.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Edge-detected binary image
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            image = self.to_grayscale(image)
            
        edges = cv2.Canny(image, self.canny_threshold1, self.canny_threshold2)
        return edges
    
    def apply_morphology(self, image: np.ndarray, 
                         dilate_iter: int = None,
                         erode_iter: int = None) -> np.ndarray:
        """
        Apply morphological operations (dilation and erosion).
        
        Args:
            image: Input binary image
            dilate_iter: Number of dilation iterations
            erode_iter: Number of erosion iterations
            
        Returns:
            Processed binary image
        """
        d_iter = dilate_iter if dilate_iter is not None else self.dilate_iterations
        e_iter = erode_iter if erode_iter is not None else self.erode_iterations
        
        # Dilate to connect broken edges
        dilated = cv2.dilate(image, self.morph_kernel, iterations=d_iter)
        
        # Erode to remove noise
        eroded = cv2.erode(dilated, self.morph_kernel, iterations=e_iter)
        
        return eroded
    
    def find_contours(self, image: np.ndarray) -> List:
        """
        Find contours in a binary image.
        
        Args:
            image: Input binary image
            
        Returns:
            List of contours
        """
        contours, _ = cv2.findContours(
            image, 
            cv2.RETR_TREE, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contours
    
    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in consistent order: top-left, top-right, bottom-right, bottom-left.
        
        Args:
            pts: Array of 4 points
            
        Returns:
            Ordered array of 4 points
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum of coordinates: top-left has smallest, bottom-right has largest
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Difference of coordinates: top-right has smallest, bottom-left has largest
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def get_perspective_transform(self, 
                                   image: np.ndarray, 
                                   corners: np.ndarray,
                                   output_size: int = 300) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply perspective transformation to get bird's eye view.
        
        Args:
            image: Input image
            corners: Four corner points of the region to transform
            output_size: Size of output square image
            
        Returns:
            Tuple of (warped image, transformation matrix)
        """
        # Order corners consistently
        ordered_corners = self.order_points(corners.reshape(4, 2))
        
        # Define destination points
        dst = np.array([
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        M = cv2.getPerspectiveTransform(ordered_corners, dst)
        
        # Apply transformation
        warped = cv2.warpPerspective(image, M, (output_size, output_size))
        
        return warped, M
    
    def inverse_perspective_transform(self,
                                       point: Tuple[int, int],
                                       M: np.ndarray) -> Tuple[int, int]:
        """
        Transform a point from warped space back to original image space.
        
        Args:
            point: Point in warped image (x, y)
            M: Original transformation matrix
            
        Returns:
            Point in original image space (x, y)
        """
        # Invert the matrix
        M_inv = np.linalg.inv(M)
        
        # Create homogeneous coordinates
        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
        
        # Transform
        transformed = cv2.perspectiveTransform(pt, M_inv)
        
        return int(transformed[0][0][0]), int(transformed[0][0][1])
    
    def preprocess(self, image: np.ndarray, 
                   return_intermediate: bool = False) -> dict:
        """
        Apply full preprocessing pipeline.
        
        Args:
            image: Input BGR image
            return_intermediate: If True, return all intermediate results
            
        Returns:
            Dictionary containing preprocessed images
        """
        results = {'original': image.copy()}
        
        # Grayscale
        gray = self.to_grayscale(image)
        results['grayscale'] = gray
        
        # Blur
        blurred = self.apply_blur(gray)
        results['blurred'] = blurred
        
        # Edge detection
        edges = self.detect_edges(blurred)
        results['edges'] = edges
        
        # Morphological operations
        morphed = self.apply_morphology(edges)
        results['morphed'] = morphed
        
        # Find contours
        contours = self.find_contours(morphed)
        results['contours'] = contours
        
        if not return_intermediate:
            return {
                'processed': morphed,
                'contours': contours
            }
        
        return results
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Contrast-enhanced image
        """
        if len(image.shape) == 3:
            image = self.to_grayscale(image)
            
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    
    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive thresholding.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Binary threshold image
        """
        if len(image.shape) == 3:
            image = self.to_grayscale(image)
            
        return cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )


def test_preprocessing():
    """
    Test function for image preprocessing module.
    """
    print("=" * 50)
    print("Testing Image Preprocessing Module")
    print("=" * 50)
    
    # Create a test image (checkerboard pattern)
    test_image = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Draw a simple grid
    for i in range(3):
        for j in range(3):
            color = (200, 200, 200) if (i + j) % 2 == 0 else (100, 100, 100)
            x1, y1 = j * 100, i * 100
            x2, y2 = x1 + 100, y1 + 100
            cv2.rectangle(test_image, (x1, y1), (x2, y2), color, -1)
    
    # Draw grid lines
    for i in range(4):
        cv2.line(test_image, (i * 100, 0), (i * 100, 300), (0, 255, 0), 2)
        cv2.line(test_image, (0, i * 100), (300, i * 100), (0, 255, 0), 2)
    
    preprocessor = ImagePreprocessor()
    
    # Apply preprocessing
    results = preprocessor.preprocess(test_image, return_intermediate=True)
    
    print(f"\n[INFO] Original shape: {test_image.shape}")
    print(f"[INFO] Grayscale shape: {results['grayscale'].shape}")
    print(f"[INFO] Contours found: {len(results['contours'])}")
    
    # Display results
    if config.SHOW_DEBUG_WINDOWS:
        cv2.imshow("Original", results['original'])
        cv2.imshow("Grayscale", results['grayscale'])
        cv2.imshow("Blurred", results['blurred'])
        cv2.imshow("Edges", results['edges'])
        cv2.imshow("Morphed", results['morphed'])
        
        print("\nPress any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("\n[INFO] Preprocessing test completed")


if __name__ == "__main__":
    test_preprocessing()
