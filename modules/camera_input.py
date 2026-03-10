"""
Module 1: Camera Input
Handles webcam capture using OpenCV
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import sys
import os

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class CameraInput:
    """
    Camera input handler for capturing video frames from webcam.
    Supports multiple camera indices and frame buffering.
    """
    
    def __init__(self, 
                 camera_id: int = config.CAMERA_ID,
                 width: int = config.CAMERA_WIDTH,
                 height: int = config.CAMERA_HEIGHT,
                 fps: int = config.CAMERA_FPS):
        """
        Initialize camera input.
        
        Args:
            camera_id: Camera index (0 for default webcam)
            width: Desired frame width
            height: Desired frame height
            fps: Desired frames per second
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """
        Initialize the camera capture.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                print(f"[ERROR] Cannot open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Verify actual settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            print(f"[INFO] Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            # Warm up camera (skip initial frames)
            self._warmup()
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"[ERROR] Camera initialization failed: {e}")
            return False
    
    def _warmup(self, num_frames: int = config.CAMERA_WARMUP_FRAMES):
        """
        Warm up camera by reading and discarding initial frames.
        
        Args:
            num_frames: Number of frames to skip
        """
        print(f"[INFO] Warming up camera ({num_frames} frames)...")
        for _ in range(num_frames):
            self.cap.read()
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Capture a single frame from the camera.
        
        Returns:
            Tuple of (success, frame) where frame is None if capture failed
        """
        if not self.is_initialized:
            print("[WARNING] Camera not initialized")
            return False, None
        
        ret, frame = self.cap.read()
        
        if not ret:
            print("[WARNING] Failed to capture frame")
            return False, None
        
        return True, frame
    
    def get_frame_continuous(self):
        """
        Generator for continuous frame capture.
        
        Yields:
            Frame captured from camera
        """
        while self.is_initialized:
            ret, frame = self.get_frame()
            if ret:
                yield frame
            else:
                break
    
    def display_frame(self, 
                      frame: np.ndarray, 
                      window_name: str = config.WINDOW_NAME,
                      scale: float = config.DISPLAY_SCALE) -> int:
        """
        Display a frame in a window.
        
        Args:
            frame: Frame to display
            window_name: Name of the display window
            scale: Scale factor for display
            
        Returns:
            Key pressed (or -1 if no key pressed)
        """
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        cv2.imshow(window_name, frame)
        return cv2.waitKey(1)
    
    def save_frame(self, frame: np.ndarray, filepath: str) -> bool:
        """
        Save a frame to file.
        
        Args:
            frame: Frame to save
            filepath: Path to save the frame
            
        Returns:
            True if save successful
        """
        try:
            cv2.imwrite(filepath, frame)
            print(f"[INFO] Frame saved to {filepath}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save frame: {e}")
            return False
    
    def release(self):
        """
        Release camera resources.
        """
        if self.cap is not None:
            self.cap.release()
            self.is_initialized = False
            print("[INFO] Camera released")
        cv2.destroyAllWindows()
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


def test_camera():
    """
    Test function for camera input module.
    """
    print("=" * 50)
    print("Testing Camera Input Module")
    print("=" * 50)
    
    with CameraInput() as camera:
        if not camera.is_initialized:
            print("[ERROR] Camera test failed - could not initialize")
            return
        
        print("\nPress 'q' to quit, 's' to save frame...")
        frame_count = 0
        
        for frame in camera.get_frame_continuous():
            frame_count += 1
            
            # Add frame counter overlay
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            key = camera.display_frame(frame)
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                camera.save_frame(frame, f"images/outputs/camera_test_{frame_count}.jpg")
    
    print("\n[INFO] Camera test completed")


if __name__ == "__main__":
    test_camera()
