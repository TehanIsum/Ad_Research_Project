"""
Webcam Capture Module

Provides WebcamCapture class for handling video capture from system camera
with error handling and resource management.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class WebcamCapture:
    """
    Manages webcam video capture with error handling.
    
    Provides methods to:
    - Open and initialize camera
    - Read frames safely
    - Check camera availability
    - Release resources properly
    """
    
    def __init__(self, camera_index: int = 0):
        """
        Initialize webcam capture.
        
        Args:
            camera_index: Camera device index (default: 0 for primary camera)
        """
        self.camera_index = camera_index
        self.cap = None
        self.is_opened = False
        
    def open(self) -> bool:
        """
        Open the camera device.
        
        Returns:
            True if camera opened successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                print(f"✗ Failed to open camera {self.camera_index}")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_opened = True
            print(f"✓ Camera {self.camera_index} opened successfully")
            
            # Read and discard first few frames (camera warmup)
            for _ in range(5):
                self.cap.read()
            
            return True
            
        except Exception as e:
            print(f"✗ Error opening camera: {e}")
            self.is_opened = False
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.
        
        Returns:
            Tuple of (success, frame):
            - success: True if frame was read successfully
            - frame: The captured frame as numpy array, or None if failed
        """
        if not self.is_opened or self.cap is None:
            return False, None
        
        try:
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                return False, None
            
            return True, frame
            
        except Exception as e:
            print(f"Error reading frame: {e}")
            return False, None
    
    def is_available(self) -> bool:
        """
        Check if camera is available and opened.
        
        Returns:
            True if camera is ready to capture, False otherwise
        """
        return self.is_opened and self.cap is not None and self.cap.isOpened()
    
    def release(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            print(f"✓ Camera {self.camera_index} released")
    
    def get_frame_size(self) -> Optional[Tuple[int, int]]:
        """
        Get current frame dimensions.
        
        Returns:
            Tuple of (width, height) or None if camera not opened
        """
        if not self.is_available():
            return None
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        return (width, height)
    
    def __enter__(self):
        """Context manager entry: open camera."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: release camera."""
        self.release()


def check_camera_availability(camera_index: int = 0) -> bool:
    """
    Check if a camera is available without keeping it open.
    
    Args:
        camera_index: Camera device index to check
        
    Returns:
        True if camera is available, False otherwise
    """
    try:
        cap = cv2.VideoCapture(camera_index)
        is_available = cap.isOpened()
        cap.release()
        return is_available
    except Exception:
        return False


# Example usage for testing
if __name__ == '__main__':
    print("WebcamCapture Module - Test Mode")
    print("=" * 60)
    
    # Check camera availability
    print("\nChecking camera availability...")
    if check_camera_availability(0):
        print("✓ Camera 0 is available")
    else:
        print("✗ Camera 0 is not available")
        exit(1)
    
    # Test capture with context manager
    print("\nTesting camera capture... Press 'q' to quit")
    
    with WebcamCapture(0) as camera:
        if not camera.is_available():
            print("✗ Failed to open camera")
            exit(1)
        
        print(f"Frame size: {camera.get_frame_size()}")
        
        frame_count = 0
        while True:
            success, frame = camera.read()
            
            if not success:
                print("Failed to read frame")
                break
            
            frame_count += 1
            
            # Display frame info
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Webcam Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print(f"\nCaptured {frame_count} frames")
    
    cv2.destroyAllWindows()
    print("\n✓ Test completed")
