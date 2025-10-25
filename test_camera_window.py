#!/usr/bin/env python3
"""
Quick test script to verify camera window functionality.
Tests camera access and OpenCV display without running full detection.
"""

import cv2
import sys

def test_camera_window():
    """Test if camera can be opened and window can be displayed."""
    print("=" * 60)
    print("Camera Window Test")
    print("=" * 60)
    
    # Try to open camera
    print("\n1. Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ ERROR: Cannot access camera!")
        print("\nTroubleshooting:")
        print("  - Check camera permissions (System Preferences → Security)")
        print("  - Close other apps using camera")
        print("  - Try unplugging/replugging camera")
        return False
    
    print("✓ Camera opened successfully")
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Get actual resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ Resolution: {width}x{height}")
    
    # Test reading frames
    print("\n2. Testing frame capture...")
    ret, frame = cap.read()
    
    if not ret or frame is None:
        print("✗ ERROR: Cannot read frames from camera!")
        cap.release()
        return False
    
    print(f"✓ Frame captured: {frame.shape}")
    
    # Test window display
    print("\n3. Testing window display...")
    print("\nA camera window should open showing live feed.")
    print("Press 'q' to close and complete test.\n")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Add test overlay
            cv2.putText(frame, "Camera Test - Press 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Draw test rectangle
            cv2.rectangle(frame, (200, 150), (440, 330), (0, 255, 0), 2)
            cv2.putText(frame, "Test Detection Area", 
                       (210, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Display frame
            cv2.imshow('Camera Window Test', frame)
            
            # Check for 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    
    except Exception as e:
        print(f"\n✗ ERROR during test: {e}")
        cap.release()
        cv2.destroyAllWindows()
        return False
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("✓ Camera window test completed successfully!")
    print("=" * 60)
    print(f"\nProcessed {frame_count} frames")
    print("\nYour camera and OpenCV are working correctly.")
    print("You can now run the full application: python main.py")
    print("=" * 60 + "\n")
    
    return True


if __name__ == '__main__':
    success = test_camera_window()
    sys.exit(0 if success else 1)
