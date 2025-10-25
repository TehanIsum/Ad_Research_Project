"""
Real-time Age/Gender/Mood-based Ad Recommendation System

Main application that:
1. Opens webcam and captures frames
2. Every 5 seconds, detects face and predicts demographics
3. Recommends best-fit ad based on demographics
4. Logs detection events to CSV
5. Displays recommendations in terminal

Usage:
    python main.py

Stop with Ctrl+C for graceful exit.
"""

import cv2
import time
import os
import sys
from datetime import datetime
import pandas as pd
from utils.capture import WebcamCapture, check_camera_availability
from utils.models import FaceEstimator
from recommender import AdRecommender


# Configuration
DETECTION_INTERVAL = 5  # seconds between detections
ADS_CSV_PATH = 'data/ads.csv'
LOGS_DIR = 'logs'
DETECTIONS_LOG = os.path.join(LOGS_DIR, 'detections.csv')
SHOW_CAMERA_WINDOW = True  # Set to True to show live camera feed with overlays


def initialize_logs():
    """Create logs directory and initialize detections CSV if needed."""
    # Create logs directory
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
        print(f"✓ Created logs directory: {LOGS_DIR}")
    
    # Initialize detections log with headers if it doesn't exist
    if not os.path.exists(DETECTIONS_LOG):
        df = pd.DataFrame(columns=['timestamp', 'age_group', 'gender', 'mood', 'ad_id_shown'])
        df.to_csv(DETECTIONS_LOG, index=False)
        print(f"✓ Initialized detections log: {DETECTIONS_LOG}")


def log_detection(detection: dict, ad_id: str):
    """
    Log a detection event to CSV.
    
    Args:
        detection: Dictionary with age_group, gender, mood
        ad_id: ID of the recommended ad
    """
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        log_entry = {
            'timestamp': timestamp,
            'age_group': detection.get('age_group', 'unknown'),
            'gender': detection.get('gender', 'unknown'),
            'mood': detection.get('mood', 'unknown'),
            'ad_id_shown': ad_id
        }
        
        # Append to CSV
        df = pd.DataFrame([log_entry])
        df.to_csv(DETECTIONS_LOG, mode='a', header=False, index=False)
        
    except Exception as e:
        print(f"Warning: Failed to log detection: {e}")


def print_detection_result(detection: dict, ad: pd.Series):
    """
    Print formatted detection and recommendation to terminal.
    
    Args:
        detection: Dictionary with demographic data
        ad: Pandas Series with ad information
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print("\n" + "=" * 65)
    print(f"Detection Event @ {timestamp}")
    print("=" * 65)
    
    print("\nDetected Demographics:")
    print(f"  Age Group: {detection['age_group']}")
    print(f"  Gender: {detection['gender']}")
    print(f"  Mood: {detection['mood']}")
    
    print("\nRecommended Ad:")
    print(f"  ID: {ad['ad_id']}")
    print(f"  Title: {ad['title']}")
    print(f"  Description: {ad['description']}")
    print(f"  Target: {ad['target_age_groups']}, {ad['target_genders']}, {ad['target_moods']}")
    print(f"  Priority: {ad['priority']}")
    
    print("=" * 65)


def draw_detection_overlay(frame, detection: dict, ad: pd.Series, face_box=None):
    """
    Draw detection results and ad recommendation on the frame.
    
    Args:
        frame: The video frame to draw on
        detection: Dictionary with demographic data
        ad: Pandas Series with ad information
        face_box: Face bounding box [x, y, width, height] (optional)
    
    Returns:
        Frame with overlays
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    # Draw face bounding box if available
    if face_box is not None:
        x, y, width, height = face_box
        # Draw rectangle around face
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 0), 2)
        
        # Draw label above face
        label = f"{detection['gender']}, {detection['age_group']}, {detection['mood']}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(overlay, (x, y - label_size[1] - 10), 
                     (x + label_size[0], y), (0, 255, 0), -1)
        cv2.putText(overlay, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Create semi-transparent info panel at bottom
    panel_height = 140
    panel_y = h - panel_height
    cv2.rectangle(overlay, (0, panel_y), (w, h), (0, 0, 0), -1)
    
    # Add transparency
    alpha = 0.7
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    # Add detection info text
    y_offset = panel_y + 25
    cv2.putText(frame, f"Demographics: {detection['gender']}, {detection['age_group']}, {detection['mood']}", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add ad recommendation
    y_offset += 30
    cv2.putText(frame, f"Recommended Ad: {ad['ad_id']}", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    y_offset += 25
    title = ad['title'][:50] + "..." if len(ad['title']) > 50 else ad['title']
    cv2.putText(frame, f"Title: {title}", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    y_offset += 25
    cv2.putText(frame, f"Priority: {ad['priority']}", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def draw_waiting_overlay(frame, next_detection_in: float):
    """
    Draw overlay when waiting for next detection.
    
    Args:
        frame: The video frame to draw on
        next_detection_in: Seconds until next detection
    
    Returns:
        Frame with overlay
    """
    h, w = frame.shape[:2]
    
    # Add status text
    status = f"Next detection in: {next_detection_in:.1f}s"
    cv2.putText(frame, status, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add instructions
    cv2.putText(frame, "Press 'q' or Ctrl+C to quit", (10, h - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame


def main():
    """Main application loop."""
    print("\n" + "=" * 65)
    print("Real-time Age/Gender/Mood-based Ad Recommendation System")
    print("=" * 65)
    print("\nInitializing system components...\n")
    
    # Check camera availability
    if not check_camera_availability(0):
        print("✗ ERROR: Camera not found or not accessible!")
        print("\nTroubleshooting:")
        print("  1. Check that your camera is connected")
        print("  2. Ensure camera permissions are granted:")
        print("     macOS: System Preferences → Security & Privacy → Privacy → Camera")
        print("  3. Close other apps using the camera (Zoom, Skype, etc.)")
        sys.exit(1)
    
    # Initialize logs
    initialize_logs()
    
    # Initialize face estimator
    print("\nInitializing face detection and demographic estimation...")
    estimator = FaceEstimator()
    
    # Initialize ad recommender
    print(f"\nLoading ad inventory from {ADS_CSV_PATH}...")
    recommender = AdRecommender(ADS_CSV_PATH)
    
    if recommender.ads_df is None:
        print("✗ ERROR: Failed to load ads. Please check ads.csv exists.")
        sys.exit(1)
    
    # Open webcam
    print("\nOpening webcam...")
    camera = WebcamCapture(0)
    
    if not camera.open():
        print("✗ ERROR: Failed to open camera!")
        sys.exit(1)
    
    print("\n" + "=" * 65)
    print("System Ready!")
    print("=" * 65)
    print(f"\n→ Detection interval: {DETECTION_INTERVAL} seconds")
    print("→ Position yourself in front of the camera")
    if SHOW_CAMERA_WINDOW:
        print("→ Camera window will show live feed with detection overlays")
        print("→ Press 'q' in camera window or Ctrl+C to stop\n")
    else:
        print("→ Press Ctrl+C to stop\n")
    
    # Main detection loop
    last_detection_time = 0
    frame_count = 0
    current_detection = None
    current_ad = None
    current_face_box = None
    
    try:
        while True:
            # Read frame
            success, frame = camera.read()
            
            if not success:
                print("\nWarning: Failed to read frame from camera")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            current_time = time.time()
            
            # Check if it's time for detection
            if current_time - last_detection_time >= DETECTION_INTERVAL:
                print(f"\n[Frame {frame_count}] Running detection...")
                
                # Detect faces first to get bounding box
                faces = estimator.detect_faces(frame)
                
                if not faces:
                    print("  → No face detected. Please face the camera.")
                    current_detection = None
                    current_ad = None
                    current_face_box = None
                else:
                    # Get largest face
                    largest_face = estimator.get_largest_face(faces)
                    current_face_box = largest_face['box']
                    
                    # Analyze frame for demographics
                    detection = estimator.analyze_frame(frame)
                    
                    if detection is not None:
                        print(f"  → Face detected: {detection['age_group']}, "
                              f"{detection['gender']}, {detection['mood']}")
                        
                        # Get ad recommendation
                        recommended_ad = recommender.recommend_ad(detection)
                        
                        if recommended_ad is not None:
                            # Store current detection results
                            current_detection = detection
                            current_ad = recommended_ad
                            
                            # Print results
                            print_detection_result(detection, recommended_ad)
                            
                            # Log event
                            log_detection(detection, recommended_ad['ad_id'])
                        else:
                            print("  → No ad recommendation available")
                            current_detection = detection
                            current_ad = None
                
                last_detection_time = current_time
            
            # Show camera window with overlays if enabled
            if SHOW_CAMERA_WINDOW:
                display_frame = frame.copy()
                
                # Draw detection results if available
                if current_detection is not None and current_ad is not None:
                    display_frame = draw_detection_overlay(
                        display_frame, current_detection, current_ad, current_face_box
                    )
                else:
                    # Show countdown to next detection
                    time_until_next = DETECTION_INTERVAL - (current_time - last_detection_time)
                    display_frame = draw_waiting_overlay(display_frame, time_until_next)
                
                # Display frame
                cv2.imshow('Ad Recommendation System - Live Feed', display_frame)
                
                # Check for 'q' key press to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n\nQuitting (q pressed)...")
                    break
            else:
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 65)
        print("Shutting down gracefully...")
        print("=" * 65)
    
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        camera.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ Processed {frame_count} frames")
        print(f"✓ Detection log saved to: {DETECTIONS_LOG}")
        print("\nThank you for using the Ad Recommendation System!\n")


if __name__ == '__main__':
    main()
