"""
Phase 1: Pure Recording - Capture camera video and screen images without marker detection
Purpose: Maintain high FPS by avoiding real-time marker detection overhead
Output: Raw camera video + screen capture images
"""

import cv2
import numpy as np
from datetime import datetime
import os
import threading
import queue
import time
from PIL import ImageGrab

# Configuration
BUTTON_ON_COLOR = ([80, 150, 90], [120, 200, 120])  # HSV range
BUTTON_OFF_COLOR = ([20, 200, 200], [40, 255, 255])  # HSV range
BUTTON_THRESHOLD = 1000
CROP_COORDS = [(315, 170, 1822, 973), (550, 170, 1580, 920), (679, 172, 1458, 971),
               (757, 172, 1380, 972), (809, 173, 1328, 973), (846, 171, 1290, 973), 
               (874, 172, 1262, 973)]
CROP_INDEX = 1
SCREEN_CAPTURE_INTERVAL = 2  # Capture every Nth frame (reduce storage)


class PerformanceMonitor:
    """Monitor FPS dan timing."""
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.frame_times = []
        self.frame_start = None
        
    def start_frame(self):
        self.frame_start = time.time()
    
    def end_frame(self):
        if self.frame_start:
            frame_time = (time.time() - self.frame_start) * 1000  # ms
            self.frame_times.append(frame_time)
            if len(self.frame_times) > self.window_size:
                self.frame_times.pop(0)
            return frame_time
        return 0
    
    def get_fps(self):
        if not self.frame_times:
            return 0
        return 1000 / (sum(self.frame_times) / len(self.frame_times))
    
    def get_frame_time(self):
        if not self.frame_times:
            return 0
        return sum(self.frame_times) / len(self.frame_times)


def grab_button_region():
    """Grab button region for fast detection."""
    try:
        screen = ImageGrab.grab()
        h, w = screen.size[1], screen.size[0]
        bbox = (0, h-150, w, h)
        button_region = ImageGrab.grab(bbox=bbox)
        return np.array(button_region)
    except:
        return None


def detect_button(img_rgb, color_range):
    """Detect button by HSV color."""
    try:
        h, w = img_rgb.shape[:2]
        region = img_rgb[max(0, h-70):min(h, h-45), max(0, int(w/2)-50):min(w, int(w/2)+100)]
        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        lower, upper = np.array(color_range[0]), np.array(color_range[1])
        return (cv2.inRange(hsv, lower, upper) > 200).sum()
    except:
        return 0


def draw_status(frame, recording, rec_frame_count, fps, frame_ms):
    """Draw status panel on frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.7
    thickness = 2
    
    # Status text
    h, w = frame.shape[:2]
    y_offset = 30
    
    # Recording status
    rec_text = f"REC: {'●' if recording else '○'} {rec_frame_count} frames"
    cv2.putText(frame, rec_text, (10, y_offset), font, font_size, 
                (0, 255, 0) if recording else (100, 100, 100), thickness)
    
    # FPS
    fps_color = (0, 255, 0) if fps > 25 else (0, 165, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_offset + 40), font, font_size, fps_color, thickness)
    
    # Frame time
    cv2.putText(frame, f"Frame: {frame_ms:.1f}ms", (10, y_offset + 80), font, font_size, 
                (0, 200, 200), thickness)
    
    # Info: Pure recording (no marker detection)
    cv2.putText(frame, "[Pure Recording Mode - No Marker Detection]", (10, h - 20), 
                font, 0.5, (100, 200, 255), 1)


def main():
    """Main function for pure recording."""
    # Setup camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Verify resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cap = cap.get(cv2.CAP_PROP_FPS)
    print(f"[Camera] {width}x{height} @ {fps_cap:.2f} FPS")
    
    # State
    recording = False
    button_state = None
    frame_count = 0
    rec_dir = None
    rec_frame_count = 0
    video_writer = None
    screen_capture_count = 0
    
    # Performance monitor
    perf = PerformanceMonitor()
    
    print("[START] Pure Recording Mode - No Marker Detection")
    print("Press ESC to exit")
    
    while True:
        perf.start_frame()
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break
        
        # Save raw video frame jika recording
        if recording and video_writer is not None:
            video_writer.write(frame)
        
        # Display frame (resized for smaller window)
        display_frame = frame.copy()
        draw_status(display_frame, recording, rec_frame_count, perf.get_fps(), perf.get_frame_time())
        display_frame = cv2.resize(display_frame, (384, 216))  # 30% size
        cv2.imshow('Pure Recording', display_frame)
        
        # Button detection - reduce frequency
        if frame_count % 5 == 0:
            try:
                screen = grab_button_region()
                if screen is not None:
                    btn_on = detect_button(screen, BUTTON_ON_COLOR)
                    btn_off = detect_button(screen, BUTTON_OFF_COLOR)
                else:
                    btn_on = btn_off = 0
                
                # Start recording
                if btn_on > BUTTON_THRESHOLD and button_state != 'ON':
                    recording = True
                    button_state = 'ON'
                    rec_dir = f"dataMarker/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    os.makedirs(rec_dir, exist_ok=True)
                    
                    # Setup video writer
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_path = f"{rec_dir}/raw_camera.mp4"
                    video_writer = cv2.VideoWriter(video_path, fourcc, fps_cap, (width, height))
                    
                    rec_frame_count = 0
                    screen_capture_count = 0
                    print(f"[REC] Started - Session: {rec_dir}")
                
                # Stop recording
                elif btn_off > BUTTON_THRESHOLD and button_state != 'OFF':
                    if recording and video_writer is not None:
                        video_writer.release()
                        video_writer = None
                        
                        print(f"[SAVED] raw_camera.mp4 ({rec_frame_count} frames)")
                        print(f"[INFO] Session ready for post-processing: {rec_dir}")
                        print(f"[NEXT] Run: python processor_video.py -s {rec_dir}")
                    
                    recording = False
                    button_state = 'OFF'
            except Exception as e:
                print(f"Button detection error: {e}")
        
        # Capture screen images periodically (reduce frequency)
        if recording and frame_count % SCREEN_CAPTURE_INTERVAL == 0:
            try:
                screen = np.array(ImageGrab.grab())
                screen_gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
                
                crop = CROP_COORDS[CROP_INDEX]
                screen_crop = screen_gray[crop[1]:crop[3], crop[0]:crop[2]]
                
                # Save screen capture
                img_dir = f"{rec_dir}/screen_captures"
                os.makedirs(img_dir, exist_ok=True)
                img_path = f"{img_dir}/screen_{screen_capture_count:06d}.png"
                cv2.imwrite(img_path, screen_crop)
                screen_capture_count += 1
            except Exception as e:
                pass  # Silent fail for screen capture
        
        if recording:
            rec_frame_count += 1
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        
        frame_time = perf.end_frame()
        frame_count += 1
        
        # Log performance
        if frame_count % 60 == 0 and frame_count > 0:
            fps = perf.get_fps()
            status = "REC" if recording else "IDLE"
            print(f"[PERF] Frame {frame_count} | {status} | FPS: {fps:.1f} | FrameTime: {frame_time:.1f}ms")
    
    # Cleanup
    if video_writer is not None:
        video_writer.release()
        print(f"[SAVED] raw_camera.mp4 ({rec_frame_count} frames)")
    
    cap.release()
    cv2.destroyAllWindows()
    print("[END] Program closed")


if __name__ == '__main__':
    main()
