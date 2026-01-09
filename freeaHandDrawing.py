import cv2
from cv2 import aruco
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
import os
import csv
from PIL import ImageGrab


def load_camera_calibration():
    """Loads pre-calibrated camera matrix and distortion coefficients."""
    # These values are specific to the camera used to record the video.
    cameraMatrix = np.array(
        [
            [1156.575220787158, 0.0, 641.235796583754],
            [0.0, 1159.917623853068, 403.13243457956634],
            [0.0, 0.0, 1.0],
        ],
        dtype='double'
    )
    distCoeffs = np.array(
        [
            [0.21463756498771683],
            [-0.8166333234031081],
            [0.016152523688770088],
            [-0.00042204457570052013],
            [0.2851862067489753],
        ],
        dtype='double'
    )
    return cameraMatrix, distCoeffs


def setup_video_file(path):
    """Opens a video file and returns the capture object and FPS."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Video opened: {width:.0f}x{height:.0f} @ {fps:.2f} FPS")
    return cap, fps


def setup_aruco():
    """Initializes the ArUco detector with specific parameters."""
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    return detector


def estimatePoseGlobal(model_pts, img_pts, cameraMatrix, distCoeffs):
    """
    Estimates the camera's pose relative to the global coordinate system
    defined by all visible markers using RANSAC for robustness.
    """
    if len(model_pts) < 4:
        return None, None
    try:
        _, rvec, tvec, _ = cv2.solvePnPRansac(np.array(model_pts), np.array(img_pts), cameraMatrix, distCoeffs)
        return rvec, tvec
    except cv2.error:
        return None, None


def rotation_matrix_to_degrees(rvec):
    """Converts rotation vector to Euler angles in degrees."""
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0
    
    return np.degrees([x, y, z])


def rvec_to_axis_angle(rvec):
    """
    Converts rotation vector to axis-angle representation.
    Returns (angle_deg, axis_x, axis_y, axis_z) where axis is scaled by angle in degrees.
    """
    angle = np.linalg.norm(rvec)
    if angle < 1e-6:
        return 0.0, 0.0, 0.0, 0.0
    
    axis = rvec / angle
    angle_deg = np.degrees(angle)
    
    # Scale axis by angle (as per recording guide format)
    axis_scaled = axis * angle_deg
    
    return angle_deg, axis_scaled[0], axis_scaled[1], axis_scaled[2]


def get_crop_button(img_rgb, lower_color, upper_color):
    """
    Detects button in application based on HSV color range.
    
    Args:
        img_rgb (np.ndarray): Input image in RGB format
        lower_color (list): Lower HSV bound [H, S, V]
        upper_color (list): Upper HSV bound [H, S, V]
    
    Returns:
        int: Number of pixels matching the color range
    
    Process:
        1. Extract bottom-center region from screen (button location)
        2. Convert RGB → HSV
        3. Create mask with cv2.inRange()
        4. Count pixels > 200 in mask
    """
    try:
        h, w = img_rgb.shape[:2]
        
        # Region box for button (bottom-center)
        # Adjust these coordinates based on your application
        top_left_x = int(w / 2) - 50
        top_left_y = h - 70
        bottom_right_x = int(w / 2) + 100
        bottom_right_y = h - 45
        
        # Ensure coordinates are within bounds
        top_left_x = max(0, top_left_x)
        top_left_y = max(0, top_left_y)
        bottom_right_x = min(w, bottom_right_x)
        bottom_right_y = min(h, bottom_right_y)
        
        # Convert to HSV
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        lower_bound = np.array(lower_color, dtype=np.uint8)
        upper_bound = np.array(upper_color, dtype=np.uint8)
        
        # Extract region and create mask
        region = hsv[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        mask = cv2.inRange(region, lower_bound, upper_bound)
        pixel_count = (mask > 200).sum()
        
        return pixel_count
    except Exception as e:
        print(f"[WARNING] Button detection error: {e}")
        return 0


def safe_int_tuple(point, debug_name=""):
    """Safely convert a point to a pure Python int tuple."""
    try:
        result = []
        # Handle numpy array or list input
        if hasattr(point, '__iter__'):
            for i, val in enumerate(point):
                # First check for NaN and Inf
                try:
                    float_val = float(val)
                    if np.isnan(float_val) or np.isinf(float_val):
                        return None
                    # Check for reasonable bounds (image is typically 640x480)
                    if abs(float_val) > 10000:
                        return None
                except (ValueError, OverflowError):
                    return None
                
                # Check if it's already a Python int
                if isinstance(val, (int, np.integer)):
                    result.append(int(val))
                # Check if it's a numpy type with .item()
                elif hasattr(val, 'item'):
                    result.append(int(val.item()))
                # Try direct conversion
                else:
                    result.append(int(float(val)))
        else:
            return None
        
        # Verify all elements are pure Python ints
        pt = tuple(result)
        for i, v in enumerate(pt):
            if not isinstance(v, int) or isinstance(v, bool):
                return None
        
        return pt
    except (TypeError, ValueError, AttributeError):
        return None


class PositionFilter:
    """Exponential moving average filter for position smoothing."""
    def __init__(self, alpha=0.3):
        self.alpha = alpha  # Smoothing factor (0-1), higher = more responsive
        self.filtered_pos = None
    
    def filter(self, new_pos):
        """Apply exponential moving average filter."""
        if self.filtered_pos is None:
            self.filtered_pos = np.array(new_pos, dtype=float)
            return self.filtered_pos
        
        new_pos_arr = np.array(new_pos, dtype=float)
        self.filtered_pos = self.alpha * new_pos_arr + (1 - self.alpha) * self.filtered_pos
        return self.filtered_pos
    
    def reset(self):
        """Reset filter state."""
        self.filtered_pos = None


class RecordingSession:
    """Manages recording sessions with CSV output and image saving."""
    
    def __init__(self, base_dir="dataMarker"):
        """Initialize recording session with timestamp."""
        self.base_dir = base_dir
        self.session_dir = None
        self.csv_path = None
        self.images_dir = None
        self.rows = []
        self.recording = False
        self.initial_tip_cam = None
        self.initial_marker_pos = None
        self.frame_idx = 0
        self._create_session()
    
    def _create_session(self):
        """Create new session folder with timestamp."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(self.base_dir, f"session_{timestamp}")
        self.images_dir = os.path.join(self.session_dir, "images")
        
        os.makedirs(self.images_dir, exist_ok=True)
        
        csv_filename = f"aruco_tip_tracking_{timestamp}.csv"
        self.csv_path = os.path.join(self.session_dir, csv_filename)
        
        print(f"[SESSION] Created: {self.session_dir}")
    
    def start_recording(self):
        """Start recording session."""
        self.recording = True
        self.initial_tip_cam = None
        self.initial_marker_pos = None
        self.frame_idx = 0
        self.rows = []
        print(f"[REC ON] Recording started...")
    
    def stop_recording(self):
        """Stop recording and save to CSV."""
        if self.recording:
            self.recording = False
            self._save_csv()
            self.rows = []
            self.frame_idx = 0
            print(f"[REC OFF] Recording stopped.")
    
    def add_frame_data(self, marker_id, tip_pos_cam, marker_pos_cam, rvec):
        """
        Add a frame to recording buffer.
        
        Args:
            marker_id (int): Detected marker ID
            tip_pos_cam (np.array): Pen tip position in camera frame [x, y, z] in mm
            marker_pos_cam (np.array): Marker position in camera frame [x, y, z] in m
            rvec (np.array): Rotation vector
        """
        if not self.recording:
            return
        
        # Initialize reference positions on first recording frame
        if self.initial_tip_cam is None:
            self.initial_tip_cam = tip_pos_cam.copy()
            self.initial_marker_pos = marker_pos_cam.copy()
        
        self.frame_idx += 1
        timestamp = datetime.now().isoformat(timespec="milliseconds")
        
        # Calculate relative positions (offset from first frame)
        tip_relative_mm = (tip_pos_cam - self.initial_tip_cam).flatten()
        marker_relative_m = (marker_pos_cam - self.initial_marker_pos).flatten()
        
        # Convert axis-angle representation
        angle_deg, ax, ay, az = rvec_to_axis_angle(rvec)
        
        # Append to buffer
        self.rows.append([
            timestamp,
            self.frame_idx,
            int(marker_id),
            float(tip_relative_mm[0]),
            float(tip_relative_mm[1]),
            float(tip_relative_mm[2]),
            float(marker_relative_m[0]),
            float(marker_relative_m[1]),
            float(marker_relative_m[2]),
            float(angle_deg),
            float(ax),
            float(ay),
            float(az)
        ])
    
    def save_image(self):
        """
        Save screen capture (cropped region) with standard naming convention.
        Captures screen via ImageGrab and saves as PNG.
        """
        if not self.recording:
            return
        
        try:
            # Capture screen
            screen_img = ImageGrab.grab()
            img_arr = np.array(screen_img)
            
            # Convert RGB (from PIL) to BGR (for OpenCV)
            img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
            
            # Save image
            filename = f"frame_{self.frame_idx:06d}.png"
            filepath = os.path.join(self.images_dir, filename)
            cv2.imwrite(filepath, img_bgr)
        except Exception as e:
            print(f"[WARNING] Image save error: {e}")
    
    def _save_csv(self):
        """Save recorded data to CSV file."""
        if not self.rows:
            print("[CSV] No data to save.")
            return
        
        headers = [
            "timestamp", "frame_idx", "marker_id",
            "tip_x_mm", "tip_y_mm", "tip_z_mm",
            "marker_x_m", "marker_y_m", "marker_z_m",
            "rot_angle_deg", "rot_axis_x", "rot_axis_y", "rot_axis_z"
        ]
        
        try:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(self.rows)
            
            print(f"[CSV] Saved: {self.csv_path} ({len(self.rows)} frames)")
        except Exception as e:
            print(f"[ERROR] CSV save failed: {e}")
    
    def clear_buffer(self):
        """Clear buffer without saving."""
        self.rows = []
        self.frame_idx = 0
        self.initial_tip_cam = None
        self.initial_marker_pos = None
        print("[BUFFER] Cleared (data discarded).")
    
    def get_frame_count(self):
        """Get current number of frames recorded."""
        return self.frame_idx





def main():
    """Main function to run the video processing and pose estimation loop."""
    video_path = 0
    cap, fps = setup_video_file(video_path)
    cameraMatrix, distCoeffs = load_camera_calibration()
    detector = setup_aruco()

    try:
        data = pd.read_csv('markers/model_points_4x4.csv')
    except FileNotFoundError:
        print("Error: 'markers/model_points_4x4.csv' not found.")
        return

    pts = data[['x', 'y', 'z']].values.tolist()
    model_pts_by_id = [pts[i:i + 4] for i in range(0, len(pts), 4)]

    # --- Initialize Recording Session ---
    recording_session = RecordingSession()

    # --- Button Detection Parameters ---
    # Color ranges in HSV (Hue, Saturation, Value)
    button_on_color = [80, 150, 90]           # Blue for ON button
    button_on_color_upper = [120, 200, 120]
    button_off_color = [20, 200, 200]         # Yellow/Orange for OFF button
    button_off_color_upper = [40, 255, 255]
    button_threshold = 1000                   # Pixel count threshold
    button_detection_interval = 10            # Check every N frames
    
    prev_button_state = None                  # Track state for transition detection

    # --- Trajectory and Pen Tip Setup ---
    global_origin_pts = deque(maxlen=1000)
    pen_tip_path = deque(maxlen=1000)
    initial_origin = None
    
    # --- Position Filters for Stability ---
    origin_2d_filter = PositionFilter(alpha=0.4)
    pen_tip_2d_filter = PositionFilter(alpha=0.4)
    origin_3d_filter = PositionFilter(alpha=0.3)
    pen_tip_3d_filter = PositionFilter(alpha=0.3)

    # --- Define Pen Tip Location ---
    pen_tip_loc_mm = np.array([[-0.02327], [-102.2512], [132.8306]])
    pen_tip_3d = pen_tip_loc_mm.reshape(1, 1, 3)

    print(f"\nUsing final extended pen tip location (mm):")
    print(f"  X={pen_tip_loc_mm[0][0]:.4f}, Y={pen_tip_loc_mm[1][0]:.4f}, Z={pen_tip_loc_mm[2][0]:.4f}")
    print(f"  Total length from center: {np.linalg.norm(pen_tip_loc_mm):.2f} mm")
    
    print("\n=== Keyboard Controls ===")
    print("Manual Recording:")
    print("  'r' - Toggle recording ON/OFF")
    print("  'c' - Capture single frame (1 data point)")
    print("  'w' - Write CSV & create new session")
    print("  'x' - Clear buffer (discard data)")
    print("\nPlayback:")
    print("  'SPACE' - Pause/Play")
    print("  'f' - Play forward")
    print("  'b' - Play backward")
    print("  'p' - Toggle pen tip display")
    print("  'ESC' - Exit program")
    print("=" * 40)

    # --- Playback and Mode Control ---
    playing = True
    direction = 1
    plot_pen_tip = True
    ret, last_frame = cap.read()
    if not ret:
        print("Could not read the first frame.")
        return

    # --- Create a persistent canvas for drawing trajectories ---
    trajectory_canvas = np.zeros_like(last_frame)
    frame_count = 0
    num_markers_detected = 0
    sufficient_markers = False

    while True:
        if playing:
            if direction == 1:
                ret, frame = cap.read()
                if not ret:
                    playing = False
                    frame = last_frame.copy()
                else:
                    last_frame = frame.copy()
            else:
                current_frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
                new_idx = max(int(current_frame_idx) - 2, 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_idx)
                ret, frame = cap.read()
                if ret:
                    last_frame = frame.copy()
                else:
                    playing = False
                    frame = last_frame.copy()
        else:
            frame = last_frame.copy()

        # --- ArUco Detection and Pose Estimation ---
        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            image_points_collected = []
            model_points_collected = []
            num_markers_detected = len(ids)
            
            # Only track pen tip if we have at least 2 markers
            sufficient_markers = num_markers_detected >= 2

            for i, corner in enumerate(corners):
                marker_id = ids[i][0]
                if marker_id < len(model_pts_by_id):
                    for point_2d in corner[0]:
                        image_points_collected.append(point_2d)
                    for point_3d in model_pts_by_id[marker_id]:
                        model_points_collected.append(point_3d)

            r_glob, t_glob = estimatePoseGlobal(model_points_collected, image_points_collected, cameraMatrix,
                                                distCoeffs)

            if r_glob is not None and t_glob is not None:
                if np.isnan(r_glob).any() or np.isnan(t_glob).any():
                    continue
                
                cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, r_glob, t_glob, 30, 4)

                euler_degrees = rotation_matrix_to_degrees(r_glob)
                
                origin_2d, _ = cv2.projectPoints(np.array([[0., 0., 0.]]), r_glob, t_glob, cameraMatrix, distCoeffs)
                
                if not np.isnan(origin_2d).any() and not np.isinf(origin_2d).any():
                    filtered_origin_2d = origin_2d_filter.filter(origin_2d[0][0])
                    origin_pt = safe_int_tuple(filtered_origin_2d)
                    if origin_pt is not None:
                        global_origin_pts.append(origin_pt)

                    if plot_pen_tip and sufficient_markers:
                        pen_tip_2d, _ = cv2.projectPoints(pen_tip_3d, r_glob, t_glob, cameraMatrix, distCoeffs)
                        
                        if not np.isnan(pen_tip_2d).any() and not np.isinf(pen_tip_2d).any():
                            filtered_pen_tip_2d = pen_tip_2d_filter.filter(pen_tip_2d[0][0])
                            pen_pt = safe_int_tuple(filtered_pen_tip_2d)
                            if pen_pt is not None:
                                pen_tip_path.append(pen_pt)
                                cv2.circle(frame, pen_pt, 5, (0, 255, 0), -1)
                                if len(pen_tip_path) > 1:
                                    pt1 = safe_int_tuple(pen_tip_path[-2])
                                    pt2 = safe_int_tuple(pen_tip_path[-1])
                                    if pt1 is not None and pt2 is not None:
                                        cv2.line(trajectory_canvas, pt1, pt2, (0, 255, 0), 5)
                            
                            rotation_matrix, _ = cv2.Rodrigues(r_glob)
                            pen_tip_global_3d = rotation_matrix @ pen_tip_loc_mm + t_glob
                            filtered_pen_tip_3d = pen_tip_3d_filter.filter(pen_tip_global_3d)
                            
                            if not (np.isnan(t_glob).any() or np.isnan(filtered_pen_tip_3d).any() or 
                                   np.isinf(t_glob).any() or np.isinf(filtered_pen_tip_3d).any()):
                                
                                # === RECORDING DATA ===
                                if recording_session.recording and sufficient_markers:
                                    recording_session.add_frame_data(
                                        marker_id=ids[0][0],
                                        tip_pos_cam=filtered_pen_tip_3d.flatten(),
                                        marker_pos_cam=t_glob.flatten(),
                                        rvec=r_glob
                                    )
                                    # Save screen capture image for this frame
                                    recording_session.save_image()

        # --- Combine frame with trajectory canvas ---
        img2gray = cv2.cvtColor(trajectory_canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        lines_fg = cv2.bitwise_and(trajectory_canvas, trajectory_canvas, mask=mask)
        display_frame = cv2.add(frame_bg, lines_fg)

        # --- UI Text Overlay ---
        cv2.rectangle(display_frame, (0, 0), (550, 130), (0, 0, 0), -1)
        
        # Markers count
        num_markers_text = f"Markers: {num_markers_detected if ids is not None else 0}"
        cv2.putText(display_frame, num_markers_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Pen tip status
        tip_color = (0, 255, 0) if (plot_pen_tip and sufficient_markers) else (128, 128, 128)
        tip_status = "ON" if (plot_pen_tip and sufficient_markers) else ("OFF (need 2+ markers)" if not sufficient_markers else "OFF")
        cv2.putText(display_frame, f"Pen Tip: {tip_status}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, tip_color, 1)
        
        # Recording status
        record_color = (0, 255, 255) if recording_session.recording else (200, 200, 200)
        record_symbol = "● REC" if recording_session.recording else "○ REC"
        cv2.putText(display_frame, record_symbol, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, record_color, 2)
        
        # Frames recorded
        frame_text = f"Frames: {recording_session.get_frame_count()}"
        cv2.putText(display_frame, frame_text, (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # --- Button Detection (every N frames) ---
        if frame_count % button_detection_interval == 0:
            try:
                screen_img = ImageGrab.grab()
                img_arr = np.array(screen_img)
                
                # Convert BGR (from PIL) to match OpenCV color space
                #img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2RGB)
                
                # Detect buttons
                nb = get_crop_button(img_arr, button_on_color, button_on_color_upper)
                ny = get_crop_button(img_arr, button_off_color, button_off_color_upper)
                
                # Detect ON button press (transition)
                if nb > button_threshold:
                    if prev_button_state != 'ON':
                        recording_session.start_recording()
                        prev_button_state = 'ON'
                        print("[BUTTON: ON] Recording started automatically.")
                
                # Detect OFF button press (transition)
                elif ny > button_threshold:
                    if prev_button_state != 'OFF':
                        recording_session.stop_recording()
                        recording_session = RecordingSession()  # Create new session
                        prev_button_state = 'OFF'
                        print("[BUTTON: OFF] Recording stopped. New session created.")
            
            except Exception as e:
                print(f"[WARNING] Button detection error: {e}")

        # --- Display and Handle Keyboard Input ---
        cv2.imshow('Global Pose Estimation with Pen Tip', display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC key
            break
        elif key == ord(' '):  # SPACE - Pause/Play
            playing = not playing
        elif key == ord('f'):  # Forward
            direction = 1
            playing = True
        elif key == ord('b'):  # Backward
            direction = -1
            playing = True
        elif key == ord('r'):  # Toggle recording
            if not recording_session.recording:
                recording_session.start_recording()
            else:
                recording_session.stop_recording()
        elif key == ord('c'):  # Capture single frame
            if not recording_session.recording:
                recording_session.start_recording()
                print("[CAPTURE] Single frame mode activated.")
            else:
                recording_session.stop_recording()
                print("[CAPTURE] Frame captured. Recording stopped.")
        elif key == ord('w'):  # Write CSV & create new session
            recording_session.stop_recording()
            recording_session = RecordingSession()
            print("[SAVE] Session saved. New session created.")
        elif key == ord('x'):  # Clear buffer
            recording_session.clear_buffer()
        elif key == ord('p'):  # Toggle pen tip display
            plot_pen_tip = not plot_pen_tip
            if not plot_pen_tip:
                pen_tip_path.clear()
                pen_tip_2d_filter.reset()
                trajectory_canvas[:] = 0

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # Save remaining data if still recording
    if recording_session.recording:
        recording_session.stop_recording()
    
    print("\n[EXIT] Program closed.")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
