import cv2
from cv2 import aruco
import numpy as np
import pandas as pd
from datetime import datetime
import os
import csv
from PIL import ImageGrab

# Camera Calibration
CAMERA_MATRIX = np.array([
    [1156.575220787158, 0.0, 641.235796583754],
    [0.0, 1159.917623853068, 403.13243457956634],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

DIST_COEFFS = np.array([
    [0.21463756498771683],
    [-0.8166333234031081],
    [0.016152523688770088],
    [-0.00042204457570052013],
    [0.2851862067489753]
], dtype=np.float32)

PEN_TIP_LOC = np.array([[-0.02327], [-102.2512], [132.8306]], dtype=np.float32)  # mm
BUTTON_ON_COLOR = ([80, 150, 90], [120, 200, 120])  # HSV range
BUTTON_OFF_COLOR = ([20, 200, 200], [40, 255, 255])  # HSV range
BUTTON_THRESHOLD = 1000
CROP_COORDS = [(315, 170, 1822, 973), (550, 170, 1580, 920), (679, 172, 1458, 971),
               (757, 172, 1380, 972), (809, 173, 1328, 973), (846, 171, 1290, 973), 
               (874, 172, 1262, 973)]
CROP_INDEX = 1

def setup_aruco():
    """Initialize ArUco detector."""
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_4X4_50), params)

def estimate_pose(model_pts, img_pts):
    """Estimate camera pose using RANSAC."""
    if len(model_pts) < 4:
        return None, None
    try:
        _, rvec, tvec, _ = cv2.solvePnPRansac(np.array(model_pts), np.array(img_pts), 
                                              CAMERA_MATRIX, DIST_COEFFS)
        return rvec, tvec
    except:
        return None, None

def safe_point(p):
    """Convert point to int tuple safely."""
    try:
        if hasattr(p, '__iter__'):
            result = tuple(int(float(v)) for v in p if not (np.isnan(float(v)) or np.isinf(float(v))))
            return result if len(result) == len(list(p)) and all(abs(x) < 10000 for x in result) else None
        return None
    except:
        return None

def apply_filter(filtered_val, new_val, alpha=0.3):
    """Simple exponential moving average filter."""
    if filtered_val is None:
        return np.array(new_val, dtype=float)
    return alpha * np.array(new_val, dtype=float) + (1 - alpha) * filtered_val

def detect_button(img_rgb, color_range):
    """Detect button by HSV color in bottom-center region."""
    try:
        h, w = img_rgb.shape[:2]
        region = img_rgb[max(0, h-70):min(h, h-45), max(0, int(w/2)-50):min(w, int(w/2)+100)]
        hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        lower, upper = np.array(color_range[0]), np.array(color_range[1])
        return (cv2.inRange(hsv, lower, upper) > 200).sum()
    except:
        return 0





def main():
    """Main function - simplified."""
    # Load data
    try:
        data = pd.read_csv('markers/model_points_4x4.csv')
        pts = data[['x', 'y', 'z']].values.tolist()
        model_pts_by_id = [pts[i:i+4] for i in range(0, len(pts), 4)]
    except FileNotFoundError:
        print("Error: markers/model_points_4x4.csv not found")
        return
    
    # Initialize
    cap = cv2.VideoCapture(0)
    detector = setup_aruco()
    
    # State
    recording = False
    button_state = None
    frame_count = 0
    num_markers = 0
    pen_tip_3d = PEN_TIP_LOC.astype(np.float32).reshape(1, 1, 3)
    
    # Filters
    origin_2d_filter = None
    pen_tip_2d_filter = None
    pen_tip_3d_filter = None
    
    # Trajectory
    trajectory_canvas = None
    pen_tip_path = []
    
    # Recording
    rec_dir = None
    rec_rows = []
    rec_frame_count = 0
    initial_tip = None
    initial_marker = None
    
    print("[START] Camera tracking initialized")
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Cannot read camera")
        return
    
    trajectory_canvas = np.zeros_like(frame)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect markers
        corners, ids, _ = detector.detectMarkers(frame)
        num_markers = len(ids) if ids is not None else 0
        
        if num_markers >= 2 and ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Collect points
            img_pts, model_pts = [], []
            for i, corner in enumerate(corners):
                mid = ids[i][0]
                if mid < len(model_pts_by_id):
                    img_pts.extend(corner[0])
                    model_pts.extend(model_pts_by_id[mid])
            
            # Estimate pose
            rvec, tvec = estimate_pose(model_pts, img_pts)
            
            if rvec is not None and tvec is not None and not (np.isnan(rvec).any() or np.isnan(tvec).any()):
                cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, 30, 4)
                
                # Filter positions
                origin_2d, _ = cv2.projectPoints(np.array([[0, 0, 0]], dtype=np.float32), rvec, tvec, CAMERA_MATRIX, DIST_COEFFS)
                if origin_2d_filter is None:
                    origin_2d_filter = origin_2d[0][0]
                else:
                    origin_2d_filter = apply_filter(origin_2d_filter, origin_2d[0][0], 0.4)
                
                # Pen tip tracking
                pen_tip_2d, _ = cv2.projectPoints(pen_tip_3d, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS)
                if pen_tip_2d_filter is None:
                    pen_tip_2d_filter = pen_tip_2d[0][0]
                else:
                    pen_tip_2d_filter = apply_filter(pen_tip_2d_filter, pen_tip_2d[0][0], 0.4)
                
                pen_pt = safe_point(pen_tip_2d_filter)
                if pen_pt:
                    pen_tip_path.append(pen_pt)
                    if len(pen_tip_path) > 1000:
                        pen_tip_path.pop(0)
                    cv2.circle(frame, pen_pt, 5, (0, 255, 0), -1)
                    if len(pen_tip_path) > 1:
                        pt1, pt2 = pen_tip_path[-2], pen_tip_path[-1]
                        if pt1 and pt2:
                            cv2.line(trajectory_canvas, pt1, pt2, (0, 255, 0), 5)
                
                # 3D pen tip
                rot_mat, _ = cv2.Rodrigues(rvec)
                pen_tip_3d_global = rot_mat @ PEN_TIP_LOC + tvec
                if pen_tip_3d_filter is None:
                    pen_tip_3d_filter = pen_tip_3d_global.flatten()
                else:
                    pen_tip_3d_filter = apply_filter(pen_tip_3d_filter, pen_tip_3d_global.flatten(), 0.3)
                
                # Recording
                if recording:
                    if initial_tip is None:
                        initial_tip = pen_tip_3d_filter.copy()
                        initial_marker = tvec.flatten()
                    
                    rec_frame_count += 1
                    tip_rel = pen_tip_3d_filter - initial_tip
                    marker_rel = tvec.flatten() - initial_marker
                    angle = np.linalg.norm(rvec)
                    axis = rvec.flatten() / angle * np.degrees(angle) if angle > 1e-6 else np.zeros(3)
                    
                    rec_rows.append([
                        datetime.now().isoformat(timespec="milliseconds"),
                        rec_frame_count,
                        int(ids[0][0]),
                        *tip_rel,
                        *marker_rel,
                        np.degrees(angle),
                        *axis
                    ])
                    
                    # Save image
                    try:
                        screen = np.array(ImageGrab.grab())
                        screen_gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
                        crop = CROP_COORDS[CROP_INDEX]
                        screen_crop = screen_gray[crop[1]:crop[3], crop[0]:crop[2]]
                        os.makedirs(f"{rec_dir}/images", exist_ok=True)
                        cv2.imwrite(f"{rec_dir}/images/frame_{rec_frame_count:06d}.png", screen_crop)
                    except:
                        pass
        
        # Combine trajectory with frame
        gray = cv2.cvtColor(trajectory_canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame = cv2.bitwise_and(frame, frame, mask=mask_inv)
        frame = cv2.add(frame, cv2.bitwise_and(trajectory_canvas, trajectory_canvas, mask=mask))
        
        # UI
        cv2.rectangle(frame, (0, 0), (450, 130), (0, 0, 0), -1)
        cv2.putText(frame, f"Markers: {num_markers}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tip_status = "ON" if num_markers >= 2 else f"OFF ({2-num_markers} needed)"
        cv2.putText(frame, f"Pen Tip: {tip_status}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if num_markers >= 2 else (128, 128, 128), 1)
        rec_color = (0, 255, 255) if recording else (200, 200, 200)
        cv2.putText(frame, "● REC" if recording else "○ REC", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, rec_color, 2)
        cv2.putText(frame, f"Frames: {rec_frame_count}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Button detection
        if frame_count % 2 == 0:
            try:
                screen = np.array(ImageGrab.grab())
                btn_on = detect_button(screen, BUTTON_ON_COLOR)
                btn_off = detect_button(screen, BUTTON_OFF_COLOR)
                
                if btn_on > BUTTON_THRESHOLD and button_state != 'ON':
                    recording = True
                    button_state = 'ON'
                    rec_dir = f"dataMarker/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    rec_rows, rec_frame_count, initial_tip, initial_marker = [], 0, None, None
                    print("[REC] Started")
                
                elif btn_off > BUTTON_THRESHOLD and button_state != 'OFF':
                    if recording:
                        os.makedirs(rec_dir, exist_ok=True)
                        csv_file = f"{rec_dir}/data.csv"
                        with open(csv_file, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(['timestamp', 'frame', 'marker_id', 'tip_x', 'tip_y', 'tip_z', 'marker_x', 'marker_y', 'marker_z', 'angle', 'axis_x', 'axis_y', 'axis_z'])
                            writer.writerows(rec_rows)
                        print(f"[SAVED] {csv_file}")
                    recording = False
                    button_state = 'OFF'
                    pen_tip_3d_filter = None
            except:
                pass
        
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
        frame_count += 1
    
    # Cleanup
    if recording:
        os.makedirs(rec_dir, exist_ok=True)
        csv_file = f"{rec_dir}/data.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'frame', 'marker_id', 'tip_x', 'tip_y', 'tip_z', 'marker_x', 'marker_y', 'marker_z', 'angle', 'axis_x', 'axis_y', 'axis_z'])
            writer.writerows(rec_rows)
    
    cap.release()
    cv2.destroyAllWindows()
    print("[END] Program closed")


if __name__ == '__main__':
    main()
