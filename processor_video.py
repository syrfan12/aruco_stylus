"""
Phase 2: Post-Processing - Process recorded video and generate marker data + CSV
Purpose: Perform marker detection and analysis on pre-recorded video files
Input: raw_camera.mp4 (from Phase 1)
Output: tracked.mp4, data.csv
"""

import cv2
from cv2 import aruco
import numpy as np
import pandas as pd
from datetime import datetime
import os
import csv
import argparse
from pathlib import Path

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

PEN_TIP_LOC = np.array([[5.210180842491079], [-121.40040855432804], [153.18664070080274]], dtype=np.float32)


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


def process_video(session_dir):
    """Process recorded video file with marker detection."""
    
    print(f"\n[PROCESSING] Session: {session_dir}")
    
    # Check if raw_camera.mp4 exists
    video_path = os.path.join(session_dir, 'raw_camera.mp4')
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found")
        return False
    
    # Load marker points
    try:
        data = pd.read_csv('markers/model_points_4x4.csv')
        pts = data[['x', 'y', 'z']].values.tolist()
        model_pts_by_id = [pts[i:i+4] for i in range(0, len(pts), 4)]
        print(f"[LOADED] {len(model_pts_by_id)} marker model points")
    except FileNotFoundError:
        print("Error: markers/model_points_4x4.csv not found")
        return False
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[VIDEO] {width}x{height} @ {fps:.2f} FPS | Total frames: {total_frames}")
    
    # Setup detector
    detector = setup_aruco()
    pen_tip_3d = PEN_TIP_LOC.astype(np.float32).reshape(1, 1, 3)
    
    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join(session_dir, 'tracked.mp4')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Data collection
    rec_rows = []
    rec_frame_count = 0
    initial_tip = None
    initial_marker = None
    
    # Filters
    origin_2d_filter = None
    pen_tip_2d_filter = None
    pen_tip_3d_filter = None
    
    # Trajectory visualization
    trajectory_canvas = None
    pen_tip_path = []
    
    frame_idx = 0
    
    print("[PROCESSING] Starting frame-by-frame analysis...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if trajectory_canvas is None:
            trajectory_canvas = np.zeros_like(frame)
        
        # Detect markers
        corners, ids, _ = detector.detectMarkers(frame)
        num_markers = len(ids) if ids is not None else 0
        
        if num_markers >= 2 and ids is not None:
            # Filter for specific marker IDs
            my_id = [1, 5, 7]
            mask_id = np.isin(ids, my_id)
            indices_id = np.where(mask_id)[0]
            corners_mask = [corners[i] for i in indices_id]
            corners = corners_mask
            ids = ids[indices_id]
            
            # Draw detected markers
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
                # Draw frame axes
                cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, 30, 4)
                
                # Filter positions
                origin_2d, _ = cv2.projectPoints(np.array([[0, 0, 0]], dtype=np.float32), rvec, tvec, 
                                                 CAMERA_MATRIX, DIST_COEFFS)
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
                
                # Track pen tip path
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
                
                # Record data
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
        
        # Combine trajectory with frame
        gray = cv2.cvtColor(trajectory_canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame = cv2.bitwise_and(frame, frame, mask=mask_inv)
        frame = cv2.add(frame, cv2.bitwise_and(trajectory_canvas, trajectory_canvas, mask=mask))
        
        # Add status text
        cv2.putText(frame, f"Processing: {frame_idx+1}/{total_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracked frames: {rec_frame_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Write frame
        video_writer.write(frame)
        
        frame_idx += 1
        
        # Progress
        if frame_idx % 30 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"  Progress: {progress:.1f}% | Tracked: {rec_frame_count} frames")
    
    # Cleanup
    cap.release()
    video_writer.release()
    
    print(f"\n[SAVED] tracked.mp4 ({frame_idx} frames processed)")
    
    # Save CSV
    if rec_rows:
        csv_file = os.path.join(session_dir, 'data.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'frame', 'marker_id', 'tip_x', 'tip_y', 'tip_z', 
                           'marker_x', 'marker_y', 'marker_z', 'angle', 'axis_x', 'axis_y', 'axis_z'])
            writer.writerows(rec_rows)
        print(f"[SAVED] data.csv ({len(rec_rows)} tracked frames)")
    else:
        print("[WARNING] No tracked data collected")
    
    print(f"[COMPLETE] Session processing finished\n")
    return True


def main():
    parser = argparse.ArgumentParser(description='Post-process recorded video for marker detection')
    parser.add_argument('-s', '--session', type=str, help='Session directory to process')
    parser.add_argument('-l', '--latest', action='store_true', help='Process latest session')
    parser.add_argument('-a', '--all', action='store_true', help='Process all sessions (yang belum ada tracked.mp4)')
    
    args = parser.parse_args()
    
    if args.all:
        # Process all sessions
        base_dir = 'dataMarker/for_calib'
        if not os.path.exists(base_dir):
            print("Error: dataMarker directory not found")
            return
        
        sessions = sorted([d for d in os.listdir(base_dir) if d.startswith('session_')])
        if not sessions:
            print("Error: No sessions found")
            return
        
        # Filter sessions that don't have tracked.mp4 yet
        pending_sessions = []
        for session in sessions:
            session_path = os.path.join(base_dir, session)
            tracked_path = os.path.join(session_path, 'tracked.mp4')
            raw_path = os.path.join(session_path, 'raw_camera.mp4')
            
            # Only process if raw_camera.mp4 exists but tracked.mp4 doesn't
            if os.path.exists(raw_path) and not os.path.exists(tracked_path):
                pending_sessions.append(session_path)
        
        if not pending_sessions:
            print("All sessions already processed!")
            return
        
        print(f"\n[BATCH PROCESSING] Found {len(pending_sessions)} pending sessions")
        print("=" * 60)
        
        for i, session_path in enumerate(pending_sessions, 1):
            print(f"\n[{i}/{len(pending_sessions)}] Processing: {os.path.basename(session_path)}")
            print("-" * 60)
            process_video(session_path)
        
        print("\n" + "=" * 60)
        print(f"✓ Batch processing completed! {len(pending_sessions)} sessions processed.")
        print("=" * 60)
    
    elif args.latest:
        # Find latest session
        base_dir = 'dataMarker'
        if not os.path.exists(base_dir):
            print("Error: dataMarker directory not found")
            return
        
        sessions = [d for d in os.listdir(base_dir) if d.startswith('session_')]
        if not sessions:
            print("Error: No sessions found")
            return
        
        latest_session = sorted(sessions)[-1]
        session_dir = os.path.join(base_dir, latest_session)
        
        if not os.path.exists(session_dir):
            print(f"Error: Session directory not found: {session_dir}")
            return
        
        process_video(session_dir)
    
    elif args.session:
        session_dir = args.session
        if not os.path.exists(session_dir):
            print(f"Error: Session directory not found: {session_dir}")
            return
        
        process_video(session_dir)
    else:
        print("Error: Specify session with -s <dir>, use -l for latest, or -a for all pending")
        return


if __name__ == '__main__':
    main()
