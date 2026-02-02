"""
Phase 2: Post-Processing with Dual Marker Tracking
- Process dodecahedron markers (RANSAC multi-marker pose)
- Process single marker (ID 15, individual pose estimation)
- Compare tip projection from both methods in single CSV
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
import math

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

# Dodecahedron pen tip offset (mm)
DODECA_PEN_TIP_LOC = np.array([[3.950217091], [-122.019756294], [143.423864236]], dtype=np.float32)

# Single marker pen tip offset (meter)
#SINGLE_TIP_OFFSET_MARKER = np.array([[-45.352322], [0], [-137.492666]], dtype=np.float32) # id 22
#SINGLE_TIP_OFFSET_MARKER = np.array([[-79.004558], [50.944913], [-163.322381]], dtype=np.float32) # id 10
#SINGLE_TIP_OFFSET_MARKER = np.array([[2.610715], [-55.398142], [-156.996888]], dtype=np.float32) # id 15
SINGLE_TIP_OFFSET_MARKER = np.array([[ 0], [0], [ -180]], dtype=np.float32)
# Single marker setup
MARKER_ID_TARGET = 15
SINGLE_MARKER_SIZE_MM = 0.016 * 1000


def setup_aruco():
    """Initialize ArUco detector."""
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_4X4_50), params)


def estimate_pose_dodeca(model_pts, img_pts):
    """Estimate camera pose using RANSAC (for dodecahedron)."""
    if len(model_pts) < 4:
        return None, None
    try:
        _, rvec, tvec, _ = cv2.solvePnPRansac(
            np.array(model_pts, dtype=np.float32), 
            np.array(img_pts, dtype=np.float32), 
            CAMERA_MATRIX, DIST_COEFFS,
            iterationsCount=500,
            reprojectionError=8.0,
            confidence=0.99
        )
        if rvec is None or tvec is None:
            return None, None
        # Refine with Levenberg-Marquardt
        rvec, tvec = cv2.solvePnPRefineLM(np.array(model_pts, dtype=np.float32),
                                          np.array(img_pts, dtype=np.float32),
                                          CAMERA_MATRIX, DIST_COEFFS, rvec, tvec)
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


def apply_filter(filtered_val, new_val, alpha=0.3, max_delta=100):
    """EMA filter with outlier rejection."""
    new_val = np.array(new_val, dtype=float)
    if filtered_val is None:
        return new_val
    
    # Reject outliers: if delta > max_delta, keep previous value
    delta = np.linalg.norm(new_val - filtered_val)
    if delta > max_delta:
        return filtered_val
    
    return alpha * new_val + (1 - alpha) * filtered_val


def check_existing_files(session_dir):
    """Check if output files already exist."""
    tracked_combined = os.path.join(session_dir, 'tracked_combined.mp4')
    data_dodeca = os.path.join(session_dir, 'data_dodeca.csv')
    data_single = os.path.join(session_dir, 'data_single.csv')
    
    existing = []
    if os.path.exists(tracked_combined):
        existing.append('tracked_combined.mp4')
    if os.path.exists(data_dodeca):
        existing.append('data_dodeca.csv')
    if os.path.exists(data_single):
        existing.append('data_single.csv')
    
    return existing


def prompt_overwrite(session_dir):
    """Prompt user if existing files should be overwritten."""
    existing = check_existing_files(session_dir)
    if not existing:
        return True
    
    print(f"\n[WARNING] Existing output files found:")
    for f in existing:
        print(f"    ├─ {f}")
    
    while True:
        choice = input(f"\n[?] Overwrite? (y/n): ").strip().lower()
        if choice in ['y', 'yes']:
            print("[→] Overwriting existing files...")
            return True
        elif choice in ['n', 'no']:
            print("[→] Skipping session")
            return False
        else:
            print("[!] Invalid choice. Please enter 'y' or 'n'")


def process_combined_phase(session_dir, allow_overwrite=True):
    """Combined Phase: Process both dodecahedron and single marker in one pass."""
    
    print(f"\n{'='*60}")
    print(f"[COMBINED] DUAL MARKER TRACKING (DODECA + SINGLE)")
    print(f"[SESSION] {session_dir}")
    print(f"{'='*60}")
    
    # Check if raw_camera.mp4 exists
    video_path = os.path.join(session_dir, 'raw_camera.mp4')
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found")
        return False
    
    # Check for existing files and skip if overwrite not allowed
    if not allow_overwrite and check_existing_files(session_dir):
        print("[→] Skipping (existing files found, use --overwrite to reprocess)")
        return False
    
    # Load dodecahedron marker points
    try:
        data = pd.read_csv('markers/model_points_4x4.csv')
        pts = data[['x', 'y', 'z']].values.tolist()
        model_pts_by_id = [pts[i:i+4] for i in range(0, len(pts), 4)]
        print(f"[LOADED] {len(model_pts_by_id)} dodeca marker model points")
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
    print(f"[MODE] Dual tracking: DODECAHEDRON (Green) + SINGLE MARKER (Blue)")
    
    # Setup detector
    detector = setup_aruco()
    dodeca_pen_tip_3d = DODECA_PEN_TIP_LOC.astype(np.float32).reshape(1, 1, 3)
    
    # Setup output video writer (single combined output)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join(session_dir, 'tracked_combined.mp4')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Data collection for both markers
    dodeca_rows = []
    single_rows = []
    frame_idx = 0
    
    # Dodecahedron tracking state
    dodeca_frame_count = 0
    dodeca_initial_tip = None
    dodeca_initial_marker = None
    dodeca_pen_tip_2d_filter = None
    dodeca_pen_tip_3d_filter = None
    dodeca_pen_tip_path = []
    dodeca_trajectory_canvas = None
    
    # Single marker tracking state
    single_frame_count = 0
    single_initial_tip = None
    single_initial_marker = None
    single_pen_tip_2d_filter = None
    single_pen_tip_3d_filter = None
    single_pen_tip_path = []
    single_trajectory_canvas = None
    
    single_marker_points = np.array([
        [-SINGLE_MARKER_SIZE_MM/2,  SINGLE_MARKER_SIZE_MM/2, 0],
        [ SINGLE_MARKER_SIZE_MM/2,  SINGLE_MARKER_SIZE_MM/2, 0],
        [ SINGLE_MARKER_SIZE_MM/2, -SINGLE_MARKER_SIZE_MM/2, 0],
        [-SINGLE_MARKER_SIZE_MM/2, -SINGLE_MARKER_SIZE_MM/2, 0]
    ], dtype=np.float32)
    
    print("[PROCESSING] Starting combined frame-by-frame analysis...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if dodeca_trajectory_canvas is None:
            dodeca_trajectory_canvas = np.zeros_like(frame)
        if single_trajectory_canvas is None:
            single_trajectory_canvas = np.zeros_like(frame)
        
        # Detect all markers (single pass)
        corners, ids, _ = detector.detectMarkers(frame)
        
        dodeca_detected = False
        dodeca_data = {}
        single_detected = False
        single_data = {}
        
        # ===== DODECAHEDRON TRACKING =====
        if ids is not None and len(ids) >= 2:
            my_id = list(range(0, 12))
            mask_id = np.isin(ids, my_id)
            indices_id = np.where(mask_id)[0]
            
            if len(indices_id) > 0:
                corners_mask = [corners[i] for i in indices_id]
                ids_mask = ids[indices_id]
                
                aruco.drawDetectedMarkers(frame, corners_mask, ids_mask, (0, 255, 0))
                
                img_pts, model_pts = [], []
                for i, corner in enumerate(corners_mask):
                    mid = ids_mask[i][0]
                    if mid < len(model_pts_by_id):
                        img_pts.extend(corner[0])
                        model_pts.extend(model_pts_by_id[mid])
                
                rvec, tvec = estimate_pose_dodeca(model_pts, img_pts)
                
                if rvec is not None and tvec is not None and not (np.isnan(rvec).any() or np.isnan(tvec).any()):
                    dodeca_detected = True
                    cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, 30, 4)
                    
                    pen_tip_2d, _ = cv2.projectPoints(dodeca_pen_tip_3d, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS)
                    if dodeca_pen_tip_2d_filter is None:
                        dodeca_pen_tip_2d_filter = pen_tip_2d[0][0]
                    else:
                        dodeca_pen_tip_2d_filter = apply_filter(dodeca_pen_tip_2d_filter, pen_tip_2d[0][0], 0.5, max_delta=50)
                    
                    pen_pt = safe_point(dodeca_pen_tip_2d_filter)
                    if pen_pt:
                        dodeca_pen_tip_path.append(pen_pt)
                        if len(dodeca_pen_tip_path) > 1000:
                            dodeca_pen_tip_path.pop(0)
                        cv2.circle(frame, pen_pt, 5, (0, 255, 0), -1)
                        if len(dodeca_pen_tip_path) > 1:
                            pt1, pt2 = dodeca_pen_tip_path[-2], dodeca_pen_tip_path[-1]
                            if pt1 and pt2:
                                cv2.line(dodeca_trajectory_canvas, pt1, pt2, (0, 255, 0), 5)
                    
                    rot_mat, _ = cv2.Rodrigues(rvec)
                    pen_tip_3d_global = rot_mat @ DODECA_PEN_TIP_LOC + tvec
                    if dodeca_pen_tip_3d_filter is None:
                        dodeca_pen_tip_3d_filter = pen_tip_3d_global.flatten()
                    else:
                        dodeca_pen_tip_3d_filter = apply_filter(dodeca_pen_tip_3d_filter, pen_tip_3d_global.flatten(), 0.4, max_delta=200)
                    
                    if dodeca_initial_tip is None:
                        dodeca_initial_tip = dodeca_pen_tip_3d_filter.copy()
                        dodeca_initial_marker = tvec.flatten()
                    
                    dodeca_frame_count += 1
                    tip_rel = dodeca_pen_tip_3d_filter - dodeca_initial_tip
                    marker_rel = tvec.flatten() #- dodeca_initial_marker
                    angle = np.linalg.norm(rvec)
                    axis = rvec.flatten() / angle * np.degrees(angle) if angle > 1e-6 else np.zeros(3)
                    
                    dodeca_data = {
                        'marker_id': int(ids_mask[0][0]),
                        'tip_x': tip_rel[0],
                        'tip_y': tip_rel[1],
                        'tip_z': tip_rel[2],
                        'marker_x': marker_rel[0],
                        'marker_y': marker_rel[1],
                        'marker_z': marker_rel[2],
                        'angle': np.degrees(angle),
                        'axis_x': axis[0],
                        'axis_y': axis[1],
                        'axis_z': axis[2]
                    }
        
        # ===== SINGLE MARKER TRACKING (ID 15) =====
        if ids is not None:
            for i in range(ids.size):
                marker_id = int(ids[i][0])
                
                if marker_id == MARKER_ID_TARGET:
                    corner = corners[i][0]
                    
                    success, rvec, tvec = cv2.solvePnP(
                        single_marker_points,
                        corner,
                        CAMERA_MATRIX,
                        DIST_COEFFS,
                        False,
                        cv2.SOLVEPNP_IPPE_SQUARE
                    )
                    
                    if success and rvec is not None and tvec is not None:
                        # Refine with iterative method
                        rvec, tvec = cv2.solvePnPRefineLM(
                            single_marker_points,
                            corner,
                            CAMERA_MATRIX,
                            DIST_COEFFS,
                            rvec, tvec
                        )
                    
                    if success and rvec is not None and tvec is not None and not (np.isnan(rvec).any() or np.isnan(tvec).any()):
                        single_detected = True
                        
                        pts = corner.astype(np.int32)
                        cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
                        cv2.putText(frame, f'ID {MARKER_ID_TARGET}', tuple(pts[0]),
                                    cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
                        
                        cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, 8, 2)
                        
                        rot_mat, _ = cv2.Rodrigues(rvec)
                        pen_tip_3d = rot_mat @ SINGLE_TIP_OFFSET_MARKER + tvec.reshape(3, 1)
                        pen_tip_3d_global = pen_tip_3d.flatten()
                        
                        if single_pen_tip_3d_filter is None:
                            single_pen_tip_3d_filter = pen_tip_3d_global
                        else:
                            single_pen_tip_3d_filter = apply_filter(single_pen_tip_3d_filter, pen_tip_3d_global, 0.4, max_delta=200)
                        
                        pen_tip_2d, _ = cv2.projectPoints(
                            single_pen_tip_3d_filter.reshape(1, 1, 3).astype(np.float32),
                            np.zeros((3, 1), dtype=np.float32),
                            np.zeros((3, 1), dtype=np.float32),
                            CAMERA_MATRIX, DIST_COEFFS
                        )
                        
                        if single_pen_tip_2d_filter is None:
                            single_pen_tip_2d_filter = pen_tip_2d[0][0]
                        else:
                            single_pen_tip_2d_filter = apply_filter(single_pen_tip_2d_filter, pen_tip_2d[0][0], 0.5, max_delta=50)
                        
                        pen_pt = safe_point(single_pen_tip_2d_filter)
                        if pen_pt:
                            single_pen_tip_path.append(pen_pt)
                            if len(single_pen_tip_path) > 1000:
                                single_pen_tip_path.pop(0)
                            cv2.circle(frame, pen_pt, 5, (255, 0, 0), -1)
                            if len(single_pen_tip_path) > 1:
                                pt1, pt2 = single_pen_tip_path[-2], single_pen_tip_path[-1]
                                if pt1 and pt2:
                                    cv2.line(single_trajectory_canvas, pt1, pt2, (255, 0, 0), 5)
                        
                        if single_initial_tip is None:
                            single_initial_tip = single_pen_tip_3d_filter.copy()
                            single_initial_marker = tvec.flatten()
                        
                        single_frame_count += 1
                        tip_rel = single_pen_tip_3d_filter - single_initial_tip
                        marker_rel = tvec.flatten() #- single_initial_marker
                        angle = np.linalg.norm(rvec)
                        axis = abs(rvec.flatten() / angle * np.degrees(angle) if angle > 1e-6 else np.zeros(3) )
                        
                        single_data = {
                            'marker_id': marker_id,
                            'tip_x': tip_rel[0],
                            'tip_y': tip_rel[1],
                            'tip_z': tip_rel[2],
                            'marker_x': marker_rel[0],
                            'marker_y': marker_rel[1],
                            'marker_z': marker_rel[2],
                            'angle': np.degrees(angle),
                            'axis_x': axis[0],
                            'axis_y': axis[1],
                            'axis_z': axis[2]
                        }
                    break
        
        # Combine both trajectories onto frame
        gray = cv2.cvtColor(dodeca_trajectory_canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame = cv2.bitwise_and(frame, frame, mask=mask_inv)
        frame = cv2.add(frame, cv2.bitwise_and(dodeca_trajectory_canvas, dodeca_trajectory_canvas, mask=mask))
        
        gray = cv2.cvtColor(single_trajectory_canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame = cv2.bitwise_and(frame, frame, mask=mask_inv)
        frame = cv2.add(frame, cv2.bitwise_and(single_trajectory_canvas, single_trajectory_canvas, mask=mask))
        
        # Add status text
        cv2.putText(frame, f"Frame: {frame_idx+1}/{total_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"DODECA (Green): {'ON' if dodeca_detected else 'OFF'} ({dodeca_frame_count})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"SINGLE (Blue): {'ON' if single_detected else 'OFF'} ({single_frame_count})",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Write frame
        video_writer.write(frame)
        
        # Record data for both
        if dodeca_detected:
            now = datetime.now().isoformat(timespec="milliseconds")
            row = [
                now, frame_idx + 1,
                dodeca_data['marker_id'],
                dodeca_data['tip_x'], dodeca_data['tip_y'], dodeca_data['tip_z'],
                dodeca_data['marker_x'], dodeca_data['marker_y'], dodeca_data['marker_z'],
                dodeca_data['angle'],
                dodeca_data['axis_x'], dodeca_data['axis_y'], dodeca_data['axis_z']
            ]
            dodeca_rows.append(row)
        
        if single_detected:
            now = datetime.now().isoformat(timespec="milliseconds")
            row = [
                now, frame_idx + 1,
                single_data['marker_id'],
                single_data['tip_x'], single_data['tip_y'], single_data['tip_z'],
                single_data['marker_x'], single_data['marker_y'], single_data['marker_z'],
                single_data['angle'],
                single_data['axis_x'], single_data['axis_y'], single_data['axis_z']
            ]
            single_rows.append(row)
        
        frame_idx += 1
        
        # Progress
        if frame_idx % 30 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"  Progress: {progress:.1f}% | DODECA: {dodeca_frame_count} | SINGLE: {single_frame_count}")
    
    # Cleanup
    cap.release()
    video_writer.release()
    
    print(f"\n[SAVED] tracked_combined.mp4 ({frame_idx} frames processed)")
    print(f"  ├─ Dodecahedron tracking: {dodeca_frame_count} frames (Green)")
    print(f"  └─ Single marker tracking: {single_frame_count} frames (Blue)")
    
    # Save CSV files
    if dodeca_rows:
        csv_file = os.path.join(session_dir, 'data_dodeca.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [
                'timestamp', 'frame',
                'marker_id', 'tip_x', 'tip_y', 'tip_z',
                'marker_x', 'marker_y', 'marker_z',
                'angle', 'axis_x', 'axis_y', 'axis_z'
            ]
            writer.writerow(header)
            writer.writerows(dodeca_rows)
        print(f"[SAVED] data_dodeca.csv ({len(dodeca_rows)} rows)")
    else:
        print("[WARNING] No dodecahedron tracking data collected")
    
    if single_rows:
        csv_file = os.path.join(session_dir, 'data_single.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [
                'timestamp', 'frame',
                'marker_id', 'tip_x', 'tip_y', 'tip_z',
                'marker_x', 'marker_y', 'marker_z',
                'angle', 'axis_x', 'axis_y', 'axis_z'
            ]
            writer.writerow(header)
            writer.writerows(single_rows)
        print(f"[SAVED] data_single.csv ({len(single_rows)} rows)")
    else:
        print("[WARNING] No single marker tracking data collected")
    
    print(f"[COMBINED PHASE COMPLETE]\n")
    return True


def main():
    parser = argparse.ArgumentParser(description='Post-process video with two-phase marker tracking (Dodeca then Single)')
    parser.add_argument('-s', '--session', type=str, help='Session directory to process')
    parser.add_argument('-l', '--latest', action='store_true', help='Process latest session')
    parser.add_argument('-a', '--all', default=True, action='store_true', help='Process all pending sessions')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output files without prompting')
    parser.add_argument('--no-overwrite', action='store_true', help='Skip sessions with existing output files')
    parser.add_argument('-i', '--interactive', action='store_true', help='Prompt for overwrite on each session')
    
    args = parser.parse_args()
    
    # Determine overwrite behavior
    allow_overwrite = not args.no_overwrite  # Default is True (allow overwrite)
    if args.overwrite:
        allow_overwrite = True
    
    def process_session_full(session_path):
        """Process session with combined tracking: dodeca + single in one video."""
        print(f"\n{'#'*60}")
        print(f"# COMBINED PROCESSING: {os.path.basename(session_path)}")
        print(f"{'#'*60}")
        
        # Check for existing files and prompt if interactive mode
        if args.interactive:
            if not prompt_overwrite(session_path):
                return False
        
        # Combined phase: dodeca + single in one pass
        success = process_combined_phase(session_path, allow_overwrite=allow_overwrite)
        if not success:
            print("[ERROR] Combined phase failed!")
            return False
        
        print(f"\n{'#'*60}")
        print(f"# ✓ SESSION COMPLETE: {os.path.basename(session_path)}")
        print(f"{'#'*60}\n")
        return True
    
    if args.all:
        # Process all sessions
        base_dir = 'dataMarker'
        if not os.path.exists(base_dir):
            print("Error: dataMarker directory not found")
            return
        
        sessions = sorted([d for d in os.listdir(base_dir) if d.startswith('session_')])
        if not sessions:
            print("Error: No sessions found")
            return
        
        # Filter sessions based on overwrite settings
        pending_sessions = []
        for session in sessions:
            session_path = os.path.join(base_dir, session)
            tracked_combined = os.path.join(session_path, 'tracked_combined.mp4')
            raw_path = os.path.join(session_path, 'raw_camera.mp4')
            
            has_output = os.path.exists(tracked_combined)
            has_input = os.path.exists(raw_path)
            
            if has_input:
                if allow_overwrite or not has_output:
                    pending_sessions.append(session_path)
        
        if not pending_sessions:
            if allow_overwrite:
                print("No sessions found with raw_camera.mp4!")
            else:
                print("All sessions already processed! (use --overwrite to reprocess)")
            return
        
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING: {len(pending_sessions)} sessions")
        print(f"Mode: COMBINED (Dodeca + Single in one video)")
        print(f"Overwrite: {'ON' if allow_overwrite else 'OFF'}")
        print(f"Interactive: {'ON' if args.interactive else 'OFF'}")
        print(f"{'='*60}")
        
        for i, session_path in enumerate(pending_sessions, 1):
            print(f"\n[{i}/{len(pending_sessions)}]")
            process_session_full(session_path)
        
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
        
        process_session_full(session_dir)
    
    elif args.session:
        session_dir = args.session
        if not os.path.exists(session_dir):
            print(f"Error: Session directory not found: {session_dir}")
            return
        
        process_session_full(session_dir)
    else:
        print("Error: Specify session with -s <dir>, use -l for latest, or -a for all pending")
        return


if __name__ == '__main__':
    main()
