"""
Calibration Program: Find Pen Tip Offset from Video Recording
- Detects markers in video frames
- Calculates transformation between marker pose and pen tip position
- Estimates optimal tip offset using multiple reference points

python calibrate_tip_offset_from_video.py raw_camera.mp4 -m 15 -s 16
"""

import cv2
from cv2 import aruco
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from scipy.optimize import least_squares
import json

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


def setup_aruco():
    """Initialize ArUco detector."""
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(
        aruco.getPredefinedDictionary(aruco.DICT_4X4_50), 
        params
    )


def get_marker_pose(marker_corners, marker_size_mm):
    """
    Estimate marker pose from corners.
    
    Args:
        marker_corners: 4 corner points of marker in image
        marker_size_mm: Marker size in millimeters
    
    Returns:
        rvec, tvec: Rotation and translation vectors
    """
    marker_size_m = marker_size_mm / 1000.0
    
    # 3D points of marker in marker coordinate system
    marker_3d = np.array([
        [-marker_size_m/2,  marker_size_m/2, 0],
        [ marker_size_m/2,  marker_size_m/2, 0],
        [ marker_size_m/2, -marker_size_m/2, 0],
        [-marker_size_m/2, -marker_size_m/2, 0]
    ], dtype=np.float32)
    
    success, rvec, tvec = cv2.solvePnP(
        marker_3d,
        marker_corners.astype(np.float32),
        CAMERA_MATRIX,
        DIST_COEFFS,
        False,
        cv2.SOLVEPNP_IPPE_SQUARE
    )
    
    return success, rvec, tvec


def extract_marker_data_from_video(video_path, marker_id, marker_size_mm, 
                                   num_frames=None, visualization=False):
    """
    Extract marker pose data from video.
    
    Args:
        video_path: Path to video file
        marker_id: Target marker ID to track
        marker_size_mm: Marker size in millimeters
        num_frames: Number of frames to process (None = all)
        visualization: Show detected markers
    
    Returns:
        List of (rvec, tvec) tuples for detected markers
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return []
    
    detector = setup_aruco()
    poses = []
    frame_count = 0
    detected_count = 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video: {Path(video_path).name}")
    print(f"Target marker ID: {marker_id}, Size: {marker_size_mm}mm")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if num_frames and frame_count >= num_frames:
            break
        
        # Detect markers
        corners, ids, _ = detector.detectMarkers(frame)
        
        if ids is not None:
            for i, detected_id in enumerate(ids.flatten()):
                if detected_id == marker_id:
                    corner = corners[i][0]
                    success, rvec, tvec = get_marker_pose(corner, marker_size_mm)
                    
                    if success and rvec is not None and tvec is not None:
                        if not (np.isnan(rvec).any() or np.isnan(tvec).any()):
                            poses.append({
                                'frame': frame_count,
                                'rvec': rvec.flatten().copy(),
                                'tvec': tvec.flatten().copy()
                            })
                            detected_count += 1
                            
                            if visualization:
                                cv2.polylines(frame, [corner.astype(np.int32)], True, (0, 255, 0), 2)
                                cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, 20, 2)
                                cv2.putText(frame, f"ID {marker_id} (Frame {frame_count})", 
                                           tuple(corner[0].astype(int)), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    break
        
        if visualization and detected_count > 0:
            cv2.imshow(f"Marker {marker_id} Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {progress:.1f}% | Detected: {detected_count} frames")
    
    cap.release()
    if visualization:
        cv2.destroyAllWindows()
    
    print(f"✓ Extracted {detected_count} frames with marker {marker_id}\n")
    return poses


def calibrate_tip_offset_single_marker(video_path, marker_id, marker_size_mm, 
                                       reference_positions=None, visualization=False):
    """
    Calibrate tip offset for a single marker using video data.
    
    Args:
        video_path: Path to video file
        marker_id: Marker ID to calibrate
        marker_size_mm: Marker size in millimeters
        reference_positions: Known tip positions (optional, for multi-point calibration)
        visualization: Show detection visualization
    
    Returns:
        Estimated tip offset in meters
    """
    
    poses = extract_marker_data_from_video(
        video_path, marker_id, marker_size_mm, 
        visualization=visualization
    )
    
    if len(poses) < 3:
        print(f"Error: Need at least 3 frames with marker {marker_id}, got {len(poses)}")
        return None
    
    # Convert all poses to array form
    rvecs = np.array([p['rvec'] for p in poses])  # (N, 3)
    tvecs = np.array([p['tvec'] for p in poses])  # (N, 3)
    
    # Get rotation matrices
    rot_mats = []
    for rvec in rvecs:
        rot_mat, _ = cv2.Rodrigues(rvec)
        rot_mats.append(rot_mat)
    rot_mats = np.array(rot_mats)  # (N, 3, 3)
    
    print(f"Analyzing {len(poses)} frames with detected marker...")
    
    # Method 1: Simple triangulation from multiple poses
    # If pen tip position is constant in world coordinates:
    # tip_world = R_i @ tip_marker + tvec_i (for all i)
    # This means: tip_marker = R_i^T @ (tip_world - tvec_i)
    # All these should be equal, so we find tip_marker that minimizes error
    
    def residuals(tip_offset, rot_mats, tvecs):
        """Calculate residuals for tip offset estimation."""
        errors = []
        tip_offset = tip_offset.reshape(3, 1)
        
        # Project tip to world coordinates
        tip_world_estimates = []
        for i, (R, tvec) in enumerate(zip(rot_mats, tvecs)):
            tip_world = R @ tip_offset + tvec.reshape(3, 1)
            tip_world_estimates.append(tip_world.flatten())
        
        tip_world_estimates = np.array(tip_world_estimates)
        tip_world_mean = tip_world_estimates.mean(axis=0)
        
        # Calculate distance from each estimate to mean
        errors = np.linalg.norm(tip_world_estimates - tip_world_mean, axis=1)
        return errors
    
    # Initial guess: center of markers
    initial_guess = np.array([0, 0, 0])
    
    result = least_squares(
        residuals,
        initial_guess,
        args=(rot_mats, tvecs),
        verbose=0
    )
    
    tip_offset_estimated = result.x.reshape(3, 1)
    residual_norm = np.linalg.norm(result.fun)
    
    print(f"✓ Estimated tip offset (meters):")
    print(f"  X: {tip_offset_estimated[0, 0]:.6f} m ({tip_offset_estimated[0, 0]*1000:.2f} mm)")
    print(f"  Y: {tip_offset_estimated[1, 0]:.6f} m ({tip_offset_estimated[1, 0]*1000:.2f} mm)")
    print(f"  Z: {tip_offset_estimated[2, 0]:.6f} m ({tip_offset_estimated[2, 0]*1000:.2f} mm)")
    print(f"  Residual error: {residual_norm:.6f} m ({residual_norm*1000:.2f} mm)\n")
    
    # Calculate confidence (std of estimates)
    confidence = []
    tip_world_all = []
    for i, (R, tvec) in enumerate(zip(rot_mats, tvecs)):
        tip_world = R @ tip_offset_estimated + tvec.reshape(3, 1)
        tip_world_all.append(tip_world.flatten())
    
    tip_world_all = np.array(tip_world_all)
    std_x = tip_world_all[:, 0].std()
    std_y = tip_world_all[:, 1].std()
    std_z = tip_world_all[:, 2].std()
    
    print(f"Position stability (std):")
    print(f"  X: {std_x:.6f} m ({std_x*1000:.2f} mm)")
    print(f"  Y: {std_y:.6f} m ({std_y*1000:.2f} mm)")
    print(f"  Z: {std_z:.6f} m ({std_z*1000:.2f} mm)\n")
    
    return tip_offset_estimated.flatten()


def calibrate_dodecahedron_tip(video_path, model_points_csv, visualization=False):
    """
    Calibrate tip offset for dodecahedron marker setup.
    
    Args:
        video_path: Path to video file
        model_points_csv: Path to CSV with marker 3D model points
        visualization: Show detection visualization
    
    Returns:
        Estimated tip offset in mm
    """
    
    # Load marker points
    try:
        data = pd.read_csv(model_points_csv)
        pts = data[['x', 'y', 'z']].values.tolist()
        model_pts_by_id = [pts[i:i+4] for i in range(0, len(pts), 4)]
        print(f"Loaded {len(model_pts_by_id)} dodecahedron markers")
    except FileNotFoundError:
        print(f"Error: {model_points_csv} not found")
        return None
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None
    
    detector = setup_aruco()
    poses = []
    frame_count = 0
    detected_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Processing video: {Path(video_path).name}")
    print(f"Detecting dodecahedron markers (IDs 0-11)...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        corners, ids, _ = detector.detectMarkers(frame)
        
        if ids is not None:
            my_ids = list(range(0, 12))
            mask = np.isin(ids.flatten(), my_ids)
            indices = np.where(mask)[0]
            
            if len(indices) > 0:
                corners_mask = [corners[i] for i in indices]
                ids_mask = ids[indices]
                
                img_pts, model_pts = [], []
                for i, corner in enumerate(corners_mask):
                    mid = ids_mask[i][0]
                    if mid < len(model_pts_by_id):
                        img_pts.extend(corner[0])
                        model_pts.extend(model_pts_by_id[mid])
                
                if len(model_pts) >= 4:
                    try:
                        _, rvec, tvec, _ = cv2.solvePnPRansac(
                            np.array(model_pts),
                            np.array(img_pts),
                            CAMERA_MATRIX, DIST_COEFFS
                        )
                        
                        if rvec is not None and tvec is not None:
                            if not (np.isnan(rvec).any() or np.isnan(tvec).any()):
                                poses.append({
                                    'frame': frame_count,
                                    'rvec': rvec.flatten().copy(),
                                    'tvec': tvec.flatten().copy(),
                                    'num_markers': len(indices)
                                })
                                detected_count += 1
                                
                                if visualization:
                                    cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, 30, 4)
                    except:
                        pass
        
        if visualization and detected_count > 0:
            cv2.imshow("Dodecahedron Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Progress: {progress:.1f}% | Detected: {detected_count} frames")
    
    cap.release()
    if visualization:
        cv2.destroyAllWindows()
    
    if len(poses) < 3:
        print(f"Error: Need at least 3 frames with dodecahedron, got {len(poses)}")
        return None
    
    rvecs = np.array([p['rvec'] for p in poses])
    tvecs = np.array([p['tvec'] for p in poses])
    
    rot_mats = []
    for rvec in rvecs:
        rot_mat, _ = cv2.Rodrigues(rvec)
        rot_mats.append(rot_mat)
    rot_mats = np.array(rot_mats)
    
    print(f"✓ Extracted {detected_count} frames with dodecahedron\n")
    
    def residuals(tip_offset, rot_mats, tvecs):
        errors = []
        tip_offset = tip_offset.reshape(3, 1)
        
        tip_world_estimates = []
        for i, (R, tvec) in enumerate(zip(rot_mats, tvecs)):
            tip_world = R @ tip_offset + tvec.reshape(3, 1)
            tip_world_estimates.append(tip_world.flatten())
        
        tip_world_estimates = np.array(tip_world_estimates)
        tip_world_mean = tip_world_estimates.mean(axis=0)
        
        errors = np.linalg.norm(tip_world_estimates - tip_world_mean, axis=1)
        return errors
    
    initial_guess = np.array([0, 0, 0])
    
    result = least_squares(
        residuals,
        initial_guess,
        args=(rot_mats, tvecs),
        verbose=0
    )
    
    tip_offset_estimated = result.x.reshape(3, 1)
    residual_norm = np.linalg.norm(result.fun)
    
    print(f"✓ Estimated tip offset for dodecahedron (mm):")
    print(f"  X: {tip_offset_estimated[0, 0]*1000:.2f} mm")
    print(f"  Y: {tip_offset_estimated[1, 0]*1000:.2f} mm")
    print(f"  Z: {tip_offset_estimated[2, 0]*1000:.2f} mm")
    print(f"  Residual error: {residual_norm*1000:.2f} mm\n")
    
    # Calculate position stability
    tip_world_all = []
    for i, (R, tvec) in enumerate(zip(rot_mats, tvecs)):
        tip_world = R @ tip_offset_estimated + tvec.reshape(3, 1)
        tip_world_all.append(tip_world.flatten())
    
    tip_world_all = np.array(tip_world_all)
    std_x = tip_world_all[:, 0].std() * 1000
    std_y = tip_world_all[:, 1].std() * 1000
    std_z = tip_world_all[:, 2].std() * 1000
    
    print(f"Position stability (std in mm):")
    print(f"  X: {std_x:.2f} mm")
    print(f"  Y: {std_y:.2f} mm")
    print(f"  Z: {std_z:.2f} mm\n")
    
    return tip_offset_estimated.flatten() * 1000  # Convert to mm


def save_calibration_result(output_file, tip_offset, marker_type="single", marker_id=None):
    """Save calibration result to JSON file."""
    result = {
        'marker_type': marker_type,
        'marker_id': marker_id,
        'tip_offset': {
            'x': float(tip_offset[0]),
            'y': float(tip_offset[1]),
            'z': float(tip_offset[2])
        },
        'units': 'meters' if marker_type == 'single' else 'millimeters'
    }
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✓ Saved calibration result to {output_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate pen tip offset from video recording'
    )
    parser.add_argument('video', type=str, help='Path to video file')
    parser.add_argument('-m', '--marker-id', type=int, default=15, 
                       help='Marker ID to calibrate (default: 15)')
    parser.add_argument('-s', '--marker-size', type=float, default=16, 
                       help='Marker size in mm (default: 16)')
    parser.add_argument('-d', '--dodeca', action='store_true', 
                       help='Calibrate dodecahedron instead of single marker')
    parser.add_argument('-mp', '--model-points', type=str, 
                       default='markers/model_points_4x4.csv',
                       help='Path to model points CSV for dodecahedron')
    parser.add_argument('-o', '--output', type=str, help='Output JSON file')
    parser.add_argument('-v', '--visualization', action='store_true',
                       help='Show detection visualization')
    
    args = parser.parse_args()
    
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        return
    
    print("=" * 60)
    print("PEN TIP OFFSET CALIBRATION")
    print("=" * 60 + "\n")
    
    if args.dodeca:
        print("MODE: Dodecahedron (multi-marker)\n")
        tip_offset = calibrate_dodecahedron_tip(
            args.video,
            args.model_points,
            visualization=args.visualization
        )
        if tip_offset is not None:
            print(f"Result array (for code):")
            print(f"  np.array([[{tip_offset[0]:.6f}], [{tip_offset[1]:.6f}], [{tip_offset[2]:.6f}]], dtype=np.float32)")
            
            if args.output:
                save_calibration_result(args.output, tip_offset, "dodecahedron")
    else:
        print(f"MODE: Single Marker (ID {args.marker_id})\n")
        tip_offset = calibrate_tip_offset_single_marker(
            args.video,
            args.marker_id,
            args.marker_size,
            visualization=args.visualization
        )
        if tip_offset is not None:
            print(f"Result array (for code, in meters):")
            print(f"  np.array([[{tip_offset[0]:.6f}], [{tip_offset[1]:.6f}], [{tip_offset[2]:.6f}]], dtype=np.float64)")
            print(f"\nResult array (in mm):")
            print(f"  np.array([[{tip_offset[0]*1000:.6f}], [{tip_offset[1]*1000:.6f}], [{tip_offset[2]*1000:.6f}]], dtype=np.float32)")
            
            if args.output:
                save_calibration_result(args.output, tip_offset, "single", args.marker_id)
    
    print("=" * 60)


if __name__ == '__main__':
    main()
