import cv2
from cv2 import aruco
import numpy as np
import pandas as pd
from datetime import datetime
import os
import csv
from PIL import ImageGrab
import time

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

#PEN_TIP_LOC = np.array([[0.02327], [-102.2512], [190]], dtype=np.float32)  # mm
PEN_TIP_LOC = np.array([[-5.8151941806526235], [-117.69125023287766], [142.09479755195974]], dtype=np.float32)
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


def grab_button_region():
    """Grab hanya region tombol, bukan full screen (jauh lebih cepat)."""
    try:
        # Dapatkan screen size
        screen = ImageGrab.grab()
        h, w = screen.size[1], screen.size[0]
        # Grab hanya area bawah (bottom 150 pixels)
        bbox = (0, h-150, w, h)
        button_region = ImageGrab.grab(bbox=bbox)
        return np.array(button_region)
    except Exception as e:
        # Fallback - return None jika gagal
        return None


class PerformanceMonitor:
    """Monitor FPS dan timing dari setiap proses."""
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.frame_times = []
        self.timings = {}
        
    def start_frame(self):
        """Tandai awal frame."""
        self.frame_start = time.time()
        self.current_timings = {}
    
    def mark(self, label):
        """Tandai waktu untuk operasi tertentu."""
        self.current_timings[label] = time.time()
    
    def end_frame(self):
        """Hitung timing untuk frame ini."""
        frame_time = (time.time() - self.frame_start) * 1000  # ms
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
        
        # Calculate segment timings
        if len(self.current_timings) > 0:
            times = sorted(self.current_timings.items(), key=lambda x: x[1])
            self.timings = {}
            for i, (label, t) in enumerate(times):
                if i == 0:
                    self.timings[label] = t - self.frame_start
                else:
                    self.timings[label] = (t - times[i-1][1]) * 1000
        
        return frame_time
    
    def get_fps(self):
        """Ambil FPS rata-rata."""
        if not self.frame_times:
            return 0
        return 1000 / (sum(self.frame_times) / len(self.frame_times))
    
    def get_frame_time(self):
        """Ambil frame time rata-rata dalam ms."""
        if not self.frame_times:
            return 0
        return sum(self.frame_times) / len(self.frame_times)


class UIDisplay:
    """Helper untuk menampilkan UI dengan layout terstruktur."""
    
    def __init__(self, frame):
        self.frame = frame
        self.h, self.w = frame.shape[:2]
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_size = 0.5
        self.font_thickness = 1
        self.line_height = 20
        self.text_color = (255, 255, 255)
        self.bg_color = (0, 0, 0)
        self.bg_alpha = 0.7
    
    def draw_text_with_bg(self, x, y, text, color=None, bg_color=None):
        """Draw text dengan background semi-transparent."""
        if color is None:
            color = self.text_color
        if bg_color is None:
            bg_color = self.bg_color
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(text, self.font, self.font_size, self.font_thickness)
        
        # Draw semi-transparent background
        overlay = self.frame.copy()
        cv2.rectangle(overlay, (x - 5, y - text_h - 5), (x + text_w + 5, y + baseline + 5), bg_color, -1)
        cv2.addWeighted(overlay, self.bg_alpha, self.frame, 1 - self.bg_alpha, 0, self.frame)
        
        # Draw text
        cv2.putText(self.frame, text, (x, y), self.font, self.font_size, color, self.font_thickness)
    
    def draw_panel(self, title, items, x, y, color=(0, 200, 0)):
        """Draw panel dengan title dan list items."""
        # Draw panel background
        panel_h = len(items) * self.line_height + 40
        overlay = self.frame.copy()
        cv2.rectangle(overlay, (x, y), (x + 300, y + panel_h), self.bg_color, -1)
        cv2.addWeighted(overlay, self.bg_alpha, self.frame, 1 - self.bg_alpha, 0, self.frame)
        
        # Draw border
        cv2.rectangle(self.frame, (x, y), (x + 300, y + panel_h), color, 2)
        
        # Draw title
        cv2.putText(self.frame, title, (x + 10, y + 25), self.font, self.font_size + 0.1, color, 2)
        
        # Draw items
        for i, item in enumerate(items):
            item_y = y + 45 + i * self.line_height
            cv2.putText(self.frame, item, (x + 15, item_y), self.font, self.font_size - 0.1, self.text_color, 1)
        
        return y + panel_h + 10
    
    def draw_status_panel(self, num_markers, recording, rec_frame_count):
        """Draw status panel di kiri atas."""
        items = [
            f"Markers: {num_markers}",
            f"Pen Tip: {'ON' if num_markers >= 2 else 'OFF'}",
            f"REC: {'●' if recording else '○'} {rec_frame_count} frames" if recording else f"REC: ○ Idle"
        ]
        return self.draw_panel("STATUS", items, 10, 10, (0, 200, 100))
    
    def draw_performance_panel(self, fps, frame_ms):
        """Draw performance panel di kanan atas."""
        fps_color = (0, 255, 0) if fps > 20 else (0, 165, 255)
        items = [
            f"FPS: {fps:.1f}",
            f"Frame: {frame_ms:.1f}ms"
        ]
        return self.draw_panel("PERFORMANCE", items, self.w - 310, 10, fps_color)
    
    def draw_recording_panel(self, perf):
        """Draw recording details panel."""
        grab_time = (perf.current_timings.get('grab_screen', perf.frame_start) - perf.frame_start) * 1000 if 'grab_screen' in perf.current_timings else 0
        buffer_time = (perf.current_timings.get('buffer_frame', perf.frame_start) - perf.current_timings.get('grab_screen', perf.frame_start)) * 1000 if 'buffer_frame' in perf.current_timings else 0
        
        items = [
            f"Grab: {grab_time:.1f}ms",
            f"Buffer: {buffer_time:.1f}ms",
            f"Images: {len(perf.current_timings)} captured"
        ]
        return self.draw_panel("RECORDING STATS", items, self.w - 310, 120, (0, 165, 255))





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
    rec_frames = []  # Buffer untuk menyimpan images
    rec_frame_count = 0
    initial_tip = None
    initial_marker = None
    video_raw = None  # VideoWriter untuk raw camera
    video_tracked = None  # VideoWriter untuk tracked/processed frame
    
    # UI Display
    ui = None
    
    print("[START] Camera tracking initialized")
    
    # Initialize performance monitor
    perf = PerformanceMonitor()
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Cannot read camera")
        return
    
    trajectory_canvas = np.zeros_like(frame)
    ui = UIDisplay(frame)
    prm = detector.getDetectorParameters()
    
    while True:
        perf.start_frame()
        
        ret, frame = cap.read()

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # thresh_frame = cv2.adaptiveThreshold(
        #     gray, 
        #     255, 
        #     cv2.ADAPTIVE_THRESH_MEAN_C, 
        #     cv2.THRESH_BINARY_INV, # ArUco sering membalik warna untuk mencari kontur
        #     prm.adaptiveThreshWinSizeMin, 
        #     prm.adaptiveThreshConstant
        # )
        # cv2.imshow("ArUco-Style Thresholding", thresh_frame)
        
        perf.mark("read_frame")
        if not ret:
            break
        
        # Save raw video frame jika recording
        if recording and video_raw is not None:
            video_raw.write(frame)
        
        # Detect markers
        corners, ids, _ = detector.detectMarkers(frame)
        perf.mark("detect_markers")
        num_markers = len(ids) if ids is not None else 0
        
        if num_markers >= 2 and ids is not None:
            my_id = [1, 5, 7]
            mask_id = np.isin(ids, my_id)
            indices_id = np.where(mask_id)[0]
            corners_mask = [corners[i] for i in indices_id]
            corners = corners_mask
            ids = ids[indices_id]

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
            perf.mark("estimate_pose")
            
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
                
                # Only track pen tip when recording
                if recording:
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
                    
                    # Buffer image untuk disimpan nanti
                    # Hanya capture jika frame tertentu (reduce frequency)
                    if True:  # Capture setiap frame ganjil (50% reduction)
                        try:
                            screen = np.array(ImageGrab.grab())
                            perf.mark("grab_screen")
                            screen_gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)

                            crop = CROP_COORDS[CROP_INDEX]
                            screen_crop = screen_gray[crop[1]:crop[3], crop[0]:crop[2]]
                            img_processed = screen_crop.astype(np.float32) + 20
                            img_processed[img_processed > 255] = 255
                            img_processed = img_processed.astype(np.uint8)
                            rec_frames.append((rec_frame_count, img_processed))
                            perf.mark("buffer_frame")
                        except:
                            pass
        
        # Combine trajectory with frame
        gray = cv2.cvtColor(trajectory_canvas, cv2.COLOR_BGR2GRAY)
        perf.mark("trajectory_combine")
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame = cv2.bitwise_and(frame, frame, mask=mask_inv)
        frame = cv2.add(frame, cv2.bitwise_and(trajectory_canvas, trajectory_canvas, mask=mask))
        
        # Initialize UI Display untuk frame ini
        ui = UIDisplay(frame)
        
        # Draw Status Panel (kiri atas)
        ui.draw_status_panel(num_markers, recording, rec_frame_count)
        
        # Draw Performance Panel (kanan atas)
        fps = perf.get_fps()
        frame_ms = perf.get_frame_time()
        ui.draw_performance_panel(fps, frame_ms)
        
        # Draw Recording Panel jika sedang recording (kanan bawah)
        if recording:
            ui.draw_recording_panel(perf)
        
        perf.mark("ui_render")
        
        # Button detection - reduce frequency dari setiap frame menjadi setiap 5 frame
        if frame_count % 5 == 0:
            try:
                screen = grab_button_region()
                perf.mark("button_grab")
                if screen is not None:
                    btn_on = detect_button(screen, BUTTON_ON_COLOR)
                    btn_off = detect_button(screen, BUTTON_OFF_COLOR)
                    perf.mark("button_detect")
                else:
                    btn_on = btn_off = 0
                
                if btn_on > BUTTON_THRESHOLD and button_state != 'ON':
                    recording = True
                    button_state = 'ON'
                    rec_dir = f"dataMarker/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    os.makedirs(rec_dir, exist_ok=True)

                    # Setup video writers
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    fps_video = 30  # Target FPS untuk video
                    frame_size = (frame.shape[1], frame.shape[0])
                    video_raw = cv2.VideoWriter(f"{rec_dir}/raw_camera.mp4", fourcc, fps_video, frame_size)
                    video_tracked = cv2.VideoWriter(f"{rec_dir}/tracked.mp4", fourcc, fps_video, frame_size)

                    # Reset semua buffer dan state
                    rec_rows = []
                    rec_frames = []
                    rec_frame_count = 0
                    initial_tip = None
                    initial_marker = None
                    trajectory_canvas = np.zeros_like(frame)  # Clear visualization
                    pen_tip_path = []
                    pen_tip_3d_filter = None
                    pen_tip_2d_filter = None
                    origin_2d_filter = None
                    print("[REC] Started")
                
                elif btn_off > BUTTON_THRESHOLD and button_state != 'OFF':
                    if recording:
                        # Release video writers
                        if video_raw is not None:
                            video_raw.release()
                            video_raw = None
                        if video_tracked is not None:
                            video_tracked.release()
                            video_tracked = None
                        
                        os.makedirs(rec_dir, exist_ok=True)
                        # Save CSV
                        csv_file = f"{rec_dir}/data.csv"
                        with open(csv_file, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(['timestamp', 'frame', 'marker_id', 'tip_x', 'tip_y', 'tip_z', 'marker_x', 'marker_y', 'marker_z', 'angle', 'axis_x', 'axis_y', 'axis_z'])
                            writer.writerows(rec_rows)
                        print(f"[SAVED] {csv_file} ({len(rec_rows)} frames)")
                        print(f"[SAVED] raw_camera.mp4 & tracked.mp4 (video files)")
                        # Save all images from buffer
                        if rec_frames:
                            img_dir = f"{rec_dir}/images"
                            os.makedirs(img_dir, exist_ok=True)
                            for frame_idx, frame_img in rec_frames:
                                cv2.imwrite(f"{img_dir}/frame_{frame_idx:06d}.png", frame_img)
                            print(f"[SAVED] {len(rec_frames)} images to {img_dir}")
                    recording = False
                    button_state = 'OFF'
                    rec_rows, rec_frames = [], []  # Clear buffers
                    pen_tip_path = []  # Clear trajectory
                    pen_tip_3d_filter = None
            except:
                pass
        
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
        # Save tracked video frame jika recording
        if recording and video_tracked is not None:
            video_tracked.write(frame)
        
        frame_time = perf.end_frame()
        
        # Log performance setiap 30 frame
        if frame_count % 30 == 0 and frame_count > 0:
            fps = perf.get_fps()
            status = "REC" if recording else "IDLE"
            print(f"[PERF] Frame {frame_count} | {status} | FPS: {fps:.1f} | FrameTime: {frame_time:.1f}ms")
        
        frame_count += 1
    
    # Cleanup - save jika masih recording
    if recording and rec_rows:
        # Release video writers
        if video_raw is not None:
            video_raw.release()
        if video_tracked is not None:
            video_tracked.release()
        
        os.makedirs(rec_dir, exist_ok=True)
        csv_file = f"{rec_dir}/data.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'frame', 'marker_id', 'tip_x', 'tip_y', 'tip_z', 'marker_x', 'marker_y', 'marker_z', 'angle', 'axis_x', 'axis_y', 'axis_z'])
            writer.writerows(rec_rows)
        print(f"[SAVED] {csv_file} ({len(rec_rows)} frames)")
        print(f"[SAVED] raw_camera.mp4 & tracked.mp4 (video files)")
        if rec_frames:
            img_dir = f"{rec_dir}/images"
            os.makedirs(img_dir, exist_ok=True)
            for frame_idx, frame_img in rec_frames:
                cv2.imwrite(f"{img_dir}/frame_{frame_idx:06d}.png", frame_img)
            print(f"[SAVED] {len(rec_frames)} images to {img_dir}")
        pen_tip_path = []  # Clear trajectory after saving
    
    cap.release()
    cv2.destroyAllWindows()
    print("[END] Program closed")


if __name__ == '__main__':
    main()
