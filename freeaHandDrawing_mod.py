import cv2
from cv2 import aruco
import numpy as np
import pandas as pd
import csv
from datetime import datetime
from collections import deque


def load_camera_calibration():
    cameraMatrix = np.array(
        [
            [1156.575220787158, 0.0, 641.235796583754],
            [0.0, 1159.917623853068, 403.13243457956634],
            [0.0, 0.0, 1.0],
        ],
        dtype="double",
    )
    distCoeffs = np.array(
        [
            [0.21463756498771683],
            [-0.8166333234031081],
            [0.016152523688770088],
            [-0.00042204457570052013],
            [0.2851862067489753],
        ],
        dtype="double",
    )
    return cameraMatrix, distCoeffs


def setup_video_source(src):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Video opened: {width:.0f}x{height:.0f} @ {fps:.2f} FPS")
    return cap, fps


def setup_aruco():
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    return detector


def estimatePoseGlobal(model_pts, img_pts, cameraMatrix, distCoeffs):
    if len(model_pts) < 4:
        return None, None
    try:
        ok, rvec, tvec, _ = cv2.solvePnPRansac(
            np.array(model_pts, dtype=np.float32),
            np.array(img_pts, dtype=np.float32),
            cameraMatrix,
            distCoeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None, None
        return rvec, tvec
    except cv2.error:
        return None, None


def rvec_to_axis_angle_deg(rvec):
    """
    rvec = axis * angle(rad). Return angle in deg + axis components.
    """
    r = rvec.reshape(3).astype(float)
    angle_rad = float(np.linalg.norm(r))
    if angle_rad < 1e-12:
        return 0.0, 0.0, 0.0, 0.0
    axis = r / angle_rad
    angle_deg = angle_rad * 180.0 / np.pi
    return float(angle_deg), float(axis[0]), float(axis[1]), float(axis[2])


def write_csv(csv_name, rows):
    with open(csv_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "frame_idx",
                "markers_used",

                # TIP dalam GLOBAL (mm)
                "tip_global_x_mm",
                "tip_global_y_mm",
                "tip_global_z_mm",

                # Pose global->camera (t dalam meter)
                "t_glob_x_m",
                "t_glob_y_m",
                "t_glob_z_m",

                # Orientasi global->camera (axis-angle dari r_glob)
                "rot_angle_deg",
                "rot_axis_x",
                "rot_axis_y",
                "rot_axis_z",
            ]
        )
        writer.writerows(rows)


def main():
    # ========== SOURCE ==========
    video_src = 0  # 0 = webcam. Bisa ganti path video file.
    cap, fps = setup_video_source(video_src)

    cameraMatrix, distCoeffs = load_camera_calibration()
    detector = setup_aruco()

    # ========== LOAD MODEL POINTS ==========
    try:
        data = pd.read_csv("markers/model_points_4x4.csv")
    except FileNotFoundError:
        print("Error: 'markers/model_points_4x4.csv' not found.")
        return

    pts = data[["x", "y", "z"]].values.tolist()  # diasumsikan dalam satuan yang konsisten (umumnya mm atau m)
    model_pts_by_id = [pts[i : i + 4] for i in range(0, len(pts), 4)]

    # ========== TIP DEFINITION (CAMERA FRAME) ==========
    # Kamu bilang mau ambil data dari "tip global position".
    # Program ini menganggap tip didefinisikan di frame kamera (mm), lalu diubah ke global.
    pen_tip_loc_mm = np.array([[-0.02327], [-102.2512], [132.8306]], dtype=np.float64)
    pen_tip_cam_m = pen_tip_loc_mm / 1000.0  # (3,1) meter

    print("Pen tip (camera frame):")
    print(f"  X={pen_tip_loc_mm[0,0]:.4f} mm, Y={pen_tip_loc_mm[1,0]:.4f} mm, Z={pen_tip_loc_mm[2,0]:.4f} mm")
    print(f"  |tip|={np.linalg.norm(pen_tip_loc_mm):.2f} mm")

    # ========== TRAJECTORY ==========
    plot_pen_tip = True
    pen_tip_path = deque(maxlen=2000)

    # canvas trajectory
    ret, last_frame = cap.read()
    if not ret:
        print("Could not read first frame.")
        return
    trajectory_canvas = np.zeros_like(last_frame)

    # ========== RECORDING ==========
    recording = False
    rows = []
    frame_idx = 0
    csv_name = f"tip_global_record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        corners, ids, _ = detector.detectMarkers(frame)

        pose_valid = False
        tip_global_mm = None
        markers_used = 0
        r_glob = None
        t_glob = None

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            image_points_collected = []
            model_points_collected = []

            for i, corner in enumerate(corners):
                marker_id = int(ids[i][0])
                if 0 <= marker_id < len(model_pts_by_id):
                    # 4 image corners
                    for p2d in corner[0]:
                        image_points_collected.append(p2d)
                    # 4 model points
                    for p3d in model_pts_by_id[marker_id]:
                        model_points_collected.append(p3d)

            markers_used = len(image_points_collected) // 4

            r_glob, t_glob = estimatePoseGlobal(
                model_points_collected, image_points_collected, cameraMatrix, distCoeffs
            )

            if r_glob is not None and t_glob is not None:
                pose_valid = True

                # draw axes (skala 30 "unit model")
                cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, r_glob, t_glob, 30, 4)

                # global->camera: X_cam = R*X_glob + t
                R_glob, _ = cv2.Rodrigues(r_glob)
                t_glob = t_glob.reshape(3, 1)

                # invert to get tip in global:
                # X_glob = R^T * (X_cam - t)
                tip_global_m = R_glob.T @ (pen_tip_cam_m - t_glob)  # (3,1)
                tip_global_mm = (tip_global_m * 1000.0).reshape(3)  # (3,)

                # project tip_global to draw on image (optional)
                if plot_pen_tip:
                    tip_global_3d = tip_global_m.reshape(1, 1, 3).astype(np.float64)
                    tip_2d, _ = cv2.projectPoints(tip_global_3d, r_glob, t_glob, cameraMatrix, distCoeffs)
                    u, v = tip_2d.reshape(2)
                    if not (np.isnan(u) or np.isnan(v)):
                        pt = (int(round(u)), int(round(v)))
                        pen_tip_path.append(pt)

                        # draw new segment on persistent canvas
                        if len(pen_tip_path) > 1:
                            cv2.line(trajectory_canvas, pen_tip_path[-2], pen_tip_path[-1], (0, 255, 0), 3)
                        cv2.circle(frame, pt, 5, (0, 255, 0), -1)

                # ===== RECORDING =====
                if recording:
                    now = datetime.now().isoformat(timespec="milliseconds")
                    angle_deg, ax, ay, az = rvec_to_axis_angle_deg(r_glob)
                    rows.append(
                        [
                            now,
                            frame_idx,
                            markers_used,
                            float(tip_global_mm[0]),
                            float(tip_global_mm[1]),
                            float(tip_global_mm[2]),
                            float(t_glob[0, 0]),
                            float(t_glob[1, 0]),
                            float(t_glob[2, 0]),
                            angle_deg,
                            ax,
                            ay,
                            az,
                        ]
                    )

        # ===== Combine frame + trajectory canvas =====
        img2gray = cv2.cvtColor(trajectory_canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        lines_fg = cv2.bitwise_and(trajectory_canvas, trajectory_canvas, mask=mask)
        display_frame = cv2.add(frame_bg, lines_fg)

        # ===== UI =====
        cv2.rectangle(display_frame, (0, 0), (820, 95), (0, 0, 0), -1)
        cv2.putText(
            display_frame,
            "q=quit | t=toggle REC | c=capture once | w=write+clear | p=traj on/off | x=clear",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display_frame,
            f"REC: {'ON' if recording else 'OFF'} | rows: {len(rows)} | markers_used: {markers_used} | pose: {'OK' if pose_valid else 'NO'}",
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        if tip_global_mm is not None:
            xg, yg, zg = tip_global_mm
            cv2.putText(
                display_frame,
                f"TIP GLOBAL (mm): X={xg:.1f}  Y={yg:.1f}  Z={zg:.1f}",
                (10, 74),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                display_frame,
                "TIP GLOBAL (mm): -",
                (10, 74),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (128, 128, 128),
                2,
            )

        cv2.imshow("Multi-marker Global Tip Recording", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # toggle recording
        if key == ord("t"):
            recording = not recording

        # capture once (only if pose valid)
        if key == ord("c"):
            if pose_valid and tip_global_mm is not None and r_glob is not None and t_glob is not None:
                now = datetime.now().isoformat(timespec="milliseconds")
                angle_deg, ax, ay, az = rvec_to_axis_angle_deg(r_glob)
                rows.append(
                    [
                        now,
                        frame_idx,
                        markers_used,
                        float(tip_global_mm[0]),
                        float(tip_global_mm[1]),
                        float(tip_global_mm[2]),
                        float(t_glob[0, 0]),
                        float(t_glob[1, 0]),
                        float(t_glob[2, 0]),
                        angle_deg,
                        ax,
                        ay,
                        az,
                    ]
                )

        # write + clear
        if key == ord("w"):
            if len(rows) > 0:
                write_csv(csv_name, rows)
                print(f"[OK] Saved: {csv_name} (rows={len(rows)})")
                rows.clear()
                pen_tip_path.clear()
                trajectory_canvas[:] = 0
                print("[OK] Cleared buffer & trajectory after save.")
            else:
                print("[INFO] No data to write.")

        # toggle trajectory display (stop drawing new points)
        if key == ord("p"):
            plot_pen_tip = not plot_pen_tip
            if not plot_pen_tip:
                # clear path + canvas (biar tidak numpuk)
                pen_tip_path.clear()
                trajectory_canvas[:] = 0

        # clear without saving
        if key == ord("x"):
            rows.clear()
            pen_tip_path.clear()
            trajectory_canvas[:] = 0
            print("[OK] Cleared buffer & trajectory.")

    # autosave on exit
    if len(rows) > 0:
        write_csv(csv_name, rows)
        print(f"[OK] Auto-saved: {csv_name} (rows={len(rows)})")
        rows.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
