import cv2
from cv2 import aruco
import numpy as np
import pandas as pd

CAMERA_MATRIX = np.array([
    [1156.575220787158, 0.0, 641.235796583754],
    [0.0, 1159.917623853068, 403.13243457956634],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

DIST_COEFFS = np.array([
    0.21463756498771683,
   -0.8166333234031081,
    0.016152523688770088,
   -0.00042204457570052013,
    0.2851862067489753
], dtype=np.float32)

clicked_point = None
samples = []

def mouse_cb(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"[CLICK] {x,y}")

def setup_aruco():
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    return aruco.ArucoDetector(
        aruco.getPredefinedDictionary(aruco.DICT_4X4_50),
        params
    )

def main():
    global clicked_point, samples

    data = pd.read_csv("markers/model_points_4x4.csv")
    model_pts = data[['x','y','z']].values
    model_pts_by_id = [model_pts[i:i+4] for i in range(0,len(model_pts),4)]

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    detector = setup_aruco()

    cv2.namedWindow("Dodeca Pen Calib")
    cv2.setMouseCallback("Dodeca Pen Calib", mouse_cb)

    print("\n=== DODECAHEDRON PEN CALIBRATION ===")
    print("Klik ujung pen dari berbagai pose")
    print("Tekan C untuk solve\n")

    while True:
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #print(f"[VIDEO] {width}x{height} @ {fps:.2f} FPS | Total frames: {total_frames}")
        if not ret:
            break

        corners, ids, _ = detector.detectMarkers(frame)
        vis = frame.copy()

        if ids is not None and len(ids) >= 3:
            img_pts, obj_pts = [], []

            for i, mid in enumerate(ids.flatten()):
                if mid < len(model_pts_by_id):
                    img_pts.extend(corners[i][0])
                    obj_pts.extend(model_pts_by_id[mid])

            img_pts = np.array(img_pts, np.float32)
            obj_pts = np.array(obj_pts, np.float32)

            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts,
                CAMERA_MATRIX, DIST_COEFFS,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if ok:
                cv2.drawFrameAxes(vis, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, 40)

                if clicked_point is not None:
                    x,y = clicked_point
                    pt = np.array([[[x,y]]],np.float32)
                    norm = cv2.undistortPoints(pt, CAMERA_MATRIX, DIST_COEFFS)[0][0]
                    ray = np.array([norm[0], norm[1], 1.0])
                    ray /= np.linalg.norm(ray)

                    R,_ = cv2.Rodrigues(rvec)
                    C = np.zeros(3)

                    samples.append({
                        "R": R,
                        "t": tvec.flatten(),
                        "d": ray,
                        "C": C
                    })

                    print(f"[SAMPLE {len(samples)}]")
                    clicked_point = None

        cv2.imshow("Dodeca Pen Calib", vis)

        k = cv2.waitKey(1)&0xFF
        if k == 27:
            break
        elif k in [ord('c'),ord('C')] and len(samples)>=5:
            solve(samples)
            break

    cap.release()
    cv2.destroyAllWindows()

def solve(samples):
    A = np.zeros((3*len(samples),3))
    b = np.zeros(3*len(samples))

    for i,s in enumerate(samples):
        R,t,d,C = s["R"],s["t"],s["d"],s["C"]
        P = np.eye(3) - np.outer(d,d)
        A[3*i:3*i+3,:] = P @ R
        b[3*i:3*i+3] = P @ (C - t)

    tip = np.linalg.lstsq(A,b,rcond=None)[0]
    print("\n=== RESULT ===")
    print("Pen tip in body frame (mm):")
    print(tip)
    print("\nPython:")
    print(f"PEN_TIP_LOC = np.array({tip.reshape(3,1).tolist()}, dtype=np.float32)")

if __name__=="__main__":
    main()
