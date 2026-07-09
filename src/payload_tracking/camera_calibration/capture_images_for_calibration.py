import cv2
import os
from ChArUco_board import ARUCO_DICT

output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Error: Could not open camera.")

# Request full sensor resolution. MJPG is often required for 4K over USB.
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Capturing at {actual_w}x{actual_h}")

dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

print("SPACE = save frame,  q = quit")
saved = 0
while True:
    ok, frame = cap.read()
    if not ok:
        print("Error: Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    n = 0 if ids is None else len(ids)

    preview = frame.copy()
    if n > 0:
        cv2.aruco.drawDetectedMarkers(preview, corners, ids)
    colour = (0, 255, 0) if n >= 10 else (0, 0, 255)
    cv2.putText(preview, f"markers: {n}   saved: {saved}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)

    # Shrink just for display; the saved file stays full-res.
    disp = cv2.resize(preview, (actual_w // 2, actual_h // 2))
    cv2.imshow("calibration capture", disp)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        filename = os.path.join(output_dir, f"image_{saved}.bmp")
        cv2.imwrite(filename, frame)   # save the raw full-res frame
        print(f"Saved {filename} ({n} markers)")
        saved += 1
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Done. {saved} images saved.")
