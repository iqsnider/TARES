import cv2
import os
from ChArUco_board import ARUCO_DICT
dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

for f in sorted(os.listdir('./images/')):
    img = cv2.imread('./images/' + f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    n = 0 if ids is None else len(ids)
    print(f"{f}: {n} markers")
