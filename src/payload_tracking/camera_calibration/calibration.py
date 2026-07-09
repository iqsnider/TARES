import cv2
import os
import json

from ChArUco_board import SQUARES_HORIZONTALLY, SQUARES_VERTICALLY, ARUCO_DICT

SQUARE_LENGTH_m = 0.027                   # Square side length (in m)
MARKER_LENGTH_m = 0.0135                   # ArUco marker side length (in m)


def get_calibration_parameters(img_dir):
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY),
        SQUARE_LENGTH_m, MARKER_LENGTH_m, dictionary)
    charuco_detector = cv2.aruco.CharucoDetector(board)

    image_files = [os.path.join(img_dir, f)
                   for f in os.listdir(img_dir) if f.endswith(".bmp")]

    all_object_points = []
    all_image_points = []
    image_size = None

    for image_file in image_files:
        image = cv2.imread(image_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]  # (width, height) — correct order

        charuco_corners, charuco_ids, marker_corners, marker_ids = \
            charuco_detector.detectBoard(gray)

        if charuco_ids is not None and len(charuco_ids) > 3:
            obj_points, img_points = board.matchImagePoints(
                charuco_corners, charuco_ids)
            if obj_points is not None and len(obj_points) > 3:
                all_object_points.append(obj_points)
                all_image_points.append(img_points)
        else:
            print(f"Board not detected in {image_file}")

    if len(all_object_points) < 4:
        raise RuntimeError(
            f"Only {len(all_object_points)} usable views. Fix image quality "
            f"before calibrating.")

    reproj_error, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_object_points, all_image_points, image_size, None, None)
    print(f"Reprojection error: {reproj_error:.4f} px")
    return mtx, dist


SENSOR = 'AR0821_onsemi'
LENS = 'See3CAM_CU81'
OUTPUT_JSON = 'calibration.json'

mtx, dist = get_calibration_parameters(img_dir='./images/')
data = {"sensor": SENSOR, "lens": LENS,
        "mtx": mtx.tolist(), "dist": dist.tolist()}

with open(OUTPUT_JSON, 'w') as json_file:
    json.dump(data, json_file, indent=4)

print(f'Data has been saved to {OUTPUT_JSON}')
