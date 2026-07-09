"""
Detect ArUco markers from a USB camera and print the results to the console.
"""

import argparse
import time

import cv2
import numpy as np

# web interface stuff
import socketio
import threading
from flask import Flask, send_from_directory
import os


sio = socketio.Server(cors_allowed_origins="*", async_mode="threading")
flask_app = Flask(__name__)
flask_app.wsgi_app = socketio.WSGIApp(sio, flask_app.wsgi_app)

HERE = os.path.dirname(os.path.abspath(__file__))


@flask_app.route("/")
def index():
    return send_from_directory(HERE, "index.html")


@flask_app.route("/viewer.js")
def viewer_js():
    return send_from_directory(HERE, "viewer.js")


def start_web_server(host="127.0.0.1", port=5000):
    """
    Starts a local web server for live viewing
    """
    flask_app.run(host=host, port=port, threaded=True, use_reloader=False)


def make_detector(dict_name):
    """
    Specify dictionary and marker id range limit
    e.g. DICT_6X6_50
    """

    # get the dict id
    dict_id = getattr(cv2.aruco, dict_name, None)

    # note that the OpenCV aruco diciontary initalization has changed in recent versions
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()

    return cv2.aruco.ArucoDetector(aruco_dict, params)


def open_camera(device, width, height, fourcc):
    """
    Opens the camera and sets fourcc and resolution settings for the See3CAM_CU81
    """
    # force video capture device to open dev/video#
    cap = cv2.VideoCapture(device)

    # See3CAM_CU81 outputs to MJPEG so will need to specify the fourcc
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

    # set resolution to allowed resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # print the resolution
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"camera open: {aw}x{ah}")

    return cap, aw, ah


def detect(frame, detector):
    """
    Return (corners, ids) as plain arrays.
    """
    # faster and more accurate with gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    # if no marker is detected
    if ids is None:
        return [], np.empty((0,), dtype=int)

    # reshape the corners
    corners = [c.reshape(4, 2) for c in corners]
    return corners, ids.flatten()


def get_marker_pose(corners, ids):
    """
    Determine the pose the detected ArUco marker wrt the camera
    (we're gonna assume that the camera is fixed in the drone frame)
    """


def print_detections(corners, ids):
    """
    Console printing for detected markers
    """
    for c, i in zip(corners, ids):
        center = c.mean(axis=0)
        # one-line-per-marker summary
        corner_str = " ".join(f"({x:.0f},{y:.0f})" for x, y in c)
        print(f"  id {int(i):<3d} center ({center[0]:.1f}, {center[1]:.1f})  "
              f"corners {corner_str}")


def main():
    # set arguments
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=0,
                    help="camera index or /dev/video path")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fourcc", default="MJPG",
                    help="pixel format; '' for default")
    ap.add_argument("--dict", default="DICT_6X6_50", help="ArUco dictionary")
    ap.add_argument("--once", action="store_true",
                    help="grab a single frame, print, and exit")
    ap.add_argument("--verbose", default=True)
    ap.add_argument("--viewer", default=False)
    args = ap.parse_args()

    # set the device id
    device = int(args.device)

    # initialize the aruco detector with the specified aruco size and id range
    detector = make_detector(args.dict)

    # initialize the camera capture source
    cap, aw, ah = open_camera(device, args.width, args.height, args.fourcc)

    # initialize web interface
    if args.viewer:
        threading.Thread(target=start_web_server, daemon=True).start()

    while True:
        # get the frame
        ok, frame = cap.read()

        # check if frame grab failed
        if not ok:
            print("frame grab failed")
            if args.once:
                break
            continue

        # live view for debugging focus
        cv2.imshow('camera view', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        # aruco corners and ids
        corners, ids = detect(frame, detector)

        # web interface
        if args.viewer:
            markers = [
                {"id": int(i), "corners": c.tolist(),
                 "center": c.mean(axis=0).tolist()}
                for c, i in zip(corners, ids)
            ]
            sio.emit("markers", {"w": aw, "h": ah, "markers": markers})

        # console verbosity
        if args.verbose:
            if len(ids):
                print(f"[{time.strftime('%H:%M:%S')}] {len(ids)} marker(s): "
                      f"ids {ids.tolist()}")
                print_detections(corners, ids)
            else:
                print(f"[{time.strftime('%H:%M:%S')}] no markers")

        if args.once:
            break

    # release the image capture source
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
