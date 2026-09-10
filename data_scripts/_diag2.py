import cv2, numpy as np

src = "../data/test_09092026/ardupilot_icra_test_20260909_132351/recording.avi"
cap = cv2.VideoCapture(src)
want = {2400: None, 2496: None}
i = 0
while True:
    ok, f = cap.read()
    if not ok or i > max(want):
        break
    if i in want:
        want[i] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
    i += 1
cap.release()

a, b = want[2400], want[2496]
# central patch, away from the propeller arms
h, w = a.shape
cy, cx = h//2, w//2
A = a[cy-400:cy+400, cx-400:cx+400]
B = b[cy-400:cy+400, cx-400:cx+400]
win = cv2.createHanningWindow((800, 800), cv2.CV_32F)
shift, resp = cv2.phaseCorrelate(A*win, B*win)
print("observed ground shift 2400->2496 (px, full res):", shift, "response", resp)
