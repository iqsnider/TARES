import cv2, numpy as np

src = "../data/test_09092026/ardupilot_icra_test_20260909_132351/recording.avi"
cap = cv2.VideoCapture(src)
want = {2400: None, 2412: None, 2424: None, 1200: None, 1212: None}
i = 0
while True:
    ok, f = cap.read()
    if not ok or i > max(want):
        break
    if i in want:
        want[i] = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
    i += 1
cap.release()

win = cv2.createHanningWindow((800, 800), cv2.CV_32F)
def patch(g):
    h, w = g.shape
    return g[h//2-400:h//2+400, w//2-400:w//2+400]

for a, b in [(2400, 2412), (2412, 2424), (2400, 2424), (1200, 1212)]:
    sh, resp = cv2.phaseCorrelate(patch(want[a])*win, patch(want[b])*win)
    print(f"{a}->{b}: ground content shift {np.round(sh,1)}  resp {resp:.2f}")
