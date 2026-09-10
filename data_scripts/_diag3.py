import cv2, numpy as np

# convention check: shift a synthetic image by a known amount
rng = np.random.default_rng(0)
img = rng.random((400, 400)).astype(np.float32)
img = cv2.GaussianBlur(img, (0, 0), 3)
M = np.float32([[1, 0, 10], [0, 1, 5]])          # move content +10 x, +5 y
img2 = cv2.warpAffine(img, M, (400, 400))
win = cv2.createHanningWindow((400, 400), cv2.CV_32F)
print("synthetic (content moved +10,+5):", cv2.phaseCorrelate(img*win, img2*win)[0])
