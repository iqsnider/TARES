import cv2
import os
import time

# create the iamges directory
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

# initialize the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Capturing 20 images, 1 second apart...")

    # capture exactly 20 images
    for i in range(20):
        ok, frame = cap.read()

        if ok:
            # construct file path: images/image_0.bmp, image_1.bmp, etc.
            filename = os.path.join(output_dir, f"image_{i}.bmp")

            # save the image
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
        else:
            print("Error: Failed to capture frame.")
            break

        # wait for 1 second before capturing the next image
        time.sleep(1)

    # release the camera
    cap.release()
    print("Capture complete!")
