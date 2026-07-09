import cv2

ARUCO_DICT = cv2.aruco.DICT_6X6_250
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5

# --- Rendering resolution ---
# Only the square:marker *ratio* matters to the geometry; this knob controls
# how many pixels each square is drawn with, which is what makes markers sharp.
PIXELS_PER_SQUARE = 260             # try 200-300; higher = crisper
MARGIN_PX = PIXELS_PER_SQUARE // 2  # generous white quiet-zone aids detection

# Arbitrary units for image generation keep the 2:1 ratio.
SQUARE_LENGTH = 2
MARKER_LENGTH = 1

OUTPUT_NAME = 'ChArUco_Marker.png'


def create_and_save_new_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY),
        SQUARE_LENGTH, MARKER_LENGTH, dictionary)

    # generateImage takes (width, height); width = the board's first dim (7),
    # height = the second (5) — same ordering as the original script.
    width_px = SQUARES_VERTICALLY * PIXELS_PER_SQUARE + 2 * MARGIN_PX
    height_px = SQUARES_HORIZONTALLY * PIXELS_PER_SQUARE + 2 * MARGIN_PX

    img = cv2.aruco.CharucoBoard.generateImage(
        board, (width_px, height_px), marginSize=MARGIN_PX)
    cv2.imwrite(OUTPUT_NAME, img)
    print(f"Saved {OUTPUT_NAME} at {width_px}x{height_px} px")


create_and_save_new_board()
