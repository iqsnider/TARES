import Prm.config as config

def estimate_to_px_coords(xi, P, T_IB, K, D, n_sigma=1):
    """
    Takes a payload state estimate and calculates the pixel coordinates sized to the camera frame.
    For viewing the estimate overlaid on the recording.
    """
    L = config.TETHER_LEN
    A = T_CB @ T_IB.T

    q_I = np.array([xi[IX_ALPHA_X], xi[IX_ALPHA_Y], -1])

    # payload center in the camera frame: undo the pivot shift
    p_C = L*(A @ q_I) - T_CB @ (t_BC_B - l_B)

    uv, _ = cv2.projectPoints(p_C.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K, D)
    center = uv.ravel()

    # d(pixel)/d(alpha), pinhole only
    X, Y, Z = p_C
    J_proj = np.array([[K[0, 0]/Z, 0, -K[0, 0]*X/Z**2],
                       [0, K[1, 1]/Z, -K[1, 1]*Y/Z**2]])
    J = J_proj @ (L*A[:, 0:2])
    P_px = J @ P[0:2, 0:2] @ J.T

    evals, evecs = np.linalg.eigh(P_px)
    axes = n_sigma*np.sqrt(evals[::-1])      # major first
    angle = math.degrees(math.atan2(evecs[1, -1], evecs[0, -1]))

    return center, axes, angle
