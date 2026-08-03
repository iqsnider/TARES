import Prm.config as config

import math
import numpy as np


def nonlinear_ode(x, u):
    """
    16 state nonlinear drone with taut tether-suspended payload system
    """
    mD = config.MASS_DRONE
    mP = config.MASS_PAYLOAD_EFF
    g = config.GRAVITY
    L = config.TETHER_LEN
    J = config.J

    Jxx, Jyy, Jzz = J[0, 0], J[1, 1], J[2, 2]

    phi, theta, psi = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]
    alx, aly = x[12], x[13]
    alx_d, aly_d = x[14], x[15]

    C_Sigma, n1, n2, n3 = u

    sp, cp = math.sin(phi), math.cos(phi)
    st, ct = math.sin(theta), math.cos(theta)
    sy, cy = math.sin(psi), math.cos(psi)

    sax, cax = math.sin(alx), math.cos(alx)
    say, cay = math.sin(aly), math.cos(aly)

    T_EB = np.array([[ct*cy, sp*st*cy - cp*sy, cp*st*cy + sp*sy],
                     [ct*sy, sp*st*sy + cp*cy, cp*st*sy - sp*cy],
                     [-st, sp*ct, cp*ct]])

    F_thrust_I = T_EB @ np.array([0, 0, C_Sigma])
    Ftx, Fty, Ftz = F_thrust_I

    # unit vector for the direction of the payload relative to the inertial vertical
    qI = np.array([sax*cay,
                   say,
                   -cax*cay])

    T = -(mP * (Ftz*cax*cay - Fty*say - Ftx*cay*sax
              + L*aly_d**2*mD + L*alx_d**2*mD*cay**2)) / (mD + mP)
    # tension along the cable, drone toward load
    F_cable_I = T*qI

    # translational acceleration inertial frame
    ddx = C_Sigma/mD * (cy*st*cp + sy*sp) - F_cable_I[0]/mD
    ddy = C_Sigma/mD * (sy*st*cp - cy*sp) - F_cable_I[1]/mD
    ddz = C_Sigma/mD * (ct*cp) - g - F_cable_I[2]/mD

    # euler rates
    tt = st / ct
    phi_dot = p + (sp*q + cp*r)*tt
    theta_dot = cp*q - sp*r
    psi_dot = (sp*q + cp*r)/ct

    # body angular velocities
    p_dot = (n1 - (Jyy - Jzz)*q*r) / Jxx
    q_dot = (n2 - (Jzz - Jxx)*p*r) / Jyy
    r_dot = (n3 - (Jxx - Jyy)*p*q) / Jzz


    alpha_x_ddot = (-(cax*Ftx + sax*Ftz) / (mD*L*cay)
                    + 2*(say/cay)*alx_d*aly_d)

    alpha_y_ddot = (-(cay*Fty + say*(cax*Ftz - sax*Ftx)) / (mD*L)
                    - say*cay*alx_d**2)
    # output xdot
    return np.array([x[3], x[4], x[5],
                     ddx, ddy, ddz,

                     phi_dot,
                     theta_dot,
                     psi_dot,

                     p_dot,
                     q_dot,
                     r_dot,

                     alx_d,
                     aly_d,

                     alpha_x_ddot,
                     alpha_y_ddot])

def payload_state(X):
    """
    Payload position and velocity in the world frame from a state history.
    """
    L = config.TETHER_LEN

    sax, cax = np.sin(X[:, 12]), np.cos(X[:, 12])
    say, cay = np.sin(X[:, 13]), np.cos(X[:, 13])
    alx_d, aly_d = X[:, 14], X[:, 15]

    r_rel = L*np.column_stack([sax*cay, say, -cax*cay])
    dr_rel = L*np.column_stack([cax*cay*alx_d - sax*say*aly_d,
                                cay*aly_d,
                                sax*cay*alx_d + cax*say*aly_d])

    return X[:, 0:3] + r_rel, X[:, 3:6] + dr_rel
