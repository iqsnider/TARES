import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# parameters
from sim.config import (
    MASS_DRONE, MASS_PAYLOAD, GRAVITY, TETHER_LEN, J)

# trajectory generation
from sim.simplified_mission_manager import (
    ReferenceTrajectory, equilibrium_state)

# revised control logic
from sim.SITL_dynamics import (
    OuterLoopPayloadLQR, ArduPilotFlightController, wrap_angle, HOVER_THRUST)

# plotting
import sim.plotting as plotting
from sim.plotting import plot_trajectory_3d, _state_control_plots


def nonlinear_ode(x, u):
    """
    16 state nonlinear drone with taut tether-suspended payload system
    """
    mD = MASS_DRONE
    mP = MASS_PAYLOAD
    g = GRAVITY
    L = TETHER_LEN

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
                     [-st,   sp*ct,            cp*ct]])

    F_thrust_W = T_EB @ np.array([0, 0, C_Sigma])

    # payload position relative to drone
    r_rel = L*np.array([sax*cay,
                        say,
                        -cax * cay])

    # payload relative velocity
    dr_rel = L*np.array([cax*cay*alx_d - sax*say*aly_d,
                         cay*aly_d,
                         sax*cay*alx_d + cax*say*aly_d])

    # # lagrange multiplier for taut cable
    # lam = ((1/mD)*(r_rel@F_thrust_W) - (dr_rel@dr_rel)) / \
    #     (2*L**2 * (1/mP + 1/mD))
    #
    # # cable force on drone
    # F_cable_W = -2*lam*r_rel

    n_hat = r_rel / L
    ndot_sq = cay**2 * alx_d**2 + aly_d**2
    T = (mP/(mD + mP))*(mD*L*ndot_sq - n_hat@F_thrust_W)
    # tension along the cable, drone toward load
    F_cable_W = T*n_hat

    # translational acceleration
    ddx = C_Sigma/mD * (cy*st*cp + sy*sp) + F_cable_W[0]/mD
    ddy = C_Sigma/mD * (sy*st*cp - cy*sp) + F_cable_W[1]/mD
    ddz = C_Sigma/mD * (ct*cp) - g + F_cable_W[2]/mD

    # euler rates
    tt = st / ct
    phi_dot = p + (sp*q + cp*r)*tt
    theta_dot = cp*q - sp*r
    psi_dot = (sp*q + cp*r)/ct

    # body angular velocities
    p_dot = (n1 - (Jyy - Jzz)*q*r) / Jxx
    q_dot = (n2 - (Jzz - Jxx)*p*r) / Jyy
    r_dot = (n3 - (Jxx - Jyy)*p*q) / Jzz

    mu = 1/mP + 1/mD

    # alpha_x_ddot = (2*lam*mu*L*sax - F_thrust_W[0]/mD
    #                 + L*sax*alx_d**2) / (L*cax)
    #
    # alpha_y_ddot = (2*lam*mu*L*say - F_thrust_W[1]/mD
    #                 + L*say*aly_d**2) / (L*cay)
    Ftx, Fty, Ftz = F_thrust_W

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


def control_law(x, t, outerLoopControl, innerLoopControl, ref):
    """Flight controller about the trajectory equilibrium at time t"""
    p_PL_ref, v_PL_ref = ref(t)  # current reference

    x_star = equilibrium_state(
        p_PL_ref, v_PL_ref, TETHER_LEN)  # equilibrium state

    e = x - x_star  # error

    e[6:9] = wrap_angle(e[6:9])  # angle wrapping

    # Outer-loop LQR controller
    a_des = outerLoopControl.compute_u(e)

    # Inner-loop ardupilot flight controller [C_Sigma, n1, n2, n3]
    u_pert = innerLoopControl.compute_u(x, a_des, yaw_s=0)
    u = u_pert

    return u, u_pert, e


def simulate(outerLoop, innerLoop, ref, dt=0.01, settle_time=8):
    """integrate the closed loop while tracking the reference"""
    t_end = ref.total_time_to_wp + settle_time
    t_eval = np.arange(0, t_end, dt)

    # drone is at rest above the first waypoint
    x0 = np.zeros(16)
    p0, _ = ref(0)  # payload reference at time t = 0
    x0[0:3] = p0 + np.array([0, 0, TETHER_LEN])

    def ode(t, x):
        return nonlinear_ode(x, control_law(x, t, outerLoop, innerLoop, ref)[0])

    # integrate
    sol = solve_ivp(ode, [0, t_end], x0, t_eval=t_eval, method='RK45')
    X = sol.y.T
    P_ref = np.array([ref(t)[0] for t in sol.t])  # reference trajectory

    p_PL, v_PL = payload_state(X)

    err_log = np.zeros((16, len(sol.t)))
    u_pert = np.zeros((4, len(sol.t)))

    for i, t in enumerate(sol.t):
        _, u_pert[:, i], err_log[:, i] = control_law(
            X[i], t, outerLoop, innerLoop, ref)
        p_ref, v_ref = ref(t)
        err_log[0:3, i] = p_PL[i] - p_ref  # payload position error
        err_log[3:6, i] = v_PL[i] - v_ref  # payload velocity error

    return sol.t, X, P_ref, err_log, u_pert


def payload_state(X):
    """Payload position and velocity in world frame from the full state history."""
    # sines and cosines of alpha_xy
    sax, cax = np.sin(X[:, 12]), np.cos(X[:, 12])
    say, cay = np.sin(X[:, 13]), np.cos(X[:, 13])

    # dalpha_xy for computing payload velocity
    alx_d, aly_d = X[:, 14], X[:, 15]

    r_rel = TETHER_LEN*np.column_stack([sax*cay, say, -cax*cay])
    dr_rel = TETHER_LEN*np.column_stack([cax*cay*alx_d - sax*say*aly_d,
                                         cay*aly_d,
                                         sax*cay*alx_d + cax*say*aly_d])

    return X[:, 0:3] + r_rel, X[:, 3:6] + dr_rel


if __name__ == '__main__':
    outer_loop = OuterLoopPayloadLQR(w_pos_xy=(1/1)**2,
                                     w_pos_z=(1/0.1)**2,
                                     tuning_const=1/1**2,
                                     moment_arm=0.381/np.sqrt(2),
                                     thrust_to_torque=0.02)
    inner_loop = ArduPilotFlightController()
    mission_ref = ReferenceTrajectory()
    ts, X, P_ref, err_log, u_log = simulate(
        outer_loop, inner_loop, mission_ref, settle_time=50)
    plotting.save_all(ts=ts, X=X, P_ref=P_ref, t=ts, X_err=err_log, U_pert=u_log,
                      title="Closed-Loop LQR with ME236 Flight Controller", layout='panels', fnames=('fc_states', 'fc_control'), verbose=True)
    plt.show()
