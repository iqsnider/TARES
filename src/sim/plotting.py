import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from sim.config import TETHER_LEN, GRAVITY, MASS_TOTAL
import sim.config as config

HOVER_THRUST = MASS_TOTAL * GRAVITY

# Default output directory for saved figures (relative to the working dir).
FIG_DIR = 'figs'

# ----------------------------------------------------------------------------
# Global style
# ----------------------------------------------------------------------------
# Colour-blind-safe trio (Paul Tol "vibrant"): blue / orange / red.
COLORS = ['#0077BB', '#EE7733', '#CC3311']

# Reusable semantic colours.
C_REF = '#9AA0A6'      # reference / set-point traces
C_DRONE = '#0077BB'    # drone path
C_PAYLOAD = '#EE7733'  # payload path
C_HOVER = '#CC3311'    # hover-thrust marker
GRID_GREY = '#D5D8DC'


def configure_plot_style(use_tex=False):
    """Apply a consistent, presentation-ready LaTeX-style theme.

    Parameters
    ----------
    use_tex : bool
        If True, attempt true LaTeX typesetting (needs a working
        latex + dvipng install). Falls back to mathtext on failure.
    """
    base = {
        # --- typography ---
        'font.family': 'serif',
        'font.serif': ['CMU Serif', 'STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': 13,
        'axes.titlesize': 15,
        'axes.titleweight': 'bold',
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
        # --- axes / spines ---
        'axes.edgecolor': '#3C4043',
        'axes.linewidth': 0.9,
        'axes.labelpad': 6,
        'axes.titlepad': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.axisbelow': True,
        # --- grid ---
        'axes.grid': True,
        'grid.color': GRID_GREY,
        'grid.linewidth': 0.7,
        'grid.alpha': 0.9,
        # --- ticks ---
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.color': '#3C4043',
        'ytick.color': '#3C4043',
        'xtick.major.width': 0.9,
        'ytick.major.width': 0.9,
        # --- lines ---
        'lines.linewidth': 1.7,
        'lines.solid_capstyle': 'round',
        'lines.antialiased': True,
        # --- legend ---
        'legend.frameon': True,
        'legend.framealpha': 0.92,
        'legend.edgecolor': '#D5D8DC',
        'legend.facecolor': 'white',
        'legend.fancybox': True,
        'legend.borderpad': 0.5,
        'legend.handlelength': 1.8,
        # --- figure / export ---
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'figure.dpi': 120,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
    }

    if use_tex:
        try:
            mpl.rcParams['text.usetex'] = True
            mpl.rcParams['text.latex.preamble'] = (
                r'\usepackage{amsmath}\usepackage{bm}'
            )
            fig = plt.figure()           # probe that LaTeX actually works
            fig.text(0.5, 0.5, r'$\dot\alpha_x$')
            fig.canvas.draw()
            plt.close(fig)
        except Exception:
            mpl.rcParams['text.usetex'] = False
            print('configure_plot_style: LaTeX unavailable, '
                  'falling back to mathtext (Computer Modern).')

    mpl.rcParams.update(base)


configure_plot_style(use_tex=False)


# ----------------------------------------------------------------------------
# Saving helper
# ----------------------------------------------------------------------------
def _save(fig, fname, save_dir=FIG_DIR, formats=('png',), dpi=300):
    """Write *fig* to ``save_dir`` (created if needed) as ``fname.<ext>``."""
    os.makedirs(save_dir, exist_ok=True)
    paths = []
    for ext in formats:
        path = os.path.join(save_dir, f'{fname}.{ext}')
        fig.savefig(path, dpi=dpi)
        paths.append(path)
    return paths


# ----------------------------------------------------------------------------
# 3-D trajectory
# ----------------------------------------------------------------------------
def plot_trajectory_3d(ts, X, P_ref, n_tethers=25,
                       title='Trajectory',
                       save_dir=None, fname='trajectory_3d'):
    drone = X[:, 0:3]
    sax, say = np.sin(X[:, 12]), np.sin(X[:, 13])
    cax, cay = np.cos(X[:, 12]), np.cos(X[:, 13])
    payload = drone + TETHER_LEN * np.column_stack([sax, say, -cax * cay])

    fig = plt.figure(figsize=(11, 9), constrained_layout=True)
    ax = fig.add_subplot(projection='3d')
    if title:
        fig.suptitle(title, fontsize=18, fontweight='bold')

    ax.plot(*P_ref.T, color=C_REF, linewidth=1.6, linestyle=(0, (5, 4)),
            label='Payload reference')
    ax.plot(*drone.T, color=C_DRONE, linewidth=2.0, label='Drone')
    ax.plot(*payload.T, color=C_PAYLOAD, linewidth=2.0, label='Payload')

    idx = np.linspace(0, len(ts) - 1, n_tethers).astype(int)
    for k, i in enumerate(idx):
        ax.plot([drone[i, 0], payload[i, 0]],
                [drone[i, 1], payload[i, 1]],
                [drone[i, 2], payload[i, 2]],
                color='#5F6368', linewidth=0.9, alpha=0.35,
                label='Tether (snapshots)' if k == 0 else None)

    ax.scatter(*drone[0], color='#2CA02C', edgecolors='white', linewidths=0.8,
               marker='^', s=90, depthshade=False, label='Start', zorder=5)
    ax.scatter(*drone[-1], color='#222222', edgecolors='white', linewidths=0.8,
               marker='s', s=70, depthshade=False, label='End', zorder=5)

    # equal-aspect cube
    all_pts = np.vstack([drone, payload, P_ref])
    center = (all_pts.min(axis=0) + all_pts.max(axis=0)) / 2
    half = max((all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 + 1.5, 5)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1, 1, 1))

    # softer panes + grid for a cleaner 3-D look
    ax.view_init(elev=22, azim=-58)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.985, 0.985, 0.985, 1.0))
        axis.pane.set_edgecolor((0.85, 0.85, 0.85, 1.0))
        axis.pane.set_alpha(1.0)
        try:  # private API, guard across mpl versions
            axis._axinfo['grid'].update(color=GRID_GREY, linewidth=0.6)
        except Exception:
            pass

    ax.set_xlabel('$x$ [m]', labelpad=10)
    ax.set_ylabel('$y$ [m]', labelpad=10)
    ax.set_zlabel('$z$ [m]', labelpad=10)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.92)

    if save_dir:
        _save(fig, fname, save_dir)
    return fig


# ----------------------------------------------------------------------------
# Time-history panels
# ----------------------------------------------------------------------------
def _fmt(ax, ylabel, title, legend_loc='upper right', zeroline=True):
    if zeroline:
        ax.axhline(0, color='#9AA0A6', linewidth=0.9, linestyle=(0, (4, 4)),
                   zorder=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.legend(loc=legend_loc, ncol=1)
    ax.grid(True, which='major', alpha=0.9)
    ax.margins(x=0.01)


# --- individual panel drawers: each renders one subplot onto `ax` -----------
def _p_position(ax, t, Xe, U):
    for i, lbl in enumerate(['$x$ [m]', '$y$ [m]', '$z$ [m]']):
        ax.plot(t, Xe[i], label=lbl, color=COLORS[i])
    _fmt(ax, 'Position error [m]', 'Payload Position Error')


def _p_velocity(ax, t, Xe, U):
    for i, lbl in enumerate([r'$v_x$ [m/s]', r'$v_y$ [m/s]', r'$v_z$ [m/s]']):
        ax.plot(t, Xe[3 + i], label=lbl, color=COLORS[i])
    _fmt(ax, 'Velocity error [m/s]', 'Payload Velocity Error')


def _p_orientation(ax, t, Xe, U):
    for i, lbl in enumerate(['roll [deg]', 'pitch [deg]', 'yaw [deg]']):
        ax.plot(t, np.rad2deg(Xe[6 + i]), label=lbl, color=COLORS[i])
    _fmt(ax, 'Angle error [deg]', 'Drone Orientation Error')


def _p_angrate(ax, t, Xe, U):
    for i, lbl in enumerate(['$p$ [deg/s]', '$q$ [deg/s]', '$r$ [deg/s]']):
        ax.plot(t, np.rad2deg(Xe[9 + i]), label=lbl, color=COLORS[i])
    _fmt(ax, 'Rate error [deg/s]', 'Drone Angular Velocity Error')


def _p_payload_angle(ax, t, Xe, U):
    for i, lbl in enumerate([r'$\alpha_x$ [deg]', r'$\alpha_y$ [deg]']):
        ax.plot(t, np.rad2deg(Xe[12 + i]), label=lbl, color=COLORS[i])
    _fmt(ax, 'Pendulum angle [deg]', 'Payload Angle')


def _p_payload_rate(ax, t, Xe, U):
    for i, lbl in enumerate([r'$\dot\alpha_x$ [deg/s]', r'$\dot\alpha_y$ [deg/s]']):
        ax.plot(t, np.rad2deg(Xe[14 + i]), label=lbl, color=COLORS[i])
    _fmt(ax, 'Pendulum rate [deg/s]', 'Payload Angle Rate')


def _p_thrust(ax, t, Xe, U):
    ax.plot(t, U[0], label=r'$C_\Sigma$ [N]', color=COLORS[0])
    ax.axhline(HOVER_THRUST, color=C_HOVER, linewidth=1.2,
               linestyle=(0, (1, 2)), label='Hover thrust', zorder=1)
    # no zero-line: thrust lives near hover, so let it autoscale tightly
    _fmt(ax, 'Thrust [N]', 'Control Thrust', zeroline=False)


def _p_torques(ax, t, Xe, U):
    for i, lbl in enumerate([r'$\tau_x$ [N$\cdot$m]', r'$\tau_y$ [N$\cdot$m]',
                             r'$\tau_z$ [N$\cdot$m]']):
        ax.plot(t, U[1 + i], label=lbl, color=COLORS[i])
    _fmt(ax, r'Torque [N$\cdot$m]', 'Control Torques')


# Panel groupings. Order is row-major to match the subplot grid.
_STATE_PANELS = [_p_position, _p_velocity, _p_orientation, _p_angrate]
_PAYLOAD_CONTROL_PANELS = [_p_payload_angle,
                           _p_payload_rate, _p_thrust, _p_torques]
_ALL_PANELS = _STATE_PANELS + _PAYLOAD_CONTROL_PANELS


def _assemble(panels, nrows, ncols, t, Xe, U, suptitle, figsize):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    for panel, ax in zip(panels, axes.ravel()):
        panel(ax, t, Xe, U)
    for ax in axes[-1, :]:          # only the bottom row carries the x-label
        ax.set_xlabel('Time [s]')
    if suptitle:
        fig.suptitle(suptitle, fontsize=18, fontweight='bold')
    return fig


def state_control_grid(t, X_err, U_pert, title='Nonlinear Model',
                       save_dir=None, fname='state_control'):
    """Single 4x2 sheet with all eight state/control panels.

    X_err   error-state time history (16 x N)
    U_pert  LQR perturbation control (4 x N: thrust + 3 torques)
    """
    fig = _assemble(_ALL_PANELS, 4, 2, t, X_err, U_pert, title, (16, 20))
    if save_dir:
        _save(fig, fname, save_dir)
    return fig


# Backward-compatible alias for the original function name.
_state_control_plots = state_control_grid


def state_control_panels(t, X_err, U_pert, title='Nonlinear Model',
                         sections=(None, None),
                         save_dir=None,
                         fnames=('state_errors', 'payload_control')):
    """Two 2x2 panels."""
    sep = ' '
    sup_top = f'{title}{sep}{sections[0]}' if sections[0] else title
    sup_bot = f'{title}{sep}{sections[1]}' if sections[1] else title

    fig_top = _assemble(_STATE_PANELS, 2, 2, t, X_err,
                        U_pert, sup_top, (14, 9))
    fig_bot = _assemble(_PAYLOAD_CONTROL_PANELS, 2, 2, t, X_err, U_pert,
                        sup_bot, (14, 9))

    if save_dir:
        _save(fig_top, fnames[0], save_dir)
        _save(fig_bot, fnames[1], save_dir)
    return fig_top, fig_bot


def save_all(ts, X, P_ref, t, X_err, U_pert, title='Nonlinear Model',
             traj_title='Trajectory',
             save_dir=FIG_DIR, layout='panels', n_tethers=25, fnames=None,
             verbose=False):
    """Generate every figure and write it to ``save_dir`` (default ``figs/``).

    layout : 'panels' (two 2x2, default), 'grid' (one 4x2), or 'both'.
    Returns a dict mapping a short key to each Figure.
    """
    figs = {'trajectory': plot_trajectory_3d(
        ts, X, P_ref, n_tethers=n_tethers, title=traj_title,
        save_dir=save_dir, fname='trajectory_3d')}

    if verbose:
        title += (
            f"\n$\\mathbf{{V_{{ref}}}}$={config.CRUISE_SPEED} m/s, "
            f"$\\mathbf{{d}}$={config.GRID_X_END} m, "
            f"tether={config.TETHER_LEN} m, "
            f"$\\mathbf{{m_D}}$={config.MASS_DRONE} kg, "
            f"$\\mathbf{{m_{{PL}}}}$={config.MASS_PAYLOAD} kg"
        )

    if layout in ('panels', 'both'):
        if fnames is not None:
            top, bot = state_control_panels(t, X_err, U_pert, title=title,
                                            save_dir=save_dir, fnames=fnames)
        else:
            top, bot = state_control_panels(t, X_err, U_pert, title=title,
                                            save_dir=save_dir)

        figs['state_errors'] = top
        figs['payload_control'] = bot
    if layout in ('grid', 'both'):
        figs['state_control'] = state_control_grid(
            t, X_err, U_pert, title=title, save_dir=save_dir)
    return figs
