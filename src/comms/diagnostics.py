"""
Diagnostics multiplexer for a live flight, in three tmux panes.

Reads the flight log the control program is already writing rather than
opening a second mavlink connection, so it never contends with the loop for
the link and can be started, stopped and restarted at any time. With nothing
running it waits and says so.

    uv run src/comms/diagnostics.py
"""
import glob
import math
import os
import subprocess
import sys
import time

PANES = ("state", "control", "ekf")
SESSION = "tares"
WINDOW = "diagnostics"
REFRESH_HZ = 5
TAIL_BYTES = 65536       # about 250 rows, enough to age a measurement
STALE_S = 1.5            # log older than this means nothing is writing
HOME = "\033[H"          # cursor to the top left, without erasing
EOL = "\033[K"           # erase what the last frame left on this line
BELOW = "\033[J"         # and anything it left further down


def newest_log(data_dir="data"):
    """
    The flight log being written right now, or None if there is not one
    """
    paths = [p for p in glob.glob(os.path.join(data_dir, "**", "flight_*.csv"),
                                  recursive=True) if "_events" not in p]
    if not paths:
        return None

    newest = max(paths, key=os.path.getmtime)

    return newest


def tail(path):
    """
    Header and the last few hundred complete rows of the log
    """
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(0)
        header = f.readline().decode("utf-8", "replace").strip().split(",")
        f.seek(max(len(header), size - TAIL_BYTES))
        chunk = f.read().decode("utf-8", "replace")

    width = len(header)
    rows = [dict(zip(header, ln.split(",")))
            for ln in chunk.splitlines() if ln.count(",") == width - 1]

    return header, rows


def num(row, key):
    """
    One column as a float, nan when missing or blank
    """
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return float("nan")


def deg(row, key):
    """
    One column of radians as degrees
    """
    return math.degrees(num(row, key))


def fmt(x, width=8, places=2):
    """
    A number, or dashes when there is not one
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "".rjust(width, ".")

    return f"{x:{width}.{places}f}"


def link_line(row, log_age):
    """
    Mavlink health and flight mode, on one line for every pane
    """
    if row is None:
        return "link: no flight log yet"

    hb = num(row, "hb_age_s")
    if math.isnan(hb):
        link = "link DOWN, no heartbeat"
    elif hb > 3:
        link = f"link LOST, heartbeat {hb:.1f}s old"
    else:
        link = f"link up, heartbeat {hb:.2f}s"

    mode = row.get("echoed_mode", "?")
    armed = "ARMED" if num(row, "armed") >= 1 else "disarmed"
    writing = "live" if log_age < STALE_S else f"STALE {log_age:.1f}s"

    return f"{link} | mode {mode} {armed} | log {writing}"


def pane_state(rows):
    """
    Attitude, velocity, position and altitude
    """
    r = rows[-1]

    return [
        f"  roll {fmt(deg(r, 'drone_roll'))}   "
        f"pitch {fmt(deg(r, 'drone_pitch'))}   "
        f"yaw {fmt(deg(r, 'drone_yaw'))}  deg",
        f"  yaw rate {fmt(deg(r, 'drone_yaw_rate'))} deg/s",
        "",
        f"  position   E {fmt(num(r, 'drone_px_meas'))}   "
        f"N {fmt(num(r, 'drone_py_meas'))}   "
        f"U {fmt(num(r, 'drone_pz_meas'))}  m",
        f"  velocity   E {fmt(num(r, 'drone_vx_meas'))}   "
        f"N {fmt(num(r, 'drone_vy_meas'))}   "
        f"U {fmt(num(r, 'drone_vz_meas'))}  m/s",
        f"  altitude   {fmt(num(r, 'drone_pz_meas'))} m",
        "",
        f"  GPS fix {r.get('GPS_fix_type', '?')}  "
        f"sats {r.get('sat_count', '?')}  "
        f"HDOP {fmt(num(r, 'HDOP'), 5)}",
        f"  battery {fmt(num(r, 'batt_voltage'), 6)} V  "
        f"{fmt(num(r, 'batt_current'), 6)} A  "
        f"{fmt(num(r, 'batt_rem_percent'), 4, 0)} %",
        f"  FC EKF variance  pos {fmt(num(r, 'ekf_pos_horiz_var'), 6)}  "
        f"vel {fmt(num(r, 'ekf_vel_var'), 6)}"]


def pane_control(rows):
    """
    What the outer loop is commanding, and whether the FC is taking it
    """
    r = rows[-1]

    return [
        f"  accel cmd   x {fmt(num(r, 'ux'))}   "
        f"y {fmt(num(r, 'uy'))}   z {fmt(num(r, 'uz'))}  m/s2",
        f"  yaw ref {fmt(deg(r, 'yaw_ref'))} deg   "
        f"rate {fmt(deg(r, 'yaw_rate_ref'))} deg/s",
        "",
        f"  drone ref    E {fmt(num(r, 'drone_px_ref'))}   "
        f"N {fmt(num(r, 'drone_py_ref'))}   "
        f"U {fmt(num(r, 'drone_pz_ref'))}  m",
        f"  payload ref  E {fmt(num(r, 'payload_px_ref'))}   "
        f"N {fmt(num(r, 'payload_py_ref'))}   "
        f"U {fmt(num(r, 'payload_pz_ref'))}  m",
        "",
        f"  setpoint mask sent {r.get('sent_setpoint_bitmask', '?')}  "
        f"echoed {r.get('echoed_setpoint_bitmask', '?')}  "
        f"round trip {fmt(num(r, 'setpoint_bitmask_callback_dt'), 6, 3)} s",
        f"  mode sent {r.get('sent_mode', '?')}  "
        f"echoed {r.get('echoed_mode', '?')}",
        "",
        f"  loop {fmt(num(r, 'cur_ctrl_freq'), 6, 1)} Hz   "
        f"dt {fmt(num(r, 'cur_loop_dt'), 7, 4)} s   "
        f"count {r.get('loop_count', '?')}"]


def pane_ekf(rows):
    """
    The payload filter: estimate, its own uncertainty, and the camera feeding it
    """
    r = rows[-1]
    sx = math.sqrt(max(num(r, "payload_cov_axx"), 0))
    sy = math.sqrt(max(num(r, "payload_cov_ayy"), 0))
    sp = math.sqrt(max(num(r, "payload_cov_psipsi"), 0))

    # the camera measurement the filter is folding in, and how old it is
    seen = [row for row in rows
            if not math.isnan(num(row, "payload_range_meas"))]
    if seen:
        age = num(r, "cur_time") - num(seen[-1], "cur_time")
        meas = (f"  camera range {fmt(num(seen[-1], 'payload_range_meas'), 6, 3)}"
                f" m   last measurement {fmt(age, 5, 2)} s ago")
    else:
        meas = "  camera range ......   nothing measured in these rows"

    ix, iy = num(r, "payload_innov_x"), num(r, "payload_innov_y")
    if math.isnan(ix):
        innov = "  innovation ......    no measurement this tick"
    else:
        innov = (f"  innovation  x {fmt(math.degrees(ix), 7)}   "
                 f"y {fmt(math.degrees(iy), 7)}   "
                 f"psi {fmt(deg(r, 'payload_innov_psi'), 7)}  deg")

    # sigma growing means it is coasting, shrinking means updates are landing
    trend = "steady"
    if len(rows) > 25:
        was = math.sqrt(max(num(rows[-25], "payload_cov_axx"), 0))
        trend = "growing" if sx > was*1.05 else (
            "shrinking" if sx < was*0.95 else "steady")

    held = r.get("sent_mode", "") == "HOLD"

    return [
        f"  alpha x {fmt(deg(r, 'payload_alpha_x'))} "
        f"+/- {fmt(math.degrees(sx), 5)}   "
        f"y {fmt(deg(r, 'payload_alpha_y'))} "
        f"+/- {fmt(math.degrees(sy), 5)}  deg",
        f"  alpha dot  x {fmt(deg(r, 'payload_alphadot_x'))}   "
        f"y {fmt(deg(r, 'payload_alphadot_y'))}  deg/s",
        f"  payload yaw {fmt(deg(r, 'payload_psi_p'))} "
        f"+/- {fmt(math.degrees(sp), 5)} deg",
        "",
        meas,
        innov,
        f"  measurements {'HELD, predicting only' if held else 'live'}",
        "",
        f"  sigma alpha_x {trend}",
        f"  covariance  axx {fmt(num(r, 'payload_cov_axx'), 9, 6)}  "
        f"ayy {fmt(num(r, 'payload_cov_ayy'), 9, 6)}  "
        f"axy {fmt(num(r, 'payload_cov_axy'), 9, 6)}"]


RENDER = {"state": pane_state, "control": pane_control, "ekf": pane_ekf}


def run_pane(name, data_dir="data"):
    """
    Redraw one pane until Ctrl+C, whether or not a flight is running
    """
    render = RENDER[name]
    while True:
        path = newest_log(data_dir)
        rows = []
        log_age = float("inf")
        if path is not None:
            _, rows = tail(path)
            log_age = time.time() - os.path.getmtime(path)

        lines = [f"[{name}]  {link_line(rows[-1] if rows else None, log_age)}",
                 os.path.basename(path) if path else "waiting for a flight log",
                 ""]
        lines += render(rows) if rows else [
            "  nothing is writing a log yet, so there is nothing to show"]

        # overwrite the last frame in place. Clearing the screen first flickers,
        # and a trailing newline on a full pane scrolls it up a row each time,
        # which is what made the header walk off the top
        size = (os.get_terminal_size() if sys.stdout.isatty()
                else os.terminal_size((80, 40)))
        body = [ln[:size.columns] + EOL for ln in lines[:size.lines]]
        sys.stdout.write(HOME + "\r\n".join(body) + BELOW)
        sys.stdout.flush()
        time.sleep(1/REFRESH_HZ)


def open_window(data_dir):
    """
    A new tmux window holding one pane per view, and its window id
    """
    me = os.path.abspath(__file__)
    cmds = [f"{sys.executable} {me} {pane} {data_dir}" for pane in PANES]
    first, *rest = cmds

    if os.environ.get("TMUX"):
        made = subprocess.run(
            ["tmux", "new-window", "-n", WINDOW, "-P", "-F", "#{window_id}",
             first], capture_output=True, text=True, check=True)
    else:
        made = subprocess.run(
            ["tmux", "new-session", "-d", "-s", SESSION, "-n", WINDOW,
             "-P", "-F", "#{window_id}", first],
            capture_output=True, text=True, check=True)

    window = made.stdout.strip()
    for cmd in rest:
        subprocess.run(["tmux", "split-window", "-v", "-t", window, cmd],
                       check=True)
    subprocess.run(["tmux", "select-layout", "-t", window, "even-vertical"],
                   check=True)

    return window


def existing_window():
    """
    The diagnostics window in this session, if one is already open
    """
    listed = subprocess.run(
        ["tmux", "list-windows", "-F", "#{window_name} #{window_id}"],
        capture_output=True, text=True)
    for line in listed.stdout.splitlines():
        name, _, window = line.partition(" ")
        if name == WINDOW:
            return window

    return None


def launch(data_dir="data"):
    """
    Put the panes in their own tmux window, so the shell you started from is
    left alone. Inside tmux already it becomes a window of this session,
    otherwise it starts one and attaches.
    """
    inside = bool(os.environ.get("TMUX"))
    window = existing_window() if inside else None
    if window is None:
        window = open_window(data_dir)

    if inside:
        subprocess.run(["tmux", "select-window", "-t", window], check=True)
        return

    os.execvp("tmux", ["tmux", "attach-session", "-t", SESSION])


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    where = sys.argv[2] if len(sys.argv) > 2 else "data"
    if arg in PANES:
        try:
            run_pane(arg, where)
        except KeyboardInterrupt:
            pass
    else:
        launch(where)
