This project is managed with the Python uv manager. Clone the project then run,
```
uv sync
```

## Plotting a flight

`data_scripts/plot.py` finds the data

```
uv run data_scripts/plot.py ls 0828
uv run data_scripts/plot.py 0828.last drone
uv run data_scripts/plot.py 0828.last ekf
uv run data_scripts/plot.py 0828.15 overlay
```

### overlay

Writes an `.mp4` file with a stats overlay given the data logs from a flight test.

### --post

Re-runs the EKF offline instead of drawing the logged states, and writes to `<id>_posthoc_ekf_overlay.mp4`.

### --compare

Draws the the as-flown estimates in cyan and post-hoc estimates in green. Helps with tuning the estimator. Writes to `<id>_compare_ekf_overlay.mp4`.
