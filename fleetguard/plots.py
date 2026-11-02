from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def explore_plots(train: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    sensors = [f"sensor_{i:02d}" for i in range(1, 22)]
    # Separate axes by sensor so scale differences do not flatten small channels.
    selected_sensors = ("sensor_02", "sensor_07", "sensor_14")
    units = sorted(train.unit.unique())[:3]
    fig, axes = plt.subplots(len(units), len(selected_sensors), figsize=(13, 8), sharex="col")
    for row, unit in enumerate(units):
        g = train[train.unit == unit]
        for col, name in enumerate(selected_sensors):
            ax = axes[row, col]
            ax.plot(g.cycle, g[name], linewidth=1)
            if row == 0:
                ax.set_title(name)
            if col == 0:
                ax.set_ylabel(f"unit {unit}\nraw value")
            if row == len(units)-1:
                ax.set_xlabel("cycle")
    fig.suptitle("Selected raw sensor trajectories (separate axes preserve channel scale)")
    fig.tight_layout()
    fig.savefig(out_dir / "sensor_trajectories.png", dpi=150)
    plt.close(fig)

    binned = train.copy()
    binned["setting_1_bin"] = pd.qcut(binned.setting_1, q=4, duplicates="drop")
    sensor_std = train[sensors].std()
    sensor_counts = train[sensors].nunique(dropna=False)
    variable_sensors = [c for c in sensors if sensor_counts[c] > 1 and sensor_std[c] > 1e-8]
    if variable_sensors:
        binned[variable_sensors] = (binned[variable_sensors] - train[variable_sensors].mean()) / sensor_std[variable_sensors]
    fig, ax = plt.subplots(figsize=(12, 5))
    binned.groupby("setting_1_bin", observed=True)[variable_sensors].mean().T.plot(ax=ax)
    ax.set_title("Standardized sensor means across setting_1 quartiles (FD001)")
    ax.set_xlabel("sensor")
    ax.set_ylabel("mean standardized reading")
    fig.tight_layout()
    fig.savefig(out_dir / "operating_condition_sensor_means.png", dpi=150)
    plt.close(fig)
