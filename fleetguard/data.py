from pathlib import Path
import numpy as np
import pandas as pd

SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"sensor_{i:02d}" for i in range(1, 22)]
RAW_COLS = ["unit", "cycle", *SETTING_COLS, *SENSOR_COLS]


def read_cmapss(data_dir: Path, subset: str):
    """Read official whitespace-delimited, headerless C-MAPSS files."""
    train_path = data_dir / f"train_{subset}.txt"
    test_path = data_dir / f"test_{subset}.txt"
    rul_path = data_dir / f"RUL_{subset}.txt"
    missing = [p.name for p in (train_path, test_path, rul_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {', '.join(missing)} in {data_dir}. Obtain the C-MAPSS archive "
            "from NASA's PCoE data repository when available; see README.md."
        )
    train = pd.read_csv(train_path, sep=r"\s+", header=None, names=RAW_COLS)
    test = pd.read_csv(test_path, sep=r"\s+", header=None, names=RAW_COLS)
    rul = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["rul"])["rul"].to_numpy()
    for label, frame in (("train", train), ("test", test)):
        if frame.shape[1] != 26 or frame.isna().any().any():
            raise ValueError(f"{label} must have 26 complete numeric columns; got {frame.shape}.")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"{label} contains non-finite values.")
        if (frame["cycle"] < 1).any():
            raise ValueError(f"{label} contains invalid cycle numbers.")
    if test.unit.nunique() != len(rul):
        raise ValueError("RUL file must contain one terminal RUL per test unit, in unit order.")
    return train, test, rul


def add_train_labels(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    out = frame.copy()
    last = out.groupby("unit").cycle.transform("max")
    out["rul_raw"] = last - out.cycle
    out["rul"] = out.rul_raw.clip(upper=cap)
    return out


def add_test_labels(frame: pd.DataFrame, terminal_rul: np.ndarray, cap: int) -> pd.DataFrame:
    out = frame.copy()
    offsets = dict(zip(sorted(out.unit.unique()), terminal_rul))
    last = out.groupby("unit").cycle.transform("max")
    out["rul_raw"] = out.unit.map(offsets) + last - out.cycle
    out["rul"] = out.rul_raw.clip(upper=cap)
    return out


def final_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["unit", "cycle"]).groupby("unit", as_index=False).tail(1).copy()


def validation_censoring_sample(frame: pd.DataFrame, cap: int, seed: int):
    """Sample one pseudo-censored endpoint per validation engine, without test data.

    The hidden RUL offset is sampled uniformly from 0..min(cap, available history)
    using only the validation trajectory's observed run-to-failure history.
    """
    rng = np.random.default_rng(seed)
    prefixes, manifest = [], []
    for unit, group in frame.sort_values(["unit", "cycle"]).groupby("unit", sort=True):
        last_cycle = int(group.cycle.max())
        first_cycle = int(group.cycle.min())
        max_offset = min(int(cap), last_cycle - first_cycle)
        if max_offset < 0:
            raise ValueError(f"Unit {unit} has an invalid cycle range.")
        target_rul = int(rng.integers(0, max_offset + 1))
        cutoff_cycle = last_cycle - target_rul
        prefix = group[group.cycle <= cutoff_cycle].copy()
        if prefix.empty or int(prefix.cycle.max()) != cutoff_cycle:
            raise ValueError(f"Unit {unit} is missing expected cycle {cutoff_cycle}.")
        observed_rul = float(prefix.loc[prefix.cycle == cutoff_cycle, "rul_raw"].iloc[-1])
        if observed_rul != target_rul:
            raise ValueError(f"Unit {unit}: sampled target RUL does not align to cycle labels.")
        prefixes.append(prefix)
        manifest.append({"unit": int(unit), "cutoff_cycle": cutoff_cycle,
                         "target_rul": target_rul, "max_sampled_rul": max_offset})
    if not prefixes:
        raise ValueError("No validation engines available for conformal calibration.")
    return pd.concat(prefixes, ignore_index=True), pd.DataFrame(manifest)


def window_for_unit(group: pd.DataFrame, columns: list[str], length: int) -> np.ndarray:
    x = group.sort_values("cycle")[columns].to_numpy(dtype=np.float32)
    # Left-pad short sequences by repeating their first observation.
    if len(x) < length:
        pad = np.repeat(x[:1], length - len(x), axis=0)
        x = np.concatenate([pad, x], axis=0)
    return x[-length:]


def windows_at_each_cycle(frame: pd.DataFrame, columns: list[str], length: int):
    """Build causal fixed-length examples, padding left with the first seen row."""
    xs, ys, units, cycles = [], [], [], []
    for unit, group in frame.sort_values(["unit", "cycle"]).groupby("unit"):
        arr = group[columns].to_numpy(dtype=np.float32)
        labels = group.rul.to_numpy(dtype=np.float32)
        for i in range(len(group)):
            start = max(0, i - length + 1)
            seq = arr[start:i + 1]
            if len(seq) < length:
                seq = np.concatenate([np.repeat(seq[:1], length - len(seq), axis=0), seq])
            xs.append(seq)
            ys.append(labels[i])
            units.append(int(unit))
            cycles.append(int(group.cycle.iloc[i]))
    return np.stack(xs), np.asarray(ys), np.asarray(units), np.asarray(cycles)


def tabular_latest_features(frame: pd.DataFrame, columns: list[str], length: int):
    """Recent-window level, slope, and variability features for each train row."""
    records, target = [], []
    for _, group in frame.sort_values(["unit", "cycle"]).groupby("unit"):
        x = group[columns].to_numpy(dtype=np.float32)
        y = group.rul.to_numpy(dtype=np.float32)
        for i in range(len(group)):
            w = x[max(0, i-length+1):i+1]
            records.append(np.concatenate([w[-1], w.mean(0), w.std(0),
                                           (w[-1]-w[0]) / max(1, len(w)-1)]))
            target.append(y[i])
    return np.asarray(records), np.asarray(target)


def latest_tabular(frame: pd.DataFrame, columns: list[str], length: int):
    out = []
    for _, group in frame.sort_values(["unit", "cycle"]).groupby("unit"):
        w = group[columns].to_numpy(dtype=np.float32)[-length:]
        out.append(np.concatenate([w[-1], w.mean(0), w.std(0),
                                   (w[-1]-w[0]) / max(1, len(w)-1)]))
    return np.asarray(out)
