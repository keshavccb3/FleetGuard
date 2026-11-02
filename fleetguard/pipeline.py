from pathlib import Path
import json
import platform
import random
import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .data import (SENSOR_COLS, SETTING_COLS, add_train_labels, add_test_labels,
                   final_rows, latest_tabular, read_cmapss, tabular_latest_features,
                   validation_censoring_sample,
                   window_for_unit, windows_at_each_cycle)
from .metrics import conformal_diagnostics, summarize
from .model import WindowLSTM
from .plots import explore_plots


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def run(data_dir: Path, results: Path, config: dict):
    results.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"]); set_seed(seed)
    subset = config["subset"]
    train_raw, test_raw, official_rul = read_cmapss(data_dir, subset)
    train = add_train_labels(train_raw, config["rul_cap"])
    test = add_test_labels(test_raw, official_rul, config["rul_cap"])

    units = np.array(sorted(train.unit.unique()))
    rng = np.random.default_rng(seed); rng.shuffle(units)
    n_train = int(len(units) * config["train_fraction"])
    n_val = int(len(units) * config["validation_fraction"])
    if not np.isclose(config["train_fraction"] + config["validation_fraction"], 1.0):
        raise ValueError("train_fraction and validation_fraction must sum to 1.0.")
    train_ids, val_ids, _ = np.split(units, [n_train, n_train+n_val])
    if not len(val_ids) or not len(train_ids): raise ValueError("Split produced an empty partition.")
    assert not (set(train_ids) & set(val_ids))
    assert set(train_ids) | set(val_ids) == set(units)
    train_part = train[train.unit.isin(train_ids)].copy()
    val_part = train[train.unit.isin(val_ids)].copy()
    feature_cols = SETTING_COLS + SENSOR_COLS

    # Explicit float64 columns prevent pandas incompatible-dtype assignment warnings.
    for frame in (train_part, val_part):
        frame[feature_cols] = frame[feature_cols].astype(np.float64)
    test_scaled = test.copy()
    test_scaled[feature_cols] = test_scaled[feature_cols].astype(np.float64)

    # Fit preprocessing on training engines only; held-out engine test rows are never used.
    scaler = StandardScaler().fit(train_part[feature_cols])
    for frame in (train_part, val_part, test_scaled):
        scaled_values = scaler.transform(frame[feature_cols].astype(np.float64))
        frame.loc[:, feature_cols] = scaled_values.astype(np.float64, copy=False)

    # Transparent constant baseline: training-set median of capped cycle labels.
    constant = float(train_part.rul.median())
    ridge_x, ridge_y = tabular_latest_features(train_part, feature_cols, config["window_length"])
    ridge = Ridge(alpha=config["ridge_alpha"]).fit(ridge_x, ridge_y)
    test_final = final_rows(test_scaled)
    test_ridge = np.clip(ridge.predict(latest_tabular(test_final, feature_cols, config["window_length"])), 0, config["rul_cap"])

    # Validation trajectories are complete run-to-failure sequences, unlike the
    # right-censored official test histories. Simulate one censoring point per
    # validation engine using a seeded offset from [0, RUL cap], without test data.
    cal_frame, cal_manifest = validation_censoring_sample(
        val_part, int(config["rul_cap"]), seed + 10)
    if cal_manifest.unit.nunique() != len(val_ids):
        raise ValueError("Calibration must contain exactly one endpoint for each validation engine.")
    cal_final = final_rows(cal_frame).sort_values("unit")
    cal_true_raw = cal_final.rul_raw.to_numpy(dtype=float)
    x_cal = np.stack([window_for_unit(g, feature_cols, config["window_length"])
                      for _, g in cal_frame.groupby("unit", sort=True)])
    val_ridge = np.clip(ridge.predict(latest_tabular(cal_frame, feature_cols, config["window_length"])),
                        0, config["rul_cap"])

    x_train, y_train, _, _ = windows_at_each_cycle(train_part, feature_cols, config["window_length"])
    x_test = np.stack([window_for_unit(g, feature_cols, config["window_length"])
                       for _, g in test_final.groupby("unit")])
    y_test_raw = test_final.sort_values("unit").rul_raw.to_numpy(dtype=float)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WindowLSTM(len(feature_cols), config["hidden_size"], config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
                        batch_size=config["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    loss_fn = torch.nn.MSELoss()
    # Fixed training budget from config. No test metrics or labels participate in fitting.
    for _ in range(int(config["epochs"])):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward(); optimizer.step()

    def predict(a):
        model.eval()
        with torch.no_grad():
            return model(torch.tensor(a, dtype=torch.float32, device=device)).cpu().numpy()
    val_lstm = np.clip(predict(x_cal), 0, config["rul_cap"])
    test_lstm = np.clip(predict(x_test), 0, config["rul_cap"])

    calibration = conformal_diagnostics(cal_true_raw, val_lstm, config["interval_level"])
    radius = calibration["radius_cycles"]
    test_lower = np.maximum(0, test_lstm-radius)
    test_upper = test_lstm+radius
    calibration_lower = np.maximum(0, val_lstm-radius)
    calibration_coverage = float(((cal_true_raw >= calibration_lower) &
                                  (cal_true_raw <= val_lstm+radius)).mean())

    scores = {
        "constant_median": summarize(y_test_raw, np.repeat(constant, len(y_test_raw))),
        "ridge_recent_window": summarize(y_test_raw, test_ridge),
        "lstm": summarize(y_test_raw, test_lstm),
    }
    table = pd.DataFrame({"unit": test_final.sort_values("unit").unit.to_numpy(),
                          "true_rul": y_test_raw, "constant_prediction": constant,
                          "ridge_prediction": test_ridge, "lstm_prediction": test_lstm,
                          "lstm_error": test_lstm-y_test_raw,
                          "interval_lower": test_lower, "interval_upper": test_upper})
    table.to_csv(results / "per_engine_test_predictions.csv", index=False)
    metrics = {"subset": subset, "seed": seed, "rul_cap": config["rul_cap"],
               "window_length": config["window_length"], "split": {
                   "train_engine_ids": sorted(map(int, train_ids)), "validation_engine_ids": sorted(map(int, val_ids)),
                   "official_test_engine_ids": sorted(map(int, test.unit.unique())),
                   "overlap_train_validation": False, "official_test_ids_are_separate_source": True},
               "models": scores,
               "uncertainty": {"method": "split conformal on one pseudo-censored endpoint per validation engine",
                   "calibration_design": "uniform seeded censoring offset from 0 to min(RUL cap, available validation trajectory history); no test labels used",
                   "calibration": {**calibration,
                       "engine_count": int(cal_manifest.unit.nunique()),
                       "validation_engine_ids": sorted(map(int, cal_manifest.unit.unique())),
                       "target_rul_min": int(cal_manifest.target_rul.min()),
                       "target_rul_max": int(cal_manifest.target_rul.max()),
                       "empirical_calibration_coverage": calibration_coverage},
                   "test": {"engine_count": int(len(y_test_raw)),
                       "observed_coverage": float(((y_test_raw >= test_lower) & (y_test_raw <= test_upper)).mean()),
                       "average_interval_width_cycles": float((test_upper-test_lower).mean())}},
               "data_diagnostics": {"train_rows": len(train_raw), "test_rows": len(test_raw),
                   "missing_values": int(train_raw.isna().sum().sum()),
                   "constant_sensors": [c for c in SENSOR_COLS if train_raw[c].nunique() <= 1],
                   "sensor_scale_std": train_raw[SENSOR_COLS].std().to_dict()},
               "config": config, "device": str(device),
               "runtime_versions": {"python": platform.python_version(), "numpy": np.__version__,
                   "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                   "torch": torch.__version__},
               "nondeterminism_note": "Seeds are set and deterministic torch algorithms are requested; backend/library hardware differences may still change floating point results."}

    # Sensor-group permutation on validation pseudo-censored endpoints only.
    baseline_mae = summarize(cal_true_raw, val_lstm)["mae"]
    importance = []
    rng = np.random.default_rng(seed+1)
    for si, sensor in enumerate(SENSOR_COLS):
        delta = []
        for _ in range(int(config["permutation_repeats"])):
            xp = x_cal.copy(); xp[:, :, len(SETTING_COLS)+si] = xp[rng.permutation(len(xp)), :, len(SETTING_COLS)+si]
            delta.append(summarize(cal_true_raw, np.clip(predict(xp), 0, config["rul_cap"]))["mae"]-baseline_mae)
        importance.append({"sensor": sensor, "mae_increase": float(np.mean(delta))})
    pd.DataFrame(importance).sort_values("mae_increase", ascending=False).to_csv(results/"validation_sensor_permutation_importance.csv", index=False)

    # Controlled sensor noise is an artificial stress test, reported separately.
    noisy = x_test.copy()
    noise_rng = np.random.default_rng(seed+2)
    noisy[:, :, len(SETTING_COLS):] += noise_rng.normal(0, config["noise_std"], noisy[:, :, len(SETTING_COLS):].shape)
    noisy_pred = np.clip(predict(noisy), 0, config["rul_cap"])
    metrics["robustness_simulated_sensor_noise"] = {"noise_std_in_train_scaled_sensor_units": config["noise_std"],
        "metrics": summarize(y_test_raw, noisy_pred), "baseline_test_lstm_metrics": scores["lstm"],
        "note": "Controlled synthetic perturbation only; not evidence of real-world failure or domain shift."}
    metrics["phm08_definition"] = "sum(exp(-error/13)-1) when error<0 else exp(error/10)-1; error=prediction-true RUL; lower is better."
    metrics["validation_pseudo_censored_metrics"] = {
        "ridge": summarize(cal_true_raw, val_ridge), "lstm": summarize(cal_true_raw, val_lstm)}
    pd.DataFrame({"unit": cal_final.unit.to_numpy(), "cutoff_cycle": cal_manifest.sort_values("unit").cutoff_cycle.to_numpy(),
                  "target_rul": cal_true_raw, "lstm_prediction": val_lstm,
                  "absolute_residual": np.abs(cal_true_raw-val_lstm)}).to_csv(
                      results/"validation_calibration_predictions.csv", index=False)
    (results/"metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"unit": test_final.sort_values("unit").unit.to_numpy(), "true_rul": y_test_raw,
                  "lstm_error": test_lstm-y_test_raw}).to_csv(results/"error_distribution.csv", index=False)

    # Diagnostic table: one row per column, range, standard deviation, and missing count.
    diagnostics = []
    for c in [*SETTING_COLS, *SENSOR_COLS]:
        diagnostics.append({"column": c, "min": float(train_raw[c].min()), "max": float(train_raw[c].max()),
                            "std": float(train_raw[c].std()), "unique": int(train_raw[c].nunique()),
                            "missing": int(train_raw[c].isna().sum())})
    pd.DataFrame(diagnostics).to_csv(results/"data_diagnostics.csv", index=False)
    explore_plots(train_raw, results)
    joblib.dump({"scaler": scaler, "feature_columns": feature_cols, "ridge": ridge,
                 "constant_prediction": constant,
                 "window_length": config["window_length"], "rul_cap": config["rul_cap"]}, results/"preprocessing_and_ridge.joblib")
    torch.save({"state_dict": model.cpu().state_dict(), "n_features": len(feature_cols),
                "hidden_size": config["hidden_size"], "dropout": config["dropout"]}, results/"lstm.pt")
    print(json.dumps({"test_engines": len(test_final), "models": scores,
                      "calibration": metrics["uncertainty"]["calibration"],
                      "test_interval_coverage": metrics["uncertainty"]["test"]["observed_coverage"],
                      "test_interval_width": metrics["uncertainty"]["test"]["average_interval_width_cycles"]}, indent=2))
