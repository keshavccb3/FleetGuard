import numpy as np


def summarize(y_true, y_pred):
    err = np.asarray(y_pred) - np.asarray(y_true)
    # PHM08/C-MAPSS score. Positive error means overestimated RUL (late warning).
    score = np.where(err < 0, np.expm1(-err / 13.0), np.expm1(err / 10.0))
    return {"mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt(np.square(err).mean())),
            "phm08_score_sum": float(score.sum())}


def conformal_radius(y_true, y_pred, level: float) -> float:
    diagnostics = conformal_diagnostics(y_true, y_pred, level)
    return diagnostics["radius_cycles"]


def conformal_diagnostics(y_true, y_pred, level: float) -> dict:
    residuals = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    if residuals.ndim != 1 or not len(residuals):
        raise ValueError("Conformal calibration requires at least one engine residual.")
    if not np.isfinite(residuals).all():
        raise ValueError("Conformal calibration residuals must be finite.")
    if not 0.0 < float(level) < 1.0:
        raise ValueError("interval_level must be between 0 and 1.")
    n = len(residuals)
    rank = int(np.ceil((n + 1) * float(level)))
    # Standard split-conformal finite-sample order statistic. If the requested
    # rank exceeds n, the valid radius is unbounded rather than the max residual.
    radius = float("inf") if rank > n else float(np.partition(residuals, rank - 1)[rank - 1])
    return {
        "sample_count": int(n),
        "nominal_level": float(level),
        "quantile_method": "absolute residual order statistic, rank=ceil((n+1)*level), one-based",
        "selected_rank_one_based": int(rank),
        "selected_empirical_quantile": float(rank / n) if rank <= n else None,
        "radius_cycles": radius,
        "absolute_residual_min": float(residuals.min()),
        "absolute_residual_median": float(np.median(residuals)),
        "absolute_residual_max": float(residuals.max()),
    }
