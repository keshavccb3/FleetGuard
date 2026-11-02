# FleetGuard: Predicting Turbofan Remaining Useful Life from Sensor Sequences

An end-to-end, reproducible portfolio project that asks: **Can sensor histories estimate an engine's remaining useful life (RUL), and how reliable are those estimates for engines the model has not seen?** This is an offline research pipeline, not a web app or maintenance decision system.

## Dataset, source, and terms

The project uses subset **FD001** of NASA Ames PCoE's Turbofan Engine Degradation Simulation Data Set. NASA's PCoE repository has a direct download entry [here](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) (item 6). NASA's current entry also links the archive directly: [NASA-hosted ZIP](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip). NASA's separate [data.gov record](https://catalog.data.gov/dataset/c-mapss-aircraft-engine-simulator-data) reports that C-MAPSS is unavailable through that catalog and lists rights as null. The NASA PCoE repository currently provides the data archive, so the data used in this project was retrieved from the repository link rather than a third-party mirror.

**Terms:** the NASA catalog metadata does not specify a dataset license or reuse terms. The project does not redistribute raw data. Obtain it from NASA, review the current terms at the point of use, and cite A. Saxena and K. Goebel (2008), “Turbofan Engine Degradation Simulation Data Set,” NASA Ames Prognostics Data Repository. NASA's repository describes four subsets with different operating-condition and fault-mode combinations. FD001 is one operating condition and one degradation mode.

### Observed archive format

The downloaded outer NASA ZIP contained a nested `CMAPSSData.zip`. The nested archive was inspected. Its `readme.txt` describes each row as one engine cycle, 26 whitespace-separated numeric values, no header: unit ID, cycle, three operating settings, and 21 sensor measurements. FD001 contains 20,631 train rows across 100 engines, 13,096 test rows across 100 engines, and 100 terminal test RUL offsets. The first data rows were checked and contain 26 fields. The loader validates expected columns, finite values, missing values, and one terminal RUL per test engine before training.

Raw columns are named `unit`, `cycle`, `setting_1`–`setting_3`, and `sensor_01`–`sensor_21`. A unit is an engine trajectory; cycle is its sequence index; settings describe operating condition; sensors are simulator measurements. They have different scales, so standardization is applied. The data readme says sensor noise is present. FD001's single operating regime limits conclusions about condition shifts. The diagnostics report missingness, constant channels, and ranges/standard deviations; the condition plot compares standardized variable-sensor means across setting-1 quartiles. No physical sensor units are asserted because the source readme does not provide per-column engineering units.

## Setup and run

Use Python 3.12 (Python 3.10–3.11 should also work). From the project root (the directory containing `requirements.txt`), install packages in a virtual environment. Download NASA's outer ZIP from the link above, extract it, then extract its nested `CMAPSSData.zip`. Copy `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` from that inner archive into `data/raw/`. The raw data is intentionally excluded from this GitHub package.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m fleetguard
```

On macOS/Linux, substitute `python3.12 -m venv .venv` and activate with `source .venv/bin/activate`. If using Python 3.10 or 3.11, change the interpreter selector accordingly. To use another subset, change `subset` in `configs/default.json` and ensure those three matching files are present. `results/` contains the measured FD001 summary, per-engine errors, calibration predictions, diagnostics, and plots included with this project. A new run overwrites those result files. Large model weights and preprocessing artifacts are generated locally but remain git-ignored.

After a successful run, `fleetguard.inference.predict_history(history, Path("results"))` loads the saved scaler and LSTM checkpoint. Pass one engine's history as a pandas DataFrame containing `cycle`, the three settings, and all 21 sensor columns; it returns a capped prediction for the final cycle.

## Methods

### Labels and leakage control

For every complete training trajectory, uncapped RUL at cycle *t* is `last_cycle - t`; training labels are capped at the configured 125 cycles by default. The cap reduces the influence of the early-life period where degradation may not yet be observable and reflects a common piecewise-linear target convention. It is a declared configuration choice, not chosen from test performance. Test labels use the supplied terminal offset plus the number of observed cycles remaining to the last recorded test cycle. Test labels stay uncapped for final metrics.

Training engines are shuffled once with seed 42, then partitioned into 80% model-training and 20% validation. IDs are stored in `metrics.json`; assertions check train/validation engine disjointness. The official test set is a separate source split. The scaler is fit only on model-training engine rows. Every model is evaluated against the exact same official test engines and their final observed cycles. Splitting individual rows at random would let highly similar cycles from one engine enter both training and evaluation, making generalization to a new engine appear easier than it is.

### Models and assessment

* **Constant baseline:** median capped training RUL.
* **Regularized regression:** Ridge regression on the most recent window's last value, mean, standard deviation, and endpoint slope for each setting/sensor channel. `ridge_alpha=10` is configurable.
* **Sequence model:** a compact one-layer LSTM maps a fixed-length 30-cycle window to one RUL estimate. Short histories are left-padded by repeating their first observed row; long histories are truncated to the latest 30 rows. The three operating settings are standardized and included alongside sensor channels. Training examples are causal windows aligned to their ending cycle. Predictions are clipped to `[0, RUL cap]`.

The LSTM has a fixed epoch budget and configuration in `configs/default.json`; no test metric is used for tuning or selection. Reports include MAE, RMSE, per-engine errors, and an error distribution. The PHM08 asymmetric score is included with error defined as predicted minus true RUL: `exp(-error/13)-1` if error is negative, otherwise `exp(error/10)-1`, summed over engines. Thus overestimating RUL (a later warning) receives the steeper penalty. The formula is documented in NASA's [PHM08 prognostics metrics paper](https://c3.nasa.gov/dashlink/static/media/publication/PHM_2008_Metrics.pdf); NASA's [C-MAPSS/PHM paper](https://c3.ndc.nasa.gov/dashlink/static/media/publication/2008_IEEEPHM_CMAPPSDamagePropagation.pdf) discusses asymmetric timeliness scoring and the [data repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) identifies the benchmark and source. Lower score is better.

### Intervals, interpretation, and robustness

### Conformal interval audit and calibration design

The earlier implementation calibrated on the final row of each complete validation trajectory. Those rows are failure endpoints, so their raw RUL labels are all zero. The official test endpoints are right-censored and have positive RUL offsets. That mismatch made the calibration residuals too small for the test prediction problem; the previously reported 8% coverage and 5.28-cycle average width were consistent with this failure. The quantile formula itself was not the main defect; the validation examples were unrepresentative.

The revised code uses validation engines only and creates one pseudo-censored endpoint per engine. With seed `seed + 10`, it samples an integer censoring offset uniformly from 0 through `min(RUL cap, available validation history)`, then uses the history ending at that cycle and its uncapped RUL label. Test labels are not read when choosing these calibration examples, the quantile, or the radius. This is an explicit censoring assumption; it does not establish that validation and test censoring are exchangeable.

For the `n` validation-engine absolute residuals `abs(true RUL - prediction)`, the selected one-based rank is `ceil((n + 1) * level)`. At `n=20` and nominal `level=0.90`, rank 19 selects a radius of 44.7908 cycles (the 95th empirical percentile because of the finite sample). Each test interval is `[max(0, prediction - radius), prediction + radius]`. `metrics.json` records calibration engine count and IDs, sampled RUL range, order-statistic rank/quantile, residual summaries, radius, calibration coverage, test coverage, and average width. The test coverage is descriptive only and is never used to alter calibration.

Split conformal coverage relies on calibration and future examples being exchangeable. That assumption is uncertain here because censoring offsets are simulated uniformly for validation while the official test offsets come from NASA's benchmark generation. Do not interpret nominal coverage as a guarantee under another censoring process, operating regime, or real maintenance population.

Sensor contribution is assessed by repeatedly permuting one sensor channel across validation engines' pseudo-censored endpoint windows and measuring the change in MAE. This measures model reliance under that validation distribution; it is not causal importance and correlated sensors can mask each other. The robustness report adds seeded Gaussian noise with standard deviation 0.10 in training-standardized sensor units to test windows. In this run the noise stress test yielded MAE 16.29 and RMSE 20.86 versus the main LSTM test MAE 16.23 and RMSE 20.71. This is a controlled simulated perturbation, kept separate from the main test result, and is not evidence of a real-world failure.

## Actual results and run status

The complete benchmark pipeline was rerun locally with Python 3.12.14, NumPy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, and PyTorch 2.15.0+cpu. It completed with `FutureWarning` promoted to an error, so pandas dtype FutureWarnings would have stopped the run. This session had no attached Colab tab; the following are measured local CPU results, not a claim that the rerun occurred in Colab.

| Model | Test MAE (cycles) | Test RMSE (cycles) |
| --- | ---: | ---: |
| Constant median | 39.08 | 49.82 |
| Ridge recent-window | 31.02 | 37.57 |
| LSTM | 16.23 | 20.71 |

For the 90% nominal LSTM interval, calibration used 20 validation engines. The selected rank was 19, with a 44.7908-cycle radius and 95% empirical calibration coverage (19/20). On the 100 official held-out test engines, observed coverage was 95% (95/100) and average interval width was 82.52 cycles. This meets the nominal coverage level on this one held-out benchmark split, but the intervals are broad and their coverage depends on the pseudo-censoring assumption. It supports only a descriptive result for this split, not a general reliability claim.

The previously supplied Colab point scores were close but not identical (LSTM MAE 16.36 and RMSE 20.75 versus 16.23 and 20.71 here). The Colab package versions and hardware were not provided, so the difference cannot be attributed to a specific cause. The key interval change is from calibrating on validation failure endpoints (true RUL zero) to validation pseudo-censored endpoints with positive RUL examples; the new test interval is wider and its measured coverage is 95%, not 8%.

## Limitations

* This is simulator-generated data with a small fleet, not operational aircraft maintenance data.
* FD001 has one operating condition and one simulated fault mode; this project does not establish transfer to other subsets, climates, fleets, maintenance policies, or real hardware.
* Training labels assume each training engine's final observed cycle is failure. Test labels depend on the provided terminal RUL offsets.
* Capping compresses early-life targets; final reported errors use uncapped true test RUL but predictions cannot exceed the configured cap.
* The LSTM uses a single fixed window length and no hyperparameter search. The baseline and model results may depend on this declared configuration.
* Conformal intervals use simulated uniform censoring offsets up to the RUL cap. The official test offset mechanism may differ; the measured 95% test coverage is one split-level observation, and the 82.52-cycle mean width is broad.
* GPU kernels and numerical library versions can cause small remaining nondeterminism. Seeds are set and deterministic PyTorch algorithms are requested where supported.

## Interview preparation

### Engine-level split and leakage

**Plain language:** I split whole engines into training and validation groups so the validation result answers whether a model can handle a new engine. Random rows would mix cycles from the same engine across both sides.

**Follow-up:** How did you check the split? **Answer:** I shuffle sorted training engine IDs with seed 42, allocate 80%/20%, record the IDs, assert train and validation disjointness, and use NASA's separate official test trajectories for final evaluation.

### RUL capping and labels

**Plain language:** A training engine's remaining life is the number of cycles until its last cycle; I cap large early-life values at 125 to avoid asking the model to distinguish very healthy engines whose sensors may look similar.

**Follow-up:** Did you tune the cap on test data? **Answer:** No. It is an explicit default config choice. Training labels are capped, while test labels are reconstructed from NASA's official terminal offsets and remain uncapped for final reporting.

### Windowing and sequence-to-one prediction

**Plain language:** The LSTM sees the latest 30 cycles and predicts one RUL value for the last cycle in that window.

**Follow-up:** What happens with short sequences? **Answer:** I left-pad by repeating the first observed row; long sequences are truncated to the latest 30. Settings are included as three additional standardized input channels.

### Baseline versus LSTM

**Plain language:** A constant and Ridge model show whether a sequence model adds value beyond simple estimates and recent summary statistics.

**Follow-up:** Why an LSTM? **Answer:** It is a compact way to learn ordered patterns in fixed sensor windows. This implementation does not claim it is optimal; its purpose is an interpretable sequence-model comparison under a fixed training budget.

### MAE, RMSE, and asymmetric score

**Plain language:** MAE is average absolute cycle error, RMSE gives large misses more weight, and PHM08 adds an asymmetric cost for RUL overestimates.

**Follow-up:** Why define error as prediction minus truth? **Answer:** A positive residual means predicted RUL is too high, so it implies a later warning; the score penalizes that side with the 10-cycle exponential scale versus 13 for early warnings.

### Calibration, coverage, and width

**Plain language:** I use separate validation engines to choose a shared error radius, then report how often test truths fall inside the resulting intervals and how wide the intervals are.

**Follow-up:** Is 90% coverage guaranteed for any future engine? **Answer:** No. This run observed 95% coverage on 100 FD001 test engines, with an average interval width of 82.52 cycles. Calibration uses simulated uniform censoring offsets, so its exchangeability with the benchmark test censoring process is uncertain; other regimes or fleets can differ further.

### Sensor importance

**Plain language:** I shuffle one sensor across validation engines and see how much MAE worsens, which shows whether this fitted model relies on that channel.

**Follow-up:** Is that causal? **Answer:** No. Sensors can be correlated, and the result depends on this model and validation distribution. It is a reliance diagnostic, not a causal explanation.

### Domain shift

**Plain language:** FD001's held-out engines test new units from the same simulated setting. The added noise test is only a controlled stress case.

**Follow-up:** Does this prove maintenance readiness? **Answer:** No. Simulator data, one regime, assumed failure endpoints, and no operational validation limit transfer. A real deployment would need fleet-specific validation, monitoring, and maintenance-context review.

## Resume bullet

**Before running the benchmark:** “Built a reproducible turbofan RUL modeling pipeline with engine-level leakage controls, Ridge and LSTM baselines, validation-calibrated prediction intervals, and robustness diagnostics; benchmark results pending execution.” Replace this with measured results only after a successful run.
