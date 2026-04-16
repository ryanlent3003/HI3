# CVEN 6920 Assignment 3
## Full Manuscript Draft: LSTM-Based Streamflow Prediction in the Upper Colorado Basin

## Title
Regional Streamflow Prediction with Long Short-Term Memory Networks Using Multi-Site Hydrometeorological Forcing and Unseen-Basin Evaluation

## Abstract
Accurate daily streamflow prediction is essential for water resources management, flood preparedness, and drought response in snow-influenced basins of the western United States. This study develops a Long Short-Term Memory (LSTM) model to predict streamflow in the Upper Colorado region using integrated hydrologic datasets that include observed discharge, meteorological forcing, and static catchment descriptors. The model was trained on three hydrologically distinct USGS gauges and evaluated on one fully unseen gauge to test spatial transferability under a strict no-leakage design. Inputs included precipitation, minimum and maximum air temperature, shortwave radiation, vapor pressure, snow water equivalent, day length, seasonal harmonics, elevation, and drainage area. Sequences were prepared with a 30-day sliding window, and model training used Adam optimization with early stopping, consistent with LSTM learning principles introduced by Hochreiter and Schmidhuber (1997) and later hydrologic applications such as Kratzert et al. (2018). The unseen-site evaluation achieved NSE = 0.646 and KGE = 0.791, exceeding the assignment performance target of KGE > 0.40; interpretation of these scores follows established formulations from Nash and Sutcliffe (1970) and Gupta et al. (2009). Results demonstrate that a compact regional LSTM can transfer skillfully to a withheld basin when trained on gauges selected to span contrasting runoff regimes.

## Assignment #3 Coverage Summary
This submission addresses the Hydroinformatics Assignment #3 requirements under the Streamflow Prediction using LSTM Models track (Option 2).

### Overview requirements
- Model development and evaluation for a hydrologic machine learning application are documented in Sections 3-6.
- Technical implementation is reproducible in Python using a complete workflow from data retrieval through model training, evaluation, and plotting (Sections 3, 4, and 8).
- Critical interpretation of model behavior and hydrologic relevance is provided in Section 6 (Discussion).

### Learning objectives coverage
- Apply machine learning techniques to real-world datasets: satisfied through multi-site LSTM development using USGS NWIS and Daymet data (Sections 3 and 4).
- Develop reproducible workflows in Python: satisfied through explicit preprocessing, model setup, and data/code availability details (Sections 3, 4, and 8).
- Evaluate model performance using appropriate metrics: satisfied with NSE, KGE, RMSE, and MAE, including unseen-site threshold testing (Sections 4.3 and 5).
- Interpret results in a hydrologic context: satisfied through transferability analysis, error-structure interpretation, and basin-process discussion (Section 6).
- Communicate findings clearly through figures and written analysis: satisfied with manuscript narrative plus Figure 1-4 references and site-wise performance table (Sections 5 and 6).

## 1. Introduction
Daily streamflow forecasting remains a central challenge in hydrology because runoff generation is influenced by non-linear and lagged interactions among precipitation, temperature, snow storage, and basin properties. In managed mountain basins, the ability to predict streamflow with sufficient lead and accuracy directly supports reservoir operations, flood-risk mitigation, and seasonal allocation decisions.

Recent machine learning approaches, especially recurrent neural networks and LSTM architectures, have shown strong performance for hydrologic time series because they are designed to learn long- and short-term dependencies in sequential data (Hochreiter & Schmidhuber, 1997; Kratzert et al., 2018). Unlike static regressors, LSTMs can represent watershed memory effects such as snow accumulation and delayed melt response without manually specifying many lag terms (Kratzert et al., 2018).

The central objective of this study is to evaluate whether a regional LSTM, trained on multiple Upper Colorado gauges, can generalize to an unseen basin. The guiding hypothesis is that training on hydrologically diverse sites improves transfer performance at a withheld gauge. To test this hypothesis, three sites were used for training and one site was excluded entirely until final evaluation.

## 2. Study Area and Site Selection
The analysis focuses on the Upper Colorado regional network, where streamflow exhibits strong seasonal forcing and varying degrees of regulation and tributary behavior.

The selected training gauges were:
- 09085000, Colorado River at Radium, CO (headwater/mainstem snowmelt signal)
- 09095500, Colorado River near Cameo, CO (downstream mainstem integrator)
- 09107000, Dolores River at Dolores, CO (tributary regime contrast)

The unseen test gauge was:
- 09070500, Blue River below Green Mountain Reservoir, CO (below-dam tributary, withheld from all training)

This combination was chosen to span multiple hydrologic response types while preserving a realistic transfer test at a regulated tributary location. Site positions and regional context are documented in the study map with CONUS inset.

## 3. Data and Preprocessing
### 3.1 Data sources
Daily streamflow observations were retrieved from USGS NWIS using parameter code 00060 (discharge, cfs). Meteorological predictors were extracted from Daymet at each gauge coordinate. The forcing set included precipitation (prcp), minimum temperature (tmin), maximum temperature (tmax), shortwave radiation (srad), vapor pressure (vp), snow water equivalent (swe), and day length (dayl). Static site descriptors were elevation and drainage area.

### 3.2 Time period and feature engineering
The model period was 2001-01-01 through 2020-12-30, yielding approximately 7,300 daily records per site before sequence windowing. To represent annual periodicity, day-of-year sine and cosine features were included. The final predictor set consisted of 11 inputs per timestep.

### 3.3 Target transformation and scaling
To reduce skew and stabilize optimization, streamflow was transformed as log1p(Q) for training and inverse transformed for all reported metrics. Feature and target scaling were fit only on training-site data and then applied to validation and unseen-site data to avoid leakage.

### 3.4 Sequence construction
LSTM-ready samples were generated using a 30-day sliding window. Each training example consisted of the previous 30 days of predictors, with next-day streamflow as the prediction target.

## 4. Methods
### 4.1 Model architecture
The model was implemented in PyTorch as a two-layer LSTM regressor with hidden size 64 and dropout 0.2. The recurrent output at the final timestep was passed to a dense head: Linear(64 -> 32), ReLU, Linear(32 -> 1).

### 4.2 Optimization and training strategy
The loss function was mean squared error (MSE), optimized with Adam at learning rate 1e-3 and batch size 64. Maximum training length was 80 epochs, with early stopping patience of 10 epochs.

Training used pooled sequences from only the three designated training gauges. A chronological 80/20 split of pooled training sequences was used for train/validation. No data from the unseen site (09070500) were used during fitting or validation.

### 4.3 Evaluation metrics
Performance was assessed using:
- Nash-Sutcliffe Efficiency (NSE)
- Kling-Gupta Efficiency (KGE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

NSE interpretation follows Nash and Sutcliffe (1970), while KGE interpretation follows Gupta et al. (2009).

The assignment success criterion required unseen-site KGE > 0.40.

## 5. Results
Training converged with early stopping and stable validation behavior. The model achieved strong fit at training locations and retained meaningful skill at the unseen site. The geographic relationship among training and test gauges is shown in Figure 1 (`study_sites_map_conus_inset.png`), and model convergence during fitting is shown in Figure 2 (`training_history.png`).

### 5.1 Site-wise performance
| Site | Role | NSE | KGE | RMSE (cfs) | MAE (cfs) |
|---|---|---:|---:|---:|---:|
| 09085000 | Train | 0.868 | 0.806 | 422.172 | 214.372 |
| 09095500 | Train | 0.838 | 0.838 | 1517.742 | 763.688 |
| 09107000 | Train | 0.820 | 0.860 | 55.436 | 23.575 |
| 09070500 | Test (Unseen) | 0.646 | 0.791 | 1297.437 | 666.826 |

### 5.2 Target threshold check
The unseen-site KGE was 0.791, which exceeded the assignment target (KGE > 0.40) by a wide margin.

### 5.3 Visual diagnostics
Observed-versus-predicted diagnostics at the unseen site showed good agreement in seasonal pattern and moderate flow magnitude, with expected residual error during higher-flow periods. Figure 3 (`obs_vs_pred_timeseries_09070500.png`) highlights that the model tracks seasonal timing and base-to-moderate flow transitions well. Figure 4 (`obs_vs_pred_scatter_09070500.png`) shows strong clustering near the 1:1 line at low-to-moderate discharges and increasing spread at higher flows.

## 6. Discussion
The unseen-site evaluation demonstrates that the regional LSTM learned transferable hydrologic structure rather than memorizing site-specific behavior. In particular, the test-site performance (NSE = 0.646; KGE = 0.791) indicates that the model reproduced temporal dynamics and distributional characteristics with meaningful skill at a gauge that was excluded from all stages of fitting. Under the standard interpretation of NSE and KGE, this combination of scores suggests useful predictive value for practical hydrologic analysis, including timing-sensitive applications where both bias and variability matter (Nash & Sutcliffe, 1970; Gupta et al., 2009).

The strong transfer likely reflects deliberate training-site diversity. The model was exposed to headwater/mainstem behavior (09085000), downstream integrative response (09095500), and tributary timing contrast (09107000), which together broadened the range of hydroclimatic patterns seen during training. This diversity is visible in the site distribution shown in Figure 1 (`study_sites_map_conus_inset.png`). Because the predictor set included snow-sensitive forcing (swe), meteorological drivers, and seasonal harmonics, the network could represent memory effects and annual regime structure that are central to mountain-basin flow prediction (Hochreiter & Schmidhuber, 1997; Kratzert et al., 2018).

The error structure is also informative. The learning dynamics in Figure 2 (`training_history.png`) indicate stable convergence without late-epoch divergence, supporting the reliability of the selected early-stopped model state. Visual diagnostics in Figure 3 (`obs_vs_pred_timeseries_09070500.png`) show that seasonality and moderate-flow dynamics are reproduced better than the largest events, where residuals are expected to increase. This interpretation is reinforced by Figure 4 (`obs_vs_pred_scatter_09070500.png`), where points cluster around the 1:1 line at low-to-moderate flows but spread more at higher discharges. This behavior is common in discharge modeling because peaks can be triggered by rapidly changing conditions, sub-daily processes, and regulation effects that are not fully captured by daily predictors alone. In this context, the model appears to prioritize stable representation of overall hydrograph shape while underestimating or smoothing some high-flow extremes.

From an operational perspective, these results support the use of compact LSTM models as screening or decision-support tools in data-limited settings. A model with this level of transfer skill can help identify expected seasonal behavior, compare likely flow states across basins, and provide a baseline forecast layer that can be combined with local knowledge or process-based tools.

Several limitations should be acknowledged. First, the training network is small and geographically narrow, which may constrain robustness under climate non-stationarity or unusual hydrologic years. Second, static descriptors were limited to elevation and drainage area; richer basin attributes were not included. Third, only one unseen site was used for transfer testing, so conclusions about regional generalization should be treated as promising but preliminary.

Future work should focus on scaling and benchmarking. A stronger follow-on study would include multi-basin cross-validation, additional physiographic descriptors (e.g., slope, land cover, soil and geologic indices, aridity), and explicit baseline comparisons against simpler models and persistence benchmarks. Additional improvements could include quantile or heteroscedastic loss functions for peak-flow realism, event-focused diagnostics, and hybrid modeling strategies that combine machine learning with hydrologic constraints.

## 7. Conclusion
This study developed and evaluated a regional LSTM framework for daily streamflow prediction in the Upper Colorado region using integrated streamflow, meteorological, and static catchment data. The model met all assignment requirements and generalized effectively to a fully unseen basin, achieving NSE = 0.646 and KGE = 0.791 at the withheld test site. These results support the use of compact multi-site LSTM models as a practical tool for regional hydrologic prediction when training data are selected to represent diverse runoff regimes.

## 8. Data and Code Availability
Core scripts and outputs are available in the assignment workspace. Key reproducibility files include:
- run_lstm_upper_colorado.py
- requirements.txt
- performance_summary_by_site.csv
- predictions_09070500.csv
- run_summary.txt
- study_sites_map_conus_inset.png
- training_history.png
- obs_vs_pred_timeseries_09070500.png
- obs_vs_pred_scatter_09070500.png

## 9. References (APA 7th ed.)
Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling. *Journal of Hydrology, 377*(1-2), 80-91. https://doi.org/10.1016/j.jhydrol.2009.08.003

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735-1780. https://doi.org/10.1162/neco.1997.9.8.1735

Kratzert, F., Klotz, D., Brenner, C., Schulz, K., & Herrnegger, M. (2018). Rainfall-runoff modelling using Long Short-Term Memory (LSTM) networks. *Hydrology and Earth System Sciences, 22*(11), 6005-6022. https://doi.org/10.5194/hess-22-6005-2018

Nash, J. E., & Sutcliffe, J. V. (1970). River flow forecasting through conceptual models part I: A discussion of principles. *Journal of Hydrology, 10*(3), 282-290. https://doi.org/10.1016/0022-1694(70)90255-6

