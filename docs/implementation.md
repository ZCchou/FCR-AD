# Implementation notes

## Forecasting inputs and context

Each batch uses `hist_times`, `hist_values`, `hist_mask`, `query_times`, `query_mask` and a separate `target`. Histories have shape `[B,15,L]`, evidence has shape `[B,15,Q]`, and padded observations have zero masks. Times are relative to the start of the five-second history. Values use population mean and standard deviation from healthy training observations. Constant training channels use scale 1.0.

Time-aware patch aggregation produces `[B,15,4,128]`. A shared depthwise causal temporal encoder processes the patch order, using bottleneck width 64, kernel size 3 and dilations 1 and 2. Patch Pooling adds sinusoidal patch positions, computes the uniform patch mean and applies LayerNorm. The resulting `[B,15,128]` channel context supports both the forecasting decoder and feature construction.

The decoder receives channel context and continuous query-time encoding. Real evidence values enter the residual calculation and forecasting loss, not the forecasting inputs. The loss first averages squared error over observed queries per channel, then over valid channels per window, then over windows.

## Classifier feature order

The first 34 columns are:

1. Mean of valid-channel MSEs.
2. Mean of valid-channel MAEs.
3. Maximum valid-channel MSE.
4. Mean of the largest `min(3, number of valid channels)` channel MSEs.
5. Fifteen channel MSEs, in `fcr_ad.data.SENSORS` order.
6. Fifteen channel MAEs, in the same order.

The remaining 256 columns concatenate the mean and elementwise maximum of 128-dimensional channel contexts over channels with real evidence observations. The implementation preserves the original query-mask reduction order. An absent channel's individual residual statistics remain nonfinite until the detector replaces them using classifier-training medians. Entirely missing training columns use zero.

The classifier uses event/normal-flight group weights and positive mass multiplier 0.5. Calibration uses Raw predictions only. PA is applied after thresholds are fixed and uses the original Raw predictions to determine whether an annotated event was hit.

## Checkpoint compatibility

The model retains the registered historical state-gate parameters and original construction order so the formal mean-pooling checkpoint loads strictly and seeded initialization consumes the same random numbers. The state gate does not participate in the published forward computation. The model stores 113517 parameters, of which 104937 participate in the forward path.

The runner separates training and evaluation. For a multi-seed run, all seeds finish development and freeze before the first seed's test is opened. Frozen artifact and source hashes are checked before evaluation. Missing data are represented by masks; observation counts exceeding the allocated capacity raise an error.

Only load checkpoint and joblib files obtained from a trusted source. These formats can contain executable Python objects.
