# Release verification

The independent package was checked against the formal mean-pooling FCR-AD implementation before publication.

- 33 automated tests pass, covering masks, window boundaries, training-only normalization, event weights, feature imputation, calibration ties, event adjustment, serialization, training recovery and test-phase ordering.
- The original checkpoint loads strictly. All 47 state tensors, seeded initialization and the post-initialization random state match the reference implementation.
- Forecasts and channel contexts match on generated asynchronous inputs, including missing histories.
- Three training updates with micro-batch accumulation match reference losses, parameters and RNG state exactly.
- All five data roles reproduce the reference window metadata and normalization. The 46 feature CSV checksums are verified.
- Preparation of one normal and one fault flight reproduces the reference feature CSV bytes exactly.
- On all 1123 test windows, features and classifier probabilities match a fresh reference CPU computation exactly. Raw and PA decisions and their reported reference metrics agree. These checks use the same CPU arithmetic; historical GPU caches can have ordinary floating-point differences.
- A one-update real-data integration run exercises training, fitting, calibration, freezing, loading and evaluation. This is a software check and does not replace the formal training experiment.

The published tests use generated data. Original telemetry, fitted classifier files, checkpoints and local experiment caches are excluded from the release.
