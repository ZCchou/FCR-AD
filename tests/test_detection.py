"""Tests for residual/context features, calibration and event metrics."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


def detection_modules(test):
    try:
        from fcr_ad import evaluation, features, head
    except ImportError as error:
        test.fail("The standalone detection modules are missing: " + str(error))
    return features, head, evaluation


def training_rows():
    return ([dict(label=0, file='normal-a', event_ids=[])] * 4
            + [dict(label=0, file='normal-b', event_ids=[])]
            + [dict(label=1, file='fault-a', event_ids=['a'])] * 3
            + [dict(label=1, file='fault-b', event_ids=['b'])])


class FeatureTests(unittest.TestCase):
    def test_residuals_balance_channels_and_preserve_missing_channels(self):
        features, _, _ = detection_modules(self)
        prediction = np.array([[[1., 1., np.nan], [3., 99., 99.],
                                [np.nan, np.nan, np.nan]]])
        mask = np.array([[[1, 1, 0], [1, 0, 0], [0, 0, 0]]])
        actual = features.residual_features(prediction, np.zeros_like(prediction), mask)
        np.testing.assert_allclose(actual, [[5., 2., 9., 5., 1., 9., np.nan,
                                            1., 3., np.nan]], equal_nan=True)

    def test_residuals_reject_empty_windows_and_nonfinite_observations(self):
        features, _, _ = detection_modules(self)
        prediction = np.ones((1, 2, 3))
        with self.assertRaises(ValueError):
            features.residual_features(prediction, prediction, np.zeros_like(prediction))
        prediction[0, 0, 0] = np.inf
        with self.assertRaises(ValueError):
            features.residual_features(prediction, np.zeros_like(prediction), np.ones_like(prediction))

    def test_loss_balances_channels_then_windows_and_masks_gradients(self):
        features, _, _ = detection_modules(self)
        prediction = torch.tensor([[[1., 1., 999.], [3., 999., 999.]],
                                   [[2., 999., 999.], [999., 999., 999.]]],
                                  requires_grad=True)
        mask = torch.tensor([[[1, 1, 0], [1, 0, 0]], [[1, 0, 0], [0, 0, 0]]])
        loss = features.channel_balanced_mse(prediction, torch.zeros_like(prediction), mask)
        self.assertEqual(loss.item(), 4.5)
        loss.backward()
        self.assertTrue(torch.equal(prediction.grad[mask == 0], torch.zeros(8)))
        self.assertTrue(torch.isfinite(prediction.grad).all())
        with self.assertRaises(ValueError):
            features.channel_balanced_mse(prediction, prediction, torch.zeros_like(mask))

    def test_context_uses_query_validity_without_observation_count_weighting(self):
        features, _, _ = detection_modules(self)
        context = torch.tensor([[[2.] * 128, [8.] * 128, [float('nan')] * 128]])
        mask = torch.tensor([[[1, 1, 1], [1, 0, 0], [0, 0, 0]]])
        pooled = features.context_summary(context, mask)
        torch.testing.assert_close(pooled, torch.tensor([[5.] * 128 + [8.] * 128]))
        expanded = context.unsqueeze(2).expand(-1, -1, 3, -1)
        torch.testing.assert_close(features.context_summary(expanded, mask), pooled)
        with self.assertRaises(ValueError):
            features.context_summary(context, torch.zeros_like(mask))

    def test_extract_features_keeps_dataset_order_and_model_training_mode(self):
        features, _, _ = detection_modules(self)
        from fcr_ad.model import FCRADBackbone

        torch.manual_seed(5)
        model = FCRADBackbone().train()
        dataset = []
        for index in range(3):
            dataset.append(dict(
                hist_times=torch.linspace(0, 5, 5).expand(15, -1).clone(),
                hist_values=torch.full((15, 5), float(index)),
                hist_mask=torch.ones(15, 5),
                query_times=torch.full((15, 2), 5.5),
                query_mask=torch.ones(15, 2),
                target=torch.full((15, 2), float(index)),
                row_index=torch.tensor(index),
            ))
        actual = features.extract_features(model, dataset, batch_size=2)
        self.assertEqual(actual.shape, (3, 290))
        self.assertTrue(model.training)
        model.eval()
        expected = []
        with torch.no_grad():
            for row in dataset:
                inputs = {key: row[key].unsqueeze(0) for key in
                          ('hist_times', 'hist_values', 'hist_mask', 'query_times', 'query_mask')}
                prediction, context = model.forward_with_context(**inputs)
                error = prediction - row['target'].unsqueeze(0)
                channel_mse = error.square().mean(-1)
                channel_mae = error.abs().mean(-1)
                residual = torch.cat([
                    channel_mse.mean(-1, keepdim=True),
                    channel_mae.mean(-1, keepdim=True),
                    channel_mse.amax(-1, keepdim=True),
                    channel_mse.topk(3, dim=-1).values.mean(-1, keepdim=True),
                    channel_mse, channel_mae,
                ], -1)
                expected.append(torch.cat([residual, context.mean(1), context.amax(1)], -1).numpy())
        np.testing.assert_allclose(actual, np.concatenate(expected), rtol=2e-5, atol=2e-6)
        self.assertEqual(features.extract_features(model, []).shape, (0, 290))

    def test_channel_context_retains_original_query_reduction_rounding(self):
        features, _, _ = detection_modules(self)
        context = torch.full((1, 1, 128), .1, dtype=torch.float32)
        mask = torch.ones((1, 1, 9), dtype=torch.float32)
        expected = torch.full((1, 256), .10000000894069672, dtype=torch.float32)
        torch.testing.assert_close(features.context_summary(context, mask), expected, rtol=0, atol=0)


class HeadTests(unittest.TestCase):
    def test_event_weights_balance_normal_flights_and_scale_positive_mass(self):
        _, head, _ = detection_modules(self)
        rows = training_rows()
        weights = head.weighted_rows(rows, .5)
        np.testing.assert_allclose(weights, [.75, .75, .75, .75, 3., .5, .5, .5, 1.5])
        np.testing.assert_allclose(head.sample_weights(rows, 'class').sum(), 9.)
        with self.assertRaises(ValueError):
            head.weighted_rows(rows, 0.)
        with self.assertRaises(ValueError):
            head.sample_weights([dict(label=0, file='x', event_ids=[])], 'event')
        bad = [*rows[:-1], dict(label=1, file='fault-b', event_ids=['b', 'c'])]
        with self.assertRaises(ValueError):
            head.sample_weights(bad, 'event')

    def test_head_imputes_from_training_only_and_roundtrips_package_class(self):
        _, head, _ = detection_modules(self)
        rows = training_rows()
        x = np.arange(27, dtype=float).reshape(9, 3)
        x[:, 2] = np.nan
        x[0, 0] = np.inf
        trained = head.Head().fit(x, rows, {'n_estimators': 3, 'min_samples_leaf': 1})
        np.testing.assert_allclose(trained.medians, [13.5, 13., 0.])
        self.assertEqual(trained.empty_columns, [2])
        np.testing.assert_allclose(list(trained.class_weight_totals.values()), [6., 3.])
        expected = trained.predict(np.array([[np.nan, np.inf, -np.inf], [1e6, -1e6, 0.]]))
        np.testing.assert_allclose(trained.medians, [13.5, 13., 0.])
        self.assertTrue(np.isfinite(expected).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'head.joblib'
            trained.save(path)
            restored = head.Head.load(path)
            self.assertEqual(type(restored).__module__, 'fcr_ad.head')
            np.testing.assert_array_equal(restored.predict(
                np.array([[np.nan, np.inf, -np.inf], [1e6, -1e6, 0.]])), expected)
        self.assertEqual(trained.predict(np.empty((0, 3))).shape, (0,))
        with self.assertRaises(ValueError):
            trained.predict(np.ones((1, 4)))

    def test_seeded_head_predictions_are_reproducible(self):
        _, head, _ = detection_modules(self)
        rows = training_rows()
        x = np.random.default_rng(7).normal(size=(9, 4))
        cfg = dict(n_estimators=4, min_samples_leaf=1)
        first = head.Head().fit(x, rows, cfg, seed=42)
        second = head.Head().fit(x, rows, cfg, seed=42)
        np.testing.assert_array_equal(first.predict(x), second.predict(x))


class EvaluationTests(unittest.TestCase):
    def test_threshold_is_strict_and_tied_scores_are_indivisible(self):
        _, _, evaluation = detection_modules(self)
        calibration = evaluation.choose_threshold([0, 0, 1, 1], [.1, .2, .2, .9], max_fpr=0)
        self.assertEqual(calibration['threshold'], .2)
        self.assertEqual(calibration['validation']['tp'], 1)
        self.assertEqual(calibration['validation']['fp'], 0)
        full = evaluation.choose_threshold([0, 1], [.5, .5], max_fpr=1)
        self.assertEqual(full['threshold'], np.nextafter(.5, -np.inf))
        no_hits = evaluation.choose_threshold([0, 1], [.9, .1], max_fpr=0)
        self.assertEqual(no_hits['threshold'], .9)

    def test_threshold_ties_prefer_lower_fpr_then_higher_threshold(self):
        _, _, evaluation = detection_modules(self)
        calibration = evaluation.choose_threshold([0, 1, 0, 0, 1], [0, 1, 2, 3, 4], max_fpr=1)
        self.assertEqual(calibration['threshold'], 3.)
        self.assertEqual(calibration['validation']['f1'], 2 / 3)
        with self.assertRaises(ValueError):
            evaluation.choose_threshold([0, 0], [0., 1.])
        with self.assertRaises(ValueError):
            evaluation.choose_threshold([0, 1], [0., np.nan])

    def test_event_adjustment_preserves_separate_adjacent_events_and_raw_delays(self):
        _, _, evaluation = detection_modules(self)
        rows = [dict(label=0, file='normal', event_ids=[], decision_time=0., fault_start_s=None),
                dict(label=1, file='fault', event_ids=['a'], decision_time=11., fault_start_s=10.),
                dict(label=1, file='fault', event_ids=['a'], decision_time=15., fault_start_s=10.),
                dict(label=1, file='fault', event_ids=['b'], decision_time=21., fault_start_s=20.),
                dict(label=1, file='fault', event_ids=['b'], decision_time=32., fault_start_s=20.)]
        report = evaluation.evaluate_scores(rows, [.2, .1, .9, .1, .1], .5)
        self.assertEqual(report['raw']['tp'], 1)
        self.assertEqual(report['point_adjusted']['tp'], 2)
        self.assertEqual(report['raw']['event_count'], 2)
        self.assertEqual(report['raw']['event_recall'], .5)
        self.assertEqual(report['raw']['detection_delay_median_s'], 5.)
        self.assertEqual(report['point_adjusted']['detection_delay_median_s'], 5.)
        self.assertEqual(report['point_adjusted']['first_detection_delay_mean_s'], 5.)
        self.assertEqual(report['raw']['recall_after_10s'], 0.)
        self.assertEqual(report['raw']['fpr'], 0.)
        self.assertEqual(report['raw']['roc_auc'], .25)
        self.assertAlmostEqual(report['raw']['pr_auc'], .85)

    def test_point_adjustment_does_not_cascade_across_overlapping_events(self):
        _, _, evaluation = detection_modules(self)
        rows = [dict(label=1, file='f', event_ids=['a'], decision_time=1.),
                dict(label=1, file='f', event_ids=['a', 'b'], decision_time=2.),
                dict(label=1, file='f', event_ids=['b'], decision_time=3.),
                dict(label=0, file='f', event_ids=[], decision_time=4.)]
        report = evaluation.evaluate_scores(rows, [.9, .1, .1, .1], .5)
        self.assertEqual(report['point_adjusted']['tp'], 2)
        self.assertEqual(report['raw']['event_hits'], 1)

    def test_single_class_and_empty_prediction_denominators_are_defined(self):
        _, _, evaluation = detection_modules(self)
        metrics = evaluation.binary_metrics([0, 0], [0, 0])
        self.assertEqual(metrics['f1'], 0.)
        self.assertEqual(metrics['fpr'], 0.)
        rows = [dict(label=0, file='f', event_ids=[], decision_time=0.)]
        report = evaluation.evaluate_scores(rows, [.5], .5)
        self.assertIsNone(report['raw']['roc_auc'])
        self.assertIsNone(report['raw']['pr_auc'])
        self.assertIsNone(report['raw']['detection_delay_median_s'])
        self.assertIsNone(report['raw']['event_recall'])
        self.assertIsNone(report['raw']['recall_after_10s'])
        with self.assertRaises(ValueError):
            evaluation.evaluate_scores(rows, [np.nan], .5)
        with self.assertRaises(ValueError):
            evaluation.binary_metrics([0, 2], [0, 1])


if __name__ == '__main__':
    unittest.main()
