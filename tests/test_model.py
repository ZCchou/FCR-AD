"""Behavioral contracts for the standalone forecasting backbone."""

import unittest

import torch


def example_batch():
    generator = torch.Generator().manual_seed(123)
    times = torch.rand(2, 15, 12, generator=generator).sort(dim=-1).values * 5
    values = torch.randn(2, 15, 12, generator=generator)
    mask = (torch.rand(2, 15, 12, generator=generator) > .2).float()
    mask[..., 0] = 1
    query = 5 + torch.rand(2, 15, 4, generator=generator)
    query_mask = torch.ones_like(query)
    return dict(hist_times=times, hist_values=values, hist_mask=mask,
                query_times=query, query_mask=query_mask)


class BackboneTests(unittest.TestCase):
    def model(self):
        try:
            from fcr_ad.model import FCRADBackbone
        except ImportError as error:
            self.fail("The standalone FCRADBackbone is missing: " + str(error))
        torch.manual_seed(42)
        return FCRADBackbone().eval()

    def test_prediction_and_channel_context_contract(self):
        model = self.model()
        batch = example_batch()
        with torch.no_grad():
            prediction, context = model.forward_with_context(**batch)
            direct = model(**batch)
        self.assertEqual(tuple(prediction.shape), (2, 15, 4))
        self.assertEqual(tuple(context.shape), (2, 15, 128))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(torch.isfinite(context).all())
        torch.testing.assert_close(direct, prediction, rtol=0, atol=0)

    def test_padded_history_values_do_not_change_forecast(self):
        model = self.model()
        batch = example_batch()
        altered = dict(batch)
        altered['hist_values'] = batch['hist_values'].masked_fill(batch['hist_mask'] == 0, 1000)
        altered['hist_times'] = batch['hist_times'].masked_fill(batch['hist_mask'] == 0, 1000)
        with torch.no_grad():
            expected = model(**batch)
            actual = model(**altered)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_channel_context_uses_history_only(self):
        model = self.model()
        batch = example_batch()
        altered = dict(batch)
        altered['query_times'] = batch['query_times'] + 7
        altered['query_mask'] = torch.zeros_like(batch['query_mask'])
        with torch.no_grad():
            _, original = model.forward_with_context(**batch)
            _, changed = model.forward_with_context(**altered)
        torch.testing.assert_close(changed, original, rtol=0, atol=0)

    def test_frozen_checkpoint_parameter_layout_is_retained(self):
        model = self.model()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 113517)
        self.assertIn('state_gate.classifier.2.weight', model.state_dict())
        self.assertNotIn('aggregation.query_proj.weight', model.state_dict())
        self.assertNotIn('aggregation.channel_query', model.state_dict())


if __name__ == '__main__':
    unittest.main()
