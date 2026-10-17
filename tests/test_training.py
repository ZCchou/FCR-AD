import tempfile
import unittest
from unittest.mock import patch
import io
import contextlib
from pathlib import Path

import torch

from fcr_ad.cli import _SyntheticWindows, default_config, smoke, main
from fcr_ad.training import deterministic_batch, train_backbone


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_deterministic_batches_cover_every_window(self):
        dataset = _SyntheticWindows(10)
        first = deterministic_batch(dataset, 42, 0, 4)['row_index'].tolist()
        duplicate = deterministic_batch(dataset, 42, 0, 4)['row_index'].tolist()
        self.assertEqual(first, duplicate)
        indices = [i for step in range(3)
                   for i in deterministic_batch(dataset, 42, step, 4)['row_index'].tolist()]
        self.assertEqual(sorted(indices), list(range(10)))

    def test_saved_training_reload_and_stale_config_rejection(self):
        dataset = _SyntheticWindows(4)
        dataset.identity = 'synthetic-training-test'
        config = default_config()['training']
        config.update(max_steps=1, validation_interval=1, batch_size=4, micro_batch_size=2)
        with tempfile.TemporaryDirectory() as directory:
            _, first = train_backbone(dataset,dataset,Path(directory),config,seed=42)
            _, second = train_backbone(dataset,dataset,Path(directory),config,seed=42)
            self.assertEqual(first,second)
            self.assertEqual(first['steps_completed'],1)
            self.assertTrue((Path(directory)/'backbone.pt').exists())
            with self.assertRaises(ValueError):
                train_backbone(dataset,dataset,Path(directory),{**config,'lr':.004},seed=42)

    def test_synthetic_end_to_end_smoke(self):
        result = smoke()
        self.assertTrue(result['synthetic'])
        self.assertEqual(result['features'],[24,290])
        self.assertTrue(result['finite_scores'])

    def test_last_validation_crash_recovers_history(self):
        dataset = _SyntheticWindows(4)
        dataset.identity = 'synthetic-resume-test'
        config = default_config()['training']
        config.update(max_steps=1, validation_interval=1, batch_size=4, micro_batch_size=2)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            train_backbone(dataset,dataset,output,config,seed=42)
            (output/'training.json').unlink()
            (output/'history.json').unlink()
            _, info = train_backbone(dataset,dataset,output,config,seed=42)
            self.assertEqual(info['steps_completed'],1)
            self.assertTrue((output/'history.json').exists())

    def test_all_seeds_freeze_before_any_test(self):
        calls = []
        with patch('fcr_ad.cli.fit_detector', side_effect=lambda *a,**kw:calls.append(('fit',kw['seed']))), \
             patch('fcr_ad.cli.evaluate_detector', side_effect=lambda *a,**kw:calls.append(('test',a[1].name)) or {}), \
             contextlib.redirect_stdout(io.StringIO()):
            main(['run','--data-root','unused','--output','unused','--seeds','42','43'])
        self.assertEqual(calls,[('fit',42),('fit',43),('test','seed42'),('test','seed43')])


if __name__ == '__main__':
    unittest.main()
