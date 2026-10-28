"""Tests for native observation windows and training normalization."""

import tempfile
import unittest
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from fcr_ad.data import (
    Inventory, NativeWindows, SENSORS, build_datasets, capacity_audit,
    fit_stats, make_window_rows, _relative_path,
)


def record(name="flight.csv", split="train", normal=True, onset=None, end=28.0):
    return {"file": name, "sequence": name, "source": name, "split": split,
            "normal": normal, "fault_type": "none" if normal else "engine",
            "fault_start_s": onset, "start": 0.0, "end": end,
            "events": [] if normal else [{"id": name+":event0", "start": onset, "end": end}],
            "path": name, "label_path": "labels/"+name,
            "data_sha256": "", "labels_sha256": ""}


class DataTests(unittest.TestCase):
    def write_frame(self, root, name, times, values):
        frame = pd.DataFrame({"t": times, **{s: np.asarray(values, float) for s in SENSORS}})
        frame.to_csv(root/name, index=False)
        return frame

    def test_statistics_exclude_validation_test_and_post_fault(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            times = np.arange(20.0, 29.0)
            records = [record("normal.csv"), record("fault.csv", normal=False, onset=26.2),
                       record("val.csv", "val"), record("test.csv", "test")]
            for r in records:
                values = np.arange(1.0, 10.0) if r["split"] == "train" else np.full(9, 1e9)
                if not r["normal"]:
                    values[-2:] = 1e8
                frame = self.write_frame(root, r["path"], times, values)
                if r["split"] == "train":
                    frame[SENSORS[-1]] = 7.0
                    frame.to_csv(root/r["path"], index=False)
            stats = fit_stats(Inventory(records, root))
            self.assertEqual(stats["count"], [16]*15)
            np.testing.assert_allclose(stats["mean"][:-1], 73/16)
            self.assertEqual(stats["std"][-1], 1.0)
            self.assertEqual(stats["constant_channels"], [SENSORS[-1]])
            self.assertEqual([s["file"] for s in stats["training_sources"]], ["normal.csv", "fault.csv"])
            self.assertAlmostEqual(stats["training_sources"][1]["interval"][1], 26.0)

    def test_exact_native_bounds_masks_and_no_truncation(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            times = np.array([20., 21., 22., 23., 24., 25., 25.+5e-11, 25.2, 26., 26.+2e-10])
            frame = self.write_frame(root, "flight.csv", times, times*2)
            frame.loc[1, SENSORS[1]] = np.nan
            frame.to_csv(root/"flight.csv", index=False)
            records = Inventory([record(end=26.)], root)
            stats = {"mean": [2.]*15, "std": [2.]*15}
            caps = capacity_audit(records, {"train"})
            ds = NativeWindows(records, stats, "train", "score", caps["history"], caps["query"])
            item = ds[0]
            self.assertEqual(item["hist_mask"][0].sum().item(), 7)
            self.assertEqual(item["hist_mask"][1].sum().item(), 6)
            self.assertEqual(item["query_mask"][0].sum().item(), 2)
            np.testing.assert_array_equal(item["hist_times"][0, :7], (times[:7]-20).astype(np.float32))
            np.testing.assert_array_equal(item["target"][0, :2], (times[7:9]-1).astype(np.float32))
            self.assertEqual(item["row_index"].item(), 0)
            self.assertTrue(np.all(item["query_mask"].numpy()[:, 2:] == 0))
            too_small = NativeWindows(records, stats, "train", "score", 1, 1)
            with self.assertRaisesRegex(RuntimeError, "truncate"):
                too_small[0]

    def test_window_identity_guard_and_event_boundaries(self):
        r = record(normal=False, onset=26.0, end=28.0)
        self.assertEqual(make_window_rows(r, "train"), [])
        self.assertEqual(make_window_rows(r, "validation"), [])
        rows = make_window_rows(r, "score")
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["window_id"], "flight.csv:score:PreFaultAugmented:5.0:1.0:0")
        self.assertEqual(rows[0]["start"], 20.)
        self.assertEqual(rows[0]["label"], 1)
        self.assertEqual(rows[0]["evidence_start"], 25.)
        self.assertEqual(rows[0]["decision_time"], 26.)

    def test_test_requires_frozen_statistics_before_loading(self):
        with self.assertRaisesRegex(ValueError, "frozen"):
            build_datasets("missing-data-root", include_test=True)

    def test_manifest_paths_cannot_escape_or_use_an_absolute_location(self):
        for path in ("../data.csv", "/data.csv", "C:/data.csv", "C:\\data.csv"):
            with self.assertRaises(ValueError):
                _relative_path(path)
        self.assertEqual(_relative_path("train\\csv\\flight.csv"), "train/csv/flight.csv")

    def test_modified_csv_is_rejected_and_unhashed_input_changes_identity(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            frame = self.write_frame(root, "flight.csv", np.arange(20., 29.), np.arange(1., 10.))
            r = record()
            r["data_sha256"] = hashlib.sha256((root/"flight.csv").read_bytes()).hexdigest()
            Inventory([r], root).frame(r)
            stats = {"mean": [5.]*15, "std": [2.]*15}
            unhashed = dict(r, data_sha256="")
            before = NativeWindows(Inventory([unhashed], root), stats, "train", "score", 16, 16)
            # Swapping two observations preserves every normalization moment.
            frame.loc[[0, 1], SENSORS[0]] = frame.loc[[1, 0], SENSORS[0]].to_numpy()
            frame.to_csv(root/"flight.csv", index=False)
            with self.assertRaisesRegex(ValueError, "hash differs"):
                Inventory([r], root).frame(r)
            after = NativeWindows(Inventory([unhashed], root), stats, "train", "score", 16, 16)
            self.assertEqual(before.rows, after.rows)
            self.assertNotEqual(before.identity, after.identity)


if __name__ == "__main__":
    unittest.main()
