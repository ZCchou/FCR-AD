"""Native topic preparation tests, using synthetic topic CSVs only."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from fcr_ad import prepare


class PrepareTests(unittest.TestCase):
    def topics(self, root):
        sequence = root/"source"/"flight"
        sequence.mkdir(parents=True)
        base = 1_531_943_810_000_000_000
        for index, topic in enumerate(prepare.TOPICS):
            offset = 0 if index == 0 else 1_000_000_000
            times = [base+2_000_000_000+offset, base+4_000_000_000]
            columns = {field: [float(index+1), float(index+2)]
                       for _, (source_topic, field) in prepare.FEATURE_SOURCES.items() if source_topic == topic}
            pd.DataFrame({"%time": times, **columns}).to_csv(sequence/f"flight-{topic}.csv", index=False)
        pd.DataFrame({"%time": [base, base+3_000_000_000], "field.data": [False, True]}).to_csv(
            sequence/"flight-failure_status-engines.csv", index=False)
        record = {"output_file": "flight.csv", "split": "train", "sequence": "flight",
                  "normal": "False", "fault_start_s": "3.0", "rows": "3",
                  "time_start_s": "2.0", "time_end_s": "4.0",
                  "feature_path": "train/csv/flight.csv", "feature_sha256": ""}
        return sequence, record

    def test_outer_join_native_time_origin_and_manifest_onset(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            sequence, record = self.topics(root)
            frame, info = prepare.build_sequence(sequence, record)
            self.assertEqual(frame.columns.tolist(), ["t"]+prepare.FEATURES)
            np.testing.assert_array_equal(frame.t.to_numpy(), [2., 3., 4.])
            self.assertTrue(np.isnan(frame.loc[1, "airspeed_commanded"]))
            self.assertTrue(np.isnan(frame.loc[0, "yaw_commanded"]))
            self.assertEqual(info["fault_start_s"], 3.)
            self.assertNotIn("failure_status", frame.columns)
            changed = dict(record, fault_start_s="3.2")
            with self.assertRaisesRegex(ValueError, "onset"):
                prepare.build_sequence(sequence, changed)
            with self.assertRaisesRegex(ValueError, "hash"):
                prepare.build_sequence(sequence, dict(record, feature_sha256="0"*64))

    def test_idempotent_export_refuses_different_file_and_reports_no_label_csv(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            sequence, record = self.topics(root)
            _, info = prepare.build_sequence(sequence, record)
            record["feature_sha256"] = info["feature_sha256"]
            manifest = root/"file_manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(record))
                writer.writeheader()
                writer.writerow(record)
            output = root/"output"
            with patch.object(prepare, "MANIFEST_FILE", manifest):
                result = prepare.prepare_alfa(root/"source", output)
                self.assertEqual(prepare.prepare_alfa(root/"source", output), result)
                self.assertEqual(result["files"], 1)
                self.assertFalse(result["label_csv_generated"])
                self.assertFalse((output/"train/failure_labels").exists())
                self.assertEqual((output/"metadata/file_manifest.csv").read_bytes(), manifest.read_bytes())
                target = output/record["feature_path"]
                target.write_bytes(b"existing-different-content")
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    prepare.prepare_alfa(root/"source", output)
                self.assertEqual(target.read_bytes(), b"existing-different-content")

    def test_conflicting_duplicate_times_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            sequence, _ = self.topics(Path(directory))
            path = sequence/"flight-mavros-nav_info-airspeed.csv"
            frame = pd.read_csv(path)
            duplicate = frame.iloc[[0]].copy()
            duplicate["field.commanded"] = 999.
            pd.concat([frame, duplicate]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                prepare.read_topic(sequence, "mavros-nav_info-airspeed")

    def test_source_and_output_must_differ(self):
        with self.assertRaisesRegex(ValueError, "differ"):
            prepare.prepare_alfa(Path.cwd(), Path.cwd())


if __name__ == "__main__":
    unittest.main()
