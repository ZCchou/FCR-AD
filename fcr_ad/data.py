"""Author-fixed ALFA windows with native asynchronous observations.

The input is the processed ALFA directory described in the repository README.
No telemetry is bundled with this package. Manifest paths are relative to the
data directory; input time stamps retain the original flight time reference.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


SENSORS = [
    "airspeed_commanded", "airspeed_measured", "roll_commanded", "roll_measured",
    "pitch_commanded", "pitch_measured", "tracking_error", "airspeed_error",
    "altitude_error", "velocity_commanded_x", "velocity_commanded_y",
    "velocity_commanded_z", "velocity_measured_x", "velocity_measured_y",
    "velocity_measured_z",
]
WINDOW = {"history": 5.0, "horizon": 1.0, "stride": 1.0, "patches": 4}
TRIM_START_S = 20.0
FAULT_GUARD_S = 0.2
VARIANT = "PreFaultAugmented"
SPLIT_FILE = Path(__file__).resolve().parents[1] / "splits" / "alfa_author_fixed.json"
if not SPLIT_FILE.is_file():
    SPLIT_FILE = Path(__file__).resolve().parent / "resources" / "author_split.json"
EXPECTED = {
    "healthy_train": (1475, 0), "healthy_val": (386, 0),
    "head_train": (1928, 447), "head_val": (489, 101), "test": (1123, 227),
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _relative_path(value):
    # Check Windows paths even when a clone is used on Linux or macOS.
    text = str(value).replace("\\", "/")
    path = Path(text)
    if path.is_absolute() or PureWindowsPath(text).root or PureWindowsPath(text).drive or ".." in path.parts:
        raise ValueError("Manifest paths must stay relative to the data root")
    return path.as_posix()


class Inventory(list):
    """Records with a runtime-only data location and lazy frame cache."""

    def __init__(self, records, data_root):
        super().__init__(records)
        self.data_root = Path(data_root).resolve()
        self._frames = {}
        self._file_hashes = {}

    def frame(self, record):
        key = record["path"]
        if key not in self._frames:
            target = (self.data_root / key).resolve()
            if not target.is_relative_to(self.data_root):
                raise ValueError("Feature path escapes the data root")
            digest = hashlib.sha256()
            with target.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(chunk)
            actual_hash = digest.hexdigest()
            expected_hash = record.get("data_sha256")
            if expected_hash and actual_hash != expected_hash:
                raise ValueError(f"Feature CSV hash differs from the manifest: {record['file']}")
            self._file_hashes[key] = actual_hash
            frame = pd.read_csv(target, usecols=["t"] + SENSORS)[["t"] + SENSORS]
            times = frame["t"].to_numpy(float)
            if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
                raise ValueError("Flight time stamps must be finite and nondecreasing")
            self._frames[key] = frame
        return self._frames[key]


def read_inventory(data_root):
    """Read and verify the published 25/8/13 file assignment.

    Required manifest columns follow the original processed ALFA exporter.
    Fault labels are derived from event times in this metadata; sensor values
    are read exclusively from the feature CSV files.
    """
    root = Path(data_root)
    manifest = pd.read_csv(root / "metadata" / "file_manifest.csv", keep_default_na=False)
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))["splits"]
    assignment = {name: part for part, names in split.items() for name in names}
    if len(assignment) != sum(map(len, split.values())):
        raise ValueError("Duplicate files in the published split")
    records = []
    for row in manifest.to_dict("records"):
        name = str(row["output_file"])
        if name not in assignment or str(row["split"]) != assignment[name]:
            raise ValueError("Manifest file assignment differs from the published split")
        normal_text = str(row["normal"]).lower()
        if normal_text not in {"true", "false"}:
            raise ValueError("Manifest normal values must be True or False")
        normal = normal_text == "true"
        onset = None if normal else float(row["fault_start_s"])
        start, end = float(row["time_start_s"]), float(row["time_end_s"])
        if not np.isfinite([start, end]).all() or end < start:
            raise ValueError("Invalid flight interval")
        if not normal and (not np.isfinite(onset) or onset > end):
            raise ValueError("Invalid fault onset")
        records.append({
            "file": name, "sequence": str(row["sequence"]),
            "source": str(row["source_group_id"]), "split": str(row["split"]),
            "normal": normal, "fault_type": str(row["fault_type"]),
            "fault_start_s": onset, "start": start, "end": end,
            "events": [] if normal else [{"id": name + ":event0", "start": onset, "end": end}],
            "path": _relative_path(row["feature_path"]),
            "label_path": _relative_path(row["label_path"]),
            "data_sha256": str(row.get("feature_sha256", "")),
            "labels_sha256": str(row.get("label_sha256", "")),
        })
    names = [record["file"] for record in records]
    if len(set(names)) != len(names) or set(names) != set(assignment):
        raise ValueError("Manifest must assign every published file exactly once")
    return Inventory(records, root)


def eligible_training_interval(record):
    start = max(float(record["start"]), TRIM_START_S)
    end = float(record["end"])
    if not record["normal"]:
        end = min(end, float(record["fault_start_s"]) - FAULT_GUARD_S)
    return (start, end) if end >= start else None


def fit_stats(records):
    """Fit normalization only on train healthy/pre-fault observations."""
    sums = np.zeros(len(SENSORS), dtype=np.float64)
    squares = np.zeros_like(sums)
    counts = np.zeros(len(SENSORS), dtype=np.int64)
    minima = np.full(len(SENSORS), np.inf)
    maxima = np.full(len(SENSORS), -np.inf)
    files = []
    for record in records:
        if record["split"] != "train":
            continue
        interval = eligible_training_interval(record)
        if interval is None:
            continue
        frame = records.frame(record)
        subset = frame[(frame["t"] >= interval[0] - 1e-10) &
                       (frame["t"] <= interval[1] + 1e-10)]
        files.append({"file": record["file"], "interval": list(interval)})
        for index, sensor in enumerate(SENSORS):
            values = pd.to_numeric(subset[sensor], errors="coerce").to_numpy(float)
            values = values[np.isfinite(values)]
            if values.size:
                sums[index] += values.sum()
                squares[index] += np.square(values).sum()
                counts[index] += values.size
                minima[index] = min(minima[index], values.min())
                maxima[index] = max(maxima[index], values.max())
    if np.any(counts == 0):
        raise ValueError("Healthy training observations are absent for one or more sensors")
    mean = sums / counts
    raw_std = np.sqrt(np.maximum(squares / counts - np.square(mean), 0.0))
    constant = raw_std < 1e-6
    std = np.where(constant, 1.0, np.maximum(raw_std, 1e-3))
    return {
        "mean": mean.tolist(), "std": std.tolist(), "raw_std": raw_std.tolist(),
        "count": counts.tolist(), "minimum": minima.tolist(), "maximum": maxima.tolist(),
        "constant_channels": [SENSORS[i] for i in np.flatnonzero(constant)],
        "training_sources": files, "constant_channel_scale_fallback": 1.0,
        "variant": VARIANT, "trim_start_s": TRIM_START_S, "fault_guard_s": FAULT_GUARD_S,
    }


def make_window_rows(record, usage):
    """Preserve the original source window IDs, intervals and event labels."""
    if usage not in {"train", "validation", "score"}:
        raise ValueError("Unknown window usage")
    lower, upper = max(float(record["start"]), TRIM_START_S), float(record["end"])
    if usage in {"train", "validation"}:
        interval = eligible_training_interval(record)
        if interval is None:
            return []
        lower, upper = interval
    first = lower + WINDOW["history"] + WINDOW["horizon"]
    if first > upper + 1e-10:
        return []
    count = int(np.floor((upper - first) / WINDOW["stride"] + 1e-8)) + 1
    rows = []
    for index in range(count):
        start = lower + index * WINDOW["stride"]
        hist_end = start + WINDOW["history"]
        decision = hist_end + WINDOW["horizon"]
        events = [event["id"] for event in record["events"]
                  if event["end"] > hist_end and event["start"] <= decision]
        rows.append({
            "window_id": f"{record['file']}:{usage}:{VARIANT}:{WINDOW['history']}:{WINDOW['horizon']}:{index}",
            "file": record["file"], "source": record["source"], "index": index,
            "start": float(start), "hist_end": float(hist_end),
            "evidence_start": float(hist_end), "evidence_end": float(decision),
            "decision_time": float(decision), "label": int(bool(events)), "event_ids": events,
            "normal_file": bool(record["normal"]), "fault_start_s": record["fault_start_s"],
            "fault_type": record["fault_type"],
        })
    return rows


def capacity_audit(records, parts):
    """Measure full observations before allocating padding; never truncate."""
    max_hist = max_query = 0
    for record in records:
        if record["split"] not in parts:
            continue
        frame = records.frame(record)
        channels = [frame.loc[np.isfinite(frame[s]), "t"].to_numpy(float) for s in SENSORS]
        for row in make_window_rows(record, "score"):
            for times in channels:
                h0 = np.searchsorted(times, row["start"], "left")
                h1 = np.searchsorted(times, row["hist_end"] + 1e-10, "right")
                q1 = np.searchsorted(times, row["evidence_end"] + 1e-10, "right")
                max_hist, max_query = max(max_hist, int(h1-h0)), max(max_query, int(q1-h1))
    return {"history": max(16, int(np.ceil(max_hist / 16)) * 16),
            "query": max(16, int(np.ceil(max_query / 16)) * 16),
            "max_history_observations": max_hist, "max_query_observations": max_query,
            "zero_silent_truncation": True}


class NativeWindows(Dataset):
    """Right-padded, unfilled, uninterpolated per-channel flight windows."""

    def __init__(self, records, stats, part, usage, hist_cap, query_cap):
        self.sensors = list(SENSORS)
        self.cfg = dict(WINDOW)
        self.stats = {key: np.asarray(stats[key], dtype=float) for key in ("mean", "std")}
        if any(v.shape != (len(SENSORS),) or not np.isfinite(v).all() for v in self.stats.values()):
            raise ValueError("Normalization must contain 15 finite means and scales")
        if np.any(self.stats["std"] <= 0):
            raise ValueError("Normalization scales must be positive")
        self.hist_cap, self.query_cap = int(hist_cap), int(query_cap)
        self.runs, self.rows, self.excluded_no_evidence = [], [], []
        for record in records:
            if record["split"] != part:
                continue
            frame = records.frame(record)
            channels = []
            for sensor in SENSORS:
                good = np.isfinite(frame[sensor])
                channels.append((frame.loc[good, "t"].to_numpy(float),
                                 frame.loc[good, sensor].to_numpy(float)))
            run = len(self.runs)
            self.runs.append((record, channels))
            for row in make_window_rows(record, usage):
                observed = sum(np.searchsorted(t, row["evidence_end"] + 1e-10, "right") -
                               np.searchsorted(t, row["hist_end"] + 1e-10, "right") for t, _ in channels)
                item = {**row, "run": run, "valid": bool(observed > 0)}
                (self.rows if observed else self.excluded_no_evidence).append(item)
        hashes = sorted((record["file"], records._file_hashes[record["path"]])
                        for record in records if record["split"] == part)
        self.identity = _digest({"rows": self.rows, "stats": stats, "part": part,
                                 "usage": usage, "actual_feature_hashes": hashes})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        channels = self.runs[row["run"]][1]
        n = len(channels)
        result = {key: np.zeros((n, self.hist_cap if key.startswith("hist_") else self.query_cap), np.float32)
                  for key in ("hist_times", "hist_values", "hist_mask", "query_times", "query_mask", "target")}
        for channel, (times, values) in enumerate(channels):
            h0 = np.searchsorted(times, row["start"], "left")
            h1 = np.searchsorted(times, row["hist_end"] + 1e-10, "right")
            q1 = np.searchsorted(times, row["evidence_end"] + 1e-10, "right")
            ht, hv, qt, qv = times[h0:h1], values[h0:h1], times[h1:q1], values[h1:q1]
            if len(ht) > self.hist_cap or len(qt) > self.query_cap:
                raise RuntimeError("Capacity audit failure: refusing to truncate observations")
            result["hist_times"][channel, :len(ht)] = ht - row["start"]
            result["hist_values"][channel, :len(ht)] = (hv-self.stats["mean"][channel])/self.stats["std"][channel]
            result["hist_mask"][channel, :len(ht)] = 1
            result["query_times"][channel, :len(qt)] = qt - row["start"]
            result["target"][channel, :len(qt)] = (qv-self.stats["mean"][channel])/self.stats["std"][channel]
            result["query_mask"][channel, :len(qt)] = 1
        result["row_index"] = np.asarray(index, dtype=np.int64)
        return {key: torch.from_numpy(value) for key, value in result.items()}


def build_datasets(data_root, stats=None, include_test=False):
    """Build the fixed development roles, optionally adding frozen-stat test.

    Call once without ``include_test`` to fit train-only normalization. Persist
    that returned dictionary, then explicitly supply it to any test build.
    Development and test padding are independently measured, as in the source.
    """
    if include_test and stats is None:
        raise ValueError("Test construction requires frozen training statistics")
    records = read_inventory(data_root)
    stats = fit_stats(records) if stats is None else json.loads(json.dumps(stats))
    capacities = {"development": capacity_audit(records, {"train", "val"})}
    mapping = {"healthy_train": ("train", "train"), "healthy_val": ("val", "validation"),
               "head_train": ("train", "score"), "head_val": ("val", "score")}
    if include_test:
        capacities["test"] = capacity_audit(records, {"test"})
        mapping["test"] = ("test", "score")
    sets, audits = {}, {}
    for role, (part, usage) in mapping.items():
        cap = capacities["test" if role == "test" else "development"]
        ds = NativeWindows(records, stats, part, usage, cap["history"], cap["query"])
        windows, positives = len(ds), sum(row["label"] for row in ds.rows)
        if (windows, positives) != EXPECTED[role]:
            raise ValueError(f"Unexpected {role} windows/positives: {(windows, positives)}")
        sets[role] = ds
        audits[role] = {"windows": windows, "positives": positives, "identity": ds.identity,
                        "excluded_no_evidence": ds.excluded_no_evidence}
    audit = {"sensors": list(SENSORS), "roles": audits, "capacities": capacities,
             "normalization_identity": _digest(stats),
             "files": {part: sum(r["split"] == part for r in records) for part in ("train", "val", "test")},
             "window": dict(WINDOW), "trim_start_s": TRIM_START_S,
             "fault_guard_s": FAULT_GUARD_S, "test_included": bool(include_test)}
    return sets, stats, audit
