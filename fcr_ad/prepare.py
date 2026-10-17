"""Rebuild the fixed 17D ALFA feature CSVs from official processed topics.

Only the published file manifest and feature mapping are bundled. Sensor
observations retain their original asynchronous times and missing values.
Failure-status files establish the flight time origin and validate manifest
event onsets; their values never become model input features.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd


def _resource(name):
    source = Path(__file__).resolve().parents[1] / "metadata" / name
    return source if source.is_file() else Path(__file__).resolve().parent / "resources" / name


MANIFEST_FILE = _resource("file_manifest.csv")
MAPPING_FILE = _resource("feature_mapping.json")
_MAPPING = json.loads(MAPPING_FILE.read_text(encoding="utf-8"))
FEATURES = _MAPPING["features"]
FEATURE_SOURCES = {feature: (_MAPPING["mapping"][feature]["topic"],
                             _MAPPING["mapping"][feature]["field"]) for feature in FEATURES}
TOPICS = list(dict.fromkeys(topic for topic, _ in FEATURE_SOURCES.values()))


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _timestamps(frame, path):
    times = pd.to_numeric(frame["%time"], errors="raise")
    if frame.empty or times.isna().any() or not pd.api.types.is_integer_dtype(times.dtype):
        raise ValueError(f"Expected nonempty integer nanosecond times in {path.name}")
    return times.astype(np.int64)


def read_topic(sequence_dir, topic):
    """Read one topic, collapsing consistent duplicates without interpolation."""
    matches = sorted(Path(sequence_dir).glob(f"*-{topic}.csv"))
    if len(matches) != 1:
        raise ValueError(f"Expected one {topic} topic in {Path(sequence_dir).name}")
    path = matches[0]
    mapping = {out: field for out, (source_topic, field) in FEATURE_SOURCES.items() if source_topic == topic}
    frame = pd.read_csv(path, usecols=["%time", *mapping.values()])
    frame["timestamp_ns"] = _timestamps(frame, path)
    frame = frame.drop(columns="%time").rename(columns={field: out for out, field in mapping.items()})
    for feature in mapping:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    if frame.duplicated("timestamp_ns").any():
        collapsed = []
        for time, group in frame.groupby("timestamp_ns", sort=False):
            row = {"timestamp_ns": int(time)}
            for feature in mapping:
                values = group[feature].dropna().to_numpy()
                if values.size and not np.all(values == values[0]):
                    raise ValueError(f"Conflicting duplicate timestamp in {path.name}")
                row[feature] = float(values[0]) if values.size else np.nan
            collapsed.append(row)
        frame = pd.DataFrame(collapsed, columns=["timestamp_ns", *mapping])
    return frame.sort_values("timestamp_ns", kind="mergesort").reset_index(drop=True)


def _fault_times(sequence_dir):
    first_times, positive_times = [], []
    for path in sorted(Path(sequence_dir).glob("*-failure_status-*.csv")):
        frame = pd.read_csv(path, usecols=["%time", "field.data"])
        times = _timestamps(frame, path).to_numpy(np.int64)
        if np.any(np.diff(times) < 0):
            raise ValueError(f"Failure-status times must be ordered in {path.name}")
        first_times.append(int(times[0]))
        for time, value in zip(times, frame["field.data"]):
            if pd.isna(value):
                continue
            text = str(value).strip().lower()
            value = 1.0 if text in {"true", "t", "yes"} else 0.0 if text in {"false", "f", "no"} else float(text)
            if not np.isfinite(value):
                raise ValueError(f"Nonfinite failure status in {path.name}")
            if value != 0:
                positive_times.append(int(time))
    return first_times, positive_times


def build_sequence(sequence_dir, record):
    """Return a validated frame and provenance for one published flight."""
    frames = [read_topic(sequence_dir, topic) for topic in TOPICS]
    first_fault_times, positives = _fault_times(sequence_dir)
    t0_ns = min([int(frame["timestamp_ns"].iloc[0]) for frame in frames] + first_fault_times)
    normal = str(record["normal"]).lower() == "true"
    if normal and positives or not normal and not positives:
        raise ValueError(f"Failure-status presence differs from manifest: {record['output_file']}")
    onset = None if normal else (min(positives)-t0_ns)/1e9
    expected_onset = None if normal else float(record["fault_start_s"])
    if onset is not None and not np.isclose(onset, expected_onset, rtol=0, atol=5e-10):
        raise ValueError(f"Fault onset differs from manifest: {record['output_file']}")
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="timestamp_ns", how="outer", sort=False, validate="one_to_one")
    merged = merged.sort_values("timestamp_ns", kind="mergesort").reset_index(drop=True)
    if merged["timestamp_ns"].duplicated().any() or merged[FEATURES].isna().all(axis=1).any():
        raise ValueError("Merged feature times must be unique and contain real observations")
    relative = (merged["timestamp_ns"].to_numpy(np.int64)-t0_ns)/1e9
    features = merged[FEATURES].copy()
    features.insert(0, "t", relative.astype(np.float64))
    if len(features) != int(record["rows"]):
        raise ValueError(f"Row count differs from manifest: {record['output_file']}")
    if not np.allclose([relative[0], relative[-1]],
                       [float(record["time_start_s"]), float(record["time_end_s"])], rtol=0, atol=5e-10):
        raise ValueError(f"Flight interval differs from manifest: {record['output_file']}")
    # Original feature files use UTF-8 without BOM and CRLF on every platform.
    data = features.to_csv(index=False, lineterminator="\r\n").encode("utf-8")
    digest = _sha(data)
    if record.get("feature_sha256") and digest != record["feature_sha256"]:
        raise ValueError(f"Feature hash differs from the published manifest: {record['output_file']}; "
                         "check the ALFA source files and pandas CSV serialization version")
    return features, {"file": record["output_file"], "rows": len(features),
                      "feature_sha256": digest, "reference_hash_matches": True,
                      "t0_timestamp_ns": int(t0_ns), "fault_start_s": onset,
                      "feature_bytes": data}


def _safe_target(output_root, relative):
    relative = str(relative).replace("\\", "/")
    path = Path(relative)
    if path.is_absolute() or PureWindowsPath(relative).root or PureWindowsPath(relative).drive or ".." in path.parts:
        raise ValueError("Output paths must remain relative to the output root")
    target = (output_root/path).resolve()
    if not target.is_relative_to(output_root):
        raise ValueError("Output path escapes its root")
    return target


def _ensure_bytes(path, data):
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"Existing file differs; refusing to overwrite: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against a second preparation process.
    with path.open("xb") as stream:
        stream.write(data)


def _export_sequence(source_root, output_root, record):
    sequence = str(record["sequence"])
    if Path(sequence).name != sequence or PureWindowsPath(sequence).name != sequence:
        raise ValueError("Sequence identifiers must be directory names")
    frame, info = build_sequence(source_root/sequence, record)
    target = _safe_target(output_root, record["feature_path"])
    _ensure_bytes(target, info.pop("feature_bytes"))
    return info


def prepare_alfa(source_root, output_root):
    """Prepare all 46 author-fixed flights from ``processed/<sequence>``.

    ``source_root`` can name the processed directory or its parent. Existing
    equal files are reused; different files are never overwritten. The output
    contains 17 sensor columns, with yaw omitted later by the 15D data loader.
    No label CSV is generated: the loader labels windows using the validated
    event onset/end information in metadata/file_manifest.csv.
    """
    source, output = Path(source_root).resolve(), Path(output_root).resolve()
    if source == output:
        raise ValueError("Source and output directories must differ")
    with MANIFEST_FILE.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise ValueError("Empty published file manifest")
    if not (source/records[0]["sequence"]).is_dir() and (source/"processed").is_dir():
        source = source/"processed"
    if source == output:
        raise ValueError("Source and output directories must differ")
    for record in records:
        if record["split"] not in {"train", "val", "test"}:
            raise ValueError("Unknown published split")
        expected = f"{record['split']}/csv/{record['output_file']}"
        if record["feature_path"] != expected:
            raise ValueError("Feature path differs from its fixed split")
    reports = [_export_sequence(source, output, record) for record in records]
    for name in ("file_manifest.csv", "feature_mapping.json"):
        _ensure_bytes(_safe_target(output, "metadata/"+name),
                      (MANIFEST_FILE if name == "file_manifest.csv" else MAPPING_FILE).read_bytes())
    return {"files": len(reports), "features": list(FEATURES),
            "splits": {part: sum(r["split"] == part for r in records) for part in ("train", "val", "test")},
            "flights": reports, "interpolation": False, "imputation": False,
            "timestamps": "native nanoseconds outer-joined; seconds relative to earliest sensor/failure-status message",
            "reference_feature_hashes_verified": True, "label_csv_generated": False,
            "label_source": "Window labels use validated onset/end events in metadata/file_manifest.csv; "
                            "referenced original label CSV files are not generated or required by the loader."}
