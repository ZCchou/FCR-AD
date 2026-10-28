"""Command-line interface for FCR-AD."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import build_datasets
from .evaluation import choose_threshold, evaluate_scores
from .features import MODEL_INPUTS, channel_balanced_mse, extract_features
from .head import Head
from .model import make_backbone
from .training import evaluate_detector, fit_detector, seed_all


def default_config():
    root = Path(__file__).resolve().parents[1] / 'configs' / 'alfa_author_fixed.json'
    if not root.exists():
        root = Path(__file__).parent / 'resources' / 'protocol.json'
    return json.loads(root.read_text(encoding='utf-8'))


class _SyntheticWindows(Dataset):
    """Small deterministic API example, with no ALFA measurements."""

    def __init__(self, windows=24):
        generator = torch.Generator().manual_seed(120)
        self.hist_times = torch.rand(windows, 15, 12, generator=generator).sort(-1).values * 5
        self.hist_values = torch.randn(windows, 15, 12, generator=generator)
        self.hist_mask = (torch.rand(windows, 15, 12, generator=generator) > .2).float()
        self.query_times = 5 + torch.rand(windows, 15, 4, generator=generator)
        self.query_mask = torch.ones_like(self.query_times)
        self.target = torch.randn(windows, 15, 4, generator=generator)
        self.rows = []
        for i in range(windows):
            label = i % 2
            if label:
                self.target[i] += 3
            self.rows.append(dict(label=label, file=f'synthetic-{i//4}',
                                  event_ids=[f'event-{i//4}'] if label else [],
                                  fault_start_s=5. if label else None, decision_time=6.))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        result = {name:getattr(self,name)[index] for name in MODEL_INPUTS + ('target',)}
        return {**result, 'row_index':torch.tensor(index)}


def smoke(device='cpu'):
    """Exercise a gradient update, feature map, head, calibration, and metrics."""
    seed_all(42)
    model = make_backbone().to(device)
    dataset = _SyntheticWindows()
    batch = {key:torch.stack([dataset[i][key] for i in range(4)]).to(device)
             for key in MODEL_INPUTS + ('target',)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.001)
    prediction = model(**{key:batch[key] for key in MODEL_INPUTS})
    loss = channel_balanced_mse(prediction, batch['target'], batch['query_mask'])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    x = extract_features(model, dataset, device, 4)
    # Use a small classifier for the synthetic example.
    head = Head().fit(x[:16], dataset.rows[:16], dict(n_estimators=5,min_samples_leaf=1),42)
    scores = head.predict(x[16:])
    labels = [row['label'] for row in dataset.rows[16:]]
    calibration = choose_threshold(labels, scores)
    metrics = evaluate_scores(dataset.rows[16:], scores, calibration['threshold'])
    return dict(synthetic=True, loss=float(loss.detach().cpu()), features=list(x.shape),
                finite_scores=bool(np.isfinite(scores).all()),
                classifier_trees=5, raw_f1=metrics['raw']['f1'])


def main(argv=None):
    parser = argparse.ArgumentParser(prog='fcr-ad', description='Forecasting Context Reuse for Anomaly Detection')
    parser.add_argument('--version', action='version', version='FCR-AD 0.1.0')
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare', help='Build the native CSV representation from official ALFA topic CSVs')
    prepare.add_argument('--source-root', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    audit = sub.add_parser('check-data', help='Check the fixed development windows and normalization')
    audit.add_argument('--data-root', type=Path, required=True)
    demo = sub.add_parser('smoke', help='Run a synthetic end-to-end API check without ALFA data')
    demo.add_argument('--device', default='cpu')
    demo.add_argument('--threads', type=int, default=1)
    for command in ('train','evaluate','run'):
        entry = sub.add_parser(command, help={'train':'Fit and freeze using development data',
            'evaluate':'Evaluate an already frozen detector', 'run':'Train, freeze, then evaluate'}[command])
        entry.add_argument('--data-root', type=Path, required=True)
        entry.add_argument('--output', type=Path, required=True)
        entry.add_argument('--seeds', nargs='+', type=int, default=[42])
        entry.add_argument('--device', default='cpu')
        entry.add_argument('--threads', type=int, default=1)
        if command != 'evaluate':
            entry.add_argument('--max-steps', type=int, default=2400)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        from .prepare import prepare_alfa
        result = prepare_alfa(args.source_root, args.output)
    elif args.command == 'check-data':
        _, _, result = build_datasets(args.data_root)
    else:
        if args.threads < 1:
            parser.error('--threads must be positive')
        torch.set_num_threads(args.threads)
        if str(args.device).startswith('cuda') and not torch.cuda.is_available():
            parser.error('CUDA is unavailable; choose --device cpu or install a suitable PyTorch build')
        if args.command == 'smoke':
            result = smoke(args.device)
        else:
            if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 or seed >= 2**32 for seed in args.seeds):
                parser.error('seeds must be distinct nonnegative 32-bit integers')
            config = default_config()
            if args.command != 'evaluate':
                if args.max_steps < 1:
                    parser.error('--max-steps must be positive')
                config['training']['max_steps'] = args.max_steps
            result = {}
            if args.command != 'evaluate':
                for seed in args.seeds:
                    output = args.output / f'seed{seed}'
                    fit_detector(args.data_root,output,copy.deepcopy(config),seed=seed,device=args.device)
            for seed in args.seeds:
                output = args.output / f'seed{seed}'
                result[str(seed)] = (evaluate_detector(args.data_root,output,device=args.device)
                                    if args.command != 'train' else dict(frozen=True,seed=seed))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
