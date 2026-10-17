"""Health-only forecasting, labeled detector fitting, and explicit test freeze."""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SENSORS, build_datasets
from .evaluation import choose_threshold, evaluate_scores
from .features import MODEL_INPUTS, channel_balanced_mse, extract_features
from .head import Head
from .model import make_backbone


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def _restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def _save_torch(path, state):
    temporary = Path(str(path) + '.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def deterministic_batch(dataset, seed, step, batch_size):
    if not len(dataset) or batch_size < 1:
        raise ValueError('training requires a nonempty dataset and a positive batch size')
    batches = int(np.ceil(len(dataset) / batch_size))
    cycle, offset = divmod(step, batches)
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed + cycle))
    indices = order[offset * batch_size:(offset + 1) * batch_size].tolist()
    return next(iter(DataLoader(dataset, batch_size=len(indices), sampler=indices, num_workers=0)))


def _loss(model, raw, device):
    batch = {key: value.to(device) for key, value in raw.items()}
    prediction = model(**{key: batch[key] for key in MODEL_INPUTS})
    return channel_balanced_mse(prediction, batch['target'], batch['query_mask'])


def healthy_validation(model, dataset, device, micro_batch):
    model.eval()
    total, count = 0., 0
    with torch.no_grad():
        for raw in DataLoader(dataset, batch_size=micro_batch, shuffle=False):
            size = len(raw['row_index'])
            total += float(_loss(model, raw, device)) * size
            count += size
    if not count:
        raise ValueError('healthy validation is empty')
    return total / count


def train_backbone(train_data, validation_data, output, config, *, seed=42, device='cpu'):
    """Select a forecasting checkpoint only by healthy validation loss.

    An interrupted run resumes model, optimizer, RNG, and early-stop state from
    its most recent validation checkpoint. A changed config or dataset is rejected.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = dict(config)
    required = ('max_steps', 'validation_interval', 'patience_steps', 'batch_size',
                'micro_batch_size', 'gradient_clip', 'lr', 'weight_decay')
    if any(key not in cfg or not np.isfinite(cfg[key]) or cfg[key] <= 0 for key in required):
        raise ValueError('training configuration requires positive finite schedule values')
    source = {p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')}
    identity = _identity(dict(train=train_data.identity, validation=validation_data.identity,
                              config=cfg, seed=seed, source=source))
    seed_all(seed)
    model = make_backbone().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    best_path, last_path = output / 'backbone.pt', output / 'last.pt'
    completed = output / 'training.json'
    if completed.exists():
        info = json.loads(completed.read_text(encoding='utf-8'))
        if info['identity'] != identity:
            raise ValueError('existing training belongs to a different config, dataset, or source')
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)['model'])
        return model, info
    step, history, best, best_step, stale, elapsed = 0, [], float('inf'), 0, 0, 0.
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state['identity'] != identity:
            raise ValueError('checkpoint identity differs from the current run')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        _restore_rng(state['rng'])
        step, history, best, best_step, stale, elapsed = (
            state[key] for key in ('step', 'history', 'best_value', 'best_step', 'stale', 'seconds'))
        if step == best_step:
            _save_torch(best_path, dict(model=state['model'], selected_step=step))
    start = time.perf_counter()
    while step < cfg['max_steps'] and stale < cfg['patience_steps']:
        raw = deterministic_batch(train_data, seed, step, cfg['batch_size'])
        size = len(raw['row_index'])
        model.train()
        model.zero_grad(set_to_none=True)
        for lo in range(0, size, cfg['micro_batch_size']):
            small = {key: value[lo:lo + cfg['micro_batch_size']] for key, value in raw.items()}
            loss = _loss(model, small, device)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite forecasting loss')
            (loss * len(small['row_index']) / size).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
        optimizer.step()
        step += 1
        if step % cfg['validation_interval'] and step != cfg['max_steps']:
            continue
        value = healthy_validation(model, validation_data, device, cfg['micro_batch_size'])
        if not np.isfinite(value):
            raise FloatingPointError('nonfinite healthy validation loss')
        improved = value < best
        if improved:
            best, best_step, stale = value, step, 0
        else:
            stale += cfg['validation_interval']
        history.append(dict(step=step, healthy_validation_mse=value))
        state = dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     rng=_rng_state(), step=step, history=history, best_value=best,
                     best_step=best_step, stale=stale, seconds=elapsed + time.perf_counter() - start)
        _save_torch(last_path, state)
        if improved:
            _save_torch(best_path, dict(model=model.state_dict(), selected_step=step))
        write_json(output / 'history.json', history)
        print(f'seed={seed} step={step}/{cfg["max_steps"]} healthy_MSE={value:.7g}', flush=True)
    write_json(output / 'history.json', history)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)['model'])
    info = dict(identity=identity, seed=seed, steps_completed=step, selected_step=best_step,
                best_validation_mse=best, stale_steps=stale,
                stop_reason='early_stopping' if stale >= cfg['patience_steps'] else 'max_steps',
                seconds=elapsed + time.perf_counter() - start)
    write_json(completed, info)
    return model, info


def fit_detector(data_root, output, config, *, seed=42, device='cpu'):
    """Train on development data and publish a frozen detector before test."""
    output = Path(output)
    frozen = output / 'frozen_before_test.json'
    if frozen.exists():
        verify_freeze(output)
        previous = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
        if previous['seed'] != seed or previous['config'] != config:
            raise ValueError('output already contains a detector with a different seed or configuration')
        return json.loads((output / 'calibration.json').read_text(encoding='utf-8'))
    sets, stats, audit = build_datasets(data_root)
    output.mkdir(parents=True, exist_ok=True)
    model, info = train_backbone(sets['healthy_train'], sets['healthy_val'], output,
                                 config['training'], seed=seed, device=device)
    micro = config['training']['micro_batch_size']
    train_x = extract_features(model, sets['head_train'], device, micro)
    val_x = extract_features(model, sets['head_val'], device, micro)
    head_config = {key: config['head'][key] for key in
                   ('n_estimators','max_depth','learning_rate','min_samples_leaf',
                    'subsample','max_features','positive_weight','weighting')}
    head = Head().fit(train_x, sets['head_train'].rows, head_config, seed)
    head.save(output / 'detector.joblib')
    labels = np.array([row['label'] for row in sets['head_val'].rows])
    calibration = choose_threshold(labels, head.predict(val_x),
                                    config['head']['threshold']['validation_max_fpr'])
    write_json(output / 'normalization.json', stats)
    write_json(output / 'development_data.json', audit)
    write_json(output / 'calibration.json', calibration)
    write_json(output / 'protocol.json', dict(method='FCR-AD', seed=seed, config=config,
               training=info, feature_dimensions=290, sensors=SENSORS))
    artifacts = ['backbone.pt','detector.joblib','normalization.json','calibration.json',
                 'protocol.json','development_data.json','training.json']
    write_json(frozen, dict(method='FCR-AD', artifacts={name:file_hash(output/name) for name in artifacts},
                           source={p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')}))
    return calibration


def verify_freeze(output):
    output = Path(output)
    record = json.loads((output / 'frozen_before_test.json').read_text(encoding='utf-8'))
    current_source = {p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')}
    if current_source != record['source']:
        raise ValueError('package source changed after detector freezing')
    for name, expected in record['artifacts'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or file_hash(output / name) != expected:
            raise ValueError('frozen detector artifact changed: ' + name)
    return record


def evaluate_detector(data_root, output, *, device='cpu'):
    """Evaluate the frozen backbone, detector, normalization, and threshold."""
    import pandas as pd
    output = Path(output)
    verify_freeze(output)
    stats = json.loads((output / 'normalization.json').read_text(encoding='utf-8'))
    sets, _, audit = build_datasets(data_root, stats=stats, include_test=True)
    model = make_backbone().to(device)
    model.load_state_dict(torch.load(output/'backbone.pt', map_location=device, weights_only=False)['model'])
    head = Head.load(output / 'detector.joblib')
    calibration = json.loads((output / 'calibration.json').read_text(encoding='utf-8'))
    x = extract_features(model, sets['test'], device=device)
    scores = head.predict(x)
    results = evaluate_scores(sets['test'].rows, scores, calibration['threshold'])
    destination = output / 'evaluation'
    destination.mkdir(exist_ok=True)
    write_json(destination/'metrics.json', results)
    write_json(destination/'data_audit.json', audit)
    rows = [dict(window_id=row['window_id'], file=row['file'],
                 decision_time=row['decision_time'], label=row['label'],
                 score=float(score), prediction=int(score > calibration['threshold']))
            for row, score in zip(sets['test'].rows, scores)]
    pd.DataFrame(rows).to_csv(destination/'predictions.csv', index=False)
    return results
