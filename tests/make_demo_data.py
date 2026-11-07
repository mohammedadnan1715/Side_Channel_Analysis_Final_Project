#!/usr/bin/env python3
"""Generate a LOCAL synthetic fixture to exercise every pytest option.

This is demonstration data, not measured hardware traces or Week 15 results.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aes_timing_attack as attack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('pytest_demo'))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Choose a new or empty output directory to avoid overwriting files.')
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7515)
    key = rng.integers(0, 256, 16, dtype=np.uint8).tobytes()
    pt = rng.integers(0, 256, (65536, 16), dtype=np.uint8)
    ctx = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
    ct = np.frombuffer(ctx.update(pt.tobytes()) + ctx.finalize(), np.uint8).reshape(-1, 16).copy()
    rk = np.frombuffer(attack.expand_key(key)[10], np.uint8)
    lines = attack.INV_SBOX[ct ^ rk] >> 3
    hits = np.zeros(len(pt), np.int16)
    for i, j in attack.PAIRS:
        hits += lines[:, i] == lines[:, j]
    timing = 12000.0 - 30.0 * hits + rng.normal(0, 5, len(pt))
    dataset = args.output / 'synthetic_traces.json'
    dataset.write_text(json.dumps({'description': 'Synthetic pair-collision demonstration only',
        'traces': [{'pt': p.tobytes().hex(), 'ct': c.tobytes().hex(), 't': float(t)}
                   for p, c, t in zip(pt, ct, timing)]}))
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    settings = SimpleNamespace(restarts=32, sweeps=25, seed=622, max_candidates=32,
                               upper_quantile=1.0, transform='rank', max_traces=0)
    selected_ct, selected_t, prep = attack.preprocess(ct, timing, settings)
    master_candidates = np.tile(np.frombuffer(key, np.uint8), (4096, 1))
    for i in range(16):
        master_candidates[i * 256:(i + 1) * 256, i] = np.arange(256, dtype=np.uint8)
    round_candidates = np.array([list(attack.expand_key(k.tobytes())[10]) for k in master_candidates], np.uint8)
    rankings = {'synthetic_demo': True, 'dataset': dataset.name, 'dataset_sha256': digest,
                'master_key_hex': key.hex(), 'round10_key_hex': rk.tobytes().hex(),
                'ranking_scope': 'post-recovery conditional master-key byte rankings',
                'score_mode': 'original_joint_score', 'methods': {}}
    for method in ['simple', 'expanded']:
        scores = attack.build_scores(selected_ct, selected_t, method)
        _, _, result = attack.recover(scores, method, pt, ct, settings)
        if result.get('master_key_hex') != key.hex():
            raise RuntimeError('Synthetic recovery failed; a generated fixture must not fake success.')
        result.update(method=method, dataset=dataset.name, dataset_sha256=digest,
                       preprocessing=prep, synthetic_demo=True)
        (args.output / f'{method}_result.json').write_text(json.dumps(result, indent=2))
        baseline = attack.objective(scores, rk[None, :], method)[0]
        support = (attack.objective(scores, round_candidates, method) - baseline).reshape(16, 256)
        rows = []
        ranks = []
        for i in range(16):
            order = np.lexsort((np.arange(256), -support[i]))
            rows.append([{'rank': rank, 'candidate_hex': f'{int(v):02x}', 'score_delta': float(support[i, v])}
                         for rank, v in enumerate(order, 1)])
            ranks.append(int(np.where(order == key[i])[0][0]) + 1)
        rankings['methods'][method] = {'rankings': rows, 'correct_value_ranks': ranks,
                                      'mean_correct_value_rank': sum(ranks) / 16, 'candidate_count': 4096}
        del scores
    (args.output / 'master_key_rankings.json').write_text(json.dumps(rankings, indent=2))
    print('Synthetic fixture created:', args.output)
    print('Do not submit this demonstration as your actual Week 15 result.')


if __name__ == '__main__':
    main()
