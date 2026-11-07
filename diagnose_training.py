#!/usr/bin/env python3
"""Measure assumed cache leakage on a LOCAL known-key training dataset.

Reference keys are used only for these diagnostic measurements. No files are
downloaded, and this script does not perform a blind-key recovery search.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import aes_timing_attack as attack


CONFIGS = [('raw', 1.0), ('raw', 0.99), ('raw', 0.95), ('log', 0.99), ('rank', 1.0)]


def summarize(records, hypotheses):
    ranks = np.array([r['rank'] for r in records], dtype=np.float64)
    scores = np.array([r['correct_score'] for r in records], dtype=np.float64)
    return {'pairs_tested': len(records), 'hypotheses_per_pair': hypotheses,
            'mean_correct_pair_rank': float(ranks.mean()),
            'median_correct_pair_rank': float(np.median(ranks)),
            'random_expected_mean_rank': (hypotheses + 1) / 2,
            'mean_correct_pair_percentile': float(np.mean(1 - (ranks - 1) / (hypotheses - 1))),
            'median_correct_pair_score': float(np.median(scores)),
            'positive_correct_pair_scores': int(np.sum(scores > 0)),
            'top1_pairs': int(np.sum(ranks == 1)),
            'top10_pairs': int(np.sum(ranks <= 10)),
            'note': 'Pair ranks are diagnostic; they are not the assignment\'s 16 key-byte ranks.'}


def diagnose(ct, timing, round10, pairs, configs=CONFIGS, offsets=(0,), max_traces=100000, seed=622):
    experiments = []
    kernels = {o: attack.fwht(attack.collision_kernel(o)) for o in offsets}
    for transform, quantile in configs:
        prep_args = SimpleNamespace(upper_quantile=quantile, transform=transform,
                                    max_traces=max_traces, seed=seed)
        selected_ct, selected_t, prep = attack.preprocess(ct, timing, prep_args)
        groups = {('simple', None): []}
        groups.update({('expanded', offset): [] for offset in offsets})
        for i, j in pairs:
            hist = attack.pair_histograms(selected_ct, selected_t, i, j)
            simple = attack.pair_scores(*hist, method='simple', statistic='welch')
            true_d = int(round10[i] ^ round10[j])
            true_score = float(simple[true_d])
            groups['simple', None].append({'i': i, 'j': j,
                'rank': 1 + int(np.sum(simple > true_score)), 'correct_score': true_score})
            for offset in offsets:
                expanded = attack.pair_scores(*hist, method='expanded', offset=offset,
                    statistic='welch', kernel_transform=kernels[offset])
                true_score = float(expanded[int(round10[i]), int(round10[j])])
                groups['expanded', offset].append({'i': i, 'j': j,
                    'rank': 1 + int(np.sum(expanded > true_score)), 'correct_score': true_score})
        for (method, offset), records in groups.items():
            row = {'method': method, 'offset': offset, 'preprocessing': prep,
                   'summary': summarize(records, 256 if method == 'simple' else 65536),
                   'pair_measurements': records}
            experiments.append(row)
            s = row['summary']
            attack.log(f'{method:8} offset={str(offset):4} transform={transform:4} q={quantile:.2f} '
                f'N={len(selected_t)} mean-pair-rank={s["mean_correct_pair_rank"]:.1f} '
                f'percentile={s["mean_correct_pair_percentile"]:.3f} '
                f'median-signed-score={s["median_correct_pair_score"]:.3f}')
    return experiments


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset', type=Path, help='Your local t00/t01 JSON or tar.gz with k and k10')
    p.add_argument('--output', type=Path, default=Path('training_diagnostics.json'))
    p.add_argument('--pairs', type=int, default=12, help='Number of byte pairs to inspect (1..120)')
    p.add_argument('--max-traces', type=int, default=100000, help='Seeded subset per setting; 0 uses all')
    p.add_argument('--scan-offsets', action='store_true', help='Compare all eight possible four-byte-aligned offsets')
    p.add_argument('--seed', type=int, default=622)
    args = p.parse_args(argv)
    if not 1 <= args.pairs <= 120 or args.max_traces < 0:
        p.error('--pairs must be 1..120, and --max-traces must be nonnegative')
    attack.self_test()
    pt, ct, timing, metadata = attack.load_dataset(args.dataset)
    if 'k' not in metadata or 'k10' not in metadata:
        p.error('A known-key TRAINING file with k and k10 is required. Use t00 or t01, not the blind file.')
    master = bytes.fromhex(metadata['k'])
    final = bytes.fromhex(metadata['k10'])
    if attack.expand_key(master)[10] != final:
        p.error('Provided k and k10 disagree under the AES-128 key schedule; check format and byte order.')
    checked = np.unique(np.linspace(0, len(pt) - 1, min(32, len(pt)), dtype=int))
    if not np.array_equal(attack.encrypt_blocks(master, pt[checked]), ct[checked]):
        p.error('Provided key does not reproduce the plaintext/ciphertext pairs; check dataset interpretation.')
    rng = np.random.default_rng(args.seed)
    chosen = np.sort(rng.choice(len(attack.PAIRS), args.pairs, replace=False))
    pairs = [attack.PAIRS[int(i)] for i in chosen]
    offsets = tuple(range(0, 32, 4)) if args.scan_offsets else (0,)
    quantiles = {str(q): float(np.quantile(timing, q)) for q in [0, .01, .5, .9, .95, .99, 1]}
    attack.log('AES/key-format checks passed. Timing quantiles:', quantiles)
    experiments = diagnose(ct, timing, np.frombuffer(final, np.uint8), pairs,
                           offsets=offsets, max_traces=args.max_traces, seed=args.seed)
    report = {'dataset': args.dataset.name, 'total_traces': len(timing),
              'aes_reference_checks_passed': True, 'aes_pairs_checked': len(checked),
              'timing_quantiles_cycles': quantiles, 'seed': args.seed, 'experiments': experiments,
              'interpretation': [
                  'Mean correct-pair percentile near 0.5 is approximately random; nearer 1 is better.',
                  'Positive signed scores mean predicted collisions were faster; negative scores mean slower.',
                  'Pairs share traces and bytes; these measurements are correlated, not independent significance tests.',
                  'A strong pair diagnostic does not guarantee full-key recovery.',
                  'Select settings on t00, then confirm them on t01 before using them on the blind dataset.',
                  'More heuristic restarts address search limitations, not absent or mismodeled timing leakage.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    attack.log('Wrote:', args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
