"""Synthetic checks only: no target traces, downloads, or recovered target keys.

Run: python -m unittest -v test_aes_timing_attack
Generate an optional local demo: python test_aes_timing_attack.py --make-demo demo.json
"""
import argparse
import json
import unittest
from pathlib import Path

import numpy as np

import aes_timing_attack as attack


def make_synthetic(n=65536, seed=7515):
    """Idealized cold-cache pair-collision timing model, not a MIPS simulator."""
    rng = np.random.default_rng(seed)
    key = rng.integers(0, 256, 16, dtype=np.uint8).tobytes()
    round10 = np.frombuffer(attack.expand_key(key)[10], np.uint8)
    pt = rng.integers(0, 256, (n, 16), dtype=np.uint8)
    ct = attack.encrypt_blocks(key, pt)
    lines = attack.INV_SBOX[ct ^ round10] >> 3
    collisions = np.zeros(n, np.int16)
    for i in range(16):
        for j in range(i + 1, 16):
            collisions += lines[:, i] == lines[:, j]
    timing = 12000.0 - 30.0 * collisions + rng.normal(0, 5, n)
    return pt, ct, timing, key, round10


def save_demo(path, n=65536):
    pt, ct, timing, key, round10 = make_synthetic(n)
    doc = {'description': 'Synthetic pair-collision model; not measured hardware timings',
           'k': key.hex(), 'k10': round10.tobytes().hex(), 'traces': [
               {'pt': p.tobytes().hex(), 'ct': c.tobytes().hex(), 't': float(t)}
               for p, c, t in zip(pt, ct, timing)]}
    Path(path).write_text(json.dumps(doc))
    print('Wrote synthetic demonstration:', path)


class MathChecks(unittest.TestCase):
    def test_training_diagnostic_on_synthetic_data(self):
        from diagnose_training import diagnose
        _, ct, timing, _, rk = make_synthetic()
        rows = diagnose(ct, timing, rk, [(0, 1), (4, 9), (11, 15)],
                        configs=[('raw', 1.0)], max_traces=0)
        for row in rows:
            self.assertGreater(row['summary']['mean_correct_pair_percentile'], 0.9)
            self.assertGreater(row['summary']['median_correct_pair_score'], 0)

    def test_fips197_and_schedule(self):
        attack.self_test()
        key = bytes.fromhex('000102030405060708090a0b0c0d0e0f')
        pt = np.frombuffer(bytes.fromhex('00112233445566778899aabbccddeeff'), np.uint8).reshape(1, 16)
        self.assertEqual(attack.encrypt_blocks(key, pt)[0].tobytes().hex(),
                         '69c4e0d86a7b0430d8cdb78070b4c55a')

    def test_vectorized_aes_matches_reference(self):
        rng = np.random.default_rng(134)
        for _ in range(4):
            key = rng.integers(0, 256, 16, dtype=np.uint8).tobytes()
            pt = rng.integers(0, 256, (20, 16), dtype=np.uint8)
            vector = attack.encrypt_blocks(key, pt)
            for i, row in enumerate(pt):
                self.assertEqual(vector[i].tobytes(), attack.encrypt_block(key, row.tobytes()))

    def test_fwht_scores_match_direct_trace_groups(self):
        rng = np.random.default_rng(334)
        ct = rng.integers(0, 256, (8192, 16), dtype=np.uint8)
        t = rng.normal(0, 3, len(ct))
        hist = attack.pair_histograms(ct, t, 3, 11)
        for offset in [0, 4, 28]:
            score = attack.pair_scores(*hist, method='expanded', offset=offset,
                                       statistic='difference')
            for a, b in [(0, 0), (1, 255), (33, 72), (189, 120)]:
                u = attack.INV_SBOX[ct[:, 3] ^ a].astype(np.int64)
                v = attack.INV_SBOX[ct[:, 11] ^ b].astype(np.int64)
                h = (4 * u + offset) // 32 == (4 * v + offset) // 32
                expected = t[~h].mean() - t[h].mean()
                self.assertAlmostEqual(float(score[a, b]), float(expected), places=10)
        score = attack.pair_scores(*hist, method='simple', statistic='difference')
        for d in [0, 1, 34, 255]:
            h = (ct[:, 3] ^ ct[:, 11]) == d
            self.assertAlmostEqual(float(score[d]), float(t[~h].mean() - t[h].mean()), places=10)

    def test_welch_score_matches_direct_statistic(self):
        rng = np.random.default_rng(841)
        ct = rng.integers(0, 256, (8192, 16), dtype=np.uint8)
        t = rng.normal(2, 4, len(ct))
        hist = attack.pair_histograms(ct, t, 2, 9)
        scores = attack.pair_scores(*hist, method='expanded', statistic='welch')
        a, b = 52, 107
        h = (attack.INV_SBOX[ct[:, 2] ^ a] >> 3) == (attack.INV_SBOX[ct[:, 9] ^ b] >> 3)
        one, zero = t[h], t[~h]
        expected = (zero.mean() - one.mean()) / np.sqrt(zero.var(ddof=1) / len(zero) + one.var(ddof=1) / len(one))
        self.assertAlmostEqual(float(scores[a, b]), float(expected), places=10)

    def test_wrong_key_is_not_verified(self):
        pt, ct, _, _, rk = make_synthetic(32)
        wrong = rk.copy()
        wrong[7] ^= 1
        self.assertIsNone(attack.verify_round10(wrong, pt, ct))

    def test_negative_control_has_no_recovered_key(self):
        pt, ct, timing, _, _ = make_synthetic(4096)
        rng = np.random.default_rng(915)
        timing = rng.normal(0, 1, len(timing))
        args = argparse.Namespace(restarts=4, sweeps=8, seed=622, max_candidates=1)
        for method in ['simple', 'expanded']:
            with self.subTest(method=method):
                scores = attack.build_scores(ct, timing, method)
                _, _, result = attack.recover(scores, method, pt, ct, args)
                self.assertEqual(result['status'], 'not_recovered')
                self.assertNotIn('master_key_hex', result)

    def test_end_to_end_synthetic_recovery(self):
        pt, ct, t, key, round10 = make_synthetic()
        t = (t - t.mean()) / t.std()
        args = argparse.Namespace(restarts=32, sweeps=25, seed=622, max_candidates=32)
        for method in ['simple', 'expanded']:
            with self.subTest(method=method):
                scores = attack.build_scores(ct, t, method)
                _, _, result = attack.recover(scores, method, pt, ct, args)
                self.assertEqual(result['status'], 'recovered')
                self.assertEqual(result['master_key_hex'], key.hex())
                self.assertEqual(result['round10_key_hex'], round10.tobytes().hex())


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--make-demo', type=Path)
    p.add_argument('--traces', type=int, default=65536)
    args, rest = p.parse_known_args()
    if args.make_demo:
        if args.traces < 1000:
            p.error('--traces must be at least 1000')
        save_demo(args.make_demo, args.traces)
    else:
        unittest.main(argv=['test_aes_timing_attack.py', *rest])
