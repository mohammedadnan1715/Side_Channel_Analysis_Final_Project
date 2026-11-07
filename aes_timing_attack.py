#!/usr/bin/env python3
"""Offline AES-128 cache-collision analysis for the RIT timing datasets.

Only ciphertexts and timings enter the attack scores. Known training keys are
kept separately for evaluation. Plaintexts are used to verify full candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import csv
import json
import tarfile
import time
from pathlib import Path

import numpy as np

SBOX = np.frombuffer(bytes.fromhex(
    "637c777bf26b6fc53001672bfed7ab76ca82c97dfa5947f0add4a2af9ca472c0"
    "b7fd9326363ff7cc34a5e5f171d8311504c723c31896059a071280e2eb27b275"
    "09832c1a1b6e5aa0523bd6b329e32f8453d100ed20fcb15b6acbbe394a4c58cf"
    "d0efaafb434d338545f9027f503c9fa851a3408f929d38f5bcb6da2110fff3d2"
    "cd0c13ec5f974417c4a77e3d645d197360814fdc222a908846eeb814de5e0bdb"
    "e0323a0a4906245cc2d3ac629195e479e7c8376d8dd54ea96c56f4ea657aae08"
    "ba78252e1ca6b4c6e8dd741f4bbd8b8a703eb5664803f60e613557b986c11d9e"
    "e1f8981169d98e949b1e87e9ce5528df8ca1890dbfe6426841992d0fb054bb16"
), dtype=np.uint8)
INV_SBOX = np.argsort(SBOX).astype(np.uint8)
PAIRS = [(i, j) for i in range(16) for j in range(i + 1, 16)]


def log(*args):
    print(*args, flush=True)


def expand_key(key: bytes) -> list[bytes]:
    if len(key) != 16:
        raise ValueError("AES-128 requires a 16-byte key")
    words = [list(key[i:i + 4]) for i in range(0, 16, 4)]
    rc = 1
    for i in range(4, 44):
        t = words[i - 1].copy()
        if i % 4 == 0:
            t = [int(SBOX[x]) for x in t[1:] + t[:1]]
            t[0] ^= rc
            rc = ((rc << 1) ^ (0x11B if rc & 0x80 else 0)) & 255
        words.append([a ^ b for a, b in zip(words[i - 4], t)])
    return [bytes(sum(words[i:i + 4], [])) for i in range(0, 44, 4)]


def invert_round10(round10: bytes) -> bytes:
    if len(round10) != 16:
        raise ValueError("The final-round key must have 16 bytes")
    words = [None] * 44
    words[40:44] = [list(round10[i:i + 4]) for i in range(0, 16, 4)]
    rcons = [0, 1, 2, 4, 8, 16, 32, 64, 128, 27, 54]
    for i in range(43, 3, -1):
        t = words[i - 1].copy()
        if i % 4 == 0:
            t = [int(SBOX[x]) for x in t[1:] + t[:1]]
            t[0] ^= rcons[i // 4]
        words[i - 4] = [a ^ b for a, b in zip(words[i], t)]
    return bytes(sum(words[:4], []))


def encrypt_block(key: bytes, plaintext: bytes) -> bytes:
    """Dependency-light AES reference used for candidate verification."""
    rks = expand_key(key)
    state = [a ^ b for a, b in zip(plaintext, rks[0])]
    perm = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11]
    for rnd in range(1, 11):
        state = [int(SBOX[state[i]]) for i in perm]
        if rnd != 10:
            for off in range(0, 16, 4):
                a, b, c, d = state[off:off + 4]
                v = a ^ b ^ c ^ d
                xs = [a ^ b, b ^ c, c ^ d, d ^ a]
                xs = [((x << 1) ^ (0x11B if x & 128 else 0)) & 255 for x in xs]
                state[off:off + 4] = [a ^ v ^ xs[0], b ^ v ^ xs[1],
                                      c ^ v ^ xs[2], d ^ v ^ xs[3]]
        state = [a ^ b for a, b in zip(state, rks[rnd])]
    return bytes(state)


def self_test():
    key = bytes.fromhex("000102030405060708090a0b0c0d0e0f")
    pt = bytes.fromhex("00112233445566778899aabbccddeeff")
    expected = bytes.fromhex("69c4e0d86a7b0430d8cdb78070b4c55a")
    rk = bytes.fromhex("13111d7fe3944a17f307a78b4d2b30c5")
    assert len(SBOX) == 256 and len(np.unique(SBOX)) == 256
    assert encrypt_block(key, pt) == expected
    assert expand_key(key)[10] == rk
    assert invert_round10(rk) == key
    rng = np.random.default_rng(671)
    for _ in range(20):
        k = rng.integers(0, 256, 16, dtype=np.uint8).tobytes()
        assert invert_round10(expand_key(k)[10]) == k


def load_dataset(path: str | Path):
    """Read supplied JSON/tar safely without extracting arbitrary paths."""
    path = Path(path)
    if path.suffix == '.json':
        with path.open() as f:
            doc = json.load(f)
    else:
        with tarfile.open(path, 'r:gz') as archive:
            members = [m for m in archive.getmembers() if m.isfile() and m.name.endswith('.json')]
            if len(members) != 1:
                raise ValueError('Expected exactly one JSON member')
            doc = json.load(archive.extractfile(members[0]))
    rows = doc.pop('traces')
    n = len(rows)
    if not rows or any(len(r['pt']) != 32 or len(r['ct']) != 32 for r in rows):
        raise ValueError('Each trace needs a 32-hex-character pt and ct')
    pt = np.frombuffer(bytes.fromhex(''.join(r['pt'] for r in rows)), np.uint8).reshape(n, 16).copy()
    ct = np.frombuffer(bytes.fromhex(''.join(r['ct'] for r in rows)), np.uint8).reshape(n, 16).copy()
    timing = np.fromiter((r['t'] for r in rows), np.float64, count=n)
    metadata = {k: v for k, v in doc.items() if k.startswith('k')}
    if not np.isfinite(timing).all() or np.any(timing < 0):
        raise ValueError('Invalid timing value')
    return pt, ct, timing, metadata


def encrypt_blocks(key: bytes, plaintexts: np.ndarray) -> np.ndarray:
    """Vectorized AES-128; state byte order is the usual column-major order."""
    rks = np.frombuffer(b''.join(expand_key(key)), np.uint8).reshape(11, 16)
    state = np.asarray(plaintexts, dtype=np.uint8).copy() ^ rks[0]
    perm = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11]
    for rnd in range(1, 11):
        state = SBOX[state[:, perm]]
        if rnd < 10:
            columns = state.reshape(-1, 4, 4)
            rotated = np.roll(columns, -1, axis=2)
            adjacent = columns ^ rotated
            doubled = (adjacent << 1) ^ ((adjacent >> 7) * np.uint8(0x1B))
            columns ^= np.bitwise_xor.reduce(columns, axis=2)[:, :, None] ^ doubled
        state ^= rks[rnd]
    return state


def fwht(values: np.ndarray) -> np.ndarray:
    """Unnormalized Walsh-Hadamard transform; H(H(x)) = len(x) * x."""
    out = np.asarray(values, dtype=np.float64).copy().reshape(-1)
    if len(out) & (len(out) - 1):
        raise ValueError('FWHT length must be a power of two')
    step = 1
    while step < out.size:
        view = out.reshape(-1, 2 * step)
        a, b = view[:, :step].copy(), view[:, step:].copy()
        view[:, :step] = a + b
        view[:, step:] = a - b
        step *= 2
    return out


def collision_kernel(offset: int = 0) -> np.ndarray:
    """Te4 has 4-byte entries; account for base address modulo 32."""
    lines = (4 * INV_SBOX.astype(np.int64) + offset) // 32
    return (lines[:, None] == lines[None, :]).astype(np.float64)


def pair_histograms(ct, timing, i, j):
    bins = ct[:, i].astype(np.int64) * 256 + ct[:, j]
    counts = np.bincount(bins, minlength=65536).astype(np.float64)
    sums = np.bincount(bins, weights=timing, minlength=65536)
    sums2 = np.bincount(bins, weights=timing * timing, minlength=65536)
    return counts, sums, sums2


def pair_scores(counts, sums, sums2, method='expanded', offset=0,
                statistic='welch', kernel_transform=None):
    """Return signed scores: positive means predicted collisions are faster."""
    if method == 'expanded':
        kh = fwht(collision_kernel(offset)) if kernel_transform is None else kernel_transform
        n1 = fwht(fwht(counts) * kh) / 65536
        s1 = fwht(fwht(sums) * kh) / 65536
        q1 = fwht(fwht(sums2) * kh) / 65536
    else:
        x = np.arange(256, dtype=np.int64)
        d = (x[:, None] ^ x[None, :]).reshape(-1)
        n1 = np.bincount(d, weights=counts, minlength=256)
        s1 = np.bincount(d, weights=sums, minlength=256)
        q1 = np.bincount(d, weights=sums2, minlength=256)
    total_n, total_s, total_q = counts.sum(), sums.sum(), sums2.sum()
    n0, s0, q0 = total_n - n1, total_s - s1, total_q - q1
    valid = (n1 > 2) & (n0 > 2)
    n1s, n0s = np.maximum(n1, 1), np.maximum(n0, 1)
    mu1, mu0 = s1 / n1s, s0 / n0s
    score = mu0 - mu1
    if statistic == 'welch':
        var1 = np.maximum((q1 - s1 * mu1) / np.maximum(n1 - 1, 1), 0)
        var0 = np.maximum((q0 - s0 * mu0) / np.maximum(n0 - 1, 1), 0)
        se = np.sqrt(var1 / n1s + var0 / n0s)
        score = score / np.maximum(se, 1e-12)
    score[~valid] = 0
    return score.reshape(256, 256) if method == 'expanded' else score


def build_scores(ct, timing, method='expanded', offset=0, statistic='welch'):
    shape = (16, 16, 256, 256) if method == 'expanded' else (16, 16, 256)
    scores = np.zeros(shape, dtype=np.float32)
    kh = fwht(collision_kernel(offset)) if method == 'expanded' else None
    started = time.monotonic()
    for pair_index, (i, j) in enumerate(PAIRS):
        hist = pair_histograms(ct, timing, i, j)
        s = pair_scores(*hist, method=method, offset=offset,
                        statistic=statistic, kernel_transform=kh)
        scores[i, j] = s
        scores[j, i] = s.T if method == 'expanded' else s
        if pair_index % 20 == 19:
            log(f'  scored {pair_index + 1}/120 pairs in {time.monotonic() - started:.1f}s')
    return scores


def verify_round10(round10, pt, ct, checks=8):
    key = invert_round10(bytes(round10))
    for n in range(min(checks, len(pt))):
        if encrypt_block(key, pt[n].tobytes()) != ct[n].tobytes():
            return None
    return key


def objective(scores, keys, method):
    result = np.zeros(len(keys), np.float64)
    for i, j in PAIRS:
        if method == 'expanded':
            result += scores[i, j][keys[:, i], keys[:, j]]
        else:
            result += scores[i, j][keys[:, i] ^ keys[:, j]]
    return result


def local_field(scores, keys, i, method):
    field = np.zeros((len(keys), 256), np.float64)
    values = np.arange(256, dtype=np.uint8)
    for j in range(16):
        if i == j:
            continue
        if method == 'expanded':
            field += scores[i, j][:, keys[:, j]].T
        else:
            field += scores[i, j][values[None, :] ^ keys[:, j, None]]
    return field


def optimize(scores, method, restarts=128, sweeps=40, seed=622):
    """Heuristic graph optimization; multiple restarts do not prove optimality."""
    rng = np.random.default_rng(seed)
    keys = rng.integers(0, 256, (restarts, 16), dtype=np.uint8)
    if method == 'simple':
        keys[:, 0] = 0  # Gauge: these states are K10[i] XOR K10[0].
        keys[0] = np.argmax(scores[0], axis=1)
        keys[0, 0] = 0
        nodes = np.arange(1, 16)
    else:
        strengths = np.max(scores, axis=3).sum(axis=1)
        keys[0] = np.argmax(strengths, axis=1)
        nodes = np.arange(16)
    for sweep in range(sweeps):
        changes = 0
        for i in rng.permutation(nodes):
            field = local_field(scores, keys, int(i), method)
            proposals = np.argmax(field, axis=1).astype(np.uint8)
            current = field[np.arange(restarts), keys[:, i]]
            best = field[np.arange(restarts), proposals]
            change = best > current + 1e-8
            changes += int(change.sum())
            keys[change, i] = proposals[change]
        values = objective(scores, keys, method)
        if sweep % 5 == 0 or not changes:
            log(f'  search sweep {sweep + 1}: best score {values.max():.3f}, changes {changes}')
        if changes == 0:
            break
    unique_keys = np.unique(keys, axis=0)
    values = objective(scores, unique_keys, method)
    order = np.argsort(-values, kind='stable')
    return unique_keys[order], values[order]


def profile_rankings(scores, states, values, method):
    """Approximate max-profile support across found contexts, not probabilities."""
    rankings = np.empty((16, 256), np.float64)
    for i in range(16):
        field = local_field(scores, states, i, method)
        existing = field[np.arange(len(states)), states[:, i]]
        rankings[i] = np.max(values[:, None] - existing[:, None] + field, axis=0)
    if method == 'simple':
        rankings[0] = -np.inf
        rankings[0, 0] = values.max()
    return rankings


def write_rankings(path, support, label):
    with Path(path).open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['byte_position', 'rank', 'candidate_hex', 'support', 'meaning'])
        for i in range(16):
            order = np.argsort(-support[i], kind='stable')
            for rank, candidate in enumerate(order, 1):
                writer.writerow([i, rank, f'{candidate:02x}', float(support[i, candidate]), label])


def preprocess(ct, timing, args):
    mask = np.isfinite(timing)
    threshold = float(np.quantile(timing[mask], args.upper_quantile))
    mask &= timing <= threshold
    indices = np.flatnonzero(mask)
    if args.max_traces and len(indices) > args.max_traces:
        rng = np.random.default_rng(args.seed)
        indices = np.sort(rng.choice(indices, args.max_traces, replace=False))
    if len(indices) < 1000:
        raise ValueError('At least 1000 selected traces are required for this analysis')
    t = timing[indices].astype(np.float64)
    if args.transform == 'log':
        t = np.log1p(t)
    elif args.transform == 'rank':
        _, inv, counts = np.unique(t, return_inverse=True, return_counts=True)
        midranks = (np.cumsum(counts) - 0.5 * counts) / len(t)
        t = midranks[inv]
    t -= t.mean()
    sd = t.std()
    if sd <= 0:
        raise ValueError('The timings have no variation')
    t /= sd
    return ct[indices], t, {'selected_traces': len(indices), 'total_traces': len(timing),
                           'upper_quantile': args.upper_quantile,
                           'upper_threshold_cycles': threshold, 'transform': args.transform}


def recover(scores, method, pt, ct, args):
    states, values = optimize(scores, method, args.restarts, args.sweeps, args.seed)
    states = states[:args.max_candidates]
    values = values[:args.max_candidates]
    sample_indices = np.unique(np.linspace(0, len(pt) - 1, min(16, len(pt)), dtype=int))
    tested = 0
    for state in states:
        anchors = range(256) if method == 'simple' else [0]
        for anchor in anchors:
            candidate = state ^ anchor if method == 'simple' else state
            tested += 1
            key = verify_round10(candidate, pt[sample_indices], ct[sample_indices], checks=len(sample_indices))
            if key is not None:
                return states, values, {'status': 'recovered', 'master_key_hex': key.hex(),
                    'round10_key_hex': candidate.tobytes().hex(),
                    'candidate_keys_tested': tested, 'verification_pairs': len(sample_indices)}
    return states, values, {'status': 'not_recovered', 'candidate_keys_tested': tested,
                           'reason': 'No searched candidate matched the verification pairs.'}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset', type=Path, help='Local .json or .tar.gz dataset; no downloads are performed')
    p.add_argument('--method', choices=['simple', 'expanded', 'both'], default='both')
    p.add_argument('--output', type=Path, default=Path('results'))
    p.add_argument('--upper-quantile', type=float, default=1.0, help='Keep timings up to this quantile; default: all')
    p.add_argument('--transform', choices=['raw', 'log', 'rank'], default='raw')
    p.add_argument('--statistic', choices=['welch', 'difference'], default='welch')
    p.add_argument('--offset', type=int, choices=range(0, 32, 4), default=0,
                   help='Te4 address modulo 32; default assumes a 32-byte-aligned table')
    p.add_argument('--max-traces', type=int, default=0, help='Seeded random subset; 0 uses all selected traces')
    p.add_argument('--restarts', type=int, default=128)
    p.add_argument('--sweeps', type=int, default=40)
    p.add_argument('--max-candidates', type=int, default=64)
    p.add_argument('--seed', type=int, default=622)
    p.add_argument('--evaluate', action='store_true', help='After the attack, compare results with provided training keys')
    p.add_argument('--save-scores', action='store_true', help='Save pair scores in an additional .npz file')
    p.add_argument('--verify-all', action='store_true', help='If recovered, verify every input plaintext/ciphertext pair')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not 0 < args.upper_quantile <= 1:
        raise SystemExit('--upper-quantile must be in (0, 1]')
    if min(args.restarts, args.sweeps, args.max_candidates) < 1 or args.max_traces < 0:
        raise SystemExit('Search settings must be positive; --max-traces must be nonnegative')
    self_test()
    started = time.monotonic()
    log('Loading local dataset:', args.dataset)
    pt, ct, timing, metadata = load_dataset(args.dataset)
    if len(pt) < 2:
        raise SystemExit('At least two verification pairs are required')
    selected_ct, selected_t, preprocessing = preprocess(ct, timing, args)
    log('Preprocessing:', preprocessing)
    methods = ['simple', 'expanded'] if args.method == 'both' else [args.method]
    args.output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with args.dataset.open('rb') as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    reports = []
    for method in methods:
        method_start = time.monotonic()
        log('\nMETHOD:', method)
        scores = build_scores(selected_ct, selected_t, method, args.offset, args.statistic)
        states, values, result = recover(scores, method, pt, ct, args)
        support = profile_rankings(scores, states, values, method)
        meaning = 'round10_byte' if method == 'expanded' else 'round10_byte_XOR_round10_byte_0'
        rank_path = args.output / f'{method}_rankings.csv'
        write_rankings(rank_path, support, meaning)
        result.update(method=method, dataset=args.dataset.name,
            dataset_sha256=digest.hexdigest(), preprocessing=preprocessing,
            statistic=args.statistic, table_offset=args.offset if method == 'expanded' else None,
            restarts=args.restarts, sweeps=args.sweeps, seed=args.seed,
            retained_search_states=len(states), rankings_file=rank_path.name,
            ranking_note='Approximate max-profile support over searched contexts; ties ordered by byte value. '
                         'These are not original/master-key byte rankings.',
            best_search_state_hex=states[0].tobytes().hex(), best_score=float(values[0]))
        if result['status'] == 'recovered' and args.verify_all:
            key = bytes.fromhex(result['master_key_hex'])
            for start in range(0, len(pt), 16384):
                expected = encrypt_blocks(key, pt[start:start + 16384])
                if not np.array_equal(expected, ct[start:start + 16384]):
                    raise RuntimeError('Full-dataset verification failed; refusing to report recovery')
            result['verification_pairs'] = len(pt)
        if args.evaluate:
            if 'k' in metadata:
                result['evaluation'] = {'master_key_matches': result.get('master_key_hex') == metadata['k'].lower()}
                if 'k10' in metadata:
                    truth = np.frombuffer(bytes.fromhex(metadata['k10']), np.uint8)
                    if method == 'simple':
                        truth = truth ^ truth[0]
                    ranks = [1 + int(np.sum(support[i] > support[i, truth[i]])) for i in range(16)]
                    result['evaluation'].update(reference=meaning, correct_value_ranks=ranks,
                                                mean_rank=float(np.mean(ranks)))
            else:
                result['evaluation'] = {'available': False, 'reason': 'No reference master key in this dataset'}
        if args.save_scores:
            np.savez_compressed(args.output / f'{method}_scores.npz', scores=scores)
        result['elapsed_seconds'] = round(time.monotonic() - method_start, 3)
        with (args.output / f'{method}_result.json').open('w') as f:
            json.dump(result, f, indent=2)
        log(json.dumps(result, indent=2))
        reports.append(result)
        del scores
    log(f'Finished in {time.monotonic() - started:.1f}s. Output:', args.output)
    return 0 if any(r['status'] == 'recovered' for r in reports) else 2


if __name__ == '__main__':
    raise SystemExit(main())
