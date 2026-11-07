from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope='module')
def generated_traces(attack, independent_aes):
    rng = np.random.default_rng(7515)
    master = rng.integers(0, 256, 16, dtype=np.uint8).tobytes()
    pt = rng.integers(0, 256, (65536, 16), dtype=np.uint8)
    ct = np.frombuffer(independent_aes(master, pt), np.uint8).reshape(-1, 16).copy()
    final = np.frombuffer(attack.expand_key(master)[10], np.uint8)
    lines = attack.INV_SBOX[ct ^ final] >> 3
    hits = np.zeros(len(pt), dtype=np.int16)
    for i in range(16):
        for j in range(i + 1, 16):
            hits += lines[:, i] == lines[:, j]
    timing = 12000.0 - 30.0 * hits + rng.normal(0, 5, len(pt))
    timing = (timing - timing.mean()) / timing.std()
    return pt, ct, timing, master, final


@pytest.mark.synthetic
@pytest.mark.parametrize('method', ['simple', 'expanded'])
def test_recovery_finds_planted_key(attack, generated_traces, method):
    pt, ct, timing, master, final = generated_traces
    scores = attack.build_scores(ct, timing, method)
    settings = SimpleNamespace(restarts=32, sweeps=25, seed=622, max_candidates=32)
    _, _, result = attack.recover(scores, method, pt, ct, settings)
    assert result['status'] == 'recovered'
    assert result['master_key_hex'] == master.hex()
    assert result['round10_key_hex'] == final.tobytes().hex()


@pytest.mark.synthetic
@pytest.mark.parametrize('method', ['simple', 'expanded'])
def test_unrelated_timings_do_not_report_recovery(attack, generated_traces, method):
    pt, ct, _, _, _ = generated_traces
    n = 4096
    unrelated = np.random.default_rng(915).normal(size=n)
    scores = attack.build_scores(ct[:n], unrelated, method)
    settings = SimpleNamespace(restarts=4, sweeps=8, seed=622, max_candidates=1)
    _, _, result = attack.recover(scores, method, pt[:n], ct[:n], settings)
    assert result['status'] == 'not_recovered'
    assert 'master_key_hex' not in result


@pytest.mark.parametrize('position', [0, 7, 15])
def test_a_wrong_round_key_is_rejected(attack, generated_traces, position):
    pt, ct, _, _, final = generated_traces
    bad = final.copy()
    bad[position] ^= 1
    assert attack.verify_round10(bad, pt[:8], ct[:8]) is None
