import numpy as np
import pytest


def test_fwht_xor_convolution_matches_direct_sum(attack):
    rng = np.random.default_rng(45)
    a, b = rng.normal(size=(2, 16))
    direct = np.array([sum(a[x] * b[x ^ k] for x in range(16)) for k in range(16)])
    transformed = attack.fwht(attack.fwht(a) * attack.fwht(b)) / 16
    np.testing.assert_allclose(transformed, direct, atol=1e-12)


@pytest.mark.parametrize('method,offset', [('simple', 0), ('expanded', 0), ('expanded', 4), ('expanded', 28)])
@pytest.mark.parametrize('statistic', ['difference', 'welch'])
def test_optimized_scores_match_direct_trace_groups(attack, method, offset, statistic):
    rng = np.random.default_rng(751)
    ct = rng.integers(0, 256, (8192, 16), dtype=np.uint8)
    timing = rng.normal(0, 3, len(ct))
    hist = attack.pair_histograms(ct, timing, 2, 13)
    scores = attack.pair_scores(*hist, method=method, offset=offset, statistic=statistic)
    for a, b in [(0, 0), (1, 255), (52, 107), (189, 120)]:
        if method == 'simple':
            guess = a ^ b
            collision = (ct[:, 2] ^ ct[:, 13]) == guess
            observed = scores[guess]
        else:
            u = attack.INV_SBOX[ct[:, 2] ^ a].astype(np.int64)
            v = attack.INV_SBOX[ct[:, 13] ^ b].astype(np.int64)
            collision = (4 * u + offset) // 32 == (4 * v + offset) // 32
            observed = scores[a, b]
        one, zero = timing[collision], timing[~collision]
        expected = zero.mean() - one.mean()
        if statistic == 'welch':
            expected /= np.sqrt(zero.var(ddof=1) / len(zero) + one.var(ddof=1) / len(one))
        assert float(observed) == pytest.approx(float(expected), abs=1e-10)


def test_cache_line_example_and_xor_relation(attack):
    ki, kj = 0x2A, 0x91
    ci, cj = 0x63 ^ ki, 0x63 ^ kj
    assert ci ^ cj == ki ^ kj
    # Different indices in the same cache line: expanded includes this event.
    cj = 0x7C ^ kj
    assert ci ^ cj != ki ^ kj
    assert int(attack.INV_SBOX[ci ^ ki]) == 0
    assert int(attack.INV_SBOX[cj ^ kj]) == 1
    assert int(attack.INV_SBOX[ci ^ ki]) >> 3 == int(attack.INV_SBOX[cj ^ kj]) >> 3
