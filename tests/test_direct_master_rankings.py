import numpy as np

import aes_timing_attack as attack
import generate_direct_master_rankings as direct


def test_vectorized_key_schedule_matches_scalar_implementation():
    rng = np.random.default_rng(1701)
    masters = rng.integers(0, 256, (64, 16), dtype=np.uint8)
    rounds = direct.expand_master_batch(masters)
    recovered = direct.invert_round10_batch(rounds)
    assert np.array_equal(recovered, masters)
    for i in range(16):
        assert rounds[i].tobytes() == attack.expand_key(masters[i].tobytes())[10]
        assert recovered[i].tobytes() == attack.invert_round10(rounds[i].tobytes())


def test_ranking_rows_are_complete_and_score_ordered():
    rng = np.random.default_rng(1702)
    support = rng.normal(size=(16, 256))
    counts = rng.integers(1, 20, size=(16, 256))
    tables = direct.ranking_rows(support, counts)
    assert len(tables) == 16
    for i, rows in enumerate(tables):
        assert len(rows) == 256
        assert [row['rank'] for row in rows] == list(range(1, 257))
        assert sorted(int(row['candidate_hex'], 16) for row in rows) == list(range(256))
        assert all(rows[j]['timing_support'] >= rows[j + 1]['timing_support']
                   for j in range(255))
        for row in rows:
            value = int(row['candidate_hex'], 16)
            assert row['candidate_contexts_seen'] == int(counts[i, value])


def test_profile_update_uses_best_complete_candidate_context():
    profile, counts = direct.empty_profile()
    masters = np.zeros((3, 16), dtype=np.uint8)
    masters[:, 0] = [7, 7, 9]
    values = np.array([1.0, 4.5, 3.0])
    direct.update_profile(profile, counts, masters, values)
    assert profile[0, 7] == 4.5
    assert profile[0, 9] == 3.0
    assert counts[0, 7] == 2
    assert counts[0, 9] == 1
