import math

import pytest

pytestmark = pytest.mark.rankings


def test_master_ranking_key_schedule(attack, master_rankings):
    assert 'master-key' in master_rankings['ranking_scope']
    master = bytes.fromhex(master_rankings['master_key_hex'])
    assert len(master) == 16
    assert attack.expand_key(master)[10].hex() == master_rankings['round10_key_hex']
    assert master_rankings['score_mode'] in ['saved_profile_surrogate', 'original_joint_score']


@pytest.mark.parametrize('method', ['simple', 'expanded'])
def test_all_256_values_for_every_master_byte(master_rankings, method):
    info = master_rankings['methods'][method]
    assert info['candidate_count'] == 4096
    assert len(info['rankings']) == 16
    for i, rows in enumerate(info['rankings']):
        assert len(rows) == 256, f'{method}: byte {i} has an incomplete candidate list'
        assert [r['rank'] for r in rows] == list(range(1, 257))
        assert sorted(int(r['candidate_hex'], 16) for r in rows) == list(range(256))
        assert all(len(r['candidate_hex']) == 2 for r in rows)
        assert all(math.isfinite(r['score_delta']) for r in rows)
        for previous, current in zip(rows, rows[1:]):
            assert previous['score_delta'] >= current['score_delta']
            if previous['score_delta'] == current['score_delta']:
                assert int(previous['candidate_hex'], 16) < int(current['candidate_hex'], 16)


@pytest.mark.parametrize('method', ['simple', 'expanded'])
def test_rank_summary_matches_actual_tables(master_rankings, method):
    info = master_rankings['methods'][method]
    master = bytes.fromhex(master_rankings['master_key_hex'])
    ranks = [next(row['rank'] for row in info['rankings'][i]
                  if int(row['candidate_hex'], 16) == master[i]) for i in range(16)]
    assert ranks == info['correct_value_ranks']
    assert sum(ranks) / 16 == pytest.approx(info['mean_correct_value_rank'])


@pytest.mark.parametrize('method', ['simple', 'expanded'])
def test_recovered_master_bytes_rank_first(master_rankings, method):
    # A ranking-quality check, not a general theorem about arbitrary noisy data.
    master = bytes.fromhex(master_rankings['master_key_hex'])
    for i, rows in enumerate(master_rankings['methods'][method]['rankings']):
        assert int(rows[0]['candidate_hex'], 16) == master[i], (
            f'{method}: recovered byte {i} is not ranked first. Inspect scoring; '
            'this does not by itself mean the AES implementation is wrong.')


def test_ranking_key_matches_recovery_reports(master_rankings, local_reports):
    for report in local_reports.values():
        assert master_rankings['master_key_hex'] == report['master_key_hex']
        assert master_rankings['round10_key_hex'] == report['round10_key_hex']
        assert master_rankings['dataset_sha256'] == report['dataset_sha256']
