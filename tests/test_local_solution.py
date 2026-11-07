import numpy as np
import pytest

pytestmark = pytest.mark.local_data


def test_both_recovery_reports_agree(local_reports):
    for method, report in local_reports.items():
        assert report['status'] == 'recovered', f'{method} did not report a verified recovered key'
        assert report['method'] == method
        assert len(bytes.fromhex(report['master_key_hex'])) == 16
        assert len(bytes.fromhex(report['round10_key_hex'])) == 16
    for field in ['master_key_hex', 'round10_key_hex', 'dataset_sha256', 'dataset']:
        assert local_reports['simple'][field] == local_reports['expanded'][field]


@pytest.mark.parametrize('method', ['simple', 'expanded'])
def test_local_dataset_matches_recovery_report(local_data, local_reports, method):
    report = local_reports[method]
    assert local_data['sha256'] == report['dataset_sha256'], 'Wrong dataset: SHA256 does not match this result file'
    assert len(local_data['pt']) == report['preprocessing']['total_traces']
    assert 1 <= report['verification_pairs'] <= len(local_data['pt'])


@pytest.mark.parametrize('method', ['simple', 'expanded'])
def test_reported_master_and_round10_keys_are_consistent(attack, local_reports, method):
    report = local_reports[method]
    master = bytes.fromhex(report['master_key_hex'])
    final = bytes.fromhex(report['round10_key_hex'])
    assert attack.expand_key(master)[10] == final
    assert attack.invert_round10(final) == master


@pytest.mark.parametrize('method', ['simple', 'expanded'])
def test_recovered_key_with_independent_aes(local_data, local_reports, independent_aes, method):
    report = local_reports[method]
    assert report['status'] == 'recovered'
    master = bytes.fromhex(report['master_key_hex'])
    pt, ct = local_data['pt'], local_data['ct']
    if local_data['verify_all']:
        for start in range(0, len(pt), 16384):
            expected = independent_aes(master, pt[start:start + 16384])
            assert expected == ct[start:start + 16384].tobytes(), f'Ciphertext mismatch in batch starting at {start}'
    else:
        selected = np.unique(np.linspace(0, len(pt) - 1, min(64, len(pt)), dtype=int))
        assert independent_aes(master, pt[selected]) == ct[selected].tobytes()


@pytest.mark.parametrize('method', ['simple', 'expanded'])
def test_corrupted_key_fails_on_local_data(attack, local_data, local_reports, method):
    report = local_reports[method]
    final = np.frombuffer(bytes.fromhex(report['round10_key_hex']), np.uint8).copy()
    final[7] ^= 1
    assert attack.verify_round10(final, local_data['pt'][:8], local_data['ct'][:8]) is None
