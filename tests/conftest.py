"""Local-data options and fixtures. Tests never download a dataset."""
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def pytest_addoption(parser):
    group = parser.getgroup('AES project verification')
    group.addoption('--dataset', default=None, help='Local .json/.tar.gz with pt, ct, and t')
    group.addoption('--results-dir', default=None, help='Folder containing both method result JSON files')
    group.addoption('--rankings-json', default=None, help='Optional master_key_rankings.json to validate')
    group.addoption('--verify-all', action='store_true', help='Check every local dataset block instead of 64 samples')


def pytest_configure(config):
    config.addinivalue_line('markers', 'local_data: requires an explicitly supplied local dataset and result files')
    config.addinivalue_line('markers', 'rankings: requires --rankings-json')
    config.addinivalue_line('markers', 'synthetic: tests recovery from artificial timing traces')
    if bool(config.getoption('--dataset')) != bool(config.getoption('--results-dir')):
        raise pytest.UsageError('Provide --dataset and --results-dir together for local key verification.')
    if config.getoption('--verify-all') and not config.getoption('--dataset'):
        raise pytest.UsageError('--verify-all requires --dataset and --results-dir.')


@pytest.fixture(scope='session')
def attack():
    try:
        return importlib.import_module('aes_timing_attack')
    except ModuleNotFoundError as exc:
        raise pytest.UsageError('Place tests/ beside aes_timing_attack.py in your project folder.') from exc


@pytest.fixture(scope='session')
def independent_aes():
    """Separate AES implementation: do not validate a cipher only against itself."""
    def encrypt(key, blocks):
        ctx = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
        data = np.asarray(blocks, dtype=np.uint8).tobytes()
        return ctx.update(data) + ctx.finalize()
    return encrypt


@pytest.fixture(scope='session')
def local_reports(pytestconfig):
    arg = pytestconfig.getoption('--results-dir')
    if not arg:
        pytest.skip('Local key verification not requested; provide --dataset and --results-dir.')
    folder = Path(arg)
    result = {}
    for method in ['simple', 'expanded']:
        path = folder / f'{method}_result.json'
        if not path.is_file():
            pytest.fail(f'Missing result file: {path}. Point --results-dir at your recovery output folder.')
        result[method] = json.loads(path.read_text(encoding='utf-8-sig'))
    return result


@pytest.fixture(scope='session')
def local_data(pytestconfig, attack):
    arg = pytestconfig.getoption('--dataset')
    if not arg:
        pytest.skip('Local trace data not supplied; no dataset is fetched automatically.')
    path = Path(arg)
    if not path.is_file():
        pytest.fail(f'Dataset path does not exist: {path}')
    pt, ct, timing, _ = attack.load_dataset(path)
    digest = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
    return {'pt': pt, 'ct': ct, 't': timing, 'sha256': digest.hexdigest(),
            'verify_all': pytestconfig.getoption('--verify-all')}


@pytest.fixture(scope='session')
def master_rankings(pytestconfig):
    arg = pytestconfig.getoption('--rankings-json')
    if not arg:
        pytest.skip('Master-key ranking checks not requested; provide --rankings-json.')
    path = Path(arg)
    if not path.is_file():
        pytest.fail(f'Master-key ranking JSON not found: {path}')
    return json.loads(path.read_text(encoding='utf-8-sig'))
