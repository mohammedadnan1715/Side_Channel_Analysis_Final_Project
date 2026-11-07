import numpy as np
import pytest


VECTORS = [
    ('000102030405060708090a0b0c0d0e0f', '00112233445566778899aabbccddeeff',
     '69c4e0d86a7b0430d8cdb78070b4c55a'),
    ('2b7e151628aed2a6abf7158809cf4f3c', '6bc1bee22e409f96e93d7e117393172a',
     '3ad77bb40d7a3660a89ecaf32466ef97'),
    ('00000000000000000000000000000000', '00000000000000000000000000000000',
     '66e94bd4ef8a2c3b884cfa59ca342b2e'),
]


@pytest.mark.parametrize('key_hex,pt_hex,ct_hex', VECTORS)
def test_known_aes128_vectors(attack, independent_aes, key_hex, pt_hex, ct_hex):
    key, pt, expected = map(bytes.fromhex, [key_hex, pt_hex, ct_hex])
    blocks = np.frombuffer(pt, np.uint8).reshape(1, 16)
    assert independent_aes(key, blocks) == expected
    assert attack.encrypt_block(key, pt) == expected
    assert attack.encrypt_blocks(key, blocks).tobytes() == expected


def test_fips197_round10_key(attack):
    master = bytes.fromhex('000102030405060708090a0b0c0d0e0f')
    final = bytes.fromhex('13111d7fe3944a17f307a78b4d2b30c5')
    assert attack.expand_key(master)[10] == final
    assert attack.invert_round10(final) == master


@pytest.mark.parametrize('seed', range(4))
def test_random_keys_against_independent_aes(attack, independent_aes, seed):
    rng = np.random.default_rng(seed)
    key = rng.integers(0, 256, 16, dtype=np.uint8).tobytes()
    blocks = rng.integers(0, 256, (32, 16), dtype=np.uint8)
    rks = attack.expand_key(key)
    assert len(rks) == 11 and all(len(rk) == 16 for rk in rks)
    assert attack.invert_round10(rks[10]) == key
    expected = independent_aes(key, blocks)
    assert attack.encrypt_blocks(key, blocks).tobytes() == expected
    assert b''.join(attack.encrypt_block(key, p.tobytes()) for p in blocks) == expected


@pytest.mark.parametrize('bad_length', [0, 15, 17, 24, 32])
def test_aes128_key_lengths_are_enforced(attack, bad_length):
    with pytest.raises(ValueError):
        attack.expand_key(b'\x00' * bad_length)
    with pytest.raises(ValueError):
        attack.invert_round10(b'\x00' * bad_length)


def test_inverse_sbox_is_correct(attack):
    values = np.arange(256, dtype=np.uint8)
    np.testing.assert_array_equal(attack.INV_SBOX[attack.SBOX], values)
    np.testing.assert_array_equal(attack.SBOX[attack.INV_SBOX], values)
