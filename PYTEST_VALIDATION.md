# Validation performed for this test add-on

Environment: Python 3.12.14 on Linux; pytest 9.2.0; NumPy 2.3.5;
cryptography 50.0.2. Windows/VS Code invocation examples are in PYTEST_README.md.

1. Default discovery without local inputs: **31 passed, 17 skipped**. Skips were
   explicitly reported for optional dataset/ranking inputs.
2. Complete suite using the independently encrypted synthetic demonstration,
   its actual successful recovery results, its conditional joint-score ranking
   tables, and `--verify-all`: **48 passed**.

The synthetic fixture contained 65,536 AES plaintext/ciphertext pairs and
artificial cache-collision timing values. Both attack styles recovered its
planted key before the local-input test path was exercised. The demonstration
generator refuses to fabricate a recovered result if that recovery fails.

No actual Week 15 traces were downloaded or processed for this validation.
Run the local-data command to independently verify your own recovered key.

The add-on does not change the attack implementation, guarantee recovery for
arbitrary traces, or determine whether a particular ranking method meets the
TA's interpretation. It checks the specified math, cryptographic consistency,
recovery behavior under the synthetic model, and supplied result/table contents.
