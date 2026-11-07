# Validation of the code package

Validation used standard AES-128 vectors and generated synthetic traces.
No target dataset or real target key is included in this package.

All eight automated tests passed:

1. FIPS-197 AES-128 ciphertext and final-round-key test vectors; inverse schedule.
2. Vectorized encryption compared with the separate single-block reference.
3. FWHT-accelerated group scores compared with direct trace grouping, including
   nonzero table offsets.
4. Signed Welch scores compared with a direct variance-based calculation.
5. Deliberately incorrect round keys rejected by AES verification.
6. Full synthetic recovery using both simple and expanded methods.
7. A negative control with timings unrelated to encryption: neither method
   reported a recovered key.
8. The new training diagnostic ranks the correct synthetic pair hypotheses
   strongly and reports the expected direction of the timing effect.

An additional command-line demonstration generated 65,536 valid AES
plaintext/ciphertext pairs and artificial pair-collision timings. Both methods
recovered the generated key, and `--verify-all` reproduced all 65,536
ciphertexts. Both entry-point help commands and Python compilation also passed.

The synthetic leakage follows the assumed collision model. These checks establish
that the code implements and searches that model; they do not establish recovery
performance against the real Week 15 measurements.
