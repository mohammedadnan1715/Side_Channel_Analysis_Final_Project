# AES timing attacks: simple and expanded final-round methods

This package implements both methods for local AES-128 timing datasets in the
assignment's JSON format. It does not download datasets. No actual training or
blind dataset, or real target key, is included. The tests use generated traces
and standard AES test vectors.

## Files

- `simple_final_round.py`: entry point for the simple method.
- `expanded_final_round.py`: entry point for the expanded method.
- `aes_timing_attack.py`: shared scoring, search, AES, verification, and CLI code.
- `test_aes_timing_attack.py`: independent score checks and synthetic recovery tests;
  also generates an optional demonstration dataset locally.
- `requirements.txt`: NumPy dependency.
- `diagnose_training.py`: measures correct-pair ranks, timing-effect direction,
  preprocessing choices, and optional table offsets on your local known-key data.

## If a run says "not_recovered"

This means no searched candidate reproduced the known ciphertexts. It does not
identify the cause, and a large best-search score is not proof of leakage.
First run the known-key diagnostic locally:

```powershell
python diagnose_training.py t00.tar.gz --scan-offsets --output diagnostic_t00.json
python diagnose_training.py t01.tar.gz --scan-offsets --output diagnostic_t01.json
```

The default diagnostic inspects 12 fixed seeded byte pairs and up to 100,000
selected traces per setting. For a more complete check, add
`--pairs 120 --max-traces 0`; that takes longer. Diagnostic files contain
statistics, not the plaintext/ciphertext trace arrays or reference keys.

For each method, compare the correct-pair ranks to their random expectations:
128.5 for the 256 simple XOR hypotheses and 32,768.5 for the 65,536 expanded
byte-pair hypotheses. The reported mean correct-pair percentile is around 0.5
under a random ordering and improves toward 1 with better rankings. A positive
signed timing score supports the assumed faster-collision effect. These are
correlated descriptive measurements, not an aggregate statistical proof, and
pair ranks are not the assignment's final 16 key-byte ranks.

If the correct hypotheses rank approximately randomly on independent training
data, increasing the search budget alone is unlikely to help. Investigate the
leakage model, table alignment, and timing preprocessing. If the ranks are
consistently strong but recovery fails, broaden the existing heuristic search,
for example:

```powershell
python aes_timing_attack.py t00.tar.gz --method both --evaluate --restarts 512 --sweeps 80 --max-candidates 128 --output training_search
```

Use the settings that work on independent training data when you run your blind
file. This update adds diagnostics; it does not establish or fix the cause of a
particular target-data failure without seeing its output.

Keep the Python files in the same folder. Python 3.10 or later is required.

## Setup in the VS Code terminal

Extract the ZIP and open the extracted folder in VS Code. Then run:

```powershell
python -m pip install -r requirements.txt
python -m unittest -v test_aes_timing_attack
```

On Windows, use `py` instead of `python` if that is your Python launcher. These
commands also work in a Linux/macOS shell with the appropriate Python command.

## Run with your local Week 15 file

Place your own `t15_blind.tar.gz` in this folder, or pass its full path in quotes.
The program reads the JSON inside the archive directly; manual extraction is
unnecessary. A local extracted `.json` is also accepted.

Run the simple method first:

```powershell
python simple_final_round.py t15_blind.tar.gz --output results_simple
```

Run the expanded method:

```powershell
python expanded_final_round.py t15_blind.tar.gz --output results_expanded
```

Or run both with one load of the data:

```powershell
python aes_timing_attack.py t15_blind.tar.gz --method both --output results_both --verify-all
```

For a quick code check on a subset of your own data:

```powershell
python aes_timing_attack.py t15_blind.tar.gz --method both --max-traces 50000 --restarts 32 --output results_subset
```

A subset is a debugging convenience and can have insufficient leakage for
recovery. Default runs use all traces. The search is heuristic: a completed run
without recovery does not prove that recovery is impossible.

## Validate on your local training files

```powershell
python aes_timing_attack.py t00.tar.gz --method both --evaluate --output training_t00
python aes_timing_attack.py t01.tar.gz --method both --evaluate --output training_t01
```

Known key fields are not used in scoring, initialization, or candidate search.
`--evaluate` uses them only after the attack to compare the result and estimate
rankings. Tune settings on one training file, then check the other before
applying them to the blind file.

## Outputs and what they mean

For each method, the output folder contains a result JSON and a ranking CSV.

- `status: recovered`: the candidate passed AES verification. The JSON includes
  `master_key_hex` (the original AES-128 key), `round10_key_hex`, and the count of
  checked plaintext/ciphertext pairs. Normally, up to 16 evenly spaced input
  pairs are checked; `--verify-all` checks every pair using vectorized AES.
- `status: not_recovered`: no searched candidate passed verification. The
  `best_search_state_hex` is an unverified search state, not a recovered key.
- `simple_rankings.csv` ranks **relative values**
  `K10[i] XOR K10[0]`, with byte 0 fixed to zero as a reference.
- `expanded_rankings.csv` ranks **final-round key-byte values**.

CSV support values are approximate max-profile scores over the contexts found
by the search, not probabilities. They are not original/master-key byte
rankings. A tie is listed in numerical byte order; that ordering is not extra
evidence. Evaluation ranks count strictly higher scores, so tied values share a
rank.

Check with the instructor which key representation is accepted for partial
credit. Full-key verification removes that representation ambiguity; unsuccessful
round-key rankings cannot be converted independently into master-key rankings.

The program exits with code 0 if at least one requested method recovered a key,
and code 2 if none did. A nonzero exit code after a completed search can therefore
mean insufficient statistical evidence rather than a software crash.

## Mathematics and implementation

The final AES round relates one ciphertext byte to its associated table lookup:

    c_i = S(u_i) XOR K10_i

The lookup index is associated with the output byte, so this formulation already
accounts for ShiftRows; do not apply an additional inverse ShiftRows.

### Simple method

An identical-index pair has `u_i == u_j`. Because S is a permutation:

    c_i XOR c_j == K10_i XOR K10_j

For each of the 120 unordered byte pairs, timings are grouped into 256 bins by
`c_i XOR c_j`. A candidate difference is supported when its group is faster.
The default score is a signed Welch statistic:

    score = (mean_noncollision - mean_collision)
            / sqrt(var_noncollision / n_noncollision
                   + var_collision / n_collision)

Positive scores support a timing reduction. `--statistic difference` uses the
unnormalized difference of means instead.

Search variables are relative bytes `r_i = K10_i XOR K10_0`, with `r_0 = 0`.
The search maximizes:

    F(r) = sum over i < j of pair_score[i,j,r_i XOR r_j]

For each retained solution, all 256 possibilities for `K10_0` are tried. Each
complete round-10 key is inverted and checked against plaintext/ciphertext pairs.

### Expanded method

For guesses `a,b`, predict the lookup indices:

    u_i = inverse_S[c_i XOR a]
    u_j = inverse_S[c_j XOR b]

With four-byte Te4 entries and 32-byte cache lines:

    line(u) = (4*u + table_offset) // 32

The default offset is zero, which reduces this to `u >> 3`. The collision
predicate is `line(u_i) == line(u_j)`. All 65,536 candidate pairs are scored using
the same signed statistic as above. The global objective is:

    F(K10) = sum over i < j of pair_score[i,j,K10_i,K10_j]

### Efficient scoring

The expanded implementation does not allocate a traces-by-65,536 boolean array.
For each byte pair, it creates 256-by-256 histograms of counts, timing sums, and
squared timing sums, indexed by observed ciphertext bytes. It computes the
collision-group totals for every candidate using a 16-bit XOR convolution and
the fast Walsh-Hadamard transform (FWHT).

If M[x,y] is the cache-line collision predicate for inverse_S[x], inverse_S[y],
and A[x,y] is a histogram, the required totals are:

    totals[a,b] = sum over x,y of A[x,y] * M[x XOR a,y XOR b]
               = FWHT(FWHT(A) * FWHT(M)) / 65536

The tests compare this optimized result against direct trace-by-trace grouping.
The transform changes computation cost, not the attack model.

### Search and verification

The optimizer runs multiple coordinate-ascent searches. Each update tries all
256 values for one byte while holding other bytes fixed. It retains distinct
solutions and checks up to `--max-candidates` of them.

The inverse key schedule uses `w[i-4] = w[i] XOR G_i(w[i-1])`, processing words
43 down to 4. Standard AES-128 test vectors and randomized round-trip tests check
the implementation. Verification uses the original key after this inversion.

## Settings to investigate on training data

- `--upper-quantile 0.99`: discard the slowest 1% of traces; default is no cut.
- `--transform log` or `--transform rank`: alternatives to raw timings.
- `--offset 4` (or 8, 12, ..., 28): test a different Te4 base address modulo 32.
  Check alignment from the target binary when possible. This affects only the
  expanded method.
- `--restarts 512 --sweeps 80 --max-candidates 128`: broaden the heuristic search.
- `--save-scores`: save pair scores for additional analysis.

Use distinct output folders for experiments so that results do not overwrite one
another. Do not select a filtering setting solely because it produces a large
unverified score. Check recovery and ranks on independent training data.

Both methods assume cache collisions cause a measurable reduction in encryption
time. Fully warm tables, incorrect geometry, other cache behavior, and noise can
weaken or remove this signal. More search cannot repair an absent leakage signal.

## Optional synthetic demonstration

```powershell
python test_aes_timing_attack.py --make-demo demo.json
python aes_timing_attack.py demo.json --method both --evaluate --verify-all --output demo_results
```

The demonstration generates valid AES plaintext/ciphertext pairs and adds
artificial timing values from an idealized pair-collision model. Its key is
randomly generated with a fixed seed for reproducibility. Successful synthetic
recovery validates the implementation, not its effectiveness on the Week 15
hardware measurements.

## Reference

Joseph Bonneau and Ilya Mironov, "Cache-Collision Timing Attacks Against AES,"
CHES 2006, Sections 6-7 and Appendix D.
https://www.iacr.org/archive/ches2006/16/16.pdf

This is an independent educational implementation. The Welch scores, exact
FWHT acceleration, initialization, and ranking reports are implementation choices;
they are not a claim to reproduce the paper's exact software or sample counts.
