# Pytest checks for the AES timing project

This is an add-on for the existing project. Extract its contents into your
`aes_week15` folder so that `tests/` and `requirements-test.txt` sit alongside
`aes_timing_attack.py`. It does not replace your attack implementation or result
files. Python 3.10 or later is required.

The tests never download a dataset and do not contain a hardcoded Week 15 key.

## 1. Install the test dependencies

In the VS Code terminal, in your existing project folder, run:

```powershell
python -m pip install -r requirements-test.txt
```

This installs pytest, NumPy, and `cryptography`. ReportLab is needed for PDF
generation by the separate ranking script, but not for these tests. Using
`python -m pip` and `python -m pytest` helps select the same Python environment.

## 2. Test the implementation without the actual dataset

```powershell
python -m pytest tests -v -ra -m "not local_data and not rankings"
```

This selects 31 checks:

- Standard AES-128 test vectors, including an independent AES implementation.
- Randomized encryption comparisons and forward/reverse key schedules.
- Invalid key sizes and the inverse S-box.
- FWHT scoring against direct grouping, for both statistics and multiple offsets.
- Full recovery using synthetic pair-collision timings for both methods.
- Negative controls with unrelated timings and deliberately corrupted keys.

Synthetic success tests the implementation under its assumed model. It does not
establish recovery on real hardware measurements.

## 3. Independently verify the recovered Week 15 key

The folder passed to `--results-dir` must contain your Week 15
`simple_result.json` and `expanded_result.json`. If your output folder was
`week15_rank`, run:

```powershell
python -m pytest tests/test_local_solution.py -v -ra --dataset ".\t15_blind.tar.gz" --results-dir ".\week15_rank" --verify-all
```

This runs 9 local checks. It confirms:

- Both methods report the same complete recovered key.
- The exact input file SHA256 agrees with the recovery reports.
- The master key and final-round key agree under the AES key schedule.
- The recovered key reproduces every supplied ciphertext using the separate
  `cryptography` AES implementation.
- A deliberately corrupted key fails verification.

`--verify-all` checks every block in batches, not only the report's success flag.
Without this option, it checks up to 64 evenly spaced pairs. Full verification
loads the dataset once; allow sufficient RAM for your JSON file and trace arrays.

## 4. Test everything, including the required master-key tables

After generating `rankings_exact/master_key_rankings.json`, run:

```powershell
python -m pytest tests -v -ra --dataset ".\t15_blind.tar.gz" --results-dir ".\week15_rank" --rankings-json ".\rankings_exact\master_key_rankings.json" --verify-all --junitxml=pytest_week15_results.xml
```

This selects all 48 checks. The additional ranking checks confirm all 256
candidates appear exactly once for each of the 16 master-key bytes, rank numbers
and score ordering agree, tie-breaking is consistent, the summary is accurate,
and the recovered bytes rank first. The last check measures ranking quality;
failure there alone does not prove that AES encryption is implemented wrongly.

The ranking JSON must be the output of `generate_master_key_rankings.py`, not
the earlier `simple_rankings.csv` or `expanded_rankings.csv`. The tests accept
both explicitly labeled saved-profile and original-joint-score modes. Passing
format checks does not change the conditional interpretation of those rankings
or establish that the TA accepts a particular scoring method.

`pytest_week15_results.xml` records the test results. You can retain it as
verification evidence; submitting source code or XML was not required by the TA.

## 5. Optional: exercise all options without actual target data

```powershell
python tests/make_demo_data.py --output pytest_demo
python -m pytest tests -v -ra --dataset ".\pytest_demo\synthetic_traces.json" --results-dir ".\pytest_demo" --rankings-json ".\pytest_demo\master_key_rankings.json" --verify-all
```

This generates valid AES plaintext/ciphertext pairs using an independent cipher,
adds artificial collision-dependent timings, actually recovers the planted key
with both methods, and writes demonstration result/ranking files. Choose a new
or empty output folder. The fixture is synthetic and must not be reported as a
real Week 15 measurement or result.

## Understanding the output

- **PASSED:** the particular assertion succeeded.
- **FAILED:** inspect the named assertion; the failure message identifies the
  mismatch. It can indicate bad paths/data, inconsistent files, a coding error,
  or insufficient ranking quality depending on the test.
- **SKIPPED:** required local inputs were not supplied. This is not proof that
  the recovered key was checked.
- **ERROR during collection:** often a missing package or project module.

Running `python -m pytest tests -v -ra` without options runs the 31 general checks
and skips 17 local/ranking checks. The `-ra` flag shows the skip reasons.
Providing only one of `--dataset` and `--results-dir` produces an explicit usage
error. A supplied path that does not exist fails instead of silently skipping.

To stop after the first failure:

```powershell
python -m pytest tests -x -v --dataset ".\t15_blind.tar.gz" --results-dir ".\week15_rank" --verify-all
```

If folders differ on your computer, replace the quoted paths. In particular,
do not point at the t00/t01 output folder while checking the Week 15 dataset.
You can use `py` instead of `python` if that is your Windows Python launcher.

## File roles

- `tests/conftest.py`: options, fixtures, file loading, independent AES helper.
- `tests/test_aes_math.py`: standard vectors, key schedule, basic AES checks.
- `tests/test_collision_math.py`: cache-collision predicates and statistical math.
- `tests/test_synthetic_recovery.py`: positive and negative recovery experiments.
- `tests/test_local_solution.py`: your supplied local results and dataset.
- `tests/test_submission_rankings.py`: master-key ranking contents and consistency.
- `tests/make_demo_data.py`: optional synthetic demonstration generator.

The existing unittest files remain usable. These commands explicitly target the
new `tests/` folder so unrelated legacy tests are not collected accidentally.

References for the public test APIs:
https://docs.pytest.org/en/stable/how-to/usage.html
https://cryptography.io/en/latest/hazmat/primitives/symmetric-encryption/
