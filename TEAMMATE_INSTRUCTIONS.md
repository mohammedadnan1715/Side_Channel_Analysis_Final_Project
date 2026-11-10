# Direct master-key ranking update

The professor rejected the post-recovery conditional ranking method. The
existing recovery attacks remain valid; only the ranking stage changes.

Add these files to the repository root:

- `generate_direct_master_rankings.py`
- `tests/test_direct_master_rankings.py`

The script imports the existing `aes_timing_attack.py` and is validated against
repository revision `38a4b510df74`.

## Blind ranking run

Generate the final rankings without supplying any recovered key:

```powershell
.\.venv\Scripts\python.exe generate_direct_master_rankings.py `
  t15_blind.tar.gz --output direct_rank_t15 --method both
```

This produces:

- `simple_direct_master_rankings.csv`
- `expanded_direct_master_rankings.csv`
- `direct_master_rankings.json`
- two profile `.npz` files
- `Week15_Direct_Master_Key_Rankings.pdf`

## Post-generation evaluation

To report the positions of the already verified key, run into a separate output
folder:

```powershell
.\.venv\Scripts\python.exe generate_direct_master_rankings.py `
  t15_blind.tar.gz --output direct_rank_t15_eval --method both `
  --reference-key 7ca80f7f9263ce8d3f5e91ef746e14ee
```

The reference key is consulted only after ranking generation. It does not
construct candidates or change their ordering. The audited runs with and
without the key produced identical CSV hashes.

## Expected blind results

- Simple direct K0 ranks:
  `83, 104, 12, 87, 91, 73, 129, 97, 39, 57, 98, 149, 117, 111, 21, 239`
- Simple mean rank: `94.1875`
- Expanded direct K0 ranks: all sixteen are `1`
- Expanded mean rank: `1.0`
- Expanded best timing state:
  `fd3d727ebc2eadd4eb7c196e2cbb6105`

Expected ranking CSV SHA-256 hashes:

- Simple: `c7c14c07257cc2d25802b2b903ef3fe1f72d60bd20003d65fd12a47076762c70`
- Expanded: `ebbb17ba131e83c370709fc826c6748f0f116f91f0ed55a0ebac980f51067774`

## Method summary

The script generates complete K10 candidates from the original timing
objectives, reverses each candidate to K0, and ranks each K0 byte value by the
best original timing score among complete candidates producing that value. It
does not hold the other 15 recovered master-key bytes fixed, and plaintext/AES
verification does not order the tables.
