# Neuron Quality Discriminator — v17 (frozen)

A keep/reject classifier for neuron-pair quality (Qwen2.5-0.5B-Instruct + LoRA, CPU-trained).

This repository publishes the **reproducibility package**: sha-verified corpus, eval sets,
gate scripts, and the three-dimension (text / pair_id / combo-semantic) leakage audit —
three tiers (basis63 / fitcheck_bare / val168), all three dimensions zero-intersection,
with the third tier independently measured by the reproduction party.

## Known limitations (read first)

1. **The three benchmark raw scores are single-party readings.** The independent
   reproduction party had no GPU environment and no read access to the model files;
   no third-party inference reproduction was performed. All three gates read
   *below* the historical reference level (the safe direction); a third-party
   reading can be arranged separately if needed.
2. **Three head-count conventions must be used with labels**: strict 116 (the
   criterion) / loose 118 (reference) / half-rows 2. Never quote a single bare
   number; the "151 − 33" derivation must not be used as a registered value.

## What is reproducible here

- Corpus sha-256, byte count, line count (see MANIFEST.md)
- `verify_v17_final.py` — five-item final check + three-dimension zero-intersection
  audit (no GPU required)
- Eval sets (basis63 / fitcheck_bare / val168) with source sha-256
- Training script with checkpoint/manifest/resume support
- Full gate judgment reports and leakage-audit outputs
- Independent reproduction party's leak-check script (fixed val168 branch) + its
  manifest, three-tier all-zero re-run evidence

## Layout

```
README.md                  # this file
MANIFEST.md                # all assets: sha256 / bytes / lines + original-path mapping
report/                    # signed final report + carrier table
repro/                     # 8 scripts (7 training-party + 1 independent-reproduction-party)
repro/eval_sets/           # basis63 / fitcheck_bare / val168
corpus/                    # final training corpus (sha self-verifying)
evidence/                  # gate report, leak-audit outputs, assembly audit
```

## Reproduce

1. Verify the corpus: `sha256sum corpus/v17_train.jsonl` → must equal the value in MANIFEST.md
2. Corpus-side final check (no GPU): `PYTHONUTF8=1 python repro/verify_v17_final.py`
3. Leakage audit, three tiers, independent script:
   `PYTHONUTF8=1 python repro/dsh_r91_leak_check.py --train corpus/v17_train.jsonl --basis repro/eval_sets/basis63.json --fit repro/eval_sets/fitcheck_bare79.jsonl --val168 repro/eval_sets/val168.jsonl`
   Expected: text / pair_id / combo-semantic intersections all 0 for all three tiers.
4. With a GPU: retrain per `repro/train_v01.py` (command in report §8.1) or load an
   adapter and run `repro/eval_basis63.py` / `repro/eval_v01.py`.

Full asset list with sha256, bytes, and line counts: **MANIFEST.md**.
Signed report (Chinese): `report/v17_对外终版报告_v1稿_20261021.md`.
