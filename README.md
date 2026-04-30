# Formal Verification of TinyTaxiNet using Marabou 2.0

CS 6315 (Automated Verification) class project. Formally verifies TinyTaxiNet,
a compact open-source surrogate for Boeing's runway-following perception
network, against six properties: output safety bounds (P1), local correctness
(P2), robustness to input perturbations (P3), directional correctness (P4),
deadzone behavior (P5), and per-axis reachable output bounds (P6).

The full write-up (motivation, methodology, results, related work,
limitations) is in `final_report.pdf`. This README documents the code and
artifacts.

## Contents

### VNN-LIB export

`vnnlib/` contains the 119 baseline specifications in standard SMTLIB2 format
plus an `instances.csv` mapping each spec to the network. Composition:

| Property | Files |
| --- | --- |
| P1       | 4 sub-queries (`p1a`-`p1d`) |
| P2       | 20 per-image files (`p2_img01`-`p2_img20`) |
| P3       | 40 files (`p3_eps005_img*`, `p3_eps010_img*`) |
| P4       | 2 regional files (`p4_left`, `p4_right`) |
| P5       | 1 conjunctive file (`p5_deadzone`) |
| P6       | 52 binary-search midpoints (`p6_*_iter*`) |

Every file embeds a preprocessing header (no `.nnet` normalization, Marabou
`normalize=False`, dataset mean-centered to 0.5 and clipped to [0, 1]) and the
caveat that `mean(x)=0.5` is **not** enforced in the baseline files.

### Verification scripts

| Script | Property |
| --- | --- |
| `verify_p1_p2_v2.py` | P1 (global box) and P2 (per-image L_inf) |
| `verify_prophecy_p1.py` | P1-patterned (activation-pattern partitioning, 24 patterns) |
| `verify_p3_p4.py` | P3 (unconstrained baseline) and P4 (regional sign) |
| `verify_p5_p6.py` | P5 (deadzone) and P6 (binary-search reachable bounds) |
| `verify_mean_constrained.py` | Mean-constrained reruns of P1, P4, P5 |
| `verify_p3_mean_constrained.py` | P3 with `mean(x)=0.5` and clipping enforced |
| `verify_p3_certified_radius.py` | Per-image binary search for the P3 safety radius |
| `verify_p3_label_box.py` | P3 over per-image label-conditioned per-pixel boxes |
| `verify_p3_local_pca.py` | P3 over per-image 8-D local-PCA regions |

### Plausibility-constrained pipeline

| Script | Purpose |
| --- | --- |
| `train_subset_autoencoders.py` | Train one autoencoder per input subset (centered, left, right, off-center) |
| `build_subset_onnx_with_sum.py` | Compose subset decoder with TinyTaxiNet into an ONNX network with auxiliary pixel-sum output |
| `verify_ae_with_mean_constraint.py` | Verify P1/P4/P5 over the AE-composed regions with `mean(x)=0.5` enforced |
| `verify_ae_strong_wrong_sign.py` | P4 threshold sweep on the AE-composed regions |
| `multi_seed_stability.py` | Retrain autoencoders at seeds 7, 13 and rerun the AE-composed verifications |
| `train_plausibility_classifier.py` | Train the real-vs-synthetic plausibility classifier |
| `score_counterexamples.py`, `rescore_mean_constrained_witnesses.py` | Score SAT witnesses against the plausibility classifier |
| `validate_witnesses_nearest_neighbor.py` | Compare AE-composed P4 witnesses to nearest real images |

### Calibration and analysis

| Script | Purpose |
| --- | --- |
| `calibrate_perturbation_scale.py` | Compute the perturbation calibration table (per-pixel std, same-state-NN L_inf, random-pair L_inf) |
| `analyze_box_coverage.py` | Per-pixel-box coverage statistics across the four data subsets |
| `analyze_p2_three_way.py` | P2 nominally-wrong / fragile / robust categorization |
| `analyze_margin_vs_radius.py` | Spearman correlation between nominal margin and certified radius (P3) |
| `analyze_counterexamples.py` | Summary of SAT witnesses per property |
| `plot_p6_reachable.py` | Generate the P6 reachable-set visualization from Marabou-certified bounds |
| `extract_images.py` | Decode SAT witnesses to images for visual inspection |
| `validate_prophecy_surrogate.py` | Validate the flat Keras surrogate used to extract activation patterns |
| `generate_vnnlib.py` | Produce the 119-file VNN-LIB export from the verification configurations |

### Result artifacts

JSON files capture verification outcomes (status, witness, timing) and
calibration statistics. Examples: `verification_results_v2.json`,
`verification_results_p3_p4.json`, `verification_results_p5_p6.json`,
`verification_p3_mean_constrained.json`, `verification_p3_certified_radius.json`,
`verification_p3_local_pca.json`, `verification_p3_label_box.json`,
`verification_subset_autoencoder_onnx.json`, `verification_strong_wrong_sign_sweep.json`,
`multi_seed_stability_results.json`, `mean_constrained_witness_scores.json`,
`witness_nearest_neighbor_validation.json`, `perturbation_scale_calibration.json`,
`p2_three_way_categorization.json`, `p3_margin_vs_radius.json`,
`prophecy_p1_results.json`.

Trained-model artifacts: `autoencoder_*_weights.npz`,
`plausibility_classifier.npz`, and the composed ONNX networks
(`composed_*_with_sum*.onnx`).

## External dependencies (not committed)

This project depends on two external repositories that are vendored locally
during development but kept outside this commit because of their size:

- **`VerifyGAN/`** (~46 MB): the released TinyTaxiNet model, dataset
  (`SK_DownsampledGANFocusAreaData.h5`), and reference code, from
  the [Stanford Intelligent Systems Laboratory](https://github.com/sisl/VerifyGAN).
  Required by every verification script.
- **`Prophecy/`** (~289 MB): activation-pattern extraction tooling, from
  [SRI-CSL/prophecy-papers/Prophecy](https://github.com/SRI-CSL/prophecy-papers).
  Required by `verify_prophecy_p1.py` to extract the P1-patterned activation
  signatures.

Clone both into this directory before running any verification script. The
expected paths (referenced by the scripts) are:

```
VerifyGAN/models/TinyTaxiNet.nnet
VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5
```

## Verification environment

- Python 3.11
- `maraboupy` (Marabou 2.0 Python bindings)
- `numpy`, `h5py`, `tensorflow` (for autoencoder training and ONNX export),
  `tf2onnx`, `matplotlib`

A virtual environment named `prophecy_env/` is used during development and is
not committed. Reproduce with:

```
python3.11 -m venv prophecy_env
source prophecy_env/bin/activate
pip install maraboupy numpy h5py tensorflow tf2onnx matplotlib
```

## Reproducing the main results

The baseline P1-P6 results in the report come from:

```
python verify_p1_p2_v2.py
python verify_prophecy_p1.py
python verify_p3_p4.py
python verify_p5_p6.py
```

The plausibility-constrained results require training the autoencoders first:

```
python train_subset_autoencoders.py
python build_subset_onnx_with_sum.py
python verify_ae_with_mean_constraint.py
python verify_ae_strong_wrong_sign.py
python multi_seed_stability.py
python train_plausibility_classifier.py
python rescore_mean_constrained_witnesses.py
python validate_witnesses_nearest_neighbor.py
```

The P3 follow-ups (mean+clip, certified radius, label-conditioned box,
local-PCA) and the dataset calibration:

```
python calibrate_perturbation_scale.py
python verify_p3_mean_constrained.py
python verify_p3_certified_radius.py
python verify_p3_label_box.py
python verify_p3_local_pca.py
python analyze_p2_three_way.py
python analyze_margin_vs_radius.py
python analyze_box_coverage.py
```

Each script prints a summary and writes its results to a JSON file with the
matching name.

## Notes

All verification queries are over the post-preprocessed network input space
(after the SISL preprocessing pipeline of crop, resize, grayscale,
8x16 downsample, mean-centering, and clipping); we do not re-run the raw
preprocessing pipeline. SAT/UNSAT verdicts are sound only relative to the
encoded input region. See the report for the full scope-of-claims discussion
across the five input-region types.
