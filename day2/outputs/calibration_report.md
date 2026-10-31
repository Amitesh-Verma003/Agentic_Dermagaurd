# DermaGuard calibration report

train = 7762 images, eval = 1942 images

## Agent 1 – quality thresholds (from training-image percentiles; FAIL≈worst 1 %, WARN≈worst 5 %)

- blur: warn 39.183, fail 26.778
- dark: warn 90.000, fail 60.000
- bright: warn 232.613, fail 242.449
- glare: warn 0.020, fail 0.060
- obstruction: warn 0.080, fail 0.150
- vignette: warn 0.400, fail 0.600

## Agent 3 – lesion suspicion model

segmented 7534/7762 training images; malignant-vs-benign CV AUC = 0.807; levels LOW<0.29≤MODERATE<0.65≤HIGH
strongest cues: c_red (-0.69), contrast (-0.62), solidity (-0.56), asym_mean (+0.52), compactness (-0.50), border_sharpness (+0.46)

A3 on eval: segmented 1876/1942, AUC malignant = 0.811
A4 on eval: metadata-only AUC malignant = 0.806

## Agent 2 – diagnosis models on eval

| model | no TTA acc | no TTA F1 | TTA acc | TTA F1 |
|---|---|---|---|---|
| best_diagnosis_agent.pth (efficientnet_b0@224) | 0.7889 | 0.6787 | 0.7889 | 0.6787 |
| efficientnet_b0_256.pth (efficientnet_b0@256) | 0.8285 | 0.7090 | 0.8285 | 0.7090 |

Selected ensemble: ['efficientnet_b0@224', 'efficientnet_b0@256'] → acc 0.8311, macro-F1 0.7371

## Agent 6 – how each agent changes the diagnosis (eval set)

| stage | accuracy | macro-F1 |
|---|---|---|
| Day-1 baseline (reported) | 0.7889 | – |
| A2 ensemble (CNN + TTA) | 0.8311 | 0.7371 |
| + A4 context fusion | 0.8326 | 0.7328 |
| + A3 lesion fusion | 0.8332 | 0.7363 |
| + per-class bias (final fusion) | 0.8429 | 0.7637 |
| + pool images of the same lesion | 0.8502 | 0.7739 |
| final, 5-fold cross-validated (honest estimate) | 0.8337 | 0.7351 |

fusion: w_ctx=0.5, w_les=0.5, bias={'akiec': -0.3, 'bcc': -1.7, 'bkl': 0.2, 'df': -0.6, 'mel': -0.3, 'nv': 0.0, 'vasc': -1.5}, lesion pooling=on

## Agent 5 – safety policy (target AUTO_TRIAGE precision ≥ 0.96)

tau_conf=0.45, tau_margin=0.0, tau_mal=0.25, tau_tta=0.5, tau_ctx=0.95, quality_uncertain_conf=0.6, use_model_disagree=False, use_fusion_changed=True

AUTO_TRIAGE 1170 (correct 1128, incorrect 42, precision 0.964) · REFER 700 · RE_IMAGE 72

| safety rule switched off | extra auto-triaged | of which wrong |
|---|---|---|
| LOW_CONFIDENCE | 1 | 1 |
| MALIGNANT_MASS | 37 | 20 |
| LOW_MARGIN | 0 | 0 |
| TTA_UNSTABLE | 2 | 0 |
| MODEL_DISAGREE | 0 | 0 |
| FUSION_CHANGED | 20 | 6 |
| CONTEXT_CONFLICT | 42 | 17 |
| LESION_CONFLICT | 83 | 9 |

Saved /Users/vyomvarshney2005/Desktop/health-training/day2/artifacts/calibration.json
