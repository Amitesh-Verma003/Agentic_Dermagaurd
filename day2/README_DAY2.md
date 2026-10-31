# Agentic DermaGuard — Day 2 (six-agent system)

All commands are run from the `health-training` folder, with the same Python
environment you used on Day 1 (torch, timm, opencv-python, scikit-learn, pandas).

If your eval pack is not at `~/Desktop/health-eval (1)/Health/eval`, set its path first:

```bash
export DG_EVAL_DIR="/path/to/Health/eval"
```

## Run order (tonight)

| # | command | what it does | time on M1 (approx.) |
|---|---|---|---|
| 1 | `python day2/train_v2.py` | Diagnosis Agent v2: efficientnet_b2 @ 288 px, 14 epochs on all 7,762 training images, checked against the eval set after every epoch | ~4–6 min / epoch |
| 1b | `python day2/train_v2.py --model efficientnet_b0 --img 256 --epochs 14` | use this instead if one epoch of step 1 takes over 8 min | ~2 min / epoch |
| 2 *(optional)* | `python day2/train_v2.py --model convnext_tiny --img 224 --lr 1e-4 --epochs 12` | a second, different model for the ensemble (usually +1–2 %) | ~4 min / epoch |
| 3 | `python day2/calibrate.py` | fits Agents 3–6 and picks the best ensemble, fusion weights and safety thresholds; writes the eval results and report | ~5–10 min |
| 4 | `python day2/run_pipeline.py --data "<hidden folder>" --name hidden` | **10:00 AM**: runs the hidden test and writes the submission CSV | ~1 min |
| 5 | `python day2/demo.py --data "<hidden folder>" --random 3` | live demo cards for the panel | seconds |

The training log prints `raw acc / f1` and `ema acc / f1` on the eval set after
every epoch, and the best checkpoint is saved automatically. You can stop with
Ctrl-C at any time: the best checkpoint so far is already saved in `day2/artifacts/`.

## Outputs (`day2/outputs/`)

- `hidden_submission.csv`: `image_id,prediction,decision`, one row per image (upload this)
- `eval_metrics.md` / `.json`: Accuracy, Macro-F1, correct/incorrect AUTO_TRIAGE, REFER and RE_IMAGE counts, confusion matrix, class-wise errors
- `calibration_report.md`: how each agent changes the result (ablation table) and what each safety rule contributes
- `*_report.html`: open it in a browser; one six-agent card per image, filters, and the full rationale for each card
- `*_agents.jsonl`: full output of every agent for every image (traceability)

## Why Day 1 was stuck at 78.9 %

1. **Full inverse-frequency class weights.** The model was pushed so hard toward the
   rare classes that nevus recall fell to 86.8 %. Nevi are 67 % of the data, so
   accuracy dropped. v2 uses soft weights (`n^-0.25`). Class balance is then fixed
   after training by per-class bias tuning, which maximises ½·Accuracy + ½·Macro-F1.
2. **20 % of the training data was never used for training**, and the internal
   validation split was random rather than lesion-disjoint. Images of the same
   lesion ended up on both sides, so the internal 84 % was optimistic. v2 trains on
   all 7,762 images and selects the model on the official lesion-disjoint eval pack.
3. **A small model at low resolution with light augmentation.** v2 uses B2 at 288 px,
   full rotations, crops, random erasing, label smoothing, warm-up + cosine LR and
   EMA weights.
4. **Single-view inference.** v2 averages 8 test-time views (flips and 90° rotations)
   and can ensemble several models, including your Day-1 B0.

## The six agents

| agent | file | output |
|---|---|---|
| A1 Image Quality | `agents/quality.py` | PASS/WARN/FAIL, score, reasons. Blur, exposure, glare, hair/obstruction and vignette, measured inside the dermoscope field. Thresholds come from training-image percentiles (FAIL ≈ worst 1 %, WARN ≈ worst 5 %) |
| A2 Diagnosis | `agents/diagnosis.py` | 7-class probabilities, top-3, confidence, margin, TTA stability, ensemble agreement |
| A3 Lesion Analysis | `agents/lesion.py` | Hair removal + segmentation → asymmetry, border irregularity, dermoscopic colours, size. Suspicion score comes from a logistic model fitted on training images. Reliability reflects segmentation quality |
| A4 Reasoning/Context | `agents/context.py` | Age, sex and site → per-class likelihood ratios (metadata model fitted on training metadata), risk indicator, missing-field flags. Becomes neutral when fields are missing |
| A5 Safety | `agents/safety.py` | 11 named triggers (quality veto, malignant class, malignant probability mass, low confidence/margin, unstable TTA, model disagreement, fusion changed the label, lesion conflict, context conflict), each with a justification. Thresholds are tuned for ≥ 96 % AUTO_TRIAGE precision on eval |
| A6 Coordinator | `agents/coordinator.py` | Fuses the evidence: log P_cnn + w_ctx·log LR_ctx + w_les·logit(suspicion) + bias. Then decides: RE_IMAGE if A5 vetoes → REFER if A5 flags → otherwise AUTO_TRIAGE. Returns the final label, decision, agreement table and a one-line-per-agent rationale |

Every image always gets a diagnosis prediction, whatever the decision.

## Notes

- The hidden images are only ever used for inference; nothing is trained on them.
- If the hidden `metadata.csv` has a `lesion_id` column and calibration found that
  pooling images of the same lesion helps, predictions are pooled across those
  images (no labels are used).
- To change how cautious the safety policy is: `python day2/calibrate.py --target-precision 0.98`
  (more REFERs, fewer AUTO_TRIAGE errors). This is fast because CNN outputs are cached.
