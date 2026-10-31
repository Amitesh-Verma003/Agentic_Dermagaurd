# DermaGuard results – eval

| metric | value |
|---|---|
| images | 1942 |
| **Accuracy** | **0.8502** |
| **Macro-F1** | **0.7739** |
| Correct AUTO_TRIAGE | 1128 |
| Incorrect AUTO_TRIAGE | 42 |
| AUTO_TRIAGE precision | 0.964 |
| REFER | 700 |
| RE_IMAGE | 72 |

## Confusion matrix (rows = true, cols = predicted)

| true \ pred | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| **akiec** | 45 | 5 | 5 | 0 | 6 | 2 | 0 |
| **bcc** | 3 | 81 | 10 | 0 | 4 | 3 | 0 |
| **bkl** | 9 | 14 | 133 | 2 | 28 | 26 | 0 |
| **df** | 3 | 2 | 0 | 16 | 2 | 0 | 0 |
| **mel** | 1 | 3 | 26 | 0 | 120 | 64 | 0 |
| **nv** | 2 | 0 | 24 | 0 | 46 | 1230 | 0 |
| **vasc** | 0 | 0 | 0 | 0 | 1 | 0 | 26 |

## Class-wise errors

| class | support | correct | errors | precision | recall | F1 | most confused with |
|---|---|---|---|---|---|---|---|
| akiec | 63 | 45 | 18 | 0.714 | 0.714 | 0.714 | mel (6), bcc (5), bkl (5) |
| bcc | 101 | 81 | 20 | 0.771 | 0.802 | 0.786 | bkl (10), mel (4), akiec (3) |
| bkl | 212 | 133 | 79 | 0.672 | 0.627 | 0.649 | mel (28), nv (26), bcc (14) |
| df | 23 | 16 | 7 | 0.889 | 0.696 | 0.780 | akiec (3), bcc (2), mel (2) |
| mel | 214 | 120 | 94 | 0.580 | 0.561 | 0.570 | nv (64), bkl (26), bcc (3) |
| nv | 1302 | 1230 | 72 | 0.928 | 0.945 | 0.936 | mel (46), bkl (24), akiec (2) |
| vasc | 27 | 26 | 1 | 1.000 | 0.963 | 0.981 | mel (1) |
