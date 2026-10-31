# Participant Data Pack

This directory contains the data released to participants. Every track is split into
`training/` and `eval/`. The hidden evaluation inputs are **not** included here; the
organizers release them separately at the scheduled time. No hidden labels are present in
this pack.

Integrity hashes and exact counts are in `manifest.json`.

---

## Cyber — CICIDS2017, five classes

| Partition | File | Rows |
|---|---|---:|
| Training | `Cyber/training/flows.parquet` | 456,737 |
| Eval | `Cyber/eval/flows.parquet` | 114,185 |

Columns: `record_id`, 78 numeric flow features, `label`.

| Class | Training | Eval |
|---|---:|---:|
| `BENIGN` | 240,000 | 60,001 |
| `BRUTE_FORCE` | 5,537 | 1,384 |
| `DOS` | 96,000 | 24,000 |
| `DDOS` | 57,600 | 14,400 |
| `PORT_SCAN` | 57,600 | 14,400 |

The two partitions are disjoint by source record. Flow endpoint metadata (`source_ip`,
`destination_ip`, ports, protocol, `seq`) is intentionally absent from this pack; it ships
with the hidden set because it is needed for replay policy matching.

## Health — HAM10000, seven classes

| Partition | Location | Images |
|---|---|---:|
| Training | `Health/training/{metadata.csv, images/}` | 7,762 |
| Eval | `Health/eval/{metadata.csv, images/}` | 1,942 |

Metadata columns: `image_id`, `lesion_id`, `dx`, `age`, `sex`, `localization`.

| Class | Training | Eval |
|---|---:|---:|
| `nv` | 5,212 | 1,302 |
| `mel` | 856 | 214 |
| `bkl` | 848 | 212 |
| `bcc` | 400 | 101 |
| `akiec` | 249 | 63 |
| `vasc` | 109 | 27 |
| `df` | 88 | 23 |

Splits are lesion-disjoint: no `lesion_id` appears in both partitions. Images sharing a
lesion with the hidden set were withheld from both partitions.

---

## Usage

- Train on `training/`, tune and report on `eval/`.
- `eval/` is a held-out development set, not the final ranking set. Final scores come from
  the hidden set released on the event day.
- Cyber hidden record IDs and Health hidden image IDs are different from anything in this
  pack; do not assume `eval/` identifiers carry over.
