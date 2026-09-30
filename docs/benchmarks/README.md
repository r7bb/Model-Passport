# Accuracy benchmark

How well the entity audit finds memorized personal data in a **real open model**, measured against known ground truth. Reproduce with `python scripts/benchmark.py`, which writes the JSON next to this file.

## Setup (2026-09-29)

- **Model:** EleutherAI/pythia-160m, fine-tuned for 3 epochs (learning rate 5e-5) on an Apple M-series GPU. Training loss fell from 1.89 to 0.61.
- **Data:** 1,500 US-English documents of up to 600 characters from Nemotron-PII (NVIDIA, CC BY 4.0), about 6.5 labeled personal values each. The people in it are synthetic.
- **Split by document id:**
  - 889 trained, each repeated 1, 2, 4, or 8 times
  - 307 held out for calibration
  - 304 never seen, but audited as if they were training data
- **Audit:** 1,200 sampled entities, 5% false-discovery rate, High and Critical findings re-tested.

Since the model never saw the unseen documents, every flag on them is a false alarm. Before fine-tuning, the model has seen none of the documents, so every flag is a false alarm.

## Results

| | Synthetic look-alikes | Held-out real records |
|---|---|---|
| Before fine-tuning: flagged (all false alarms) | 27% | 11% |
| After fine-tuning: false alarms on unseen documents | 41% | 7.8% |
| After fine-tuning: trained values caught | 89% | 30% |
| &nbsp;&nbsp;seen once / 2× / 4× / 8× | 75% / 89% / 96% / 97% | 22% / 24% / 34% / 42% |
| After fine-tuning: High or Critical, trained / unseen | 30% / 10% | 14% / 2.4% |
| Membership AUC against real never-seen values | — | 0.70 |

## What this means

- **Synthetic look-alikes alone are unreliable on real models.** They flag a quarter of values in a model that has memorized nothing, because real values look more natural to a model than generated ones: its AUC between held-out real values and look-alikes is 0.66–0.77, where 0.5 would be fair. The high catch rate with look-alikes is mostly this bias. Audits now calibrate against held-out real records whenever the platform trains the model.
- **Held-out calibration is much more honest but not finished.** False alarms fall to 8–11%, and High or Critical false alarms to 2.4%, but that is still above the 5% target. Catching 30% of memorized values, rising with repetition, is in line with published entity-level results on small fine-tuned models. The next steps are:
  - choosing the attack method on held-out data as well
  - a larger calibration set, so each entity type gets its own baseline
  - requiring a confirmed finding to beat held-out values, not only look-alikes

Data: Nemotron-PII by NVIDIA, licensed under CC BY 4.0 (<https://huggingface.co/datasets/nvidia/Nemotron-PII>).
