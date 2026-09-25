# MP: Model Passport

[![CI](https://github.com/r7bb/Model-Passport/actions/workflows/ci.yml/badge.svg)](https://github.com/r7bb/Model-Passport/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Find the personal data an AI model has memorized, remove it, prove it's gone, and only then release the model.**

## Why it exists

AI language models learn from huge amounts of text, and that text often contains people's names, emails, phone numbers, card numbers, and other personal details. Models can **memorize individual details** and repeat them later.

- **The research gap:** checks that ask "was this whole document in the training data?" miss single memorized details. Research published in 2026 (EL-MIA, LREC 2026) shows how to test each detail on its own.
- **The market gap:** investors and buyers of AI companies now check where training data came from, and whether it creates privacy risk.

MP checks every sensitive detail, removes the risk, verifies the fix, and records every step in a tamper-proof passport.

## How it works

```
 1. DETECT                     2. REMEDIATE                  3. VERIFY & RELEASE
 ─────────                     ────────────                  ───────────────────
 Register model and data   →   Replace risky details     →   Test on developer endpoints
 Test every sensitive          with realistic fakes          Re-test, plus human testers
 detail for memorization       Retrain the model             and AI probing
 Flag risky details            Save as a new version         Approve and release, with
 (Critical can block release)  (old versions are kept)       rollback and a kill switch
```

Every step is signed and logged, so an auditor, investor, or buyer can check the whole history.

## What it checks

| Check | In plain words |
|---|---|
| Memorized details | For each name, email, card number, and so on in the training data: does the model prefer the real value over look-alike fakes? If so, it memorized it. |
| Leakage on request | For models behind an API: does the model write out a real value when shown the text that came before it? |
| Risk level | Each finding gets a risk score from 0 to 10 and a level (Low, Medium, High, Critical), based on how certain the evidence is and how harmful that kind of detail is if leaked. |
| Personal data in datasets | Are there names, emails, phone numbers, or ID numbers in tables of training data? Could someone single out a person? |
| Tampering | Has the model, the data, or anything else changed since it was approved? |
| File safety | Could the model file run harmful code when opened? Are any software parts known to be unsafe? |
| Changing data | Is new data still similar enough to what the model was trained on? |

## See it in action

Test a model, clean the data, retrain, and test again. This is real output on made-up support tickets (commands shortened; the exact ones are in SETUP.md):

```
$ passport llm audit --model models/v1 --corpus data/corpus.jsonl
entities audited: 300; attack AUC 0.990, TPR at 1% FPR 0.707 (chance: AUC 0.5, TPR 0.01)
severity: critical 35, high 225, medium 14, low 26
  [CRITICAL] CREDITCARDNUMBER **************02         risk  9.9  record r375  confirmed, exposure 6.7 bits

$ passport llm sanitize --corpus data/corpus.jsonl --audit reports/v1.json --out data/corpus.v2.jsonl
$ passport llm finetune --corpus data/corpus.v2.jsonl --out models/v2
$ passport llm audit --model models/v2 --corpus data/corpus.jsonl
entities audited: 300; attack AUC 0.519, TPR at 1% FPR 0.003 (chance: AUC 0.5, TPR 0.01)
severity: critical 0, high 1, medium 0, low 269

# one username still leaks, so round two: sanitize, retrain as v3, audit
entities audited: 300; attack AUC 0.521, TPR at 1% FPR 0.027 (chance: AUC 0.5, TPR 0.01)
severity: critical 0, high 0, medium 0, low 268
```

How to read the rounds:

- **v1** had memorized 35 card numbers, IBANs, and ID numbers (Critical) and 225 other details (High).
- **v2:** after cleaning and retraining, the attack was back to chance, except for one username. It had looked safe in v1, so it wasn't cleaned then.
- **v3:** one more round removed it.

Every High or Critical finding is re-tested with fresh evidence before it counts. That's how v3 correctly ignored a one-off statistical fluke.

The passport also covers tabular models, with a readable report and a dashboard:

| Passed | Failed |
|---|---|
| ![Passing report](https://raw.githubusercontent.com/r7bb/Model-Passport/main/docs/images/report-pass.png) | ![Failing report](https://raw.githubusercontent.com/r7bb/Model-Passport/main/docs/images/report-fail.png) |

| How it was built | Privacy results |
|---|---|
| ![Pipeline](https://raw.githubusercontent.com/r7bb/Model-Passport/main/docs/images/dashboard-pipeline.png) | ![Privacy](https://raw.githubusercontent.com/r7bb/Model-Passport/main/docs/images/dashboard-privacy.png) |

## What's ready, and what's coming

| Ready now | Being built |
|---|---|
| Memorization testing of language models, for open models and API models | The web app: a console for administrators, and a dashboard for each organization |
| Risk scores and a release gate based on industry practice | Organizations with separate data, and roles (admin, engineer, auditor, tester, reviewer) |
| Cleaning risky details and retraining | Test endpoints, approvals, release, rollback, and a kill switch |
| Signed passports, version history, and tamper checks | Reports for investors and buyers (due diligence) |
| Checks for tabular models and datasets, drift monitoring, and a GitHub Action | Cloud deployment (Kubernetes, Terraform) |

See [ROADMAP.md](https://github.com/r7bb/Model-Passport/blob/main/ROADMAP.md) for the full plan.

## Try it

Instructions are in **[SETUP.md](https://github.com/r7bb/Model-Passport/blob/main/SETUP.md)**. They cover installing, testing a language model, the tabular demo, using your own data, and troubleshooting. The built-in examples use made-up data only, with no real people.

## Learn more

- [SETUP.md](https://github.com/r7bb/Model-Passport/blob/main/SETUP.md): installation, commands, and technical details
- [ROADMAP.md](https://github.com/r7bb/Model-Passport/blob/main/ROADMAP.md): what's done and what's next
- Research: EL-MIA (Satvaty, Verberne, and Turkmen, [LREC 2026](https://aclanthology.org/2026.lrec-1.362/)) and AIPassport (Kalokyri et al., [arXiv 2506.22358](https://arxiv.org/abs/2506.22358))

Free to use under the MIT license ([LICENSE](https://github.com/r7bb/Model-Passport/blob/main/LICENSE)). See [THIRD_PARTY_NOTICES.md](https://github.com/r7bb/Model-Passport/blob/main/THIRD_PARTY_NOTICES.md) for the open-source software it uses.
