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

These screenshots are from the real app, running on made-up data (no real people). A made-up clinic, "Northwind Health", trained a support assistant on 300 made-up tickets. MP found what the model had memorized, cleaned the data, retrained, checked again, and released the fixed version.

**1. See where every model stands.** Each organization gets its own dashboard, showing models in each stage and anything blocked by the kill switch.

![Organization dashboard](https://raw.githubusercontent.com/r7bb/Model-Passport/main/docs/images/web-dashboard.png)

**2. See exactly what leaked.** Every personal detail the model memorized is listed from riskiest to least risky, with its level and whether a second test confirmed it. Real values are never shown, only masked ones.

![Audit findings](https://raw.githubusercontent.com/r7bb/Model-Passport/main/docs/images/web-findings.png)

**3. Fix it, and keep the history.** The first version (1.0.0) had memorized 83 critical and 465 high-risk details. An attack score (AUC) of 0.98 means they were easy to spot. After cleaning and retraining, version 1.1.0 scored 0.49, which is no better than a coin flip, and was released. The old version is kept.

![Model versions](https://raw.githubusercontent.com/r7bb/Model-Passport/main/docs/images/web-model.png)

**4. Release safely.** Testers try the model on private developer endpoints first. Customers can reach only approved, released versions, and one switch blocks a version everywhere at once.

![Deployments](https://raw.githubusercontent.com/r7bb/Model-Passport/main/docs/images/web-deployments.png)

**5. Prove it to investors and buyers.** Each model has a report showing where its data came from and who consented, every version and test, signed records, and a check that the activity log hasn't been changed.

![Diligence report](https://raw.githubusercontent.com/r7bb/Model-Passport/main/docs/images/web-diligence.png)

<details>
<summary>More screens: platform console, people and roles, analytics, activity log, data sources, and sign-in</summary>

| Platform console (all organizations) | People and roles |
|---|---|
| ![Platform console](https://raw.githubusercontent.com/r7bb/Model-Passport/main/docs/images/web-admin.png) | ![People and roles](https://raw.githubusercontent.com/r7bb/Model-Passport/main/docs/images/web-members.png) |

| Risk across versions | Tamper-proof activity log |
|---|---|
| ![Analytics](https://raw.githubusercontent.com/r7bb/Model-Passport/main/docs/images/web-analytics.png) | ![Audit log](https://raw.githubusercontent.com/r7bb/Model-Passport/main/docs/images/web-audit-log.png) |

| Where the data came from | Sign in |
|---|---|
| ![Data provenance](https://raw.githubusercontent.com/r7bb/Model-Passport/main/docs/images/web-data.png) | ![Sign in](https://raw.githubusercontent.com/r7bb/Model-Passport/main/docs/images/web-login.png) |

</details>

Each person sees only what their role allows. An ML engineer can test and fix models, a compliance auditor approves releases and can use the kill switch, a canary tester reports leaks, and an outside reviewer sees only the reports.

### From the command line

The same loop works without the web app. This is real output on 400 made-up support tickets:

```
$ passport init --llm --corpus corpus.jsonl
$ passport llm remediate
v1.0.0: verdict fail, AUC 0.991, critical 107, high 638
  sanitized 501 values; retraining as v1.1.0
v1.1.0: verdict fail, AUC 0.6078, critical 0, high 0
  sanitized 302 values (all CITY, DOB, FULLNAME, IPV4, USERNAME); retraining as v1.2.0
v1.2.0: verdict pass, AUC 0.4926, critical 0, high 0
release gate passed at v1.2.0
```

After round two no single detail stood out, but the details as a group were still slightly recognizable (AUC 0.61). So MP replaced every value of those five kinds, and round three passed. Every High or Critical finding is re-tested with fresh evidence before it counts.

MP also checks tabular models and datasets, with a [readable report](https://raw.githubusercontent.com/r7bb/Model-Passport/main/docs/images/report-pass.png) and a [dashboard](https://raw.githubusercontent.com/r7bb/Model-Passport/main/docs/images/dashboard-privacy.png).

## What's ready, and what's coming

| Ready now | Being built |
|---|---|
| Memorization testing of language models, for open models and API models | Serving models behind the developer and customer endpoints, with live monitoring |
| Cleaning risky details and retraining, until the release gate passes | Automatic leak probing during verification |
| The web app: a console for platform administrators, and a dashboard for each organization | Container images for every service, and cloud deployment (Kubernetes, Terraform) |
| A service for many organizations, each with its own address, logins, roles, and encryption key, and data no other organization can see | |
| Approvals, canary testing, release, rollback, and a kill switch, run by a control plane and the `mp` command-line tool | |
| A tamper-proof activity log, signed records for every version, and a report for investors and buyers | |
| Checks for tabular models and datasets, drift monitoring, and a GitHub Action | |

See [ROADMAP.md](https://github.com/r7bb/Model-Passport/blob/main/ROADMAP.md) for the full plan.

## Try it

Instructions are in **[SETUP.md](https://github.com/r7bb/Model-Passport/blob/main/SETUP.md)**. They cover installing, testing a language model, the tabular demo, using your own data, and troubleshooting. The built-in examples use made-up data only, with no real people.

## Learn more

- [SETUP.md](https://github.com/r7bb/Model-Passport/blob/main/SETUP.md): installation, commands, and technical details
- [ROADMAP.md](https://github.com/r7bb/Model-Passport/blob/main/ROADMAP.md): what's done and what's next
- Research: EL-MIA (Satvaty, Verberne, and Turkmen, [LREC 2026](https://aclanthology.org/2026.lrec-1.362/)) and AIPassport (Kalokyri et al., [arXiv 2506.22358](https://arxiv.org/abs/2506.22358))

Free to use under the MIT license ([LICENSE](https://github.com/r7bb/Model-Passport/blob/main/LICENSE)). See [THIRD_PARTY_NOTICES.md](https://github.com/r7bb/Model-Passport/blob/main/THIRD_PARTY_NOTICES.md) for the open-source software it uses.
