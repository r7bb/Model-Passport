# Model Passport

[![CI](https://github.com/r7bb/Model-Passport/actions/workflows/ci.yml/badge.svg)](https://github.com/r7bb/Model-Passport/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**A passport for AI models: proof of where a model came from, and a safety check before it's used.**

A person's passport says who they are and where they're from, and it's hard to fake. Model Passport gives an AI model the same thing. Each time a model is trained, it creates a document that records:

- **What went into it:** the data, the code, and the settings.
- **How well it works:** its test scores.
- **Whether it's safe:** the results of privacy and security checks.

The document is digitally signed, so if anyone changes the model or its data afterward, the passport shows it.

![The Model Passport dashboard](docs/images/dashboard-simple-view.png)

## Why it matters

AI models learn from data, and that data often comes from real people. Things can go wrong:

- The data may contain **personal details** such as names, emails, or phone numbers.
- The model may **memorize** people's records and later reveal them.
- Someone may **swap or alter** the model file after it was approved.
- The world may **change** until the model's predictions quietly stop being accurate.

Model Passport checks for all four and gives a clear **PASS** or **FAIL**.

## What it checks

| Check | In plain words |
|---|---|
| Personal information | Are there names, emails, phone numbers, or ID numbers in the data? |
| Re-identification risk | Could someone single out a person by combining details like age and ZIP code? |
| Memorization | Did the model memorize its training data instead of learning general patterns? |
| File safety | Could the model file run harmful code when opened? Are any software parts known to be unsafe? |
| Leaked passwords | Were passwords or access keys left in the code? |
| Tampering | Has anything changed since the passport was signed? |
| New data | When new data arrives, is it still similar enough for the model to work well? |

## See it in action

**Unsafe data and a model that memorizes are caught and blocked:**

![A failing safety check](docs/images/cli-build-fail.png)

**A model file that was changed after signing is named:**

![Tampering detected](docs/images/cli-verify-tamper.png)

**New data that looks different from the training data triggers a retraining alert:**

![Changing data detected](docs/images/cli-monitor-drift.png)

**Every passport also comes as a readable report:**

| Passed | Failed |
|---|---|
| ![Passing report](docs/images/report-pass.png) | ![Failing report](docs/images/report-fail.png) |

**The dashboard shows how a model was built, its privacy results, and how it holds up over time:**

| How it was built | Its version history |
|---|---|
| ![Pipeline](docs/images/dashboard-pipeline.png) | ![Lineage](docs/images/dashboard-lineage.png) |
| **Privacy results** | **Checks on new data** |
| ![Privacy](docs/images/dashboard-privacy.png) | ![Monitoring](docs/images/dashboard-monitoring.png) |

## Try it

A built-in demo uses made-up data (no real people) to show a model failing the checks, then passing after it's fixed.

Step-by-step instructions are in **[SETUP.md](SETUP.md)**: installing, running the demo, using it on your own project, and troubleshooting.

## Learn more

- [SETUP.md](SETUP.md): installation, commands, and technical details
- [ROADMAP.md](ROADMAP.md): what's done and what's next
- Model Passport builds on the AIPassport research framework (Kalokyri et al., [arXiv 2506.22358](https://arxiv.org/abs/2506.22358)).

Free to use under the MIT license ([LICENSE](LICENSE)). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the open-source software it uses.
