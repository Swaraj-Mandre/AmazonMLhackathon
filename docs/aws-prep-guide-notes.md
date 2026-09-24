# AWS Builder Center, ML Challenge 2026 Prep Guide (captured notes)

Source: https://builder.aws.com/content/3HiM6zDmFrF98fRzOUETnGFDoqz/amazon-ml-challenge-2026-your-complete-prep-guide-with-live-demo
Author: Jatin, Developer Advocate @ AWS. Published 21 Sep 2026, modified 23 Sep 2026.
Captured: 24 Sep 2026. **Partial, see "Not captured" at the bottom.**

---

## Credits: what you actually get

- **$100 AWS credits** on signing up.
- **+$100 more** after completing 5 starter activities (create an EC2 instance, an RDS database, etc.) → **$200 total**.
- **+$100** for the Top 500 teams at the 48-hour mark of the hackathon.
- **Student Rewards** (launched 20 Aug 2026), separate from the challenge:
  - Verify student status → 12 months Skill Builder Premium ($449 value)
  - 7 badges on Builder Center → $10 AWS credits
  - 14 badges → $20 AWS credits
  - 21 badges → $100 AWS certification exam voucher
  - Badges come from simple activity: signing in, commenting, publishing articles, submitting feature requests. No credit card, ever.

## Free tier worth knowing (new accounts)

| Service | Free allowance |
|---|---|
| SageMaker Notebooks | 250 hours on `ml.t3.medium` (2 months) |
| SageMaker Training | 50 hours on `ml.m5.xlarge` (2 months) |
| SageMaker Inference | 125 hours on `ml.m5.xlarge` (2 months) |
| S3 | 5 GB always free |
| Lambda | 1M requests/month always free |

**Caveats called out in the guide:**
- Set billing alerts in the console immediately after creating the account.
- Always delete endpoints when done, they charge ~$0.12/hour even while idle.
- Use `us-east-1` for best compatibility.

## Services you need to understand

| Service | Role in this challenge |
|---|---|
| EC2 | SageMaker training jobs run on EC2 behind the scenes |
| S3 | Data in, trained models out, everything flows through it |
| DynamoDB | Results, metadata, feature stores at scale |
| IAM | The role that lets SageMaker read your data and spin up machines |
| VPC | Private network your SageMaker resources run in; Quick Setup creates one |

## Setup path the guide recommends for the challenge

Use a **Notebook Instance**, not SageMaker Studio. Studio needs a Domain setup that can take a while on brand-new accounts; a Notebook Instance is plain JupyterLab with no Domain and no quota issues. Same code runs on both.

Steps:
1. SageMaker AI console → sidebar → Applications and IDEs → Notebook → Notebook instances
2. Create notebook instance, any name
3. Instance type `ml.t3.medium` (free tier, 250 hours)
4. IAM Role → Create a new role → leave defaults → Create role
5. Create notebook instance, wait 2-3 min for **InService**
6. Open JupyterLab → "+" → `conda_python3` notebook

**Guide's recommendation for the challenge specifically:**
> Notebook Instance + local training + local prediction. You submit a CSV, not a running API.

i.e. don't waste time deploying endpoints, train inside the notebook, predict locally, write the CSV.

- **Local training vs Training Job:** local runs on the notebook's own CPU with data already in memory, no S3 upload. A Training Job spins up a separate, more powerful machine that pulls from S3. Use local when data is small; use Training Jobs when you need big machines or GPUs.
- **Local prediction vs Endpoint:** `model.predict()` is instant in the notebook. An endpoint is a 24/7 hosted API at $0.12/hour even when idle.

## Boilerplate from the demo

```python
!pip install xgboost scikit-learn -q
```

```python
import sagemaker, boto3, pandas as pd, numpy as np, time

session = sagemaker.Session()
role    = sagemaker.get_execution_role()   # permission pass for SageMaker to reach your data
region  = session.boto_region_name
bucket  = session.default_bucket()         # auto-created S3 bucket
```

Reading data (the demo reads a public AWS bucket; for the challenge, upload our data to *our* S3 bucket and give the notebook's IAM role permission):

```python
data = pd.read_csv(f"s3://sagemaker-example-files-prod-{region}/datasets/tabular/synthetic/churn.txt")
```

Three-way split used in the demo:

```python
train_data, validation_data = train_test_split(model_data, test_size=0.33, random_state=42)
validation_data, test_data  = train_test_split(validation_data, test_size=0.33, random_state=42)
```

XGBoost params shown:

```python
params = {"max_depth": 5, "eta": 0.2, "gamma": 4, "min_child_weight": 6, ...}
```

## Advice the guide gives that's worth keeping

- **Check the target distribution before anything else.** The demo dataset is a balanced 50/50; the guide warns the challenge data "might be heavily skewed to one side," and that changes how you build *and* evaluate.
- **The challenge data will not be clean**, expect missing values and noise, unlike the demo's 21 clean columns.
- **Drop unique IDs, serial numbers, row numbers.** They add nothing.
- **"Feature engineering often matters more than your choice of algorithm. Spend 70% of your time here."**
- **SageMaker's built-in XGBoost has two rules:** no column headers in the CSV, and the target must be the **first** column. The guide calls this the number one beginner mistake.
- **Never touch the test set until you're done experimenting.**

## Not captured

The earlier part of the article rendered fully; capture cut off partway through Step 7 (XGBoost hyperparameters). Missing: the rest of Step 7 and the remaining steps (evaluation metrics, local prediction, and any closing tips). The site is a JavaScript app whose content API (`prod.capabilities.builder.aws.com`) is blocked in the browser pane, so a re-scrape currently fails. The recording is on Twitch if the remainder matters.
