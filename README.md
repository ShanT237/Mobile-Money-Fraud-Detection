# Mobile Money Fraud Detection

## Executive Summary

This project develops a cost-sensitive fraud detection system for mobile money transactions using the PaySim synthetic dataset.

Rather than maximizing traditional machine learning metrics such as accuracy or F1-score, the objective is to minimize the company's expected financial loss by selecting an optimal decision threshold based on business costs.

The project compares an interpretable Logistic Regression baseline with an XGBoost benchmark and demonstrates how model evaluation changes when business objectives drive the optimization process instead of statistical metrics alone.

**Highlights**

- 6.3M transactions analyzed
- Cost-sensitive threshold optimization
- 99.8% fraud recall with Logistic Regression
- PR-AUC: 0.794 (Logistic Regression) vs 1.000 (XGBoost)
- Discussion of synthetic data limitations and model validation

## Architecture

```
Raw Transactions
       │
       ▼
 Data Validation
       │
       ▼
 Feature Engineering
       │
       ▼
 Model Training
       │
       ▼
 Threshold Optimization
       │
       ▼
 Fraud Decision
```

## Tech Stack

| Area | Technologies |
|------|--------------|
| Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Evaluation | Precision, Recall, PR-AUC, Cost-sensitive Threshold Optimization |
| Environment | Jupyter Notebook |

## 1. Business Context

### 1.1 Problem Statement
Fraudulent transactions drain customer accounts, causing direct financial loss (chargeback liability), customer churn from broken trust, and operational costs from manual reviews — the current rule-based system (isFlaggedFraud) catches only 0.19% of fraud, leaving the business exposed.

### 1.2 Actors
- **Victim:** The mobile money account holder. Their balance is reduced or emptied without consent. For many users, mobile money is a primary financial tool — losing access means losing funds needed for daily life.
- **Attacker:** An adversarial actor who gains unauthorized control of a customer's account for financial gain. They capture gains while the victim and business absorb the losses. Their identity is unobservable in the data — only their effects on transaction patterns are visible.
- **Business:** The mobile money operator. Absorbs direct financial loss (reimbursements), operational costs (investigation teams, support), regulatory exposure, and reputational damage.

### 1.3 Fraud Pattern
PaySim's fraud follows one specific mechanism: the attacker takes control of a customer's account and executes a two-step sequence:
1. **TRANSFER** — moves money out of the victim's account into a mule account, typically draining the full balance
2. **CASH_OUT** — immediately withdraws the money as physical cash, making it untraceable before the fraud is detected

### 1.4 Business Impact
Each undetected fraud costs = **full transaction amount + investigation cost + reputational damage**. We use the amount as our FN cost proxy, knowing the real cost is higher.

## 2. Project Objective

Build a binary classifier that predicts the probability of fraud **before the transaction is approved**, using a decision threshold that minimizes total financial loss — not accuracy or F1.

## 3. Why ML?

The current rule-based system (`isFlaggedFraud`) catches only **16 out of 8,213 frauds (0.19%)**. A simple rule like "flag transactions over $200k" misses almost everything — especially moderate amounts that blend in with legitimate activity. ML can learn complex patterns (balance errors, velocity, transaction sequences) that rules can't capture.

## 4. Cost Matrix

![Cost Matrix](img/image.png)

**FN cost** = full transaction amount (variable — the company absorbs 100%)

**FP cost** = $10 USD (estimated operational cost of manual review + customer friction — documented assumption, adjustable by the business)

**Total Cost Formula:**

![Total Cost Formula](img/image-1.png)

Because fraud represents only **0.13% of all transactions**, model evaluation focuses on **Recall, Precision, PR-AUC**, and **Expected Business Cost**. PR-AUC is preferred over ROC-AUC because it better reflects model performance on highly imbalanced datasets.

## 5. Dataset & Limitations

**Dataset:** PaySim synthetic mobile money transactions (`AIML Dataset.csv`)

### 5.1 Limitations
- **Synthetic data** — not real transactions. Fraud patterns follow simulator rules, so a model may perform perfectly here but fail on real data.
- **Simplistic fraud patterns** — fraud only occurs in TRANSFER and CASH_OUT types. Never in PAYMENT, DEBIT, or CASH_IN. Real fraud is more diverse.
- **No geography, merchant, or device data** — can't analyze by location, merchant category, IP, or device fingerprint.
- **No customer history** — no transaction frequency, velocity per user, or spending patterns (only per-row balance snapshots).
- **Time is simulated (step)** — 743 steps = 1 hour each ≈ 31 days. No real day-of-week or seasonality effects.
- **Obvious fraud signature** — `oldbalanceOrg - amount = newbalanceOrig` almost perfectly in fraud cases. A simple rule catches most of it. Real fraud is subtler.

## 6. EDA Summary

- **0 nulls**, **0 exact duplicates**
- **11 columns**, **6,362,620 rows** (534 MB)
- Fraud rate: **0.13%** (8,213 frauds out of 6.3M)
- Fraud only in **TRANSFER** and **CASH_OUT** transaction types

![Fraud by Transaction Type](img/fraud_by_type.png)
- Flagged fraud system (`isFlaggedFraud`) detects only **16 out of 8,213**
- No negative amounts or balances — data is consistent
- Balance inconsistency (`oldbalanceOrg - amount ± newbalanceOrig`) is **rare in legit, common in fraud** — strong predictive signal

## 7. ML Models

### 7.1 Logistic Regression

**First approach:**
1. `class_weight='balanced'` — the dataset is heavily imbalanced (0.13% fraud, 99.87% legit). This assigns higher weight to the minority class (fraud), penalizing the model more when it misses fraud than when it flags a legit transaction.
2. One-hot encoding for `type` + StandardScaler for numeric features, chained in a Pipeline.
3. **93% recall** for fraud (catches 93/100), but **2% precision** (98 out of 100 fraud alerts are wrong), causing ~4,500 false positives.

**Analysis:** The model catches most frauds, but the high false positive rate is costly. Next steps: tune the decision threshold using the cost matrix, add better features (balance differences, velocity), and try tree-based models.

**Second approach:**

**Motivation**

The baseline model demonstrated that Logistic Regression could detect most fraudulent transactions, but it suffered from a high false positive rate. Three improvements were introduced to better reflect a real production environment and optimize business outcomes.

**Changes from v1**
- Temporal train-test split — The model was trained on the earliest 80% of transactions (step) and evaluated on the most recent 20%. This better simulates deployment, where predictions are made on future transactions, and avoids overly optimistic evaluation that can result from random splits.
- Feature engineering — Added:

  `balanceDiffOrig = oldbalanceOrg - amount - newbalanceOrig`
  `balanceDiffDest = oldbalanceDest + amount - newbalanceDest`

  These features capture balance inconsistencies that frequently appear in fraudulent transactions, such as accounts being unexpectedly drained.
- Threshold optimization — Instead of using the default probability threshold of 0.5, the decision threshold was selected by minimizing the business cost:

  False Negative cost = transaction amount
  False Positive cost = $10

  The optimal threshold was **0.20**, reflecting that missing fraud is substantially more expensive than investigating a legitimate transaction.

**Results**

| Metric | v1 | v2 |
|---|---|---|
| Train-test split | Random | Temporal |
| Feature engineering | No | Yes |
| Threshold | 0.50 | 0.20 |
| Fraud Recall | 93.0% | 99.8% |
| Precision | 2% | 19% |
| Minimum Business Cost | — | $25.9M |

**Business Impact**

The improved model detected 99.8% of fraudulent transactions, missing only 3 out of 1,654 fraud cases in the test set.

Precision increased from 2% to 19%, meaning nearly 1 in 5 fraud alerts corresponded to an actual fraudulent transaction, compared with only 1 in 50 in the baseline model.

Although lowering the threshold generated more fraud alerts, the additional investigation cost is relatively small ($10 per false positive) compared with the financial loss associated with an undetected fraudulent transaction. As a result, the selected threshold minimizes the company's expected financial loss rather than maximizing traditional classification metrics.

### 7.2 XGBoost

XGBoost achieved **100% precision, recall, and F1-score** on the PaySim test set.

**Why this happened**

PaySim is synthetic — fraud follows deterministic rules. Three factors explain the perfect score:

| Factor | Why it helps XGBoost |
|---|---|
| Synthetic dataset | Fraud follows predefined simulator rules, not real human behavior |
| Clean fraud boundary | Fraud only occurs in TRANSFER and CASH_OUT — tree models learn this easily |
| Balance features | `balanceDiffOrig` and `balanceDiffDest` capture the simulator's drain pattern directly |

**Validation test**

We removed both balance difference features and retrained:

| Metric | With balance features | Without balance features |
|---|---|---|
| Recall | 100% | 100% |
| Precision | 100% | 91% |

Even without the engineered features, XGBoost catches every fraud. The original balance columns already encode the fraud pattern.

**Bottom line:** XGBoost establishes the upper performance bound on this dataset. These results should not be interpreted as real-world readiness — they reflect how cleanly PaySim separates fraud from legitimate transactions. A model this perfect on synthetic data likely overfits to simulator-specific patterns that won't appear in production.

## 8. Model Comparison

| Metric | Logistic Regression v2 | XGBoost |
|---|---|---|
| Precision | 19% | 100% |
| Recall | 99.8% | 100% |
| PR-AUC | 0.794 | 1.000 |
| Primary Objective | Minimize Expected Cost | Benchmark Performance |

Precision-Recall AUC (PR-AUC) is reported because fraud represents only 0.13% of all transactions. Unlike ROC-AUC, PR-AUC better reflects the trade-off between detecting fraudulent transactions and minimizing false alerts in highly imbalanced datasets.

The Logistic Regression model achieved a PR-AUC of 0.794, indicating strong ranking performance while remaining interpretable and suitable as a production baseline.

XGBoost achieved a PR-AUC of 1.000, indicating perfect separation between fraudulent and legitimate transactions on the PaySim dataset.

## 9. Key Learnings

- Business metrics can be more important than traditional ML metrics.
- Decision threshold selection should reflect financial cost rather than arbitrary defaults.
- Synthetic datasets can produce overly optimistic model performance.
- Strong predictive performance must always be interpreted in the context of data quality and dataset limitations.

## 10. Conclusion

This project demonstrates that optimizing fraud detection requires balancing predictive performance with business objectives.

Logistic Regression showed that a simple interpretable model can dramatically reduce expected financial loss when combined with cost-sensitive threshold optimization.

XGBoost established the upper performance bound on the PaySim dataset, but its near-perfect performance also exposed important limitations of synthetic data. Additional validation confirmed that these results stem from highly separable fraud patterns rather than from a single engineered feature.

Ultimately, the primary outcome of this project is not achieving perfect predictive metrics, but demonstrating a structured approach to developing, evaluating, and interpreting fraud detection models using business-driven decision making.

This repository represents the machine learning component of a larger end-to-end analytics engineering project. The next phase focuses on building the production data platform that supports model training, deployment, and monitoring.

## 11. Future Work

The current project focuses on model development and evaluation.

The next iteration extends this work into a production-oriented analytics platform by integrating:

- **Airflow** for pipeline orchestration and scheduling
- **BigQuery** as the cloud data warehouse
- **dbt** for data modeling, testing, and documentation
- **Incremental ELT pipelines** for continuously processing new transactions
- **Automated model retraining** using curated warehouse tables
- **Streamlit** for real-time fraud scoring
- **Data quality and drift monitoring** to improve reliability over time

The objective is to demonstrate how a fraud detection model can evolve from an exploratory notebook into a production-ready analytics and machine learning pipeline.
