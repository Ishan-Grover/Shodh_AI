# Shodh_AI_Assignment

This repository demonstrates how data-driven modeling can optimize loan approvals by combining **Supervised Deep Learning** for risk prediction with **Offline Reinforcement Learning (RL)** for profit-maximizing policy learning.

The project is divided into four main tasks: **EDA & Preprocessing**, **Predictive Modeling**, **Offline RL Agent**, and **Comparative Analysis** — offering both a predictive and decision-optimization perspective on financial risk modeling.

---

## 🧭 Project Overview

| **Task** | **Objective** |
|-----------|----------------|
| **Task 1 – EDA & Preprocessing** | Clean, explore, and prepare the Lending Club dataset for modeling. |
| **Task 2 – Predictive Deep Learning Model** | Build classifiers (Logistic Regression, RF, GBM, MLP) to predict loan defaults. |
| **Task 3 – Offline RL Agent (CQL)** | Train a Reinforcement Learning agent to make optimal loan approval decisions using historical data. |
| **Task 4 – Comparative Analysis** | Compare predictive and policy-based results, interpret metrics, and propose future improvements. |

---

## 🗂️ Repository Structure

Shodh_AI/
│
├── Shodh_AI_Ishan.ipynb # Main Jupyter notebook containing the full workflow
├── Shodh_AI_Report.pdf # 3-page analysis report
├── Ishan_Grover_Thapar_Institute_Of_Engineering_And_Technology_Resume.pdf # Resume file
└── README.md # This guide
---

## ⚙️ Setup Instructions

### **1️⃣ Clone the Repository**
git clone https://github.com/<your-username>/loan_approval_rl.git
cd loan_approval_rl

### **2️⃣ Install Dependencies**
pip install -r requirements.txt

**Main Libraries:**
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- tensorflow / torch
- gymnasium (replacing deprecated gym)
- d3rlpy (for Offline Reinforcement Learning)

⚠️ Important: The codebase uses gymnasium instead of gym, as Gym is deprecated and incompatible with NumPy 2.0.

---

## 🚀 How to Run the Code

### **🔹 Task 1: Exploratory Data Analysis & Preprocessing**

Run the preprocessing pipeline:
python scripts/shodh_ai_ishan.py --task eda

Or explore interactively in:
notebooks/task1_eda_preprocessing.ipynb

**This step includes:**
- Cleaning string-based columns (int_rate, emp_length, term)
- Handling missing values (median/mode)
- Encoding categorical features (home_ownership, purpose)
- Creating unified features like fico_score
- Binary target creation: {0: Fully Paid, 1: Defaulted}

---

### **🔹 Task 2: Predictive Deep Learning Model**

Train all supervised models:
python scripts/shodh_ai_ishan.py --task supervised

**Models trained:**
- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- Neural Network (MLP)

**Example Results:**
Model | Accuracy | Recall | F1 | AUC
-------|-----------|--------|----|-----
Logistic Regression | 0.8563 | 0.539 | 0.606 | 0.913
Random Forest | 0.8807 | 0.604 | 0.675 | 0.938
Gradient Boosting | 0.8843 | 0.614 | 0.686 | 0.942
Neural Network (MLP) | 0.8787 | 0.649 | 0.687 | 0.933

Outputs:
- ROC Curve comparison under results/plots/roc_curve.png
- Model results in results/model_metrics.csv

---

### **🔹 Task 3: Offline RL Agent (CQL)**

Train the RL policy with Conservative Q-Learning (CQL) using d3rlpy.

python scripts/shodh_ai_ishan.py --task rl

**Reward Function:**
if action == 0:  # Deny
    reward = 0
elif action == 1 and loan_status == "Fully Paid":
    reward = + loan_amnt * int_rate  # Profit
else:
    reward = - loan_amnt  # Loss due to default

**Outputs:**
Policy | Avg Reward | Approve Rate
--------|-------------|--------------
CQL Agent | 123,320.38 | 0.888
Always Approve | 133,979.89 | 1.000
Always Deny | 0.00 | 0.000
Supervised RF | 126,543.76 | 0.832

✅ The RL agent learns a risk-adjusted, profit-maximizing approval policy — not just minimizing default probability.

---

### **🔹 Task 4: Analysis & Comparison**

Open:
notebooks/task4_analysis_report.ipynb

This section covers:
- Metric interpretation (AUC/F1 for DL, Estimated Policy Value for RL)
- Side-by-side comparison of predictive vs decision models
- Example of applicants where both disagree:
  - DL model rejects based on high default probability
  - RL agent approves if expected interest gain outweighs risk
- Limitations and future steps

---

## 🧠 Conceptual Summary

Model Type | Objective | Metric | Key Insight
-------------|------------|---------|--------------
Deep Learning (Supervised) | Predict probability of default | AUC, F1 | Identifies risky applicants
Offline RL (CQL) | Maximize long-term profit | Estimated Policy Value | Learns optimal approval policy

- AUC/F1: Measure model’s discrimination power (predictive accuracy).  
- Estimated Policy Value: Measures expected financial return under a learned policy.  

---

## 📊 Visual Outputs

ROC Curve Comparison  
(results/plots/roc_curve.png)

Reward Distribution (RL Agent)  
(results/plots/reward_distribution.png)

---

## 🔮 Future Improvements

- Explore additional offline RL algorithms: IQL, AWAC, BCQ  
- Integrate explainability (SHAP, LIME) for both DL & RL decisions  
- Build a Streamlit/FastAPI dashboard for deployment  
- Add fairness and bias evaluation for responsible AI deployment  
- Experiment with dynamic interest rate adjustments as part of policy learning

---

## 🧩 Limitations

- RL policy heavily depends on the designed reward structure.  
- Dataset is static — true online adaptation not tested.  
- Economic assumptions (loan amount, interest rate) simplified for modeling.  
- Estimated Policy Value is based on simulation, not live testing.

---

## 👤 Author

**Ishan Grover**  
Final-Year B.E. in Electrical & Computer Engineering  
Thapar Institute of Engineering and Technology  

📧 Email: ishangrover.21@outlook.com
💼 LinkedIn: https://www.linkedin.com/in/ishan-grover-33b9b6314/
🐙 GitHub: https://github.com/Ishan-Grover

---

> *This project demonstrates how Deep Learning models can predict default risk, while Offline Reinforcement Learning can optimize approval decisions for maximum long-term profitability.*
