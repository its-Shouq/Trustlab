# TrustLab — Understanding Human Cooperation

**A Machine Learning Project | Supervised & Unsupervised Learning**

TrustLab explores human cooperation and defection in the **Iterated Prisoner's Dilemma** using machine learning to predict decisions and identify behavioral strategies.

### Live Demo

**[Explore TrustLab](https://trustlab-project-v5.onrender.com/director/dbtrnb3t)**

### Dataset

**Source:** [Dryad Dataset (2022)](https://datadryad.org/dataset/doi:10.5061/dryad.37pvmcvmk)

The dataset contains **18,800 decisions from 188 players**, each completing 100 rounds.

- **Fixed Partner:** 92 players interacting with the same partner.
- **Random Partner:** 96 players interacting with changing partners.

Each player chooses to **Cooperate (C)** or **Defect (D)**.

## 1. Exploratory Data Analysis (EDA)

We analyzed cooperation patterns, previous decisions, and differences between partner conditions.

**Key Findings:**

- Fixed Partner cooperation rate: **56.10%**
- Random Partner cooperation rate: **28.52%**
- Previous decisions were strongly associated with future cooperation.
- Cooperation patterns differed between fixed and changing partners.

## 2. Supervised Learning

**Goal:** Predict whether a player will cooperate or defect.

**Features:** `round`, `treatment`, `prev_player`, `prev_opp`

Five classification models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)

We applied preprocessing, session-based train/test splitting, Group Cross-Validation, and GridSearchCV for hyperparameter tuning.

**Selected Model: Logistic Regression**

### Final Results

| Metric | Score |
|---|---:|
| Accuracy | **83.79%** |
| Precision | 84.50% |
| Recall | 85.22% |
| F1-Score | **84.86%** |
| ROC-AUC | **0.898** |

## 3. Unsupervised Learning

**Goal:** Discover behavioral strategies without predefined labels.

We compared K-Means, Hierarchical Clustering, Gaussian Mixture Models, and DBSCAN.

**Selected Model: K-Means (k = 4)**

### Discovered Behavioral Strategies

| Strategy | Players | Description |
|---|---:|---|
| Defector | 73 (39%) | Mostly defects |
| Forgiving Cooperator | 55 (29%) | Frequently cooperates, even after betrayal |
| Erratic | 31 (16%) | Frequently changes decisions |
| Reciprocator | 29 (15%) | Responds to the opponent's previous decision |

K-Means achieved the best clustering results among the compared models, with a **Silhouette Score of 0.32**.

## 4. Model Integration

We integrated historical cooperation patterns and discovered strategy features into the supervised model to improve next-move predictions.

### Performance Comparison

| Metric | Original Model | Integrated Model |
|---|---:|---:|
| Test Accuracy | 83.8% | **85.7%** |
| Test ROC-AUC | 0.898 | **0.937** |
| CV Accuracy | 82.9% | **84.2%** |
| CV ROC-AUC | 0.845 | **0.898** |

**Key Finding:** Combining supervised and unsupervised learning improved prediction performance by incorporating historical behavior and strategy information.

## 5. Project Files

| File | Description |
|---|---|
| `EDA&supervised_notebook.ipynb` | EDA, preprocessing, supervised model training and evaluation |
| `unsupervised_strategies.ipynb` | Behavioral clustering and strategy analysis |
| `supervised_integration.ipynb` | Integration of supervised and unsupervised learning |
| `next_move_lr.joblib` | Saved integrated prediction model |
| `strategy_model.joblib` | Saved clustering model |
| `strategy_model.json` | Strategy descriptions and profiles |
| `player_strategies.csv` | Player behavioral features and cluster assignments |
| `TrustLab_Presentation.html` | Project presentation |
| `requirements.txt` | Project dependencies |

## 6. Technologies

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · SciPy · Joblib · Google Colab · Render

## 7. Conclusion

TrustLab demonstrates how machine learning can help us understand and predict human cooperation.

Our analysis found that cooperation was higher with fixed partners, identified **four distinct behavioral strategy profiles**, and improved next-move prediction by combining supervised and unsupervised learning.

The integrated model achieved **85.7% test accuracy and 0.937 ROC-AUC**, demonstrating the value of incorporating behavioral patterns into predictive models.

---

**TrustLab — Understanding Behavior, Predicting Decisions.**
