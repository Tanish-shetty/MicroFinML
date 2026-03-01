# MicroFinML

## Data-Driven Intelligence for Sustainable Economics: Machine Learning for Micro-Financial Growth

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PySpark](https://img.shields.io/badge/PySpark-3.5.4-orange.svg)](https://spark.apache.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](LICENSE)

> **Big Data Analytics — Internal Assessment (IA)**  
> Third-Year AI & DS | Springer-Level Research Chapter  
> **Topic:** Scalable Credit Scoring Framework for Microfinance Using Distributed Machine Learning

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Authors](#authors)
- [License](#license)

---

## 🎯 Overview

**MicroFinML** is a comprehensive Big Data Analytics and Machine Learning research project that addresses the critical challenge of loan default prediction in microfinance institutions (MFIs). The project demonstrates enterprise-grade implementation of distributed data processing, scalable machine learning, and emerging technologies for transparent AI decision-making.

### What Makes This Project Unique?

1. **Full Big Data Stack**: Apache Spark + Spark MLlib for distributed processing at scale
2. **Dual Implementation**: Local (scikit-learn/XGBoost) vs Distributed (Spark MLlib) comparison
3. **Scalability Analysis**: 36 benchmarks across 6 dataset sizes (10K-255K records)
4. **Blockchain Integration**: Immutable audit trail for credit decisions
5. **Research Quality**: Springer-level documentation with 35 peer-reviewed papers
6. **Social Impact**: Focus on financial inclusion for underserved populations

---

## 🔍 Problem Statement

Microfinance institutions serve **2.5 billion unbanked individuals** globally, providing small loans to entrepreneurs and families without access to traditional banking. However, MFIs face a critical challenge:

- **High Default Rates**: 10-15% average default rate threatens sustainability
- **Limited Credit History**: Traditional credit scoring models fail for underserved populations
- **Manual Assessment**: Time-consuming and subjective loan approval processes
- **Scalability Issues**: Growing loan portfolios require automated, data-driven solutions

**Our Solution**: A scalable, transparent, and ethical ML-powered credit scoring framework that:
- Predicts loan defaults with 75.8% ROC-AUC accuracy
- Processes 255K+ records using distributed computing
- Provides explainable AI with feature importance analysis
- Ensures transparency through blockchain audit trails

---

## ✨ Key Features

### 1. Big Data Analytics Framework
- **Apache Spark (PySpark 3.5.4)** for distributed data processing
- **Spark MLlib** for scalable machine learning
- **5Vs Analysis**: Volume, Velocity, Variety, Veracity, Value
- **Distributed Architecture**: HDFS-inspired data pipeline

### 2. Machine Learning Models
- **6 Models Trained**:
  - Local: Logistic Regression, Random Forest, XGBoost
  - Distributed: Spark LR, Spark RF, Spark GBT
- **Imbalanced Data Handling**: SMOTE (11.6% → 50% default rate)
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC

### 3. Scalability Benchmarking
- **36 Benchmarks**: 6 dataset sizes × 6 models
- **Performance Metrics**: Training time vs dataset size
- **Framework Comparison**: Local vs Spark at scale
- **Visualization**: 10 publication-quality plots

### 4. Blockchain Audit Trail
- **SHA-256 Hash Chain**: Immutable decision logging
- **Tamper Detection**: Cryptographic integrity verification
- **Regulatory Compliance**: Transparent AI for financial services

### 5. Research Documentation
- **25-Page LaTeX Chapter**: Springer LNCS format
- **35 Research Papers**: Categorized literature review
- **Mathematical Rigor**: Formal model definitions
- **Ethical Analysis**: Bias, privacy, and social impact

---

## 🛠️ Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **Language** | Python | 3.12 |
| **Big Data** | Apache Spark (PySpark) | 3.5.4 |
| **Distributed ML** | Spark MLlib | 3.5.4 |
| **Local ML** | scikit-learn | 1.3+ |
| **Gradient Boosting** | XGBoost | 2.0+ |
| **Imbalance Handling** | imbalanced-learn (SMOTE) | 0.11+ |
| **Data Processing** | pandas, NumPy | 2.0+, 1.24+ |
| **Visualization** | Matplotlib, Seaborn | 3.7+, 0.12+ |
| **Blockchain** | hashlib (SHA-256) | Built-in |
| **Documentation** | LaTeX (Springer LNCS) | - |
| **Notebooks** | Jupyter | 1.0+ |

---

## 📁 Project Structure

```
MicroFinML/
├── data/
│   ├── raw/
│   │   └── Loan Default.csv          # 255,347 records × 18 features
│   └── processed/
│       └── preprocessor.pkl          # Train/val/test splits + scaler
│
├── src/
│   ├── data_preprocessing.py         # Feature engineering + SMOTE
│   ├── model_training.py             # Local ML training (LR, RF, XGBoost)
│   ├── model_evaluation.py           # Metrics, plots, confusion matrices
│   ├── prediction.py                 # Inference for new applications
│   ├── spark_processing.py           # PySpark distributed preprocessing
│   ├── spark_ml_training.py          # Spark MLlib models (LR, RF, GBT)
│   ├── scalability_analysis.py       # Benchmarking framework
│   ├── blockchain_audit.py           # Blockchain audit trail
│   └── create_literature_review.py   # Literature review generator
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA with visualizations
│   ├── 02_data_preprocessing.ipynb   # Feature engineering + SMOTE
│   ├── 03_model_training.ipynb       # Model training + cross-validation
│   ├── 04_model_evaluation.ipynb     # Metrics + plots + comparison
│   └── 05_bda_analysis.ipynb         # Spark, scalability, blockchain, ethics
│
├── models/
│   ├── logisticregression_model.pkl  # Trained local LR
│   ├── randomforest_model.pkl        # Trained local RF
│   ├── xgboost_model.pkl             # Trained local XGBoost
│   └── spark/
│       ├── spark_logisticregression/ # Spark MLlib LR
│       ├── spark_randomforest/       # Spark MLlib RF
│       └── spark_gbt/                # Spark MLlib GBT
│
├── results/
│   ├── figures/
│   │   ├── feature_importance/       # 3 feature importance plots
│   │   └── model_comparison/         # 7 comparison plots
│   └── metrics/
│       ├── model_performance.csv     # Local model metrics
│       ├── framework_comparison.csv  # Local vs Spark comparison
│       └── scalability_results.csv   # 36 benchmark results
│
├── reports/
│   ├── latex/
│   │   └── main.tex                  # 25-page Springer chapter
│   └── literature_review.xlsx        # 35 categorized papers
│
├── run_pipeline.py                   # Local ML pipeline only
├── run_full_pipeline.py              # Complete BDA pipeline (all 5 phases)
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── PROJECT_VERIFICATION.md           # IA requirements checklist
├── SUBMISSION_CHECKLIST.md           # Submission guide
└── .gitignore
```

**Total**: 9 source modules | 5 notebooks | 6 models | 10 plots | 3 metrics CSVs

---

## 🚀 Installation

### Prerequisites

- **Python 3.12+**
- **Java 11+** (for PySpark)
- **8GB+ RAM** (for full dataset processing)
- **Git** (for cloning)

### Step 1: Clone Repository

```bash
git clone https://github.com/Tanish-shetty/MicroFinML.git
cd MicroFinML
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies installed:**
- pandas, numpy, scikit-learn, xgboost
- imbalanced-learn (SMOTE)
- matplotlib, seaborn
- pyspark (3.5.4)
- openpyxl (for Excel)
- jupyter, notebook

### Step 4: Verify Installation

```bash
python3 -c "import pyspark; print(f'PySpark {pyspark.__version__} installed')"
```

---

## 💻 Usage

### Option 1: Run Complete Pipeline (Recommended)

Run all 5 phases (Local ML + Spark + Scalability + Blockchain + Literature):

```bash
python3 run_full_pipeline.py
```

**Duration**: ~5-7 minutes  
**Output**: All models, metrics, plots, and reports

**Phases Executed**:
1. **Phase 1**: Local ML (preprocessing, training, evaluation)
2. **Phase 2**: Spark distributed processing + MLlib training
3. **Phase 3**: Scalability analysis (36 benchmarks)
4. **Phase 4**: Blockchain audit trail demonstration
5. **Phase 5**: Literature review generation

### Option 2: Run Local ML Only

```bash
python3 run_pipeline.py
```

**Duration**: ~2-3 minutes  
**Output**: Local models and evaluation only

### Option 3: Interactive Notebooks

```bash
jupyter notebook
```

Open notebooks in order:
1. `01_data_exploration.ipynb` - EDA with visualizations
2. `02_data_preprocessing.ipynb` - Feature engineering + SMOTE
3. `03_model_training.ipynb` - Model training + CV
4. `04_model_evaluation.ipynb` - Metrics + plots
5. `05_bda_analysis.ipynb` - **Spark, scalability, blockchain, ethics**

### Option 4: Individual Components

```python
# Preprocess data
from src.data_preprocessing import preprocess_pipeline
data = preprocess_pipeline('data/raw/Loan Default.csv', use_smote=True)

# Train models
from src.model_training import train_all_models
results = train_all_models(data['X_train'], data['y_train'])

# Evaluate
from src.model_evaluation import full_evaluation
metrics, predictions = full_evaluation(models, data['X_test'], data['y_test'])

# Spark processing
from src.spark_processing import create_spark_session, spark_preprocess_pipeline
spark = create_spark_session()
spark_data = spark_preprocess_pipeline(spark, 'data/raw/Loan Default.csv')
```

---

## 📊 Results

### Model Performance (Local)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg Precision |
|-------|----------|-----------|--------|-----|---------|---------------|
| **Logistic Regression** | 0.694 | 0.228 | 0.686 | 0.343 | **0.758** | 0.336 |
| **Random Forest** | **0.867** | **0.391** | 0.257 | 0.311 | 0.745 | 0.310 |
| **XGBoost** | 0.699 | 0.227 | 0.663 | 0.338 | 0.746 | 0.318 |

**Best Model**: Logistic Regression (ROC-AUC: 0.758) for imbalanced classification

### Framework Comparison (Local vs Spark)

| Framework | Model | Accuracy | ROC-AUC | Train Time |
|-----------|-------|----------|---------|------------|
| **scikit-learn** | LR | 0.694 | 0.758 | ~0.1s |
| **scikit-learn** | RF | 0.867 | 0.745 | ~17.7s |
| **scikit-learn** | XGBoost | 0.699 | 0.746 | ~2.1s |
| **Spark MLlib** | LR | 0.890 | 0.756 | 4.76s |
| **Spark MLlib** | RF | 0.890 | 0.739 | 23.53s |
| **Spark MLlib** | GBT | 0.888 | 0.750 | 18.41s |

**Key Insight**: Spark shows overhead at 255K scale but wins at 1M+ rows (projected)

### Scalability Analysis

**36 Benchmarks** across 6 dataset sizes (10K, 25K, 50K, 100K, 200K, 255K):

- **Local RF**: Linear scaling (0.34s → 2.33s)
- **Spark RF**: Higher overhead but constant per-record cost
- **Crossover Point**: ~500K records (Spark becomes faster)

### Blockchain Audit Trail

- **5 Loan Decisions** recorded with SHA-256 hashes
- **Tamper Detection**: 100% success rate
- **Chain Validation**: All blocks verified

---

## 📚 Documentation

### Research Documentation

1. **LaTeX Chapter** (`reports/latex/main.tex`)
   - 25 pages in Springer LNCS format
   - Sections: Abstract, Introduction, 5Vs, Literature Review, Mathematical Models, System Architecture, Methodology, Results, Ethics, Conclusion
   - 35 BibTeX references

2. **Literature Review** (`reports/literature_review.xlsx`)
   - 35 peer-reviewed papers categorized:
     - Classical Big Data (12 papers): Dean & Ghemawat 2004, White 2012, etc.
     - Modern ML (12 papers): Breiman 2001, Chen & Guestrin 2016, etc.
     - Blockchain/Quantum (11 papers): Nakamoto 2008, Grover 1996, etc.

### Technical Documentation

- **README.md**: Project overview and usage
- **PROJECT_VERIFICATION.md**: IA requirements checklist
- **SUBMISSION_CHECKLIST.md**: Grading rubric alignment
- **Docstrings**: All functions documented in source code

### Compile LaTeX Chapter

```bash
cd reports/latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Output: `main.pdf` (25-page research chapter)

---

## 🎓 Academic Context

### IA Requirements Coverage

| Requirement | Weight | Status | Evidence |
|-------------|--------|--------|----------|
| Big Data Framework | 25% | ✅ | PySpark + Spark MLlib |
| ML Implementation | 20% | ✅ | 6 models, SMOTE, evaluation |
| Scalability Analysis | 15% | ✅ | 36 benchmarks with plots |
| Emerging Technology | 10% | ✅ | Blockchain audit trail |
| Documentation | 20% | ✅ | 25-page LaTeX + 35 papers |
| Code Quality | 10% | ✅ | Modular, documented, automated |

**Total Coverage**: 100% ✅

### Learning Outcomes

1. **Big Data Processing**: Hands-on experience with Apache Spark and distributed computing
2. **Scalable ML**: Understanding trade-offs between local and distributed training
3. **Production ML**: End-to-end pipeline from data to deployment
4. **Emerging Tech**: Blockchain for transparent AI decision-making
5. **Research Skills**: Literature review, mathematical modeling, academic writing
6. **Social Impact**: Ethical AI for financial inclusion

---

## 🤝 Contributing

This is an academic project for Internal Assessment. Contributions are welcome for:

- Bug fixes
- Documentation improvements
- Additional models or features
- Performance optimizations

**To contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 👥 Authors

**Big Data Analytics — Internal Assessment Project**  
Third-Year AI & Data Science

- **Tanish Shetty** - [GitHub](https://github.com/Tanish-shetty)
- **Shreyas Gurav** - Project Implementation

**Supervisor**: [Faculty Name]  
**Institution**: [University/College Name]  
**Course**: Big Data Analytics (BDA)  
**Academic Year**: 2025-2026

---

## 📄 License

This project is licensed for **Academic Use Only**.

**Restrictions**:
- May not be used for commercial purposes
- Must cite original authors if used in research
- Modifications must be documented

---

## 🙏 Acknowledgments

- **Dataset**: Kaggle Loan Default Dataset
- **Frameworks**: Apache Spark, scikit-learn, XGBoost
- **Literature**: 35 peer-reviewed papers (see `reports/literature_review.xlsx`)
- **Inspiration**: Financial inclusion and sustainable economics

---

## 📞 Contact

For questions or collaboration:

- **GitHub Issues**: [Create an issue](https://github.com/Tanish-shetty/MicroFinML/issues)
- **Email**: [Your email]

---

## 🌟 Star This Repository

If you find this project useful for your research or learning, please ⭐ star this repository!

---

**Last Updated**: March 2, 2026  
**Version**: 1.0.0  
**Status**: ✅ Complete and Ready for Submission
