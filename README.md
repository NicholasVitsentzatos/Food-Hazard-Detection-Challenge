# Food Hazard Detection Challenge
This repository contains solutions for the SemEval 2025 Task 9: Food Hazard Detection Challenge. The task involves classifying and detecting food hazards from incident reports using NLP. It includes two sub-tasks: predicting product and hazard categories (ST1) and detecting specific products and hazards (ST2).

This project addresses the **Food Hazard Detection Challenge**, which involves benchmarking both basic and advanced machine learning algorithms to classify food hazards using textual data. Two types of text are used for classification: **short texts (titles)** and **long texts**. The ultimate goal is to identify hazard categories, hazard types, and product categories effectively.

---

## Objectives

1. Perform **benchmark analysis** using:
   - Basic Machine Learning models.
   - Advanced Machine Learning models (XLM-RoBERTa).
2. Evaluate the impact of using short texts (**titles**) versus long texts (**detailed descriptions**) for classification.
3. Analyze the best algorithms for predicting:
   - Hazard Categories
   - Hazard Types
   - Product Categories
   - Product Types

---

## Methodology

### Classify Hazard Categories Using Titles
- **Features Used**: Title column.
- **Models Benchmarked**:
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machines (SVM)
  - Random Forest
  - Gradient Boosting
  - K-Nearest Neighbors (KNN)
- **Evaluation Metrics**: Accuracy, Precision, Recall, and F1 score.
- **Process**:
  - Train basic models.
  - Use **GridSearchCV** for hyperparameter tuning.
  - Compare performance between default and tuned models.

### Classify Hazard Categories Using Long Texts
- **Features Used**: Text column.
- **Objective**: Compare the performance of models trained on the richer, more detailed text column versus the title column.

### Classify Hazard Types
- **Objective**: Use the best-performing model from previous experiments to predict hazard types (128 unique labels) using titles and long texts.

### Classify Product Categories and Types
- **Objective**: Predict the **product category** (22 unique labels) and **product type** using the same process as above.

### Advanced Benchmark Analysis
- Evaluate **XLM-RoBERTa**, a transformer-based model, for hazard and product classification:
  - Train models on both the title and text columns.
  - Perform hyperparameter tuning.
  - Compare the results with basic ML models.

---

## Results

### Subtask 1
| Category         | Algorithm Type | Algorithm Name       | Best Column | F1 Score |
|------------------|----------------|----------------------|-------------|----------|
| Hazard Category  | Basic          | Logistic Regression  | Title       | 0.68     |
| Product Category | Basic          | Logistic Regression  | Title       | 0.62     |

### Subtask 2
| Category       | Algorithm Type | Algorithm Name       | Best Column | F1 Score |
|----------------|----------------|----------------------|-------------|----------|
| Hazard Type    | Basic          | Logistic Regression  | Text        | 0.41     |
| Product Type   | Basic          | Logistic Regression  | Title       | 0.24     |

- **Logistic Regression** consistently outperformed other models in both subtasks.

---

## Key Findings

1. **Model Performance**:
   - Logistic Regression achieved the best results for most tasks, especially after hyperparameter tuning.
   - Advanced models (e.g., XLM-RoBERTa) will be benchmarked in subsequent analyses to validate improvements.

2. **Impact of Text Features**:
   - Using long text generally improved performance for hazard classification but showed mixed results for product classification.

3. **Evaluation Metrics**:
   - Macro F1 score was the preferred metric, as it accounted for label imbalances better than accuracy.

---

## Running the Notebook

### Prerequisites
Ensure the following software and packages are installed:
- Python 3.8 or later
- Jupyter Notebook
- Required Python libraries (install via `pip install package`):
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- sklearn
- wordcloud import WordCloud
- evaluate
- torch
- transformers import AutoTokenizer, AutoModelForSequenceClassification
- datasets import Dataset, ClassLabel

  

### Instructions

1. **Clone or Download Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
