# Food Hazard Detection Challenge

This assignment focuses on the **SemEval 2025 Task 9: Food Hazard Detection Challenge**, where the objective is to classify food hazard incidents based on text data. Specifically, the task involves predicting the **hazard category**, **product category**, **product**, and **hazard type** from both short texts (titles) and long texts (descriptions) provided in the dataset.

We will explore and apply various machine learning models, ranging from basic to advanced, to analyze and classify the data. The models will be evaluated and compared based on their performance, and the best-performing approach will be selected for generating final predictions.

The end goal is to submit the predictions, along with the models, evaluation metrics, and competition ranking.


## Installation of Required Packages

To install all the necessary packages for this project, you can run the following command. It's recommended to create and activate a virtual environment first:

```bash
pip install pandas numpy matplotlib seaborn wordcloud scikit-learn torch transformers datasets evaluate
```

## Installation Using `requirements.txt`

```bash
pip install -r requirements.txt
```

## Data Exploration and Preprocessing

Next step, is to load and explore the provided dataset to better understand its structure and quality. Key tasks include:

1. **Data Loading and Inspection**:
   - Load the dataset and inspect its contents (e.g., columns, sample data).
   - Check the data types of each column and ensure they are appropriate for the task.

2. **Missing Data Handling**:
   - Identify and handle any missing or null values in the dataset. This could involve imputation, removal, or other strategies depending on the extent and importance of the missing data.

3. **Label Distribution Analysis**:
   - Analyze the distribution of the target labels (hazard categories, products, etc.).
   - Check for any class imbalances that could affect model performance. This can be addressed by techniques like oversampling, undersampling, or using weighted loss functions.

4. **Data Visualization**:
   - Visualize the distribution of labels, text lengths, or other relevant features using plots (e.g., histograms, bar plots, word clouds).
   - Comment on the findings, identifying any potential challenges or patterns that could guide the modeling process.

5. **Data Cleaning**:
   - Clean and preprocess text data, such as removing stopwords, punctuation, and other irrelevant characters.

6. **Preprocessing Strategy**:
   - Based on the data exploration findings, create a preprocessing strategy that includes:
     - Tokenization and Vectorization techniques (e.g., TF-IDF, word embeddings).
     - Handling imbalanced classes.
     - Data normalization or scaling, if required for specific models.


## Benchmark Analysis 1: Basic Machine Learning Algorithms

The benchmark analysis involves classifying food hazard and product data using basic machine learning models, starting with the **hazard category** prediction. We will:

1. **Classify by Hazard Category (Using Titles)**:
   - Begin with the **title** column, applying basic machine learning models like Logistic Regression, Naive Bayes, SVM, and others with fine-tuned versions of them.
   - Evaluate models using **accuracy**, **precision**, **recall**, and **F1 score**.

2. **Classify by Hazard Category (Using Full Text)**:
   - Extend the analysis to the **text** column for more context and improved models performance.
   - Compare the results with those from the title-based models.

3. **Classify by Hazard**:
   - Predict **hazard types** using the best-performing fine tuned or vanilla model from the previous steps.

4. **Classify by Product Category**:
   - Predict **product category** using both **title** and **text** columns, and compare performance across models like we did with the Hazard Category.



## Benchmark Analysis 2: Advanced Machine Learning with XLM-RoBERTa

In **Benchmark Analysis 2**, we will explore **XLM-RoBERTa**, a transformer model, to predict hazard and product categories/types, building on the basic ML models from Benchmark Analysis 1.

#### **1. Establishing a Baseline**

- To begin, we trained the **XLM-RoBERTa** model on both the **title** and **text** columns of the dataset. This allowed us to establish a baseline for the **hazard category** prediction. Initially, we focused on training the model without any extensive hyperparameter tuning. The idea was to evaluate how the raw, pre-trained XLM-RoBERTa model performs on the title and text data and which data source (title vs. text) yields better results for hazard category classification.

#### **2. Using the Right Column**

- Once the initial models were trained, we compared the performance of the XLM-RoBERTa model on the **title** and **text** columns. We expected that the **text** column, being more comprehensive, would provide more context and improve the model’s ability to predict the **hazard category**. By evaluating the baseline scores, we could determine if the richer context of the full text resulted in a more accurate prediction, or if the shorter, more concise **title** data proved sufficient for hazard classification.

#### **3. Hyperparameter Tuning**

- With the baseline performance established, the next step was to perform **hyperparameter tuning**. We explored various configurations to fine-tune the **XLM-RoBERTa** model and improve its performance. This step involved experimenting with parameters such as learning rate, batch size, and the number of training epochs. The goal was to see if we could boost the model’s accuracy and F1 score by adjusting these hyperparameters.

#### **4. Repeating the Process for Other Tasks**

- After fine-tuning the model for hazard category prediction, we repeated the same process for predicting **hazard type**, **product category**, and **product type**. As with the hazard category task, we trained the model on both the **title** and **text** columns, evaluated the results, and compared the performance of the model across different prediction tasks.

#### **5. Final Model Selection**

- Once the transformer model was fine-tuned for all prediction tasks, we compared its performance with the **basic machine learning models** used in Benchmark Analysis #1. This comparison helped us determine the best model for predicting **hazard categories**, **hazard types**, **product categories**, and **product types**. The final model selection was based on overall performance metrics, including **precision**, **recall**, and **F1 score**.


## Benchmark Analysis Results

### **Benchmark Analysis #1: Basic Machine Learning Models**

| Features | Best Algorithm  | Target | Macro F1 Score | 
|-------|---------------------------|-----------------|------|
| Title | Tuned Logistic Regression | Hazard Category | 0.68 |
| Title | Tuned Logistic Regression | Hazard Type | 0.37 |
| Title | Tuned Logistic Regression | Product Category | 0.62 |
| Title | Tuned Logistic Regression | Product Type | 0.24 |
| Text | Tuned Logistic Regression | Hazard Category | 0.63 |
| Text | Tuned Logistic Regression | Hazard Type | 0.41 |
| Text | Tuned Logistic Regression | Product Category | 0.57 |
| Text | Tuned Logistic Regression | Product Type | 0.20 |

### **Benchmark Analysis #2: Advanced Transformer Models (XLM-RoBERTa)**

| Features | Algorithm  | Tuning | Target | Macro F1 Score | 
|-------|---------------|--|-----------------|--|
| Title | XLM - RoBERTa | Vanilla | Hazard Category | 0.18 |
| Text | XLM - RoBERTa | Vanilla | Hazard Category | 0.26 |
| Text | XLM - RoBERTa | Tuned | Hazard Category | 0.65 |
| Title | XLM - RoBERTa | Tuned | Hazard Type | 0.13 |
| Text | XLM - RoBERTa | Tuned | Hazard Type | 0.28 |
| Title | XLM - RoBERTa | Vanilla | Product Category | 0.03 |
| Text | XLM - RoBERTa | Vanilla | Product Category | 0.02 |
| Title | XLM - RoBERTa | Tuned | Product Category | 0.34 |
| Title | XLM - RoBERTa | Tuned | Product Type | 0.0004 |
| Text | XLM - RoBERTa | Tuned | Product Type | 0.0019 |

### **Best Candidates for Sub Task 1 (Hazard & Product Classification)**

| Prediction Category | Best Algorithm Type | Best Algorithm Name | Best Feature | Macro F1 Score |
|---------------------|---------------------|---------------------|--------------|----------------|
| Hazard Category     | Basic               | Tuned Logistic Regression | Title    | 0.68           |
| Product Category    | Basic               | Tuned Logistic Regression | Title    | 0.62           |

### **Best Candidates for Sub Task 2 (Hazard & Product Type Classification)**

| Prediction Category | Best Algorithm Type | Best Algorithm Name | Best Feature | Macro F1 Score |
|---------------------|---------------------|---------------------|--------------|----------------|
| Hazard Type         | Basic               | Tuned Logistic Regression | Text     | 0.41           |
| Product Type        | Basic               | Tuned Logistic Regression | Title    | 0.24           |
