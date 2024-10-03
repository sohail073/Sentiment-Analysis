# Sentiment Analysis on IMDB Movie Reviews

## Overview
This project involves training and evaluating multiple machine learning models to perform sentiment analysis on a dataset of IMDB movie reviews. The goal is to classify reviews as either positive or negative based on their content.

## Dataset
The dataset used in this project is the **IMDB Dataset of 50k Movie Reviews**, which contains 50,000 reviews with their corresponding sentiment labels (positive or negative).

## Preprocessing
The following preprocessing steps were performed on the dataset:

- Removed HTML tags
- Removed special characters and punctuations
- Converted text to lowercase
- Removed numbers
- Tokenized text
- Removed stopwords
- Performed stemming (using Porter Stemmer)

## Feature Extraction
**TF-IDF (Term Frequency-Inverse Document Frequency)** was used to extract features from the preprocessed text data.

## Model Training and Evaluation
The following models were trained and evaluated:

- Naive Bayes
- Logistic Regression
- Support Vector Classifier (SVC)

The performance of each model was evaluated using **accuracy, precision, recall,** and **F1-score**.

## Results
The results of the model evaluations are as follows:

- **Naive Bayes:** 
  - Accuracy: 0.8604
  - Precision: 0.85
  - Recall: 0.87
  - F1-score: 0.86

- **Logistic Regression:** 
  - Accuracy: 0.8922
  - Precision: 0.90
  - Recall: 0.88
  - F1-score: 0.89

- **SVC:** 
  - Accuracy: 0.8838
  - Precision: 0.91
  - Recall: 0.85
  - F1-score: 0.88

## Comparison of Model Performance
A bar chart was created to compare the performance of the three models.

## Learning Curve
A learning curve was created to visualize the performance of the Logistic Regression model as the training data size increases.

## Bias-Variance Analysis
A bias-variance analysis was performed using cross-validation to evaluate the performance of the Logistic Regression model.

## Model Deployment
The trained Logistic Regression model was saved using `pickle` for future use.

## Conclusion
The results of this project show that the **Logistic Regression** model performs the best among the three models, with an accuracy of **0.8922**. The learning curve and bias-variance analysis also indicate that the model is well-balanced and generalizes well to unseen data.
