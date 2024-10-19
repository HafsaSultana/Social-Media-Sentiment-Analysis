# Social Media Sentiment Analysis

This project focuses on sentiment analysis of social media posts, classifying them into four categories: positive, negative, neutral, and irrelevant. Several machine learning models were trained and evaluated to determine their effectiveness in accurately identifying sentiments from a dataset of tweets.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
  - [Applied Algorithms](#applied-algorithms)
  - [Train-Test Split](#train-test-split)
  - [Text Vectorization](#text-vectorization)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [References](#references)

## Introduction
Social media platforms generate vast amounts of user-generated content daily. Understanding the sentiments expressed in these posts is crucial for businesses, researchers, and policymakers. This project analyzes sentiments from tweets, classifying them into four categories: **positive**, **negative**, **neutral**, and **irrelevant**.

## Dataset
The dataset used for this project is the "Twitter Sentiment Analysis" dataset, which can be found [here](https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/twitter_sentiment.csv). It contains tweets labeled with corresponding sentiment classes:

- Negative: 22,808 tweets
- Positive: 21,109 tweets
- Neutral: 18,603 tweets
- Irrelevant: 13,162 tweets

The dataset is stored as a CSV file, and we selected the relevant columns `sentiment` and `text` for analysis.

## Data Preprocessing
Data preprocessing is critical in preparing raw data for analysis. The following steps were taken to clean and preprocess the dataset:

- **Lowercasing**: All text data was converted to lowercase for uniformity.
- **Removing Unwanted Content**: URLs, HTML tags, special characters, and retweets were removed using the [preprocess-kgptalkie library](https://github.com/laxmimerit/preprocess_kgptalkie).
- **Handling Missing Values**: Missing text entries were filled with empty strings.
- **Text Cleaning**: Further cleaning steps were applied to remove noise from the text data.

These preprocessing steps improved the quality of the input data and boosted the performance of our sentiment analysis models.

## Model Training

### Applied Algorithms
We trained four machine learning models:
1. **Naive Bayes Classifier**: A simple probabilistic model for text classification.
2. **Decision Tree Classifier**: A tree-based model that makes decisions based on feature values.
3. **Support Vector Machine (SVM)**: A robust classifier for separating sentiment classes in high-dimensional space.
4. **Random Forest Classifier**: An ensemble model combining multiple decision trees to enhance accuracy and reduce overfitting.

### Train-Test Split
The dataset was split into 80% training data and 20% test data.

### Text Vectorization
We used the `TfidfVectorizer` from `sklearn` to convert text data into a matrix of numerical features that the models could understand.

## Model Evaluation
Each model's performance was evaluated using accuracy. Here’s a summary of the results:

| Model                | Accuracy (%) |
|----------------------|--------------|
| Naive Bayes           | 72.57        |
| Decision Tree         | 78.95        |
| Support Vector Machine (SVM) | 83.14        |
| Random Forest         | 89.65        |

## Results
- **Random Forest** emerged as the most effective model, with an accuracy of **89.65%**.
- **SVM** also performed well with an accuracy of **83.14%**.
- **Decision Tree** and **Naive Bayes** classifiers were less accurate, indicating that more complex models can better handle the complexities of sentiment analysis.

## Conclusion
This project demonstrated the use of multiple machine learning algorithms for sentiment analysis of social media posts. The **Random Forest** classifier provided the best accuracy and generalization capabilities, proving to be an excellent model for this task.

## Future Work
Potential improvements for future versions of this project include:
- Advanced deep learning models such as **LSTM** or **Transformers** should be implemented to improve accuracy further.
- Expanding the dataset to include posts from multiple social media platforms for a broader range of sentiment analysis.
- Incorporating multilingual sentiment analysis to extend the project's applicability in global contexts.

## References
1. [Twitter Sentiment Analysis Dataset](https://raw.githubusercontent.com/laxmimerit/All-CSV-ML-Data-Files-Download/master/twitter_sentiment.csv)
2. [Preprocess-kgptalkie Library](https://github.com/laxmimerit/preprocess_kgptalkie)
3. [Scikit-Learn Documentation](https://scikit-learn.org/stable/supervised_learning.html)
4. [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)


## Installation
To get started, clone the repository and install the necessary libraries.

## Install dependencies:
pip install -r requirements.txt

### Clone the repository:
```bash
git clone https://github.com/HafsaSultana/Social-Media-Sentiment-Analysis.git
cd Social-Media-Sentiment-Analysis


