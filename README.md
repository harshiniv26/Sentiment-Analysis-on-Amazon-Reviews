# Sentiment Analysis on Amazon Customer Reviews

---

##  Overview

This project aims to classify Amazon customer reviews into **Positive**, **Negative**, or **Neutral** sentiments using Natural Language Processing (NLP) and classical machine learning algorithms. By analyzing text data, this system helps understand customer opinions and improves decision-making for businesses.

---

## 📊 Data Collection and Preprocessing

### 🔹 Source:
- A publicly available dataset containing Amazon product reviews and their associated sentiment labels.

### 🔹 Preprocessing Steps:
1. **Lowercasing** – Convert all text to lowercase.
2. **Removing Special Characters & Numbers** – Eliminate non-alphabetic characters.
3. **Tokenization** – Break text into individual words.
4. **Stopword Removal** – Remove common words like "is", "and", "the" using NLTK.
5. **Vectorization** – Apply **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numerical format.

---

## 🤖 Model Building and Training

Four different machine learning models were built and trained on the processed data:

- **Naïve Bayes**
- **Logistic Regression**
- **Linear Support Vector Classifier (LinearSVC)**
- **Random Forest Classifier**

The data was split into **training** and **testing** sets for evaluation.

---

## 🧪 Model Evaluation

| Model                  | Accuracy  | Precision | Recall    | F1-Score  |
|------------------------|-----------|-----------|-----------|-----------|
| Naïve Bayes            | 90.34%    | 81.61%    | 90.34%    | 85.75%    |
| Logistic Regression    | 92.68%    | 90.17%    | 92.68%    | 90.49%    |
| LinearSVC              | 98.84%    | 98.86%    | 98.84%    | 98.83%    |
| **Random Forest** ✅   | **99.74%**| **99.74%**| **99.74%**| **99.74%**|

👉 The **Random Forest Classifier** achieved the best performance and was selected as the final model.

---

## 📉 Confusion Matrix (Random Forest)

A confusion matrix helps visualize the classification accuracy:

## ✅ Conclusion

- Proper text preprocessing and the use of TF-IDF vectorization significantly improve model performance.
- Ensemble methods like **Random Forest** are highly effective for text classification tasks.
- This project demonstrates that classical ML algorithms can achieve high accuracy (99%+) in sentiment analysis when paired with appropriate preprocessing.
- Future improvements may include integrating a live interface, real-time sentiment tracking, or using deep learning models like BERT or LSTM.

---

## 🧰 Tools & Technologies

- Python
- Google Colab
- Scikit-learn
- NLTK
- Pandas, NumPy
- TF-IDF
- Matplotlib & Seaborn (for visualization)


---
