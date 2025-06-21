# 📧 AI-Based Email Spam Classifier  

This repository contains the implementation of an **Email Spam Classifier** developed using **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques. It is capable of predicting whether a given email is **Spam** or **Not Spam** based on its content.  

> 🛠 This project was developed as a part of a university semester project by a group of three members.  


## 📌 Project Overview  

Email spam has been a growing problem in digital communication, causing both inconvenience and security threats. To tackle this, we built a **machine learning model** that classifies emails as either **spam** or **not spam** using supervised learning algorithms and text analysis. We used a publicly available **email dataset from Kaggle**, containing labeled messages that allowed us to train and evaluate multiple classification models.  


### 🎯 Purpose  

The primary purpose of this project is to demonstrate the practical application of **Natural Language Processing (NLP)** and **Machine Learning (ML)** to solve a real-world problem — detecting and filtering spam emails. This classifier helps automate email filtering by intelligently analyzing the content and predicting whether an email is spam or not. It also serves as a hands-on project to solidify understanding of text preprocessing, vectorization, model selection, and evaluation techniques.  


### 🧭 Project Scope  

This project is designed as a console-based ML application with the following boundaries:  

- Focuses on **binary classification** of email text: spam or not spam.  
- Uses a publicly available email dataset from **Kaggle**.  
- Applies and compares multiple ML algorithms to identify the most effective.  
- Designed for demonstration and academic purposes, no email server or inbox integration.  
- Limited to **text-only analysis** (no attachments, links, or metadata handling).  
- Easily extendable to GUI or web-based implementations in the future.  


## 🔄 Project Workflow  

Here’s a step-by-step explanation of how the project was built:  

### 1. **Dataset Collection**
We used a **Kaggle-provided dataset** consisting of labeled emails as spam or ham (not spam). The dataset included email content and labels in structured format.  

### 2. **Data Cleaning**
- Removed unnecessary columns and handled null values.  
- Converted categorical labels (`spam`, `ham`) into binary numeric values (1 for spam, 0 for not spam).  
- Checked for duplicates and cleaned them.  

### 3. **Exploratory Data Analysis (EDA)**  
- Analyzed class distribution to identify any imbalance.  
- Visualized frequently occurring words in spam vs. ham emails using word clouds.  
- Studied email lengths, most used words, and their frequencies.  

### 4. **Text Preprocessing**  
Used **NLTK (Natural Language Toolkit)** for:  
- Converting all text to lowercase.  
- Tokenizing text into individual words.  
- Removing stopwords (like “the”, “is”, etc.) and punctuation.  
- Applying **Porter Stemming** to reduce words to their root form.  

A custom `transform_text()` function handled the entire preprocessing pipeline.  


### 5. **Feature Extraction**
- Applied **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization to convert the cleaned text into numeric features.
- This allowed our ML algorithms to understand and process the textual data efficiently.  

### 6. **Model Building & Comparison**  
We trained and evaluated the following models:  
- **Multinomial Naive Bayes**  
- **Bernoulli Naive Bayes**  
- **Gaussian Naive Bayes**  

Models were compared based on:  
- Accuracy  
- Precision  
- Recall  
- F1-Score  

> 📌 We selected the **Multinomial Naive Bayes** model as it gave the best trade-off between accuracy and precision on this dataset.  


### 7. **Model Saving and Integration**  
- The final model and vectorizer were saved using **Pickle**.  
- A simple CLI-based interface was created to input an email, transform it, and get a prediction result (Spam / Not Spam).  


## ✨ Key Features  

- Preprocessing pipeline for raw email text.  
- Real-time spam prediction using saved model.  
- High accuracy and precision on test data.  
- Multiple models trained and compared.  
- Clean and structured implementation.  


## 🧰 Technologies Used  

- **Python**  
- **NLTK** – Tokenization, stopword removal, stemming  
- **Scikit-learn** – TF-IDF vectorization, model training and evaluation  
- **Pandas & NumPy** – Data manipulation and analysis  
- **Matplotlib & Seaborn** – For EDA and visualizations  
- **Pickle** – Model serialization  


### 🚀 Future Improvements

While the current version achieves reliable spam classification through text analysis, several enhancements can be made to improve functionality and usability:

- **Add a web-based interface** using Flask or Streamlit for a user-friendly experience.
- **Integrate with email clients** to process real inbox messages in real-time.
- **Incorporate deep learning models** (e.g., LSTM or BERT) for improved accuracy on complex datasets.
- **Expand the dataset** with multilingual or more recent email data for broader coverage.
- **Add detailed performance reports** such as confusion matrix and ROC curves in the CLI.
- **Implement phishing or malicious content detection** alongside spam filtering.