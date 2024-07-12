import os
from flask import Flask, request, render_template
import pandas as pd
import math
import string
import nltk
from nltk.stem import PorterStemmer

nltk.download('punkt')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = nltk.word_tokenize(text.lower())
    stemmed_words = [stemmer.stem(word) for word in words]
    return stemmed_words

def load_training_data():
    df = pd.read_csv('trainingdata.csv')
    return df

def train_naive_bayes(training_data):
    total_spam = training_data['Spam Count'].sum()
    total_ham = training_data['Ham Count'].sum()
    total_words = training_data['Total Count'].sum()

    spam_probabilities = {}
    ham_probabilities = {}

    for index, row in training_data.iterrows():
        word = row['Word']
        spam_count = row['Spam Count']
        ham_count = row['Ham Count']
        total_count = row['Total Count']

        spam_prob = math.log((spam_count + 1) / (total_spam + total_words))
        ham_prob = math.log((ham_count + 1) / (total_ham + total_words))

        spam_probabilities[word] = spam_prob
        ham_probabilities[word] = ham_prob

    prior_spam = math.log(total_spam / (total_spam + total_ham))
    prior_ham = math.log(total_ham / (total_spam + total_ham))

    return spam_probabilities, ham_probabilities, prior_spam, prior_ham

def classify_email(email_text, spam_probabilities, ham_probabilities, prior_spam, prior_ham):
    words = preprocess_text(email_text)
    
    log_prob_spam = prior_spam
    log_prob_ham = prior_ham

    for word in words:
        if word in spam_probabilities:
            log_prob_spam += spam_probabilities[word]
        if word in ham_probabilities:
            log_prob_ham += ham_probabilities[word]

    prob_spam = math.exp(log_prob_spam - max(log_prob_spam, log_prob_ham))
    prob_ham = 1 - prob_spam

    if prob_spam > prob_ham:
        return 'spam', prob_spam
    else:
        return 'ham', prob_ham

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    probability = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            with open(file_path, 'r', encoding='latin-1') as f:
                email_text = f.read()
            
            training_data = load_training_data()
            spam_probabilities, ham_probabilities, prior_spam, prior_ham = train_naive_bayes(training_data)
            result, probability = classify_email(email_text, spam_probabilities, ham_probabilities, prior_spam, prior_ham)
            probability = round(probability, 3)

    return render_template('index.html', result=result, probability=probability)

if __name__ == '__main__':
    app.run(debug=True)