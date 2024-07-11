import os
import string
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
import math

# Download NLTK data if necessary
nltk.download('punkt')

# Initialize the Porter Stemmer
stemmer = PorterStemmer()

def preprocess_text(text):
    """
    Preprocesses the text by tokenizing, converting to lowercase, removing punctuation,
    and stemming each word.
    """
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = nltk.word_tokenize(text.lower())
    stemmed_words = [stemmer.stem(word) for word in words]
    return stemmed_words

def load_training_data():
    """
    Loads trainingdata.csv into a Pandas DataFrame.
    """
    df = pd.read_csv('trainingdata.csv')
    return df

def train_naive_bayes(training_data):
    """
    Trains a Naive Bayes classifier based on the training data.
    Returns dictionaries of word probabilities for spam and ham, and prior probabilities.
    """
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
    """
    Classifies the given email text as spam or ham using Naive Bayes.
    Returns 'spam' or 'ham' and the probability of being spam.
    """
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

def main():
    # Load training data
    training_data = load_training_data()

    # Train Naive Bayes classifier
    spam_probabilities, ham_probabilities, prior_spam, prior_ham = train_naive_bayes(training_data)

    # Example email file path
    email_file = 'test.txt'

    # Read email text
    with open(email_file, 'r', encoding='latin-1') as f:
        email_text = f.read()

    # Classify the email
    classification, probability = classify_email(email_text, spam_probabilities, ham_probabilities, prior_spam, prior_ham)

    # Print classification result
    print(f"The email is classified as {classification} with a probability of {probability:.3f}")

if __name__ == "__main__":
    main()