import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import multiprocessing
import string
from collections import Counter
from tqdm import tqdm
import nltk
from nltk.stem import PorterStemmer
import re

# Download the NLTK data
nltk.download('punkt')

base_dir = r'C:\Users\Yogesh\Documents\GitHub\Naive-Bayes'
stemmer = PorterStemmer()

def read_files_from_directory(directory):
    """
    Read all text files from a directory.
    """
    files_content = []
    for root, _, files in os.walk(directory):
        for file in tqdm(files, desc=f"Reading files from {directory}"):
            with open(os.path.join(root, file), 'r', encoding='latin-1') as f:
                files_content.append(f.read())
    return files_content

def preprocess_text(text):
    """
    Preprocess the text by converting it to lowercase, removing punctuation,
    splitting into words, and stemming.
    """
    # Convert text to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Split into words
    words = nltk.word_tokenize(text)
    # Stem words
    words = [stemmer.stem(word) for word in words]
    return words

def count_words_in_directory(directory):
    """
    Count the occurrences of each word in all files of a directory using multiprocessing.
    """
    files_content = read_files_from_directory(directory)
    with multiprocessing.Pool() as pool:
        words_list = list(tqdm(pool.imap(preprocess_text, files_content), total=len(files_content), desc=f"Processing files from {directory}"))
    counter = Counter()
    for words in words_list:
        counter.update(words)
    return counter

def main():
    # Directories
    spam_dir = os.path.join(base_dir, 'archive/spam')
    ham_dir = os.path.join(base_dir, 'archive/ham')

    # Count words in spam and ham directories
    spam_counter = count_words_in_directory(spam_dir)
    ham_counter = count_words_in_directory(ham_dir)

    # Combine counters for total count
    total_counter = spam_counter + ham_counter

     # Prepare data for DataFrame
    data = []
    all_words = set(total_counter.keys())
    for word in all_words:
        # Check if the word contains any digits
        if re.search(r'\d', word) is None:
            spam_count = spam_counter[word]
            ham_count = ham_counter[word]
            total_count = total_counter[word]
            spam_ratio = round(spam_count / total_count, 3) if total_count > 0 else 0
            data.append([word, ham_count, spam_count, total_count, spam_ratio])
    
   # Create DataFrame
    df = pd.DataFrame(data, columns=['Word', 'Ham Count', 'Spam Count', 'Total Count', 'Spam Ratio'])
    # Sort DataFrame by Total Count in descending order
    df = df.sort_values(by='Total Count', ascending=False)
    # Save to CSV
    df.to_csv('trainingdata.csv', index=False)

    # Print results
    print(f"Total words: {sum(total_counter.values())}")
    print(f"Total unique words: {len(total_counter)}")
    print(f"Top 10 words in total: {total_counter.most_common(10)}")
    print(f"Total words in spam: {sum(spam_counter.values())}")
    print(f"Total unique words in spam: {len(spam_counter)}")
    print(f"Top 10 words in spam: {spam_counter.most_common(10)}")
    print(f"Total words in ham: {sum(ham_counter.values())}")
    print(f"Total unique words in ham: {len(ham_counter)}")
    print(f"Top 10 words in ham: {ham_counter.most_common(10)}")

if __name__ == '__main__':
    main()