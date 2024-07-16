import os
from flask import Flask, request, render_template, jsonify
import pandas as pd
import math
import string
import nltk
from nltk.stem import PorterStemmer
import plotly.graph_objs as go
import plotly.utils
import json
import re
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
from datetime import datetime

nltk.download('punkt', quiet=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'fallback_secret_key')

#Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

stemmer = PorterStemmer()

#Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

#Mock user database
users = {
    "admin": generate_password_hash("admin_password")
}

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

def check_auth(username, password):
    return username in users and check_password_hash(users.get(username), password)

def authenticate():
    return ('Could not verify your access level for that URL.\n'
            'You have to login with proper credentials', 401,
            {'WWW-Authenticate': 'Basic realm="Login Required"'})

def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = nltk.word_tokenize(text.lower())
    stemmed_words = [stemmer.stem(word) for word in words]
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # remove corrupted characters
    text = re.sub(r'\s+', ' ', text).strip()  # whitespace
    return stemmed_words

def load_training_data():
    try:
        df = pd.read_csv('trainingdata.csv')
        app.logger.info("Training data loaded successfully")
    except Exception as e:
        app.logger.error(f"Error loading training data: {str(e)}")
        df = pd.DataFrame(columns=['Word', 'Ham Count', 'Spam Count', 'Total Count', 'Spam Ratio'])
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

        spam_prob = math.log((spam_count + 1) / (total_spam + total_words))
        ham_prob = math.log((ham_count + 1) / (total_ham + total_words))

        spam_probabilities[word] = spam_prob
        ham_probabilities[word] = ham_prob

    prior_spam = math.log(total_spam / (total_spam + total_ham))
    prior_ham = math.log(total_ham / (total_spam + total_ham))

    app.logger.info(f"Prior probabilities: Spam {math.exp(prior_spam)}, Ham {math.exp(prior_ham)}")
    app.logger.info("Sample word probabilities:")
    for word in ['hello', 'hi']:
        app.logger.info(f"  {word}: Spam {math.exp(spam_probabilities.get(word, 0))}, Ham {math.exp(ham_probabilities.get(word, 0))}")

    return spam_probabilities, ham_probabilities, prior_spam, prior_ham

def classify_email(email_text, spam_probabilities, ham_probabilities, prior_spam, prior_ham):
    words = preprocess_text(email_text)
    
    log_prob_spam = prior_spam
    log_prob_ham = prior_ham

    app.logger.info(f"Initial log probabilities: Spam {log_prob_spam}, Ham {log_prob_ham}")

    for word in words:
        if word in spam_probabilities:
            log_prob_spam += spam_probabilities[word]
            log_prob_ham += ham_probabilities[word]
            app.logger.info(f"Word: {word}, Spam prob: {math.exp(spam_probabilities[word])}, Ham prob: {math.exp(ham_probabilities[word])}")
        else:
            app.logger.info(f"Word: {word} not found in training data")

    app.logger.info(f"Final log probabilities: Spam {log_prob_spam}, Ham {log_prob_ham}")

    max_log_prob = max(log_prob_spam, log_prob_ham)
    prob_spam = math.exp(log_prob_spam - max_log_prob)
    prob_ham = math.exp(log_prob_ham - max_log_prob)

    total = prob_spam + prob_ham
    prob_spam /= total
    prob_ham /= total

    app.logger.info(f"Final probabilities: Spam {prob_spam}, Ham {prob_ham}")

    if prob_spam > 0.95:
        return 'spam', prob_spam
    else:
        return 'ham', prob_spam

def safe_json_dumps(obj):
    try:
        return json.dumps(obj, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        app.logger.error(f"Error in JSON encoding: {e}")
        return json.dumps({"error": str(e)})

def create_word_frequency_bar_chart(training_data, top_n=20):
    try:
        top_words = training_data.nlargest(top_n, 'Total Count')
        
        trace1 = go.Bar(x=top_words['Word'].tolist(), y=top_words['Spam Count'].tolist(), name='Spam')
        trace2 = go.Bar(x=top_words['Word'].tolist(), y=top_words['Ham Count'].tolist(), name='Ham')
        
        layout = go.Layout(
            title='Top {} Word Stems in Dataset'.format(top_n),
            xaxis=dict(title='Word'),
            yaxis=dict(title='Frequency'),
            barmode='group'
        )
        
        fig = go.Figure(data=[trace1, trace2], layout=layout)
        return safe_json_dumps(fig)
    except Exception as e:
        app.logger.error(f"Error in create_word_frequency_bar_chart: {e}")
        return safe_json_dumps({"error": str(e)})

def sanitize_string(s):
    if not isinstance(s, str):
        return str(s)
    #Remove unusable characters
    return re.sub(r'[\x00-\x1F\x7F-\x9F]', '', s)

def create_scatter_plot(training_data):
    try:
        #Remove non-numeric data
        numeric_data = training_data[training_data['Ham Count'].apply(lambda x: isinstance(x, (int, float))) & 
                                     training_data['Spam Count'].apply(lambda x: isinstance(x, (int, float))) & 
                                     training_data['Spam Ratio'].apply(lambda x: isinstance(x, (int, float)))]
        
        #Convert to list and remove non valid characters
        x = numeric_data['Ham Count'].fillna(0).tolist()
        y = numeric_data['Spam Count'].fillna(0).tolist()
        text = [sanitize_string(word) for word in numeric_data['Word'].fillna('').tolist()]
        color = numeric_data['Spam Ratio'].fillna(0).tolist()
        
        trace = go.Scatter(
            x=x,
            y=y,
            mode='markers',
            text=text,
            marker=dict(
                size=8,
                color=color,
                colorscale='Viridis',
                colorbar=dict(title='Spam Ratio'),
                showscale=True
            )
        )
        
        layout = go.Layout(
            title='Word Stem Frequencies: Spam vs Ham',
            xaxis=dict(title='Ham Frequency', type='log'),
            yaxis=dict(title='Spam Frequency', type='log'),
            hovermode='closest'
        )
        
        fig = go.Figure(data=[trace], layout=layout)
        json_data = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        #Additional sanitization of the JSON string
        json_data = sanitize_string(json_data)
        
        #Check for potential issues in the JSON string
        json.loads(json_data)  # This will raise an error if the JSON is invalid
        
        return json_data
    except Exception as e:
        app.logger.error(f"Error in create_scatter_plot: {e}")
        #Return a simplified version of the chart if there's an error
        return json.dumps({
            "data": [{"x": [0], "y": [0], "type": "scatter", "mode": "markers"}],
            "layout": {"title": "Error in Scatter Plot"}
        })

def create_pie_chart(training_data):
    try:
        total_spam = training_data['Spam Count'].sum()
        total_ham = training_data['Ham Count'].sum()
        
        labels = ['Spam', 'Ham']
        values = [total_spam, total_ham]
        
        trace = go.Pie(labels=labels, values=values)
        layout = go.Layout(title='Distribution of Spam and Ham Emails')
        
        fig = go.Figure(data=[trace], layout=layout)
        return safe_json_dumps(fig)
    except Exception as e:
        app.logger.error(f"Error in create_pie_chart: {e}")
        return safe_json_dumps({"error": str(e)})

def evaluate_accuracy(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, pos_label='spam')
    recall = recall_score(true_labels, predicted_labels, pos_label='spam')
    f1 = f1_score(true_labels, predicted_labels, pos_label='spam')
    return precision, recall, f1

@app.route('/', methods=['GET', 'POST'])
@requires_auth
def index():
    try:
        results = []
        charts = {}
        accuracy_metrics = {}
        
        training_data = load_training_data()
        
        if request.method == 'POST':
            if 'file' not in request.files:
                return 'No file part'
            files = request.files.getlist('file')
            if not files or files[0].filename == '':
                return 'No selected file'

            spam_probabilities, ham_probabilities, prior_spam, prior_ham = train_naive_bayes(training_data)

            true_labels = []
            predicted_labels = []

            for file in files:
                if file:
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                    file.save(file_path)
                    app.logger.info(f"File saved: {file_path}")
                    with open(file_path, 'r', encoding='latin-1') as f:
                        email_text = f.read()
                    app.logger.info(f"File content (first 100 chars): {email_text[:100]}")
                    
                    result, probability = classify_email(email_text, spam_probabilities, ham_probabilities, prior_spam, prior_ham)
                    probability = round(probability, 3)
                    results.append((result, probability, file.filename))

                    true_label = 'spam' if 'spam' in file.filename.lower() else 'ham'
                    true_labels.append(true_label)
                    predicted_labels.append(result)

            #Calculate accuracy metrics
            precision, recall, f1 = evaluate_accuracy(true_labels, predicted_labels)
            accuracy_metrics = {
                'precision': round(precision, 3),
                'recall': round(recall, 3),
                'f1_score': round(f1, 3)
            }

            #Log the classification activity
            app.logger.info(f"{datetime.now()} - Classified {len(files)} emails. Accuracy metrics: {accuracy_metrics}")

        #Create the three visualizations
        charts['word_frequency'] = create_word_frequency_bar_chart(training_data)
        charts['scatter_plot'] = create_scatter_plot(training_data)
        charts['pie_chart'] = create_pie_chart(training_data)
        
        return render_template('index.html', results=results, charts=charts, accuracy_metrics=accuracy_metrics)
    except Exception as e:
        app.logger.error(f"An error occurred in index route: {str(e)}")
        return "An error occurred", 500

@app.route('/search_word', methods=['POST'])
def search_word():
    word = request.form['word']
    stemmed_word = stemmer.stem(word.lower())  # Stem the search word
    training_data = load_training_data()
    word_data = training_data[training_data['Word'] == stemmed_word]
    if not word_data.empty:
        result = word_data.to_dict('records')[0]
        result['Original Word'] = word  # Include the original word in the result
        return jsonify(result)
    
    #If exact stem not found, search for words that contain the stemmed word
    containing_words = training_data[training_data['Word'].str.contains(stemmed_word, case=False, na=False)]
    if not containing_words.empty:
        results = containing_words.to_dict('records')
        return jsonify({
            "message": f"Exact stem '{stemmed_word}' not found. Showing related words:",
            "results": results
        })
    
    return jsonify({"error": f"No words found for '{word}' (stem: '{stemmed_word}')"})

@app.route('/classify_text', methods=['POST'])
def classify_text():
    text = request.form['text']
    training_data = load_training_data()
    spam_probabilities, ham_probabilities, prior_spam, prior_ham = train_naive_bayes(training_data)
    result, probability = classify_email(text, spam_probabilities, ham_probabilities, prior_spam, prior_ham)
    return jsonify({"result": result, "probability": round(probability, 3)})

@app.route('/log', methods=['GET'])
@requires_auth
def view_log():
    with open('app.log', 'r') as log_file:
        logs = log_file.readlines()
    return render_template('log.html', logs=logs)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)