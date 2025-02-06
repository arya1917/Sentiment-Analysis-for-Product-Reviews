from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your trained model and vectorizer
with open('improved_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('improved_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review_text = request.form['review']
    
    # Transform the text input
    review_vector = vectorizer.transform([review_text])
    
    # Get probability scores
    probabilities = model.predict_proba(review_vector)[0]
    
    # Define sentiment labels (adjust if needed)
    sentiment_labels = ['negative', 'neutral', 'positive']
    
    # Find the highest probability sentiment
    predicted_sentiment = sentiment_labels[np.argmax(probabilities)]
    
    # Prepare JSON response
    response = {
        'sentiment': predicted_sentiment.capitalize(),
        'probabilities': {
            'negative': float(probabilities[0]),
            'neutral': float(probabilities[1]),
            'positive': float(probabilities[2])
        }
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
