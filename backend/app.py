from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load the trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return "Fake News Detection API is running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    news_text = data.get('text')

    if not news_text:
        return jsonify({'error': 'No text provided'}), 400

    # Vectorize the input text
    transformed_text = vectorizer.transform([news_text])

    # Make prediction
    prediction = model.predict(transformed_text)[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
