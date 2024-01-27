from flask import render_template, request, jsonify
from app import app
from app.model_utils import load_model, predict

model = load_model()  # Load your pre-trained model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    user_input = data['user_input']
    prediction = predict(model, user_input)
    return jsonify({'prediction': prediction})
