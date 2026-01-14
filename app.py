from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('music_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    age = int(data['age'])
    gender = int(data['gender'])
    
    prediction = model.predict([[age, gender]])[0]
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)