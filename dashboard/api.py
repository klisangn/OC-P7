# Import libraries
import os
import pandas as pd
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

# Load the model
parent_dir = os.path.abspath('..')
model_path = os.path.join(parent_dir, 'notebooks/mlruns/966063637948665005/a2cd557f418b4b81a6f694c7dbc4d55e/artifacts/model/model.pkl')
model = pickle.load(open(model_path, "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.read_json(data)
    prediction = model.predict_proba(df)
    pred = prediction[0][0]
    print(pred)
    return jsonify(pred)

if __name__ == '__main__':
    app.run(port=5000, debug=True)

