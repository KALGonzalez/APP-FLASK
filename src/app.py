from utils import db_connect
engine = db_connect()

# your code here
from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('/workspaces/APP-FLASK/models/model_30trees_15depth.pkl')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        airline = int(request.form['airline'])
        source_city = int(request.form['source_city'])
        destination_city = int(request.form['destination_city'])
        flight_class = int(request.form['class'])
        duration = float(request.form['duration'])

        # El orden y número de columnas debe coincidir con lo que tu modelo espera
        features = np.array([[airline, source_city, destination_city, flight_class, duration]])
        prediction = model.predict(features)[0]

        return render_template('index.html', prediction=round(prediction, 2))
    except Exception as e:
        return f"Ocurrió un error: {e}"

if __name__ == '__main__':
    app.run(debug=True)

