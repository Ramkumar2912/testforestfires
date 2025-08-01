from flask import Flask, request, render_template
import pickle
import numpy as np

application = Flask(__name__)
app = application

# Load model and scaler
ridge_model = pickle.load(open('ridge.pkl', 'rb'))  # put your model here
scaler = pickle.load(open('scaler.pkl', 'rb'))      # put your scaler here

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [
            float(request.form['Temperature']),
            float(request.form['RH']),
            float(request.form['Ws']),
            float(request.form['Rain']),
            float(request.form['FFMC']),
            float(request.form['DMC']),
            float(request.form['DC']),
            float(request.form['ISI']),
            float(request.form['BUI']),
        ]

        # Scale and predict
        scaled = scaler.transform([data])
        prediction = ridge_model.predict(scaled)[0]

        return f"<h2>Predicted FWI: {prediction:.2f}</h2><br><a href='/'>Back</a>"
    
    except Exception as e:
        return f"<h2>Error occurred:</h2><pre>{e}</pre><br><a href='/'>Back</a>"

if __name__ == '__main__':
    app.run(debug=True)
