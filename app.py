from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('models/rf_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            feature_cols = [
                'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
                'Fwd Packet Length Max', 'Bwd Packet Length Max', 'Flow Bytes/s'
            ]
            user_input = [float(request.form.get(col, 0)) for col in feature_cols]
            data = pd.DataFrame([dict(zip(feature_cols, user_input))])
            prediction = model.predict(data)[0]
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template('results.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)