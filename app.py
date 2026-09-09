import pickle
# pyrefly: ignore [missing-import]
import os
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Use paths relative to this script's directory so it works from any CWD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, 'housepred.pkl'), 'rb'))
scaler = pickle.load(open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    a_data = (np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.transform(a_data)
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    raw_output = model.predict(final_input)[0]

    # Model output is in units of $100,000 (California Housing dataset)
    price_usd = raw_output * 100000
    price_inr = price_usd * 83.5   # approx 1 USD = 83.5 INR

    prediction_text = (
        "Predicted House Price:  "
        "USD $ {:,.0f}  |  INR Rs. {:,.0f}".format(price_usd, price_inr)
    )
    return render_template("home.html", prediction_text=prediction_text)


if __name__ =="__main__":
    app.run(debug=True)