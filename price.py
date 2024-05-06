from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    val1 = float(request.form['bedrooms'])
    val2 = float(request.form['bathrooms'])
    val3 = float(request.form['floors'])
    val4 = float(request.form['yr_built'])
    arr = np.array([[val1, val2, val3, val4]])
    pred = model.predict(arr)

    return render_template('index.html', prediction=int(pred[0]))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
