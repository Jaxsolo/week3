from flask import Flask, render_template, request
import numpy as np
import pickle
#import joblib
app = Flask(__name__)
filename = 'wine.pkl'
model = pickle.load(open(filename, 'rb'))    # load the model
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])  # The user input is processed here
def predict():
    Volatile_acidity = request.form['volatile acidity']
    Chlorides = request.form['chlorides']
    Density = request.form['density']
    Alcohol = request.form['alcohol']
    pred = model.predict(np.array([[Volatile_acidity, Chlorides, Density, Alcohol ]]))
    #print(pred)
    return render_template('index.html', predict=str(pred))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
