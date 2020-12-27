import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('fire_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    if output>(0.5):
        return render_template('index.html',Pred='Your Forest is in Danger.')
    else:
        return render_template('index.html',Pred='Your Forest is safe.')



if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 8080)