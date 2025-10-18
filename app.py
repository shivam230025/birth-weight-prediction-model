from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import numpy as np

app= Flask(__name__)

def form_to_dict(form_data):
    gestation= float(form_data['gestation'])
    parity= int(form_data['parity'])
    age= float(form_data['age'])
    height= float(form_data['height'])
    weight= float(form_data['weight'])
    smoke= float(form_data['smoke'])

    response= {
        "gestation": [gestation],
        "parity": [parity],
        "age": [age],
        "height": [height],
        "weight": [weight],
        "smoke": [smoke]
    }
    return response


@app.route('/', methods=["GET"])
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def prediction():

    babyData_form= request.form
    babyData_dict= form_to_dict(babyData_form)
    babyData_df= pd.DataFrame(babyData_dict)

    with open('linerRegressionModel.pkl', 'rb') as obj:
        mymodel= pickle.load(obj)

    prediction= mymodel.predict(babyData_df)
    prediction= round(float(prediction[0]), 2)

    return render_template("result.html", prediction=prediction)   


if __name__=='__main__':
    app.run(debug=True)