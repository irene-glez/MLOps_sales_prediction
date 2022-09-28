from flask import Flask, request, jsonify
import os
import pickle
from sklearn.model_selection import cross_val_score
import pandas as pd
import sqlite3

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/", methods=['GET'])
def hello():
    return "Bienvenido a mi API para el modelo advertising"

# 1. endpoint que devuelva la predicción de los nuevos datos enviados mediante argumentos en la llamada (/predict):
@app.route('/predict', methods=['GET'])
def predict():
    model = pickle.load(open('data/advertising_model','rb'))

    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)
    prediction = model.predict([[tv,radio,newspaper]])
    return "The prediction of sales investing that amount of money in TV, radio and newspaper is: " + str(round(prediction[0],2)) + 'k €'

# ejemplo input: https://ireneglez.pythonanywhere.com/predict?tv=200&radio=33&newspaper=44


# # 2. endpoint para almacenar nuevos registros en la base de datos que deberá estar previamente creada. (/ingest_data)
@app.route('/ingest_data', methods=['GET'])
def ingest_data():
    connection = sqlite3.connect("data/my_database.db")
    crsr = connection.cursor()

    tv = request.args.get('tv', 0)
    radio = request.args.get('radio', 0)
    newspaper = request.args.get('newspaper', 0)
    sales = request.args.get('sales', 0)

    insertion = '''INSERT INTO advertising (TV, radio, newspaper, sales) VALUES (?,?,?,?)'''
    crsr.execute(insertion,(tv, radio, newspaper, sales)).fetchall()
    connection.commit()

    return str(crsr.rowcount) + " record inserted."

# 3 Reentrenar de nuevo el modelo con los posibles nuevos registros que se recojan. (/retrain)
@app.route('/retrain', methods=['GET'])
def retrain():
    connection = sqlite3.connect("data/my_database.db")
    crsr = connection.cursor()

    query = '''SELECT * FROM advertising'''

    crsr.execute(query)
    data = crsr.fetchall()
    cols = [description[0] for description in crsr.description]
    df = pd.DataFrame(data, columns=cols)

    X = df.drop(columns=['sales'])
    y = df['sales']

    model = pickle.load(open('data/advertising_model','rb'))
    model.fit(X,y)
    scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_absolute_error')

    pickle.dump(model, open('advertising_model_retrain_v1','wb'))

    return "New model retrained and saved as advertising_model_retrain_v1. The results of MAE with cross validation of 10 folds is: " + str(abs(round(scores.mean(),2)))