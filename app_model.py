from flask import Flask, request, jsonify
import os
import pickle
from sklearn.model_selection import cross_val_score
import pandas as pd


os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/", methods=['GET'])
def hello():
    return "Bienvenido a mi API del modelo advertising"

# 1. Crea un endpoint que devuelva la predicción de los nuevos datos enviados mediante argumentos en la llamada
@app.route('/v1/predict', methods=['GET'])
def predict():
    # model = pickle.load(open('data/advertising_model','rb'))

    tv = request.args.get('tv', None)
    # radio = request.args.get('radio', None)
    # newspaper = request.args.get('newspaper', None)

        return tv

    # if tv is None or radio is None or newspaper is None:
    #     return "Missing args, the input values are needed to predict"
    # else:
    #     prediction = model.predict([[tv,radio,newspaper]])
    #     return "The prediction of sales investing that amount of money in TV, radio and newspaper is: " + str(round(prediction[0],2)) + 'k €'



# # 2. Crea un endpoint que reentrene de nuevo el modelo con los datos disponibles en la carpeta data, que guarde ese modelo reentrenado, devolviendo en la respuesta la media del MAE de un cross validation con el nuevo modelo
# @app.route('/v1/retrain', methods=['PUT'])
# def retrain():
#     df = pd.read_csv('data/Advertising.csv', index_col=0)
#     X = df.drop(columns=['sales'])
#     y = df['sales']

#     model = pickle.load(open('data/advertising_model','rb'))
#     model.fit(X,y)
#     pickle.dump(model, open('data/advertising_model_v1','wb'))

#     scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_absolute_error')

#     return "New model retrained and saved as advertising_model_v1. The results of MAE with cross validation of 10 folds is: " + str(abs(round(scores.mean(),2)))


# app.run()