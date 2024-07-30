from pickle import load
from sklearn.compose import ColumnTransformer
import pandas as pd


def process_data(df: pd.DataFrame):

    categorical_features = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

    with open('data/column_transformer.pkl', 'rb') as file:
        column_transformer: ColumnTransformer = load(file)

    transformed_data = column_transformer.transform(df)

    lst = list(column_transformer.transformers_[0][1].get_feature_names_out())
    lst.extend([col for col in df.columns if col not in categorical_features])

    new_df = pd.DataFrame(transformed_data, columns=lst)

    return new_df

def load_model_and_predict(df):
    with open('data/model_weights.pkl', 'rb') as file:
        model = load(file)

    prediction = model.predict(df)[0]

    prediction_proba = model.predict_proba(df)[0]

    encode_prediction_proba = {
        0: "Клиент будет доволен с вероятностью",
        1: "Клиент будет не доволен с вероятностью"
    }

    encode_prediction = {
        0: "Ура, мы сделали мир чуточку лучше",
        1: "О нет, клиент расстроен"
    }

    prediction_data = {}
    for key, value in encode_prediction_proba.items():
        prediction_data.update({value: prediction_proba[key]})

    prediction_df = pd.DataFrame(prediction_data, index=[0])
    prediction = encode_prediction[prediction]

    return prediction, prediction_df
