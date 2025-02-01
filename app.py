import gradio as gr
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Charger le modèle, l'encodeur et le scaler
model = joblib.load('random_forest_model.pkl')
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Définir les colonnes catégoriques et numériques
categorical_columns = [
    "job", "marital", "education", "default", "housing", 
    "loan", "contact", "month", "poutcome"
]
numeric_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

# Fonction de prétraitement des données d'entrée
def preprocess_input(data):
    try:
        # Convertir les données d'entrée en DataFrame
        input_df = pd.DataFrame([data])

        # Normaliser les colonnes numériques
        input_df[numeric_columns] = scaler.transform(input_df[numeric_columns])

        # Encoder les colonnes catégoriques
        encoded_features = encoder.transform(input_df[categorical_columns])

        # Combiner les colonnes numériques avec les colonnes encodées
        numerical_features = input_df[numeric_columns].values
        final_features = np.concatenate([numerical_features, encoded_features], axis=1)

        return final_features
    except Exception as e:
        raise ValueError(f"Erreur dans le prétraitement des données : {e}")

# Fonction de prédiction
def predict(age, job, marital, education, default, balance, housing, loan, contact,
            day, month, duration, campaign, pdays, previous, poutcome):
    try:
        # Organiser les données en dictionnaire
        data = {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "day": day,
            "month": month,
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome
        }

        # Prétraiter les données
        input_data = preprocess_input(data)

        # Effectuer la prédiction
        prediction = model.predict(input_data)

        # Retourner "yes" ou "no"
        return "yes" if prediction[0] == 1 else "no"
    except Exception as e:
        return f"Erreur lors de la prédiction : {e}"

# Interface utilisateur avec les nouveaux composants Gradio
with gr.Blocks() as interface:
    gr.Markdown("# Prediction Interface")
    gr.Markdown("Entrez les données du client pour prédire l'issue (yes/no).")

    with gr.Row():
        age = gr.Number(label="Age", minimum=0, maximum=120)
        job = gr.Dropdown(choices=["admin", "technician", "blue-collar", "management", "retired", "services", "student", "unemployed", "self-employed"], label="Job")
        marital = gr.Dropdown(choices=["married", "single", "divorced"], label="Marital Status")
        education = gr.Dropdown(choices=["primary", "secondary", "tertiary", "unknown"], label="Education")
        default = gr.Dropdown(choices=["yes", "no"], label="Default (Has Credit in Default)")
        balance = gr.Number(label="Balance", minimum=-100000, maximum=100000)

    with gr.Row():
        housing = gr.Dropdown(choices=["yes", "no"], label="Housing (Has Housing Loan)")
        loan = gr.Dropdown(choices=["yes", "no"], label="Loan (Has Personal Loan)")
        contact = gr.Dropdown(choices=["telephone", "cellular", "unknown"], label="Contact Type")
        day = gr.Number(label="Day", minimum=1, maximum=31)
        month = gr.Dropdown(choices=["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], label="Month")
        duration = gr.Number(label="Duration", minimum=0)

    with gr.Row():
        campaign = gr.Number(label="Campaign", minimum=1)
        pdays = gr.Number(label="Pdays", minimum=-1)
        previous = gr.Number(label="Previous", minimum=0)
        poutcome = gr.Dropdown(choices=["success", "failure", "unknown", "other"], label="Poutcome (Previous Outcome)")

    predict_button = gr.Button("Predict")
    output = gr.Textbox(label="Prediction")

    # Lier les composants à la fonction de prédiction
    predict_button.click(
        predict, 
        inputs=[age, job, marital, education, default, balance, housing, loan, contact,
                day, month, duration, campaign, pdays, previous, poutcome], 
        outputs=output
    )

# Lancer l'application
interface.launch()