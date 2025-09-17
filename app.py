# app.py

import gradio as gr
import numpy as np
import joblib
import tensorflow as tf

# ----------------- Load Model & Scaler -----------------
scaler = joblib.load("scaler.pkl")
model = tf.keras.models.load_model("churn_model.h5")

# ----------------- Prediction Function -----------------
def predict_churn(RowNumber,CreditScore, Gender, Age, Tenure, Balance, NumOfProducts,
                  HasCrCard, IsActiveMember, EstimatedSalary,
                  Geo_France, Geo_Germany, Geo_Spain):

    # Build input
    input_data = np.array([[RowNumber, CreditScore, Gender, Age, Tenure, Balance,
                            NumOfProducts, HasCrCard, IsActiveMember,
                            EstimatedSalary, Geo_France, Geo_Germany, Geo_Spain]])

    # Scale input
    input_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data, verbose=0)

    return "❌ Likely to Exit" if prediction[0][0] > 0.5 else "✅ Likely to Stay"

# ----------------- Gradio UI -----------------
inputs = [
    gr.Number(label="Months"),
    gr.Number(label="Credit Score"),
    gr.Radio([0,1], label="Gender (0=Male, 1=Female)"),
    gr.Number(label="Age"),
    gr.Number(label="Tenure"),
    gr.Number(label="Balance"),
    gr.Number(label="Number of Products"),
    gr.Radio([0,1], label="Has Credit Card"),
    gr.Radio([0,1], label="Is Active Member"),
    gr.Number(label="Estimated Salary"),
    gr.Radio([0,1], label="Geo: France"),
    gr.Radio([0,1], label="Geo: Germany"),
    gr.Radio([0,1], label="Geo: Spain"),
]

app = gr.Interface(
    fn=predict_churn,
    inputs=inputs,
    outputs=gr.Label(label="Prediction"),
    title="Bank Customer Churn Prediction",
    description="Enter customer details to predict if the customer will churn."
)

if __name__ == "__main__":
    app.launch()
