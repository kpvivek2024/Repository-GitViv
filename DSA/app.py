from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load your trained model and model columns
model = joblib.load('loan_prediction_model.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    return "✅ Bank Loan Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Convert JSON to DataFrame
    input_df = pd.DataFrame([data])

    # One-hot encode categorical features
    input_df = pd.get_dummies(input_df)

    # Reindex to match model training columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Predict
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Construct response
    result = {
        "input": data,
        "predicted_class": int(prediction[0]),
        "predicted_label": '✅ Likely to Repay Loan' if prediction[0] == 0 else '⚠️ High Risk of Default',
        "prediction_probability": {
            "Repay": float(prediction_proba[0][0]),
            "Default": float(prediction_proba[0][1])
        }
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
