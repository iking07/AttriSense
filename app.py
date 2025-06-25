from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

default_values = {
    'Age_Group': 2,
    'Distance_Category': 2,
    'Experience_Ratio': 0.5,
    'Income_Per_Age': 100,
    'EnvironmentSatisfaction': 3,
    'JobInvolvement': 2,
    'JobLevel': 2,
    'StockOptionLevel': 1
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        satisfaction = int(request.form["satisfaction"])
        income = float(request.form["income"])
        years = int(request.form["years"])
        overtime = int(request.form["overtime"])
        balance = int(request.form["balance"])
    except Exception as e:
        return f"Error in form input: {e}", 400

    input_data = {
        'JobSatisfaction': satisfaction,
        'MonthlyIncome': income,
        'YearsAtCompany': years,
        'OverTime': overtime,
        'WorkLifeBalance': balance
    }

    full_input = []
    for col in model_columns:
        value = input_data.get(col, default_values.get(col, 0))
        full_input.append(float(value))

    scaled_input = scaler.transform([full_input])
    prediction = model.predict(scaled_input)[0]
    result = "Likely to Leave" if prediction == 1 else "Likely to Stay"

    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
