from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# -------------------------------
# Load model and scaler safely
# -------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "kmeans_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# -------------------------------
# Cluster Interpretation
# -------------------------------

cluster_dict = {
    0: "High Value Customer - High balance & high purchases",
    1: "Low Usage Customer - Low transactions",
    2: "Cash Advance Heavy User - Risk segment",
    3: "Installment Based Buyer",
    4: "Premium Credit User"
}

# -------------------------------
# Routes
# -------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect input features
        features = [
            float(request.form["BALANCE"]),
            float(request.form["BALANCE_FREQUENCY"]),
            float(request.form["PURCHASES"]),
            float(request.form["ONEOFF_PURCHASES"]),
            float(request.form["INSTALLMENTS_PURCHASES"]),
            float(request.form["CASH_ADVANCE"]),
            float(request.form["PURCHASES_FREQUENCY"]),
            float(request.form["ONEOFF_PURCHASES_FREQUENCY"]),
            float(request.form["PURCHASES_INSTALLMENTS_FREQUENCY"]),
            float(request.form["CASH_ADVANCE_FREQUENCY"]),
            float(request.form["CASH_ADVANCE_TRX"]),
            float(request.form["PURCHASES_TRX"]),
            float(request.form["CREDIT_LIMIT"]),
            float(request.form["PAYMENTS"]),
            float(request.form["MINIMUM_PAYMENTS"]),
            float(request.form["PRC_FULL_PAYMENT"]),
            float(request.form["TENURE"]),
        ]

        # Convert to numpy array
        input_array = np.array([features])

        # Scale input
        input_scaled = scaler.transform(input_array)

        # Predict cluster
        cluster = model.predict(input_scaled)[0]

        # Get interpretation
        interpretation = cluster_dict.get(cluster, "Unknown Cluster")

        return render_template(
            "result.html",
            cluster=cluster,
            interpretation=interpretation
        )

    except Exception as e:
        return f"Error occurred: {str(e)}"


# -------------------------------
# Run App
# -------------------------------

if __name__ == "__main__":
    app.run(debug=True)