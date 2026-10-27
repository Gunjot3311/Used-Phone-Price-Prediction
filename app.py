import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
model = joblib.load("model.joblib")
meta = json.load(open("meta.json"))

@app.route("/")
def home():
    return render_template("index.html", meta=meta)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    row = {}
    for col in meta["columns"]:
        val = data.get(col)
        row[col] = float(val) if col in meta["num"] else val
    price = float(model.predict(pd.DataFrame([row]))[0])
    return jsonify({"price": round(max(price, 0), 2)})

if __name__ == "__main__":
    app.run(debug=True)
