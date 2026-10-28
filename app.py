import json
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
model = joblib.load("model.joblib")
meta = json.load(open("meta.json"))
extras = json.load(open("extras.json")) if os.path.exists("extras.json") else None

LABELS = {
    "brand": "Brand", "model": "Phone Model", "os_type": "OS Type", "condition": "Physical Condition",
    "city_tier": "City Tier", "seller_type": "Seller Type",
    "release_year": "Release Year", "purchase_year": "Purchase Year",
    "ram_gb": "RAM (GB)", "storage_gb": "Storage (GB)",
    "screen_size_inches": "Screen Size (Inches)", "battery_capacity": "Battery Capacity (mAh)",
    "processor_score": "Processor Score", "camera_score": "Camera Score",
    "has_5g": "Has 5G?", "original_price": "Original Price (units)",
    "age_months": "Age (Months)", "usage_hours_per_day": "Daily Usage Hours",
    "battery_health": "Battery Health (%)", "screen_cracked": "Is Screen Cracked?",
    "body_damage": "Body Damage?", "repair_history": "Repaired Before?",
    "water_damage": "Water Damage?", "warranty_remaining_months": "Warranty Remaining (Months)",
    "box_available": "Box Available?", "charger_available": "Charger Available?",
    "market_demand_score": "Market Demand Score",
}

# Which fields go in which panel (any column not listed lands in the last panel)
GROUPS = [
    ("Phone Specs", ["brand", "model", "os_type", "release_year", "ram_gb", "storage_gb",
                     "screen_size_inches", "battery_capacity"]),
    ("Condition & Age", ["condition", "original_price", "purchase_year", "age_months",
                         "battery_health", "city_tier"]),
    ("Scores & Usage", ["processor_score", "camera_score", "usage_hours_per_day",
                        "warranty_remaining_months", "market_demand_score"]),
    ("Seller & Extras", ["seller_type", "has_5g", "screen_cracked", "body_damage",
                         "repair_history", "water_damage", "box_available", "charger_available"]),
]
NUMBER_BOXES = {"release_year", "purchase_year", "ram_gb", "storage_gb"}


def fmt(x):
    x = float(x)
    return int(x) if x.is_integer() else round(x, 2)


def build_field(col):
    label = LABELS.get(col, col.replace("_", " ").title())
    if col == "model" and meta.get("brand_models"):
        return {"name": col, "label": label, "kind": "model_select"}
    if col in meta["cat"]:
        opts = meta["cat"][col]
        return {"name": col, "label": label, "kind": "select", "options": opts, "default": opts[0]}
    n = meta["num"][col]
    lo, hi, d = fmt(n["min"]), fmt(n["max"]), fmt(n["default"])
    if n["binary"]:
        return {"name": col, "label": label, "kind": "radio", "no": lo, "yes": hi, "default": d}
    if col in NUMBER_BOXES:
        return {"name": col, "label": label, "kind": "number", "min": lo, "max": hi, "default": d}
    whole = all(float(v).is_integer() for v in (n["min"], n["max"], n["default"]))
    return {"name": col, "label": label, "kind": "slider", "min": lo, "max": hi,
            "default": d, "step": 1 if whole else 0.01}


def build_groups():
    used, groups = set(), []
    for title, cols in GROUPS:
        fields = [build_field(c) for c in cols if c in meta["columns"]]
        used.update(cols)
        groups.append({"title": title, "fields": fields})
    leftover = [c for c in meta["columns"] if c not in used]
    groups[-1]["fields"] += [build_field(c) for c in leftover]
    return groups


@app.route("/")
def home():
    return render_template("index.html", groups=build_groups(), extras=extras,
                           brand_models=meta.get("brand_models", {}))


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
