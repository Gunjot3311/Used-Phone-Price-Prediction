# ---- Run this in Google Colab (one cell, after uploading the CSV) ----
import json, joblib, sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

CSV = "/content/used_phone_price_prediction_1M (3) (1).csv"   # change if your file name differs
df = pd.read_csv(CSV)

# helper / leaky columns + very high-cardinality 'model' are dropped
drop_cols = ["resale_ratio", "age_group", "demand_group", "model"]
data = df.sample(n=min(100_000, len(df)), random_state=42).drop(columns=drop_cols, errors="ignore")

X = data.drop(columns=["resale_price"])
y = data["resale_price"]
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

prep = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])
model = Pipeline([("prep", prep), ("model", Ridge(alpha=1.0))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("MAE:", round(mean_absolute_error(y_test, pred), 2), "| R2:", round(r2_score(y_test, pred), 3))

# Save the model + form metadata (dropdown options, numeric ranges) for the web app
meta = {"cat": {c: sorted(X[c].dropna().unique().tolist()) for c in cat_cols}, "num": {}}
for c in num_cols:
    meta["num"][c] = {"min": float(X[c].min()), "max": float(X[c].max()),
                      "default": float(X[c].median()), "binary": bool(X[c].nunique() <= 2)}
meta["columns"] = X.columns.tolist()

joblib.dump(model, "model.joblib", compress=3)
json.dump(meta, open("meta.json", "w"))
print("scikit-learn version to pin in requirements.txt:", sklearn.__version__)

from google.colab import files
files.download("model.joblib"); files.download("meta.json")
