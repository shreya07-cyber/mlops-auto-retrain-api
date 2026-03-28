import pandas as pd
import pickle
import json
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = "data/data.csv"
MODEL_PATH = "model.pkl"
METRICS_PATH = "metrics.json"


def train():
    print("📥 Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"   Rows: {len(df)}, Columns: {list(df.columns)}")

    X = df[["area", "bedrooms", "age"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🏋️  Training RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"✅ MAE  : ₹{mae:,.0f}")
    print(f"✅ R²   : {r2:.4f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"💾 Model saved -> {MODEL_PATH}")

    metrics = {"mae": round(mae, 2), "r2": round(r2, 4), "rows_trained": len(df)}
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"📊 Metrics saved -> {METRICS_PATH}")

    return metrics


if __name__ == "__main__":
    train()
