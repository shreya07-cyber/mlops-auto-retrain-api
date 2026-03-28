# 🏠 MLOps: Automated House Price Retraining Pipeline

A production-grade MLOps system that **automatically retrains and redeploys** a house price prediction model whenever new data is pushed to the repository.

---

## 📁 Project Structure

```
mlops-auto-retrain-api/
│
├── data/
│   ├── data.csv               ← Training data (push new rows here to trigger retraining)
│   └── generate_data.py       ← Script to generate synthetic data
│
├── model/
│   └── train.py               ← Trains model, saves model.pkl + metrics.json
│
├── app/
│   └── main.py                ← FastAPI prediction server
│
├── model.pkl                  ← Serialized trained model (auto-updated by CI/CD)
├── metrics.json               ← MAE, R², rows trained (auto-updated by CI/CD)
├── Dockerfile                 ← Containerizes the API
├── requirements.txt
├── .github/workflows/
│   └── train.yml              ← GitHub Actions: auto-retrain + deploy on data push
└── README.md
```

---

## ⚙️ Local Setup

### 1. Clone & install
```bash
git clone https://github.com/YOUR_USERNAME/mlops-auto-retrain-api.git
cd mlops-auto-retrain-api
pip install -r requirements.txt
```

### 2. Train the model
```bash
python model/train.py
```
Output:
```
📥 Loading data...
🏋️  Training RandomForestRegressor...
✅ MAE  : ₹12,345
✅ R²   : 0.9712
💾 Model saved -> model.pkl
📊 Metrics saved -> metrics.json
```

### 3. Start the API
```bash
uvicorn app.main:app --reload
```
API runs at: http://localhost:8000

### 4. Make a prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"area": 1500, "bedrooms": 3, "age": 10}'
```
Response:
```json
{
  "predicted_price": 248500.0,
  "currency": "INR",
  "model_version": "latest"
}
```

### 5. Check model metrics
```bash
curl http://localhost:8000/metrics
```

---

## 🐳 Docker

### Build & run
```bash
docker build -t house-price-api .
docker run -p 8000:8000 house-price-api
```

---

## 🔁 Automated Retraining (CI/CD)

The GitHub Actions workflow (`.github/workflows/train.yml`) triggers automatically when:

| Trigger | What happens |
|---|---|
| `data/data.csv` is updated | Model retrains on new data |
| `model/train.py` is changed | Model retrains with new logic |
| `app/main.py` is changed | API is rebuilt and tested |
| Manual dispatch | Retrain on demand from GitHub UI |

### Steps performed automatically:
1. ✅ Install dependencies
2. 🏋️ Retrain `model/train.py`
3. 💾 Commit updated `model.pkl` + `metrics.json` back to repo
4. 🐳 Build new Docker image
5. 🔍 Smoke test: hit `/health` and `/predict`
6. 🚀 (Optional) Push image to Docker Hub

### To trigger retraining:
```bash
# Add new house price rows to data.csv and push
echo "2000,4,5,320000" >> data/data.csv
git add data/data.csv
git commit -m "Add new housing data"
git push
# → GitHub Actions automatically retrains and redeploys!
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Model status |
| GET | `/metrics` | MAE, R², training rows |
| POST | `/predict` | Get house price prediction |
| GET | `/docs` | Swagger UI (auto-generated) |

---

## 🔑 Optional: Deploy to Docker Hub

Add these secrets to your GitHub repo (`Settings → Secrets`):
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

Then uncomment the push step in `.github/workflows/train.yml`.
