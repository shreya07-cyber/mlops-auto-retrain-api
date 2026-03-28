# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY data/       data/
COPY model/      model/
COPY app/        app/
COPY model.pkl   model.pkl

# Train at build time if model.pkl is missing (optional safety net)
# RUN python model/train.py

# Expose FastAPI port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
