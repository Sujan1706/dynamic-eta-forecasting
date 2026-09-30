# Dynamic ETA Forecasting Engine

A lightweight, data-driven Expected Time of Arrival (ETA) forecasting system for Indian Railways coaching trains (Problem Statement ID 26028).

---

## Project Structure

```text
dynamic-eta-forecasting/
│
├── backend/
│   ├── api/
│   ├── database/
│   ├── simulator/
│   ├── features/
│   ├── ml/
│   ├── services/
│   └── main.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
│
├── tests/
├── notebooks/
├── docs/
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Backend Directory Responsibilities

- **`backend/api/`**  
  Houses FastAPI route definitions and request/response schema contracts (e.g., endpoints for `/trains`, `/train/{id}`, `/station/{code}/arrivals`, `/simulate/event`, and `/model/metrics`).

- **`backend/database/`**  
  Manages SQLite database connections, SQLAlchemy ORM models (stations, routes, route stations, trains, journey logs), and corridor seed data.

- **`backend/simulator/`**  
  Contains the discrete-event train running simulator engine that advances trains along routes on a scheduled tick, injects trackside disruption events (signal halts, temporary speed restrictions, congestion), and emits normalized train-running states.

- **`backend/features/`**  
  Implements the tabular feature engineering pipeline that transforms normalized running state and timetable context into model-ready features (static, dynamic, historical, and contextual features per checkpoint).

- **`backend/ml/`**  
  Contains machine learning model definitions, training pipelines (LightGBM/XGBoost segment regressors), evaluation metrics calculations (MAE/RMSE vs baseline), and multi-station segment chaining logic.

- **`backend/services/`**  
  Encapsulates core business logic and orchestration services, including the deterministic baseline ETA heuristic calculation (`schedule + delay - recovery`) and state management caches.

- **`backend/main.py`**  
  The primary entry point for the FastAPI application. Sets up CORS, initializes database tables, registers routers, and manages the simulation background process lifecycle.

---

## Supporting Directories

- **`data/raw/`**: Unprocessed external datasets or raw simulator event logs.
- **`data/processed/`**: Cleaned, transformed tabular feature datasets ready for ML training.
- **`data/models/`**: Serialized model weights and evaluation metadata artifacts.
- **`tests/`**: Automated unit and integration test suite (simulator advancement, baseline calculations, feature extraction, API routes).
- **`notebooks/`**: Exploratory data analysis, validation reports, and model comparison notebooks.
- **`docs/`**: Product Requirements Document (PRD) and architecture specifications.
