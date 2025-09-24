# NYC Taxi Demand Prediction - Phase 1: Data Versioning & MLflow Integration

Complete project structure with DagsHub integration for multi-developer collaboration.

## Project Structure
```
nyc-taxi-demand-prediction/
├── .dvc/
├── .dagshub/
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
├── models/
├── notebooks/
├── src/
│   ├── data/
│   ├── models/
│   ├── utils/
│   └── production/
├── config/
├── metrics/
├── plots/
├── logs/
├── scripts/
└── docker/
```

## Quick Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup DagsHub
python scripts/setup_dagshub.py

# 3. Initialize DVC
dvc init
dvc remote add origin https://dagshub.com/YOUR_USERNAME/nyc-taxi-demand-prediction.dvc

# 4. Run data pipeline
dvc repro

# 5. Push to DagsHub
dvc push
git add .
git commit -m "Phase 1: Initial setup"
git push origin main
```

## Team Development Workflow
Each developer can work on different components simultaneously without conflicts.

## Features
- ✅ Data versioning with DVC
- ✅ Experiment tracking with MLflow
- ✅ Multi-developer collaboration
- ✅ Reproducible pipelines
- ✅ Production integration