# NYC Taxi Demand Prediction - Team Collaboration Guide

Complete guide for multi-developer collaboration using DagsHub, DVC, and MLflow.

## Quick Start

### 1. Clone Repository
```bash
git clone https://dagshub.com/YOUR_USERNAME/nyc-taxi-demand-prediction.git
cd nyc-taxi-demand-prediction
```

### 2. Setup Environment
```bash
# Run setup script
chmod +x scripts/setup_phase1.sh
./scripts/setup_phase1.sh

# Setup DagsHub integration
python scripts/setup_dagshub.py --repo-owner YOUR_USERNAME
```

### 3. Authentication
```bash
export DAGSHUB_USER_TOKEN="your_token_here"
```

### 4. Download Data
```bash
dvc pull
```

## Development Workflow

### Starting New Feature
```bash
git checkout main
git pull origin main
dvc pull  # Get latest data
git checkout -b feature/your-feature-name
```

### Working with Data
```bash
# After modifying data pipeline
dvc repro
dvc push  # Push data changes
```

### Model Development
```bash
# Train new model
python src/models/train_model.py

# Evaluate model
python src/models/evaluate_model.py

# Check results in MLflow dashboard
```

### Submitting Changes
```bash
git add .
git commit -m "feat: descriptive message"
git push origin feature/your-feature-name
# Create pull request on DagsHub
```

## Team Roles

### Data Engineers
- Maintain data pipelines (`src/data/`)
- Ensure data quality and validation
- Manage DVC configurations

### Data Scientists
- Develop models (`src/models/`)
- Run experiments and track in MLflow
- Create model evaluation reports

### ML Engineers
- Manage production code (`src/production/`)
- Deploy models and monitoring
- Maintain infrastructure scripts

## Best Practices

### Data Management
- Use DVC for all data files >10MB
- Document data sources and transformations
- Validate data quality in pipelines

### Experiment Tracking
- Log all experiments to MLflow
- Use descriptive run names and tags
- Document model parameters and metrics

### Code Quality
- Follow PEP 8 style guidelines
- Add docstrings and type hints
- Write unit tests for critical functions

### Collaboration
- Use feature branches for development
- Review code before merging
- Communicate changes that affect team

## File Structure

```
nyc-taxi-demand-prediction/
├── README.md                   # Project overview
├── COLLABORATION.md           # This guide
├── requirements.txt           # Dependencies
├── dvc.yaml                  # DVC pipeline
├── config/                   # Configuration files
├── data/                     # Data (tracked by DVC)
├── src/                      # Source code
│   ├── data/                 # Data processing
│   ├── models/              # Model training
│   ├── utils/               # Utilities
│   └── production/          # Production code
├── models/                   # Trained models
├── metrics/                  # Evaluation metrics
├── plots/                    # Visualization outputs
├── notebooks/                # Jupyter notebooks
└── scripts/                  # Setup and utility scripts
```

## Troubleshooting

### DVC Issues
```bash
# Reset DVC cache
dvc cache dir ~/.dvc/cache

# Fix corrupted files
dvc checkout --force
```

### MLflow Issues
```bash
# Reset experiment tracking
rm -rf mlruns/
python scripts/setup_dagshub.py --repo-owner YOUR_USERNAME
```

### Git Issues
```bash
# Reset to clean state
git reset --hard origin/main
git clean -fd
dvc checkout
```

## Support Resources

- [DagsHub Documentation](https://dagshub.com/docs)
- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- Project Issues: Create issues on DagsHub repository

## Commands Reference

### DVC Commands
```bash
dvc init                    # Initialize DVC
dvc add data/file.csv      # Track file with DVC
dvc push                   # Push data to remote
dvc pull                   # Pull data from remote
dvc repro                  # Reproduce pipeline
dvc status                 # Check pipeline status
```

### MLflow Commands
```bash
mlflow ui                  # Start MLflow UI
mlflow experiments list    # List experiments
mlflow runs list           # List runs
```

### Git + DVC Workflow
```bash
git add .
dvc push
git commit -m "message"
git push origin branch-name
```