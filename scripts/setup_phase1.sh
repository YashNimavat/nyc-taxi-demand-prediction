#!/bin/bash

# Phase 1 Setup Script for NYC Taxi Demand Prediction Project
# This script sets up the complete development environment

echo "🚀 Setting up NYC Taxi Demand Prediction - Phase 1"
echo "================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate || source venv/Scripts/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p data/{raw,processed,features}
mkdir -p models
mkdir -p metrics
mkdir -p plots
mkdir -p logs
mkdir -p config
mkdir -p src/{data,models,utils,production}
mkdir -p notebooks
mkdir -p scripts
mkdir -p docker

echo "✅ Directory structure created"

# Initialize Git if not already done
if [ ! -d ".git" ]; then
    echo "🎯 Initializing Git repository..."
    git init
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Setup .gitignore
echo "📝 Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/

# Jupyter Notebooks
.ipynb_checkpoints

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Data files (tracked by DVC)
data/raw/*.parquet
data/raw/*.csv
data/processed/*.parquet
models/*.pkl

# Logs
logs/*.log
*.log

# MLflow
mlruns/

# DVC
.dvc/cache

# Environment variables
.env

# Temporary files
*.tmp
*.temp
EOF

echo "✅ .gitignore created"

# Initialize DVC if not already done
if [ ! -d ".dvc" ]; then
    echo "📊 Initializing DVC..."
    dvc init --no-scm
    echo "✅ DVC initialized"
else
    echo "✅ DVC already initialized"
fi

# Create initial commit
if [ ! -f ".git/refs/heads/main" ] && [ ! -f ".git/refs/heads/master" ]; then
    echo "💾 Creating initial commit..."
    git add .
    git commit -m "Initial project setup - Phase 1"
    
    # Rename default branch to main if needed
    git branch -M main
    echo "✅ Initial commit created"
fi

echo ""
echo "🎉 Phase 1 setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Set up DagsHub repository:"
echo "   python scripts/setup_dagshub.py --repo-owner YOUR_USERNAME"
echo ""
echo "2. Configure DagsHub authentication:"
echo "   export DAGSHUB_USER_TOKEN='your_token_here'"
echo ""
echo "3. Add DagsHub as remote:"
echo "   git remote add origin https://dagshub.com/YOUR_USERNAME/nyc-taxi-demand-prediction.git"
echo ""
echo "4. Run data pipeline:"
echo "   dvc repro"
echo ""
echo "5. Push to DagsHub:"
echo "   dvc push"
echo "   git push -u origin main"
echo ""
echo "📚 Read COLLABORATION.md for team development workflow"