#!/usr/bin/env python3
"""
DagsHub Setup and Configuration Script
Initializes DagsHub integration for multi-developer collaboration
"""

import os
import sys
import dagshub
import mlflow
import yaml
import logging
from pathlib import Path
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DagsHubSetup:
    """Setup DagsHub integration for the project"""
    
    def __init__(self, repo_owner: str, repo_name: str):
        """Initialize DagsHub setup"""
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.repo_url = f"https://dagshub.com/{repo_owner}/{repo_name}"
        
    def initialize_dagshub(self):
        """Initialize DagsHub repository connection"""
        logger.info(f"🚀 Initializing DagsHub connection to {self.repo_url}")
        
        try:
            # Initialize DagsHub with MLflow support
            dagshub.init(
                repo_owner=self.repo_owner,
                repo_name=self.repo_name,
                mlflow=True
            )
            
            logger.info("✅ DagsHub initialization successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ DagsHub initialization failed: {e}")
            return False
    
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        logger.info("🔧 Setting up MLflow tracking...")
        
        # Set MLflow tracking URI
        tracking_uri = f"{self.repo_url}.mlflow"
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create experiment
        experiment_name = "taxi-demand-prediction"
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"✅ Created MLflow experiment: {experiment_name}")
        except mlflow.exceptions.MlflowException:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id
            logger.info(f"✅ Using existing MLflow experiment: {experiment_name}")
        
        mlflow.set_experiment(experiment_name)
        
        return experiment_id
    
    def create_team_config(self):
        """Create team collaboration configuration"""
        logger.info("👥 Creating team collaboration configuration...")
        
        team_config = {
            'project': {
                'name': 'NYC Taxi Demand Prediction',
                'description': 'Real-time taxi demand prediction using H3 spatial indexing and XGBoost',
                'repository': self.repo_url,
                'version': '1.0.0'
            },
            'team': {
                'roles': {
                    'data_engineers': ['Responsible for data pipelines and feature engineering'],
                    'data_scientists': ['Model development and experimentation'],
                    'ml_engineers': ['Production deployment and monitoring'],
                    'maintainers': ['Project oversight and permissions']
                },
                'permissions': {
                    'data_access': 'all_team_members',
                    'model_deployment': 'ml_engineers_maintainers',
                    'experiment_tracking': 'data_scientists_ml_engineers',
                    'repository_admin': 'maintainers'
                }
            },
            'development_workflow': {
                'branching_strategy': 'feature_branches',
                'code_review': 'required',
                'ci_cd': 'github_actions',
                'data_validation': 'dvc_pipeline',
                'model_validation': 'mlflow_tracking'
            }
        }
        
        # Save team configuration
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        config_path = config_dir / "team_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(team_config, f, default_flow_style=False, indent=2)
        
        logger.info(f"✅ Team configuration saved to {config_path}")
        return team_config
    
    def setup_git_hooks(self):
        """Setup Git hooks for data validation"""
        logger.info("🪝 Setting up Git hooks...")
        
        hooks_dir = Path(".git/hooks")
        if not hooks_dir.exists():
            logger.warning("⚠️ Git repository not initialized. Run 'git init' first.")
            return
        
        # Pre-commit hook for data validation
        pre_commit_hook = """#!/bin/bash
# Pre-commit hook for data validation and DVC checks

echo "🔍 Running pre-commit validation..."

# Check if DVC files are tracked
if git diff --cached --name-only | grep -E "\\.(dvc|lock)$"; then
    echo "✅ DVC files detected in commit"
    
    # Validate DVC pipeline
    if ! dvc status; then
        echo "❌ DVC pipeline is not up to date. Run 'dvc repro' first."
        exit 1
    fi
fi

# Check for large files
large_files=$(git diff --cached --name-only | xargs ls -la 2>/dev/null | awk '$5 > 10485760 {print $9 " (" $5 " bytes)"}')
if [ ! -z "$large_files" ]; then
    echo "❌ Large files detected (>10MB). Please use DVC to track these files:"
    echo "$large_files"
    exit 1
fi

echo "✅ Pre-commit validation passed"
"""
        
        hook_path = hooks_dir / "pre-commit"
        with open(hook_path, 'w') as f:
            f.write(pre_commit_hook)
        
        # Make executable
        os.chmod(hook_path, 0o755)
        
        logger.info("✅ Git hooks setup completed")
    
    def create_collaboration_guide(self):
        """Create collaboration guide for team members"""
        logger.info("📚 Creating collaboration guide...")
        
        guide_content = f"""# Team Collaboration Guide

## Repository Setup
```bash
# Clone repository
git clone {self.repo_url}.git
cd {self.repo_name}

# Install dependencies
pip install -r requirements.txt

# Setup DagsHub authentication
export DAGSHUB_USER_TOKEN="your_token_here"

# Initialize DVC
dvc pull  # Download latest data
```

## Development Workflow

### 1. Starting New Feature
```bash
git checkout main
git pull origin main
dvc pull
git checkout -b feature/your-feature-name
```

### 2. Working with Data
```bash
# After modifying data processing
dvc repro  # Reproduce pipeline
dvc push   # Push data changes
```

### 3. Model Experiments
```bash
# Train new model
python src/models/train_model.py

# Check MLflow dashboard at:
# {self.repo_url}.mlflow
```

### 4. Submitting Changes
```bash
# Add changes
git add .
git commit -m "feat: descriptive commit message"

# Push changes
git push origin feature/your-feature-name

# Create pull request on DagsHub
```

## Best Practices

### Data Management
- Always use DVC for datasets >10MB
- Run `dvc repro` after data pipeline changes
- Document data sources and transformations

### Experiment Tracking
- Log all experiments to MLflow
- Use descriptive run names
- Tag important runs for easy reference

### Code Quality
- Follow PEP 8 style guidelines
- Add docstrings to functions
- Write unit tests for critical functions

### Collaboration
- Use feature branches for development
- Write clear commit messages
- Review code before merging

## Troubleshooting

### DVC Issues
```bash
# Reset DVC cache
dvc cache dir --unset
dvc cache dir ~/.dvc/cache

# Fix corrupted files
dvc checkout --force
```

### MLflow Issues
```bash
# Reset MLflow tracking
rm -rf mlruns/
python scripts/setup_dagshub.py
```

## Support
- DagsHub Documentation: https://dagshub.com/docs
- Project Issues: {self.repo_url}/issues
- Team Chat: [Add your team communication channel]
"""
        
        with open("COLLABORATION.md", 'w') as f:
            f.write(guide_content)
        
        logger.info("✅ Collaboration guide created as COLLABORATION.md")
    
    def run(self):
        """Run complete DagsHub setup"""
        logger.info("🚀 Starting complete DagsHub setup...")
        
        # Initialize DagsHub
        if not self.initialize_dagshub():
            logger.error("❌ Failed to initialize DagsHub")
            return False
        
        # Setup MLflow
        experiment_id = self.setup_mlflow()
        
        # Create team configuration
        team_config = self.create_team_config()
        
        # Setup Git hooks
        self.setup_git_hooks()
        
        # Create collaboration guide
        self.create_collaboration_guide()
        
        logger.info("✅ DagsHub setup completed successfully!")
        logger.info(f"🔗 Repository: {self.repo_url}")
        logger.info(f"📊 MLflow: {self.repo_url}.mlflow")
        logger.info(f"📚 Collaboration guide: COLLABORATION.md")
        
        return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Setup DagsHub integration')
    parser.add_argument('--repo-owner', required=True,
                       help='DagsHub repository owner username')
    parser.add_argument('--repo-name', default='nyc-taxi-demand-prediction',
                       help='Repository name')
    args = parser.parse_args()
    
    try:
        setup = DagsHubSetup(args.repo_owner, args.repo_name)
        success = setup.run()
        
        if success:
            logger.info("🎉 DagsHub setup completed successfully!")
            logger.info("\nNext steps:")
            logger.info("1. Set DAGSHUB_USER_TOKEN environment variable")
            logger.info("2. Run: dvc repro")
            logger.info("3. Run: git add . && git commit -m 'Initial setup' && git push")
        else:
            logger.error("❌ DagsHub setup failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()