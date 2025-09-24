# Copy your existing production files to the new structure

# Move your existing files to the new organized structure:

# From root to src/production/
cp fastapi_production_server_fixed.py src/production/
cp shadow_mode_validator.py src/production/
cp drift_detection_system.py src/production/
cp production_validation_system.py src/production/
cp streaming_backfill_system.py src/production/
cp master_control_dashboard_fixed.py src/production/
cp taxi_dashboard.py src/production/
cp websocket_server.py src/production/
cp interactive_taxi_system.py src/production/
cp nyc_taxi_production.py src/production/
cp nyc_taxi_demand_production.py src/production/

# Move notebooks to notebooks/
cp nyc_taxi_demand_prediction-3.ipynb notebooks/
cp nyc_taxi_demand_prediction-4.ipynb notebooks/

# Move test files
cp test_api.py tests/
cp api.py src/production/  # Legacy API file

# Move shell scripts to scripts/
cp start_api.sh scripts/
cp setup_production_validation.sh scripts/
cp start_complete_system.py scripts/

# Create updated requirements.txt with your original dependencies plus Phase 1 additions
# (This is already done in the requirements.txt file created above)