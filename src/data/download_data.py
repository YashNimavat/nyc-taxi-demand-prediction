#!/usr/bin/env python3
"""
NYC TLC Data Download Script
Downloads NYC Yellow Taxi trip data and zone lookup files
"""

import os
import sys
import urllib.request
import pandas as pd
import logging
import yaml
from pathlib import Path
from typing import List, Dict
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TLCDataDownloader:
    """Downloads NYC TLC taxi data"""
    
    def __init__(self, config_path: str = "config/data_config.yaml"):
        """Initialize downloader with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.raw_data_dir = Path("data/raw")
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_tlc_data(self) -> List[str]:
        """Download NYC TLC data for specified years and months"""
        downloaded_files = []
        
        base_url = self.config['data']['raw']['tlc_base_url']
        years = self.config['data']['raw']['years']
        months = self.config['data']['raw']['months']
        
        for year in years:
            for month in months:
                month_str = f'{month:02d}'
                filename = f'yellow_tripdata_{year}-{month_str}.parquet'
                url = f'{base_url}{filename}'
                filepath = self.raw_data_dir / filename
                
                if filepath.exists():
                    logger.info(f"✅ File already exists: {filepath}")
                    downloaded_files.append(str(filepath))
                    continue
                
                try:
                    logger.info(f"⬇️ Downloading {filename}...")
                    urllib.request.urlretrieve(url, filepath)
                    
                    # Verify file
                    df = pd.read_parquet(filepath)
                    logger.info(f"✅ Downloaded {filename} - {len(df):,} records")
                    downloaded_files.append(str(filepath))
                    
                except Exception as e:
                    logger.error(f"❌ Failed to download {filename}: {e}")
                    if filepath.exists():
                        filepath.unlink()
        
        return downloaded_files
    
    def download_zone_lookup(self) -> str:
        """Download taxi zone lookup table"""
        url = self.config['data']['raw']['zone_lookup_url']
        filepath = self.raw_data_dir / 'taxi_zone_lookup.csv'
        
        if filepath.exists():
            logger.info(f"✅ Zone lookup already exists: {filepath}")
            return str(filepath)
        
        try:
            logger.info("⬇️ Downloading taxi zone lookup...")
            urllib.request.urlretrieve(url, filepath)
            
            # Verify file
            df = pd.read_csv(filepath)
            logger.info(f"✅ Downloaded zone lookup - {len(df)} zones")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"❌ Failed to download zone lookup: {e}")
            if filepath.exists():
                filepath.unlink()
            raise
    
    def validate_downloads(self, filepaths: List[str]) -> Dict[str, dict]:
        """Validate downloaded files"""
        validation_results = {}
        
        for filepath in filepaths:
            try:
                if filepath.endswith('.parquet'):
                    df = pd.read_parquet(filepath)
                else:
                    df = pd.read_csv(filepath)
                
                validation_results[filepath] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'size_mb': os.path.getsize(filepath) / (1024*1024),
                    'status': 'valid'
                }
                
            except Exception as e:
                validation_results[filepath] = {
                    'error': str(e),
                    'status': 'invalid'
                }
                
        return validation_results
    
    def run(self) -> Dict[str, any]:
        """Run the download process"""
        logger.info("🚀 Starting TLC data download...")
        
        # Download trip data
        trip_files = self.download_tlc_data()
        
        # Download zone lookup
        zone_file = self.download_zone_lookup()
        
        # Combine all files
        all_files = trip_files + [zone_file]
        
        # Validate downloads
        validation_results = self.validate_downloads(all_files)
        
        # Summary
        total_size_mb = sum(
            result.get('size_mb', 0) 
            for result in validation_results.values() 
            if 'size_mb' in result
        )
        
        valid_files = [
            f for f, r in validation_results.items() 
            if r.get('status') == 'valid'
        ]
        
        summary = {
            'files_downloaded': len(all_files),
            'files_valid': len(valid_files),
            'total_size_mb': round(total_size_mb, 2),
            'files': validation_results
        }
        
        logger.info(f"✅ Download complete!")
        logger.info(f"📊 Downloaded {len(valid_files)} valid files ({total_size_mb:.1f} MB)")
        
        return summary

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Download NYC TLC data')
    parser.add_argument('--config', default='config/data_config.yaml',
                       help='Path to data configuration file')
    args = parser.parse_args()
    
    try:
        downloader = TLCDataDownloader(args.config)
        summary = downloader.run()
        
        # Save summary
        summary_path = Path("metrics/download_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📋 Summary saved to {summary_path}")
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()