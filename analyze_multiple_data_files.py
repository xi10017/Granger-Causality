#!/usr/bin/env python3
"""
Script to demonstrate analyzing multiple data files using the generalized Granger causality pipeline.

This script shows how to modify the configuration and run analysis for different data files.
"""

import os
import subprocess
import sys
from confs import *

def analyze_data_file(data_file_name):
    """Analyze a specific data file by temporarily modifying the configuration"""
    print(f"\n{'='*60}")
    print(f"ANALYZING DATA FILE: {data_file_name}")
    print(f"{'='*60}")
    
    # Check if data file exists
    data_file_path = os.path.join(data_dir, data_file_name)
    if not os.path.exists(data_file_path):
        print(f"Error: Data file {data_file_path} not found")
        return False
    
    # Temporarily modify the configuration
    original_file_name = file_name
    
    # Update the configuration file
    with open('confs.py', 'r') as f:
        config_content = f.read()
    
    # Replace the file_name line
    config_content = config_content.replace(
        f'file_name = "{original_file_name}"',
        f'file_name = "{data_file_name}"'
    )
    
    with open('confs.py', 'w') as f:
        f.write(config_content)
    
    try:
        # Run the analysis
        print(f"Running Granger causality analysis for {data_file_name}...")
        result = subprocess.run([sys.executable, 'granger_causality_pipeline.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Analysis completed successfully for {data_file_name}")
            return True
        else:
            print(f"❌ Analysis failed for {data_file_name}")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running analysis for {data_file_name}: {e}")
        return False
    
    finally:
        # Restore original configuration
        config_content = config_content.replace(
            f'file_name = "{data_file_name}"',
            f'file_name = "{original_file_name}"'
        )
        
        with open('confs.py', 'w') as f:
            f.write(config_content)

def main():
    """Main function to demonstrate multiple data file analysis"""
    print("=== MULTIPLE DATA FILE ANALYSIS DEMONSTRATION ===")
    print("This script shows how to analyze different data files using the generalized pipeline.")
    
    # List of data files to analyze (you can modify this)
    data_files_to_analyze = [
        'Alabama_2010_2020.csv',
        'California_2010_2020.csv', 
        'Texas_2010_2020.csv',
        'Florida_2010_2020.csv',
        'New_York_2010_2020.csv'
    ]
    
    print(f"\nData files to analyze: {', '.join(data_files_to_analyze)}")
    print(f"Current configuration: file_name = {file_name}")
    
    # Ask user if they want to proceed
    response = input("\nDo you want to proceed with analyzing these data files? (y/n): ")
    if response.lower() != 'y':
        print("Analysis cancelled.")
        return
    
    # Analyze each data file
    successful_analyses = 0
    for data_file in data_files_to_analyze:
        if analyze_data_file(data_file):
            successful_analyses += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total data files attempted: {len(data_files_to_analyze)}")
    print(f"Successful analyses: {successful_analyses}")
    print(f"Failed analyses: {len(data_files_to_analyze) - successful_analyses}")
    
    if successful_analyses > 0:
        print(f"\nResults saved to: {result_dir}")
        print("Check the results directory for output files.")
    
    print(f"\nConfiguration restored to: file_name = {file_name}")

if __name__ == "__main__":
    main()
