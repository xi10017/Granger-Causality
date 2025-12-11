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
    original_response_var = response_var
    
    # Extract response variable name from data file name (e.g., "Alabama_2010_2020.csv" -> "Alabama")
    response_var_name = data_file_name.split('_')[0]
    
    # Update the configuration file
    with open('confs.py', 'r') as f:
        config_content = f.read()
    
    # Replace the file_name line
    config_content = config_content.replace(
        f'file_name = "{original_file_name}"',
        f'file_name = "{data_file_name}"'
    )
    
    # Replace the response_var line
    config_content = config_content.replace(
        f'response_var = "{original_response_var}"',
        f'response_var = "{response_var_name}"'
    )
    
    with open('confs.py', 'w') as f:
        f.write(config_content)
    
    try:
        # Step 1: Run the Granger causality analysis
        print(f"Step 1: Running Granger causality analysis for {data_file_name}...")
        result1 = subprocess.run([sys.executable, 'granger_causality_pipeline_refactored.py'], 
                               capture_output=True, text=True)
        
        if result1.returncode != 0:
            print(f"ERROR: Granger causality analysis failed for {data_file_name}")
            print(f"Error: {result1.stderr}")
            return False
        
        print(f"SUCCESS: Granger causality analysis completed for {data_file_name}")
        
        # Step 2: Create comprehensive significant terms summary
        print(f"Step 2: Creating comprehensive summary for {data_file_name}...")
        result2 = subprocess.run([sys.executable, 'create_comprehensive_significant_terms_summary.py'], 
                               capture_output=True, text=True)
        
        if result2.returncode != 0:
            print(f"ERROR: Summary creation failed for {data_file_name}")
            print(f"Error: {result2.stderr}")
            return False
        
        print(f"SUCCESS: Comprehensive summary created for {data_file_name}")
        
        # Step 3: Generate time series analysis plots
        print(f"Step 3: Generating time series plots for {data_file_name}...")
        result3 = subprocess.run([sys.executable, 'time_series_analysis.py'], 
                               capture_output=True, text=True)
        
        if result3.returncode != 0:
            print(f"ERROR: Time series analysis failed for {data_file_name}")
            print(f"Error: {result3.stderr}")
            return False
        
        print(f"SUCCESS: Time series analysis completed for {data_file_name}")
        
        # Step 4: Generate comprehensive results analysis
        print(f"Step 4: Generating comprehensive results analysis for {data_file_name}...")
        result4 = subprocess.run([sys.executable, 'analyze_comprehensive_results.py'], 
                               capture_output=True, text=True)
        
        if result4.returncode != 0:
            print(f"ERROR: Comprehensive results analysis failed for {data_file_name}")
            print(f"Error: {result4.stderr}")
            return False
        
        print(f"SUCCESS: Comprehensive results analysis completed for {data_file_name}")
        print(f"SUCCESS: Complete analysis pipeline finished for {data_file_name}")
        return True
            
    except Exception as e:
        print(f"ERROR: Error running analysis for {data_file_name}: {e}")
        return False
    
    finally:
        # Restore original configuration
        config_content = config_content.replace(
            f'file_name = "{data_file_name}"',
            f'file_name = "{original_file_name}"'
        )
        config_content = config_content.replace(
            f'response_var = "{response_var_name}"',
            f'response_var = "{original_response_var}"'
        )
        
        with open('confs.py', 'w') as f:
            f.write(config_content)

def main():
    """Main function to demonstrate multiple data file analysis"""
    print("=== MULTIPLE DATA FILE COMPLETE ANALYSIS PIPELINE ===")
    print("This script runs the complete analysis pipeline for different data files:")
    print("1. Granger causality analysis")
    print("2. Comprehensive significant terms summary")
    print("3. Time series analysis plots")
    print("4. Comprehensive results analysis (bar graphs and statistics)")
    
    # List of data files to analyze (imported from configs)
    # You can modify the list in confs.py to change which files are analyzed
    
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
    print(f"COMPLETE PIPELINE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total data files attempted: {len(data_files_to_analyze)}")
    print(f"Successful complete analyses: {successful_analyses}")
    print(f"Failed analyses: {len(data_files_to_analyze) - successful_analyses}")
    
    if successful_analyses > 0:
        print(f"\nResults saved to: {result_dir}")
        print("Each successful analysis includes:")
        print("  - Granger causality test results and visualizations")
        print("  - Comprehensive significant terms summary")
        print("  - Time series analysis plots")
        print("  - Comprehensive results analysis (bar graphs and statistics)")
        print(f"  - All files organized in: {result_dir}/granger_causality_results/")
    
    print(f"\nConfiguration restored to: file_name = {file_name}, response_var = {response_var}")

if __name__ == "__main__":
    main()
