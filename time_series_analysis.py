import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import os
from datetime import datetime
import re
from confs import *

def parse_state_summary(filepath):
    """Parse the response variable significant terms summary file"""
    significant_combinations = []
    
    # Check if it's a CSV file or TXT file
    if filepath.endswith('.csv'):
        # Parse CSV format
        try:
            import pandas as pd
            df = pd.read_csv(filepath)
            
            for _, row in df.iterrows():
                significant_combinations.append({
                    'term': row['Term'],
                    'state': response_var,  # Use current response_var from config
                    'maxlag': int(row['Lag']),  # Lag is now just the number
                    'p_value': row['P_Value'],
                    'significance_type': row['Significance']
                })
        except Exception as e:
            print(f"Error reading CSV file {filepath}: {e}")
            return []
    else:
        # Parse TXT format (legacy support)
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        current_term = None
        for i, line in enumerate(lines):
            line_original = line
            line = line.strip()
            
            # Check for the format: "term significant combinations:"
            if ' significant combinations:' in line:
                current_term = line.split(' significant combinations:')[0].strip()
            elif line and current_term and line_original.startswith('  ') and not line.startswith('===') and not line.startswith('Total') and not line.startswith('Unique') and not line.startswith('States') and not line.startswith('Direction'):
                # Parse: "  {response_var}_lag3: p=0.001234 (Bonferroni)"
                if ':' in line and 'p=' in line:
                    # Extract response variable and lag
                    response_var_lag_part = line.split(':')[0]
                    response_var_lag_match = re.search(r'(\w+)_lag(\d+)', response_var_lag_part)
                    if response_var_lag_match:
                        response_var_name = response_var_lag_match.group(1)
                        maxlag = int(response_var_lag_match.group(2))
                        
                        # Extract p-value and significance type
                        p_value_match = re.search(r'p=([\d.]+)', line)
                        significance_match = re.search(r'\(([^)]+)\)', line)
                        
                        if p_value_match and significance_match:
                            p_value = float(p_value_match.group(1))
                            significance_type = significance_match.group(1)
                            
                            significant_combinations.append({
                                'term': current_term,
                                'state': response_var_name,
                                'maxlag': maxlag,
                                'p_value': p_value,
                                'significance_type': significance_type
                            })
                        else:
                            pass  # Skip lines that don't parse correctly
                    else:
                        pass  # Skip lines that don't match pattern
                else:
                    pass  # Skip lines that don't contain required elements
    
    return significant_combinations

def load_state_data(response_var_name, start_year=None, end_year=None):
    """Load response variable data with optional date filtering"""
    # For now, use the configured file_name since we're analyzing one file at a time
    # In the future, this could be extended to handle multiple state files
    state_file = os.path.join(data_dir, file_name)
    
    if not os.path.exists(state_file):
        print(f"Warning: State file {state_file} not found")
        return None
    
    # Load state data
    df_state = pd.read_csv(state_file)
    
    # Parse date
    df_state['date'] = pd.to_datetime(df_state['date'])
    df_state['YEAR'] = df_state['date'].dt.year
    
    # Filter by date range if specified
    if start_year is not None:
        df_state = df_state[df_state['YEAR'] >= start_year]
    if end_year is not None:
        df_state = df_state[df_state['YEAR'] <= end_year]
    
    # Sort by date
    df_state = df_state.sort_values('date').reset_index(drop=True)
    
    return df_state

def create_state_analysis_plot(term, df_state, significant_info, output_dir, response_var_name):
    """Create response variable analysis plot for a single term"""
    fig, axes = plt.subplots(max_lags_to_test, 1, figsize=(15, 4 * max_lags_to_test))
    
    # Get significant info for title
    if significant_info:
        title_parts = []
        for s in significant_info:
            title_parts.append(f"maxlag {s['maxlag']}, p={s['p_value']:.6f} ({s['significance_type']})")
        title_info = ', '.join(title_parts)
        fig.suptitle(f'{term} → {response_var} Activity (Significant: {title_info})', fontsize=10, fontweight='bold', y=0.98)
    else:
        fig.suptitle(f'{term} → {response_var} Activity', fontsize=12, fontweight='bold', y=0.98)
    
    for lag in range(1, max_lags_to_test + 1):
        ax = axes[lag-1]
        
        # Create lagged term data - this shifts the search term data backward in time
        df_state[f'{term}_lag{lag}'] = df_state[term].shift(lag)
        
        # Plot data - use the same x-axis (current time points)
        x = df_state['date']
        
        # Plot lagged search term (highlight if significant) - this shows what was searched X weeks ago
        is_significant = any(s['maxlag'] == lag for s in significant_info)
        
        if is_significant:
            ax.plot(x, df_state[f'{term}_lag{lag}'], label=f'{term} Search Volume ({lag} week(s) ago)', color='red', alpha=0.8, linewidth=2)
        else:
            ax.plot(x, df_state[f'{term}_lag{lag}'], label=f'{term} Search Volume ({lag} week(s) ago)', color='blue', alpha=0.7, linewidth=1)
        
        # Create dual y-axis for response variable activity (Y variable)
        ax_twin = ax.twinx()
        
        # Plot response variable activity (Y variable) - right y-axis - this is what we're predicting
        ax_twin.plot(x, df_state[response_var], label=f'{response_var} Activity (current)', color='green', alpha=0.8, linewidth=2)
        
        # Customize plot
        ax.set_title(f'Lag {lag} - Testing if search term from {lag} week(s) ago predicts current {response_var} activity', fontweight='bold')
        ax.set_ylabel(f'{term} Search Volume (X variable)', color='blue')
        ax_twin.set_ylabel(f'{response_var} Activity (Y variable)', color='green')
        ax.grid(True, alpha=0.3)
        
        # Add legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Format x-axis
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave more space at top
    
    # Clean term name for file naming
    clean_term = term.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
    filename = os.path.join(output_dir, f"{clean_term}_{response_var.lower()}_analysis.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def create_term_plots(term, significant_combinations, output_dir):
    """Create analysis plots for a single term for the response variable"""
    
    # Filter significant combinations for this term
    term_significant = [s for s in significant_combinations if s['term'] == term]
    
    if not term_significant:
        print(f"No significant combinations found for term: {term}")
        return
    
    # Create plots for the response variable
    # Load data
    df_state = load_state_data(response_var)
    if df_state is None:
        return
    
    # Check if term exists in the data
    if term not in df_state.columns:
        print(f"    Warning: Term '{term}' not found in {response_var} dataset")
        return
    
    # Get significant info for this response variable
    response_var_significant = [s for s in term_significant if s['state'] == response_var]
    
    # Create analysis plot directly in output_dir (no subfolder)
    create_state_analysis_plot(term, df_state, response_var_significant, output_dir, response_var)

def main(start_year=None, end_year=None):
    """Main function to generate all time series plots with optional date range filtering"""
    print("Loading significant terms...")
    
    # Parse significant terms using configuration - look in granger causality subfolder
    granger_results_dir = os.path.join(result_dir, granger_causality_prefix, response_var)
    
    # Try to find CSV files first, then fall back to TXT files
    csv_files = []
    for lag in range(1, max_lags_to_test + 1):
        csv_file = os.path.join(granger_results_dir, f"{results_prefix}_{response_var}_lag{lag}.csv")
        if os.path.exists(csv_file):
            csv_files.append(csv_file)
    
    if not csv_files:
        # Fall back to summary file
        summary_file = os.path.join(granger_results_dir, f"{summary_prefix}_{response_var}.txt")
        if not os.path.exists(summary_file):
            print(f"No CSV files or summary file found in: {granger_results_dir}")
            print("Please run granger_causality_pipeline_refactored.py first")
            return
        summary_files = [summary_file]
    else:
        summary_files = csv_files
    
    # Parse all files (CSV or TXT)
    significant_combinations = []
    for file_path in summary_files:
        file_combinations = parse_state_summary(file_path)
        significant_combinations.extend(file_combinations)
    
    print(f"Found {len(significant_combinations)} significant combinations")
    
    # Get unique terms
    unique_terms = list(set([s['term'] for s in significant_combinations]))
    print(f"Unique terms: {unique_terms}")
    
    # Get unique response variables
    unique_response_vars = list(set([s['state'] for s in significant_combinations]))
    print(f"Response variables analyzed: {unique_response_vars}")
    
    print("Loading data...")
    
    # Create output directory based on date range - use granger causality subfolder
    if start_year is not None or end_year is not None:
        # Create date-specific folder name
        start_str = str(start_year) if start_year is not None else "start"
        end_str = str(end_year) if end_year is not None else "end"
        output_dir = os.path.join(result_dir, granger_causality_prefix, response_var, time_series_prefix, f"_{start_str}-{end_str}_")
        print(f"Filtering data from year {start_year} to {end_year}")
    else:
        output_dir = os.path.join(result_dir, granger_causality_prefix, response_var, time_series_prefix)
        print("No date range specified - using full dataset")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots for each term
    for i, term in enumerate(unique_terms, 1):
        try:
            create_term_plots(term, significant_combinations, output_dir)
        except Exception as e:
            print(f"Error processing term '{term}': {e}")
            import traceback
            traceback.print_exc()
    
    print(f"All plots saved to: {output_dir}")
    print(f"Total terms processed: {len(unique_terms)}")

if __name__ == "__main__":
    # You can call main() with optional start_year and end_year parameters
    # Examples:
    # main()  # No date range - saves to time_series_plots
    # main(start_year=2015)  # From 2015 onwards - saves to time_series_plots_2015-end
    # main(end_year=2020)  # Up to 2020 - saves to time_series_plots_start-2020
    # main(start_year=2015, end_year=2020)  # 2015-2020 - saves to time_series_plots_2015-2020

    main()  # Default: no date range filtering
