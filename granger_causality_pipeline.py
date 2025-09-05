import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tools.sm_exceptions import InfeasibleTestError
from scipy.stats import f, t
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_hac_simple, cov_hac
import matplotlib.pyplot as plt
import warnings
import os
import glob
from confs import *
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare data from the specified file"""
    print(f"=== LOADING AND PREPARING DATA ===")
    
    # Check if the specified file exists
    data_file_path = os.path.join(data_dir, file_name)
    
    if not os.path.exists(data_file_path):
        print(f"Error: Data file {data_file_path} not found")
        return None, None, None
    
    # Load the data
    data = pd.read_csv(data_file_path)
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {list(data.columns[:5])}...")
    
    # Check if response_var is specified in configuration
    if not response_var:
        print("Error: response_var must be specified in confs.py")
        return None, None, None
    
    # Check if the response variable exists in the data
    if response_var not in data.columns:
        print(f"Error: Response variable '{response_var}' not found in data columns: {list(data.columns)}")
        return None, None, None
    
    # Use the explicitly specified response variable
    response_column = response_var
    print(f"Response variable specified: {response_column}")
    
    # Get search terms (all columns except date and response variable)
    search_terms = [col for col in data.columns if col not in ['date', response_column]]
    
    return data, search_terms, response_column

def perform_data_diagnostics(data, search_terms, response_column):
    """Perform diagnostic analysis on data"""
    print("\n=== DIAGNOSTIC ANALYSIS ===")
    
    if data is None:
        print("No data available for diagnostics")
        return []
    
    # Check for constant columns
    constant_columns = []
    low_variance_columns = []
    
    for col in search_terms:
        if col in data.columns:
            if data[col].nunique() == 1:
                constant_columns.append(col)
            elif data[col].std() < low_variance_threshold:
                low_variance_columns.append(col)
    
    print(f"Constant columns: {len(constant_columns)}")
    print(f"Low variance columns (std < {low_variance_threshold}): {len(low_variance_columns)}")
    
    # Filter out problematic columns
    filtered_columns = [col for col in search_terms 
                       if col not in constant_columns and col not in low_variance_columns]
    
    print(f"After filtering: {len(filtered_columns)} terms remaining")
    return filtered_columns

def prepare_merged_data(data, search_terms, response_column, max_lag):
    """Prepare dataset with lagged variables"""
    print(f"\n=== PREPARING DATA WITH LAGGED VARIABLES ===")
    
    if data is None:
        print("No data available")
        return None, None, None, None
    
    # Parse date and add YEAR/WEEK columns
    data['date'] = pd.to_datetime(data['date'])
    data['YEAR'] = data['date'].dt.isocalendar().year
    data['WEEK'] = data['date'].dt.isocalendar().week
    
    # Create a copy for processing
    df_processed = data.copy()
    
    # Create lagged variables
    response_lags = []
    all_lags = []
    
    for lag in range(1, max_lag + 1):
        # Create lag for the response variable (target)
        df_processed[f'{response_column}_lag{lag}'] = df_processed[response_column].shift(lag)
        response_lags.append(f'{response_column}_lag{lag}')
        
        # Create lags for search terms
        for term in search_terms:
            if term in df_processed.columns:
                lag_col = f'{term}_lag{lag}'
                df_processed[lag_col] = df_processed[term].shift(lag)
                all_lags.append(lag_col)
    
    # Clean data - drop rows where we can't compute the target variable
    df_processed = df_processed.dropna(subset=[response_column])
    
    print(f"Data points after cleaning: {len(df_processed)}")
    print(f"Response variable lag columns: {len(response_lags)}")
    print(f"Search term lag columns: {len(all_lags)}")
    
    # Check how many complete cases we have
    complete_cases = df_processed.dropna()
    print(f"Complete cases (no NaN values): {len(complete_cases)}")
    
    return df_processed, response_lags, all_lags, search_terms

def perform_granger_causality_test(df_processed, response_lags, all_lags, response_column):
    """Perform the main Granger causality test"""
    print("\n" + "="*60)
    print("MULTIPLE LINEAR REGRESSION GRANGER CAUSALITY TEST")
    print("="*60)
    
    try:
        # Create complete case dataset for regression (drop rows with any NaN values)
        regression_data = df_processed.dropna()
        
        if len(regression_data) == 0:
            print("No complete cases available for regression after dropping NaN values")
            return None, None, None, None, None, None
        
        print(f"Complete cases for regression: {len(regression_data)}")
        
        # Restricted model (only response variable lags)
        X_restricted = sm.add_constant(regression_data[response_lags])
        y = regression_data[response_column]
        model_restricted = sm.OLS(y, X_restricted).fit()
        
        # Unrestricted model (response variable lags + all search term lags)
        X_unrestricted = sm.add_constant(regression_data[response_lags + all_lags])
        model_unrestricted = sm.OLS(y, X_unrestricted).fit()
        
        # Calculate F-statistic
        rss_restricted = np.sum(model_restricted.resid ** 2)
        rss_unrestricted = np.sum(model_unrestricted.resid ** 2)
        df1 = len(all_lags)
        df2 = len(regression_data) - X_unrestricted.shape[1]
        
        if df1 > 0 and df2 > 0 and rss_unrestricted > 0:
            F = ((rss_restricted - rss_unrestricted) / df1) / (rss_unrestricted / df2)
            p_value = 1 - f.cdf(F, df1, df2)
            
            print(f"Testing if ALL search terms together Granger-cause '{response_column}'")
            print(f"Number of search terms included: {len(all_lags) // len(response_lags)}")
            print(f"Number of lags: {len(response_lags)}")
            print(f"Sample size: {len(regression_data)}")
            print(f"F-statistic: {F:.4f}")
            print(f"p-value: {p_value:.6f}")
            print(f"Degrees of freedom (numerator): {df1}")
            print(f"Degrees of freedom (denominator): {df2}")
            
            if p_value < 0.001:
                significance = "***"
            elif p_value < 0.01:
                significance = "**"
            elif p_value < alpha_level:
                significance = "*"
            else:
                significance = ""
            
            print(f"Significance: {significance}")
            
            if p_value < alpha_level:
                print(f"CONCLUSION: Search terms collectively Granger-cause {response_column} (p < {alpha_level})")
            else:
                print(f"CONCLUSION: No evidence that search terms collectively Granger-cause {response_column}")
                
            # Model fit statistics
            print(f"\nModel Fit Statistics:")
            print(f"Restricted model R²: {model_restricted.rsquared:.4f}")
            print(f"Unrestricted model R²: {model_unrestricted.rsquared:.4f}")
            print(f"R² improvement: {model_unrestricted.rsquared - model_restricted.rsquared:.4f}")
            
            return model_restricted, model_unrestricted, F, p_value, X_unrestricted, y
            
        else:
            print("Error: Cannot compute F-statistic (check degrees of freedom or RSS)")
            return None, None, None, None, None, None
            
    except Exception as e:
        print(f"Error in Granger causality test: {e}")
        return None, None, None, None, None, None

def analyze_individual_terms(model_unrestricted, search_terms, max_lag, F, p_value, response_column):
    """Analyze individual term significance from the multiple regression model"""
    print("\n" + "="*60)
    print("INDIVIDUAL TERM SIGNIFICANCE FROM MULTIPLE REGRESSION")
    print("="*60)
    
    # Extract individual term significance from the unrestricted model
    term_significance = []
    
    # Extract coefficients and p-values for search term lags
    for term in search_terms:
        term_lags = [f'{term}_lag{lag}' for lag in range(1, max_lag + 1)]
        term_pvals = []
        
        for lag_col in term_lags:
            if lag_col in model_unrestricted.params.index:
                # Get the coefficient and p-value for this lag
                coef = model_unrestricted.params[lag_col]
                try:
                    pval = model_unrestricted.pvalues[lag_col]
                except (IndexError, TypeError):
                    # For HAC results, pvalues might be a numpy array
                    if hasattr(model_unrestricted.pvalues, 'loc'):
                        pval = model_unrestricted.pvalues.loc[lag_col]
                    else:
                        # Find the index position
                        param_index = list(model_unrestricted.params.index).index(lag_col)
                        pval = model_unrestricted.pvalues[param_index]
                term_pvals.append(pval)
        
        if term_pvals:
            # Use the minimum p-value across all lags for this term
            min_p = min(term_pvals)
            term_significance.append((term, min_p))
    
    # Sort by significance
    term_significance.sort(key=lambda x: x[1])
    
    # Calculate Bonferroni-corrected significance threshold
    num_tests = len(term_significance)
    bonferroni_threshold = bonferroni_alpha / num_tests if num_tests > 0 else bonferroni_alpha
    
    # Perform FDR correction (Benjamini-Hochberg)
    if num_tests > 0:
        search_term_pvalues = [pval for term, pval in term_significance]
        fdr_rejected, fdr_pvalues, _, _ = multipletests(search_term_pvalues, method='fdr_bh', alpha=fdr_alpha)
        
        # Create mapping of terms to FDR significance
        fdr_significant_terms = set()
        for i, (term, pval) in enumerate(term_significance):
            if fdr_rejected[i]:
                fdr_significant_terms.add(term)
    else:
        fdr_significant_terms = set()
    
    print(f"Number of search terms tested: {num_tests}")
    print(f"Bonferroni-corrected significance threshold: {bonferroni_threshold:.6f} ({bonferroni_alpha}/{num_tests})")
    print(f"FDR correction applied (Benjamini-Hochberg method, alpha={fdr_alpha})")
    
    # Identify significant terms with different thresholds
    significant_uncorrected = [term for term, pval in term_significance if pval < alpha_level]
    significant_bonferroni = [term for term, pval in term_significance if pval < bonferroni_threshold]
    
    print(f"Significant terms (uncorrected p < {alpha_level}): {len(significant_uncorrected)}")
    print(f"Significant terms (Bonferroni-corrected p < {bonferroni_threshold:.6f}): {len(significant_bonferroni)}")
    print(f"Significant terms (FDR-corrected): {len(fdr_significant_terms)}")
    
    return term_significance, significant_uncorrected, significant_bonferroni, fdr_significant_terms, bonferroni_threshold

def create_visualization(model_unrestricted, search_terms, max_lag, term_significance, 
                        significant_uncorrected, significant_bonferroni, fdr_significant_terms, 
                        bonferroni_threshold, response_column):
    """Create visualization of individual term significance"""
    print(f"\n=== CREATING VISUALIZATION ===")
    
    # Check for valid p-values (not nan)
    valid_pvals = [(term, pval) for term, pval in term_significance if not np.isnan(pval)]
    
    if not valid_pvals:
        print("\nWARNING: All p-values are NaN. This indicates the multiple regression model failed to fit properly.")
        return
    
    # Use only valid p-values for plotting
    valid_terms_plot = [term for term, pval in valid_pvals]
    granger_pvals_plot = [pval for term, pval in valid_pvals]
    
    # Dynamic figure sizing based on number of terms
    num_terms = len(valid_terms_plot)
    if num_terms <= 20:
        fig_width = 16
        fig_height = 8
        font_size = 8
        value_font_size = 6
    elif num_terms <= 50:
        fig_width = 20
        fig_height = 10
        font_size = 6
        value_font_size = 5
    else:
        fig_width = 24
        fig_height = 12
        font_size = 4
        value_font_size = 4
    
    plt.figure(figsize=(fig_width, fig_height))
    
    # Create bars with colors for different significance levels
    colors = []
    for i, term in enumerate(valid_terms_plot):
        pval = granger_pvals_plot[i]
        if term in fdr_significant_terms:
            colors.append('purple')  # FDR significant
        elif pval < bonferroni_threshold:
            colors.append('darkred')  # Bonferroni significant
        elif pval < alpha_level:
            colors.append('red')      # Uncorrected significant
        else:
            colors.append('orange')   # Not significant
    
    bars = plt.bar(valid_terms_plot, granger_pvals_plot, color=colors, alpha=0.7)
    
    plt.ylabel('Min p-value (across lags)', fontsize=12)
    plt.title(f'Individual Term Significance for {response_column} - Max Lag = {max_lag}', fontsize=14, pad=20)
    plt.axhline(alpha_level, color='red', linestyle='--', label=f'p={alpha_level} (uncorrected)', linewidth=2)
    
    # Add custom legend entries for bar colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='purple', alpha=0.7, label='FDR significant'),
        Patch(facecolor='darkred', alpha=0.7, label='Bonferroni significant'),
        Patch(facecolor='red', alpha=0.7, label='Uncorrected significant'),
        Patch(facecolor='orange', alpha=0.7, label='Not significant')
    ]
    
    # Improve x-axis labels with better rotation and positioning
    plt.xticks(rotation=45, fontsize=font_size, ha='right')
    
    # Add value labels on bars with better positioning
    for bar, pval in zip(bars, granger_pvals_plot):
        height = bar.get_height()
        # Position text above bar with small offset
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{pval:.3f}', ha='center', va='bottom', fontsize=value_font_size, rotation=90)
    
    # Set y-axis limits to ensure visibility
    if granger_pvals_plot:
        plt.ylim(0, max(granger_pvals_plot) * 1.1)
    
    plt.legend(handles=legend_elements, fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Better layout with more space
    plt.tight_layout(pad=2.0)
    
    # Create results directory and granger causality subfolder
    granger_results_dir = os.path.join(result_dir, granger_causality_prefix, response_column)
    os.makedirs(granger_results_dir, exist_ok=True)
    
    # Save visualization
    viz_filename = os.path.join(granger_results_dir, f"{visualization_prefix}_{response_column}_lag{max_lag}.png")
    plt.savefig(viz_filename, dpi=figure_dpi, bbox_inches=figure_bbox_inches)
    plt.close()
    
    print(f"Visualization saved to {viz_filename}")
    
    # Print summary statistics
    significant_uncorrected_plot = [term for term, pval in valid_pvals if pval < alpha_level]
    significant_bonferroni_plot = [term for term, pval in valid_pvals if pval < bonferroni_threshold]
    significant_fdr_plot = [term for term in valid_terms_plot if term in fdr_significant_terms]
    
    print(f"\nSummary:")
    print(f"Uncorrected (p < {alpha_level}): {len(significant_uncorrected_plot)} terms")
    print(f"Bonferroni-corrected (p < {bonferroni_threshold:.6f}): {len(significant_bonferroni_plot)} terms")
    print(f"FDR-corrected: {len(significant_fdr_plot)} terms")
    
    if significant_fdr_plot:
        print(f"\nTop 5 FDR-significant terms:")
        sorted_valid = sorted(valid_pvals, key=lambda x: x[1])
        for i, (term, pval) in enumerate(sorted_valid[:5]):
            significance = "***" if term in fdr_significant_terms else "**" if pval < alpha_level else ""
            print(f"{i+1}. {term}: p = {pval:.4f}{significance}")

def save_results(term_significance, significant_uncorrected, significant_bonferroni, 
                fdr_significant_terms, bonferroni_threshold, F, p_value, 
                model_unrestricted, max_lag, response_column):
    """Save comprehensive results to a text file"""
    print(f"\n=== SAVING COMPREHENSIVE RESULTS ===")
    
    # Create results directory and granger causality subfolder
    granger_results_dir = os.path.join(result_dir, granger_causality_prefix, response_column)
    os.makedirs(granger_results_dir, exist_ok=True)
    
    # Save results
    txt_filename = os.path.join(granger_results_dir, f"{results_prefix}_{response_column}_lag{max_lag}.txt")
    with open(txt_filename, "w") as f:
        # Write summary at the top
        f.write(f"=== COMPREHENSIVE GRANGER CAUSALITY ANALYSIS SUMMARY ===\n")
        f.write(f"Data file: {file_name}\n")
        f.write(f"Response variable: {response_column}\n")
        f.write(f"Max lag: {max_lag}\n")
        f.write(f"Number of tests: {len(term_significance)}\n")
        f.write(f"Bonferroni threshold: {bonferroni_threshold:.6f}\n")
        f.write(f"FDR correction applied (Benjamini-Hochberg method, alpha={fdr_alpha})\n")
        f.write(f"Overall Granger causality F-statistic: {F:.4f}\n")
        f.write(f"Overall Granger causality p-value: {p_value:.6f}\n")
        f.write(f"Model R-squared: {model_unrestricted.rsquared:.4f}\n\n")
        
        f.write(f"=== SIGNIFICANCE SUMMARY ===\n")
        f.write(f"Uncorrected significant (p < {alpha_level}): {len(significant_uncorrected)} terms\n")
        f.write(f"Bonferroni significant (p < {bonferroni_threshold:.6f}): {len(significant_bonferroni)} terms\n")
        f.write(f"FDR significant: {len(fdr_significant_terms)} terms\n\n")
        
        # Get all significant terms (any type) and determine most conservative significance
        all_significant_terms = set()
        for term, pval in term_significance:
            if pval < alpha_level:  # Any type of significance
                all_significant_terms.add(term)
        
        if all_significant_terms:
            f.write(f"=== ALL SIGNIFICANT TERMS (n={len(all_significant_terms)}) ===\n")
            f.write(f"Term\tMin_p_value\tMost_Conservative_Significance\n")
            
            # Sort by p-value (most significant first)
            significant_terms_sorted = []
            for term in all_significant_terms:
                pval = [p for t, p in term_significance if t == term][0]
                # Determine most conservative significance
                if pval < bonferroni_threshold:
                    most_conservative = "Bonferroni"
                elif term in fdr_significant_terms:
                    most_conservative = "FDR"
                else:
                    most_conservative = "Uncorrected"
                
                significant_terms_sorted.append((term, pval, most_conservative))
            
            # Sort by p-value
            significant_terms_sorted.sort(key=lambda x: x[1])
            
            for term, pval, most_conservative in significant_terms_sorted:
                f.write(f"{term}\t{pval:.6f}\t{most_conservative}\n")
        else:
            f.write("No terms were significant at any level.\n")
    
    print(f"Comprehensive results saved to {txt_filename}")

def main():
    """Main function to run the complete Granger causality analysis"""
    print("=== GENERALIZED GRANGER CAUSALITY ANALYSIS ===")
    print(f"Configuration loaded from confs.py")
    print(f"Data directory: {data_dir}")
    print(f"Results directory: {result_dir}")
    print(f"Granger causality subfolder: {granger_causality_prefix}")
    print(f"Data file to analyze: {file_name}")
    print(f"Max lags to test: {max_lags_to_test}")
    
    if not file_name:
        print("Error: Please specify file_name in confs.py")
        return
    
    # Load and prepare data from the specified file
    data, search_terms, response_column = load_and_prepare_data()
    
    if data is None:
        print(f"Failed to load data from {file_name}")
        return
    
    # Perform data diagnostics
    filtered_columns = perform_data_diagnostics(data, search_terms, response_column)
    
    # Use filtered columns for the analysis
    search_terms = filtered_columns[:max_terms] if max_terms else filtered_columns
    print(f"Final number of search terms to use: {len(search_terms)}")
    
    # Run analysis for configured max lags
    for max_lag in range(1, max_lags_to_test + 1):
        print(f"\n{'='*80}")
        print(f"RUNNING ANALYSIS FOR {file_name} WITH MAX LAG = {max_lag}")
        print(f"{'='*80}")
        
        # Prepare data with lagged variables
        df_processed, response_lags, all_lags, search_terms_simple = prepare_merged_data(
            data, search_terms, response_column, max_lag
        )
        
        if df_processed is None:
            print("Failed to prepare data. Continuing to next lag.")
            continue
        
        # Check for constant values in response variable
        if df_processed[response_column].nunique() <= 1:
            print(f"Error: Response variable '{response_column}' has constant values")
            continue
        
        # Perform Granger causality test
        model_restricted, model_unrestricted, F, p_value, X_unrestricted, y = perform_granger_causality_test(
            df_processed, response_lags, all_lags, response_column
        )
        
        if model_unrestricted is None:
            print("Granger causality test failed. Continuing to next lag.")
            continue
        
        # Perform individual term analysis
        term_significance, significant_uncorrected, significant_bonferroni, fdr_significant_terms, bonferroni_threshold = analyze_individual_terms(
            model_unrestricted, search_terms_simple, max_lag, F, p_value, response_column
        )
        
        # Create visualization
        create_visualization(
            model_unrestricted, search_terms_simple, max_lag, term_significance,
            significant_uncorrected, significant_bonferroni, fdr_significant_terms,
            bonferroni_threshold, response_column
        )
        
        # Save comprehensive results
        save_results(
            term_significance, significant_uncorrected, significant_bonferroni,
            fdr_significant_terms, bonferroni_threshold, F, p_value,
            model_unrestricted, max_lag, response_column
        )
        
        print(f"\n=== ANALYSIS COMPLETE FOR {file_name} WITH MAX LAG = {max_lag} ===")
    
    # Show final output location
    granger_results_dir = os.path.join(result_dir, granger_causality_prefix)
    print(f"\n=== ALL ANALYSES COMPLETE FOR {file_name} ===")
    print(f"Results saved to: {granger_results_dir}")

if __name__ == "__main__":
    main()
