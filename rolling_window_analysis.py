"""
Rolling Window Granger Causality Analysis

This module implements rolling window analysis for Granger causality testing
using configurable time periods and sliding windows.

Author: Rolling Window Analysis Implementation
Date: 2024
"""

import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Import the refactored Granger causality pipeline
from granger_causality_pipeline_refactored import (
    GrangerCausalityAnalyzer, 
    AnalysisConfig, 
    GrangerResults
)
from confs import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


@dataclass
class TimeWindow:
    """Represents a time window for analysis."""
    start_year: int
    start_month: int
    end_year: int
    end_month: int
    window_id: str


class RollingWindowAnalyzer:
    """Main class for rolling window Granger causality analysis."""
    
    def __init__(self, data_file: str, response_var: str):
        """Initialize the rolling window analyzer."""
        self.data_file = data_file
        self.response_var = response_var
        self.data: Optional[pd.DataFrame] = None
        self.windows: List[TimeWindow] = []
        self.results: List[Dict] = []
        
        # Create output directory structure
        self.base_output_dir = Path(result_dir) / granger_causality_prefix / response_var / rolling_window_folder_name
        self.csv_dir = self.base_output_dir / "csvs"
        self.text_dir = self.base_output_dir / "text_summaries"
        
        # Create all directories
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.text_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized RollingWindowAnalyzer for {response_var}")
        logger.info(f"Output directory: {self.base_output_dir}")
    
    def load_data(self) -> bool:
        """Load and prepare the data for analysis."""
        logger.info("=== LOADING DATA FOR ROLLING WINDOW ANALYSIS ===")
        
        try:
            data_path = Path(data_dir) / self.data_file
            if not data_path.exists():
                logger.error(f"Data file not found: {data_path}")
                return False
            
            self.data = pd.read_csv(data_path)
            logger.info(f"Loaded data shape: {self.data.shape}")
            
            # Parse date column
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.sort_values('date').reset_index(drop=True)
            
            # Validate response variable
            if self.response_var not in self.data.columns:
                logger.error(f"Response variable '{self.response_var}' not found in data")
                return False
            
            logger.info(f"Date range: {self.data['date'].min()} to {self.data['date'].max()}")
            logger.info(f"Total data points: {len(self.data)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def create_rolling_windows(self) -> List[TimeWindow]:
        """Create rolling windows based on configuration."""
        logger.info("=== CREATING ROLLING WINDOWS ===")
        
        windows = []
        current_start_year = rolling_window_start_year
        current_start_month = rolling_window_start_month
        window_id = 1
        
        while True:
            # Calculate window end date
            end_year = current_start_year + rolling_window_years
            end_month = current_start_month
            
            # Check if we exceed the end year
            if end_year > rolling_window_end_year:
                break
            
            # Create window
            window = TimeWindow(
                start_year=current_start_year,
                start_month=current_start_month,
                end_year=end_year,
                end_month=end_month,
                window_id=f"window_{window_id}"
            )
            
            # Check if we have enough data for this window
            start_date = pd.Timestamp(f"{current_start_year}-{current_start_month:02d}-01")
            end_date = pd.Timestamp(f"{end_year}-{end_month:02d}-01") + pd.DateOffset(months=1) - pd.DateOffset(days=1)
            
            window_data = self.data[
                (self.data['date'] >= start_date) & 
                (self.data['date'] <= end_date)
            ]
            
            if len(window_data) >= rolling_window_min_data_points:
                windows.append(window)
                logger.info(f"Window {window_id}: {current_start_year}-{current_start_month:02d} to {end_year}-{end_month:02d} ({len(window_data)} points)")
            else:
                logger.warning(f"Window {window_id}: Insufficient data points ({len(window_data)} < {rolling_window_min_data_points})")
            
            # Move to next window
            current_start_year += rolling_window_step_years
            window_id += 1
        
        self.windows = windows
        logger.info(f"Created {len(windows)} valid rolling windows")
        return windows
    
    def analyze_time_window(self, window: TimeWindow) -> Optional[Dict]:
        """Analyze a specific time window using the refactored Granger causality pipeline."""
        logger.info(f"=== ANALYZING {window.window_id}: {window.start_year}-{window.start_month:02d} to {window.end_year}-{window.end_month:02d} ===")
        
        try:
            # Filter data for this window
            start_date = pd.Timestamp(f"{window.start_year}-{window.start_month:02d}-01")
            end_date = pd.Timestamp(f"{window.end_year}-{window.end_month:02d}-01") + pd.DateOffset(months=1) - pd.DateOffset(days=1)
            
            window_data = self.data[
                (self.data['date'] >= start_date) & 
                (self.data['date'] <= end_date)
            ].copy()
            
            if len(window_data) < rolling_window_min_data_points:
                logger.warning(f"Insufficient data for {window.window_id}: {len(window_data)} points")
                return None
            
            # Save window data to temporary file
            temp_data_file = self.base_output_dir / f"temp_{window.window_id}_{self.response_var}.csv"
            window_data.to_csv(temp_data_file, index=False)
            
            # Create configuration for this window
            config = AnalysisConfig(
                data_dir=str(self.base_output_dir),
                result_dir=str(self.base_output_dir),
                max_terms=max_terms,
                response_var=self.response_var,
                max_lags_to_test=max_lags_to_test,
                low_variance_threshold=low_variance_threshold,
                zero_ratio_threshold=zero_ratio_threshold,
                alpha_level=alpha_level,
                bonferroni_alpha=bonferroni_alpha,
                fdr_alpha=fdr_alpha,
                figure_dpi=figure_dpi,
                figure_bbox_inches=figure_bbox_inches,
                file_name=f"temp_{window.window_id}_{self.response_var}.csv",
                results_prefix=f"rolling_{results_prefix}",
                visualization_prefix=f"rolling_{visualization_prefix}",
                summary_prefix=f"rolling_{summary_prefix}",
                time_series_prefix=f"rolling_{time_series_prefix}",
                granger_causality_prefix=f"rolling_{granger_causality_prefix}",
                comprehensive_analysis_prefix=f"rolling_{comprehensive_analysis_prefix}"
            )
            
            # Create analyzer instance
            analyzer = GrangerCausalityAnalyzer(config)
            
            # Get search terms (all columns except date and response variable)
            search_terms = [col for col in window_data.columns if col not in ['date', self.response_var]]
            
            # Perform data diagnostics
            filtered_terms = analyzer.perform_data_diagnostics(window_data, search_terms)
            
            if not filtered_terms:
                logger.warning(f"No valid search terms for {window.window_id}")
                # Clean up temp file
                if temp_data_file.exists():
                    temp_data_file.unlink()
                return None
            
            # Prepare data with lagged variables
            df_processed, response_lags, all_lags, search_terms_simple = analyzer.prepare_merged_data(
                window_data, filtered_terms, self.response_var, max_lags_to_test
            )
            
            if df_processed is None:
                logger.warning(f"Failed to prepare data for {window.window_id}")
                # Clean up temp file
                if temp_data_file.exists():
                    temp_data_file.unlink()
                return None
            
            # Perform Granger causality test
            granger_results = analyzer.perform_granger_causality_test(
                df_processed, response_lags, all_lags, self.response_var
            )
            
            if granger_results is None:
                logger.warning(f"Granger causality test failed for {window.window_id}")
                # Clean up temp file
                if temp_data_file.exists():
                    temp_data_file.unlink()
                return None
            
            # Extract significant terms with proper hierarchy
            raw_terms = set(granger_results.significant_uncorrected)
            fdr_terms = set(granger_results.significant_fdr)
            bonferroni_terms = set(granger_results.significant_bonferroni)
            
            # Count terms (Bonferroni ⊆ FDR ⊆ Raw)
            raw_count = len(raw_terms)
            fdr_count = len(fdr_terms)
            bonferroni_count = len(bonferroni_terms)
            
            # Create result dictionary
            result = {
                'window_id': window.window_id,
                'start_year': window.start_year,
                'start_month': window.start_month,
                'end_year': window.end_year,
                'end_month': window.end_month,
                'data_points': len(window_data),
                'num_terms_tested': len(filtered_terms),
                'raw_significant_count': raw_count,
                'fdr_significant_count': fdr_count,
                'bonferroni_significant_count': bonferroni_count,
                'raw_terms': list(raw_terms),
                'fdr_terms': list(fdr_terms),
                'bonferroni_terms': list(bonferroni_terms),
                'granger_results': granger_results,
                'success': True
            }
            
            # Save window-specific results to text file
            self._save_window_results(result, window)
            
            # Clean up temp file
            if temp_data_file.exists():
                temp_data_file.unlink()
            
            logger.info(f"{window.window_id} results:")
            logger.info(f"  Raw significant terms: {raw_count}")
            logger.info(f"  FDR significant terms: {fdr_count}")
            logger.info(f"  Bonferroni significant terms: {bonferroni_count}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {window.window_id}: {e}")
            # Clean up temp file
            temp_data_file = self.base_output_dir / f"temp_{window.window_id}_{self.response_var}.csv"
            if temp_data_file.exists():
                temp_data_file.unlink()
            return None
    
    def _save_window_results(self, result: Dict, window: TimeWindow) -> None:
        """Save window-specific results to text file and CSV with p-values."""
        try:
            results_file = self.text_dir / f"{window.window_id}_significant_terms_summary.txt"
            
            # Calculate percentages
            total_terms = result['num_terms_tested']
            raw_percent = (result['raw_significant_count'] / total_terms * 100) if total_terms > 0 else 0
            fdr_percent = (result['fdr_significant_count'] / total_terms * 100) if total_terms > 0 else 0
            bonferroni_percent = (result['bonferroni_significant_count'] / total_terms * 100) if total_terms > 0 else 0
            
            with open(results_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write(f"ROLLING WINDOW GRANGER CAUSALITY ANALYSIS - {window.window_id.upper()}\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Time Period: {window.start_year}-{window.start_month:02d} to {window.end_year}-{window.end_month:02d}\n")
                f.write(f"Data Points: {result['data_points']}\n")
                f.write(f"Terms Tested: {result['num_terms_tested']}\n\n")
                
                f.write("SIGNIFICANT TERMS COUNTS:\n")
                f.write(f"  Raw significant terms: {result['raw_significant_count']} ({raw_percent:.1f}%)\n")
                f.write(f"  FDR significant terms: {result['fdr_significant_count']} ({fdr_percent:.1f}%)\n")
                f.write(f"  Bonferroni significant terms: {result['bonferroni_significant_count']} ({bonferroni_percent:.1f}%)\n\n")
                
                f.write("SIGNIFICANT TERMS LISTS:\n")
                f.write(f"Raw significant terms ({result['raw_significant_count']}):\n")
                for term in sorted(result['raw_terms']):
                    f.write(f"  - {term}\n")
                
                f.write(f"\nFDR significant terms ({result['fdr_significant_count']}):\n")
                for term in sorted(result['fdr_terms']):
                    f.write(f"  - {term}\n")
                
                f.write(f"\nBonferroni significant terms ({result['bonferroni_significant_count']}):\n")
                for term in sorted(result['bonferroni_terms']):
                    f.write(f"  - {term}\n")
                
                f.write("\n" + "="*80 + "\n")
            
            # Save p-values CSV for all lags
            self._save_pvalues_csv(result, window)
            
            logger.info(f"Window results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Error saving window results: {e}")
    
    def _save_pvalues_csv(self, result: Dict, window: TimeWindow) -> None:
        """Save p-values for all lags to CSV file."""
        try:
            granger_results = result['granger_results']
            
            # Create DataFrame with p-values for all lags
            pvalues_data = []
            
            # Get all terms that were tested (from the raw terms list)
            all_terms = result['raw_terms']
            
            for lag in range(1, max_lags_to_test + 1):
                for term in all_terms:
                    # Get p-value from term_significance_by_lag
                    pvalue = None
                    
                    if (hasattr(granger_results, 'term_significance_by_lag') and 
                        term in granger_results.term_significance_by_lag and 
                        lag in granger_results.term_significance_by_lag[term]):
                        pvalue = granger_results.term_significance_by_lag[term][lag]
                    
                    # If not found, set to NaN
                    if pvalue is None:
                        pvalue = float('nan')
                    
                    pvalues_data.append({
                        'term': term,
                        'lag': lag,
                        'pvalue': pvalue,
                        'significant_raw': pvalue < alpha_level if not pd.isna(pvalue) else False,
                        'significant_fdr': term in result['fdr_terms'],
                        'significant_bonferroni': term in result['bonferroni_terms']
                    })
            
            if pvalues_data:
                pvalues_df = pd.DataFrame(pvalues_data)
                pvalues_file = self.csv_dir / f"{window.window_id}_pvalues_all_lags.csv"
                pvalues_df.to_csv(pvalues_file, index=False)
                logger.info(f"P-values CSV saved to {pvalues_file}")
            
        except Exception as e:
            logger.error(f"Error saving p-values CSV: {e}")
    
    def run_rolling_analysis(self) -> None:
        """Run the complete rolling window analysis."""
        logger.info("=== STARTING ROLLING WINDOW GRANGER CAUSALITY ANALYSIS ===")
        
        # Load data
        if not self.load_data():
            logger.error("Failed to load data. Exiting.")
            return
        
        # Create windows
        self.create_rolling_windows()
        if not self.windows:
            logger.error("No valid windows created. Exiting.")
            return
        
        # Analyze each window
        self.results = []
        for window in self.windows:
            result = self.analyze_time_window(window)
            if result:
                self.results.append(result)
        
        # Generate summary
        self._generate_summary()
        
        logger.info("=== ROLLING WINDOW ANALYSIS COMPLETE ===")
    
    def _generate_summary(self) -> None:
        """Generate summary of all rolling window results."""
        logger.info("=== GENERATING SUMMARY ===")
        
        if not self.results:
            logger.warning("No successful results to summarize")
            return
        
        # Calculate summary statistics
        total_windows = len(self.windows)
        successful_windows = len(self.results)
        failed_windows = total_windows - successful_windows
        
        # Count terms across all windows and track which windows they appear in
        all_raw_terms = set()
        all_fdr_terms = set()
        all_bonferroni_terms = set()
        
        # Track term significance across windows
        term_window_counts = {
            'raw': {},
            'fdr': {},
            'bonferroni': {}
        }
        
        for result in self.results:
            all_raw_terms.update(result['raw_terms'])
            all_fdr_terms.update(result['fdr_terms'])
            all_bonferroni_terms.update(result['bonferroni_terms'])
            
            # Count windows for each term
            for term in result['raw_terms']:
                term_window_counts['raw'][term] = term_window_counts['raw'].get(term, 0) + 1
            for term in result['fdr_terms']:
                term_window_counts['fdr'][term] = term_window_counts['fdr'].get(term, 0) + 1
            for term in result['bonferroni_terms']:
                term_window_counts['bonferroni'][term] = term_window_counts['bonferroni'].get(term, 0) + 1
        
        # Calculate rates
        total_terms_tested = sum(result['num_terms_tested'] for result in self.results)
        total_raw_significant = sum(result['raw_significant_count'] for result in self.results)
        total_fdr_significant = sum(result['fdr_significant_count'] for result in self.results)
        total_bonferroni_significant = sum(result['bonferroni_significant_count'] for result in self.results)
        
        raw_rate = total_raw_significant / total_terms_tested if total_terms_tested > 0 else 0
        fdr_rate = total_fdr_significant / total_terms_tested if total_terms_tested > 0 else 0
        bonferroni_rate = total_bonferroni_significant / total_terms_tested if total_terms_tested > 0 else 0
        
        # Save summary
        summary_file = self.text_dir / "rolling_window_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ROLLING WINDOW GRANGER CAUSALITY ANALYSIS - SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write("ANALYSIS CONFIGURATION:\n")
            f.write(f"  Data file: {self.data_file}\n")
            f.write(f"  Response variable: {self.response_var}\n")
            f.write(f"  Window length: {rolling_window_years} years\n")
            f.write(f"  Step size: {rolling_window_step_years} years\n")
            f.write(f"  Time period: {rolling_window_start_year}-{rolling_window_start_month:02d} to {rolling_window_end_year}-{rolling_window_end_month:02d}\n")
            f.write(f"  Max lags: {max_lags_to_test}\n\n")
            
            f.write("WINDOW SUMMARY:\n")
            f.write(f"  Total windows: {total_windows}\n")
            f.write(f"  Successful windows: {successful_windows}\n")
            f.write(f"  Failed windows: {failed_windows}\n\n")
            
            f.write("SIGNIFICANCE RATES:\n")
            f.write(f"  Raw significance rate: {raw_rate:.3f}\n")
            f.write(f"  FDR significance rate: {fdr_rate:.3f}\n")
            f.write(f"  Bonferroni significance rate: {bonferroni_rate:.3f}\n\n")
            
            f.write("TERMS SIGNIFICANT AT LEAST ONCE:\n")
            f.write(f"  Raw significant terms: {len(all_raw_terms)}\n")
            f.write(f"  FDR significant terms: {len(all_fdr_terms)}\n")
            f.write(f"  Bonferroni significant terms: {len(all_bonferroni_terms)}\n\n")
            
            f.write("TERMS SIGNIFICANT ACROSS WINDOWS:\n")
            f.write(f"  Raw significant terms (showing window counts out of {successful_windows} total):\n")
            for term, count in sorted(term_window_counts['raw'].items(), key=lambda x: x[1], reverse=True):
                f.write(f"    {term}: {count}/{successful_windows} windows\n")
            
            f.write(f"\n  FDR significant terms (showing window counts out of {successful_windows} total):\n")
            for term, count in sorted(term_window_counts['fdr'].items(), key=lambda x: x[1], reverse=True):
                f.write(f"    {term}: {count}/{successful_windows} windows\n")
            
            f.write(f"\n  Bonferroni significant terms (showing window counts out of {successful_windows} total):\n")
            for term, count in sorted(term_window_counts['bonferroni'].items(), key=lambda x: x[1], reverse=True):
                f.write(f"    {term}: {count}/{successful_windows} windows\n")
            
            f.write("\nDETAILED RESULTS BY WINDOW:\n")
            for result in self.results:
                f.write(f"  {result['window_id']} ({result['start_year']}-{result['start_month']:02d} to {result['end_year']}-{result['end_month']:02d}):\n")
                f.write(f"    Raw: {result['raw_significant_count']}, FDR: {result['fdr_significant_count']}, Bonferroni: {result['bonferroni_significant_count']}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        # Create visualization
        self._create_term_significance_visualization(term_window_counts, successful_windows)
        
        logger.info(f"Summary saved to {summary_file}")
        logger.info(f"Total windows: {total_windows}")
        logger.info(f"Successful windows: {successful_windows}")
        logger.info(f"Raw significance rate: {raw_rate:.3f}")
        logger.info(f"FDR significance rate: {fdr_rate:.3f}")
        logger.info(f"Bonferroni significance rate: {bonferroni_rate:.3f}")
    
    def _create_term_significance_visualization(self, term_window_counts: Dict, total_windows: int) -> None:
        """Create visualization showing term significance across windows."""
        try:
            logger.info("=== CREATING TERM SIGNIFICANCE VISUALIZATION ===")
            
            # Set up the plot style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(14, 16))
            fig.suptitle(f'Term Significance Across Rolling Windows\n{self.response_var} ({self.data_file})', 
                        fontsize=16, fontweight='bold')
            
            # Define colors for each significance level
            colors = {'raw': '#1f77b4', 'fdr': '#ff7f0e', 'bonferroni': '#2ca02c'}
            significance_names = {'raw': 'Raw (α=0.05)', 'fdr': 'FDR Corrected', 'bonferroni': 'Bonferroni Corrected'}
            
            for idx, (sig_type, counts) in enumerate(term_window_counts.items()):
                if not counts:
                    axes[idx].text(0.5, 0.5, f'No {sig_type} significant terms found', 
                                 ha='center', va='center', transform=axes[idx].transAxes, fontsize=12)
                    axes[idx].set_title(f'{significance_names[sig_type]} - No Significant Terms')
                    continue
                
                # Sort terms by window count (descending)
                sorted_terms = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                terms = [item[0] for item in sorted_terms]
                window_counts = [item[1] for item in sorted_terms]
                
                # Create horizontal bar plot
                bars = axes[idx].barh(range(len(terms)), window_counts, color=colors[sig_type], alpha=0.7)
                
                # Customize the plot
                axes[idx].set_yticks(range(len(terms)))
                axes[idx].set_yticklabels(terms, fontsize=10)
                axes[idx].set_xlabel(f'Number of Windows (out of {total_windows})', fontsize=12)
                axes[idx].set_title(f'{significance_names[sig_type]} - {len(terms)} Significant Terms', 
                                  fontsize=14, fontweight='bold')
                
                # Add value labels on bars
                for i, (bar, count) in enumerate(zip(bars, window_counts)):
                    axes[idx].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                                 f'{count}/{total_windows}', va='center', fontsize=9)
                
                # Add grid for better readability
                axes[idx].grid(axis='x', alpha=0.3)
                axes[idx].set_xlim(0, total_windows + 1)
                
                # Invert y-axis to show highest counts at top
                axes[idx].invert_yaxis()
            
            # Adjust layout
            plt.tight_layout()
            
            # Save the plot
            plot_file = self.base_output_dir / "term_significance_across_windows.png"
            plt.savefig(plot_file, dpi=figure_dpi, bbox_inches=figure_bbox_inches)
            plt.close()
            
            logger.info(f"Term significance visualization saved to {plot_file}")
            
        except Exception as e:
            logger.error(f"Error creating term significance visualization: {e}")


def main():
    """Main function to run rolling window analysis for a single dataset."""
    # Use configuration from confs.py
    data_file = file_name
    response_variable = response_var
    
    # Create analyzer and run analysis
    analyzer = RollingWindowAnalyzer(data_file, response_variable)
    analyzer.run_rolling_analysis()
    
    print("\n" + "="*80)
    print("ROLLING WINDOW ANALYSIS COMPLETE")
    print("="*80)
    print(f"Data file: {data_file}")
    print(f"Response variable: {response_variable}")
    print(f"Results saved to: {analyzer.base_output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
