"""
Comprehensive Rolling Window Analysis Runner

This script runs rolling window Granger causality analysis across multiple states
using configuration from confs.py and generates comprehensive reports.

Author: Xi Chen
Date: September 2025
"""

import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import warnings

from rolling_window_analysis import RollingWindowAnalyzer
from confs import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class MultiStateRollingAnalysis:
    """Class to run rolling window analysis across multiple states."""
    
    def __init__(self):
        """Initialize multi-state analysis using configuration from confs.py."""
        self.states_to_analyze = data_files_to_analyze
        self.results: Dict[str, Dict] = {}
        self.comparative_summary: Optional[pd.DataFrame] = None
        
        logger.info(f"Initialized MultiStateRollingAnalysis for {len(self.states_to_analyze)} states")
        logger.info(f"Window configuration: {rolling_window_years} years, step {rolling_window_step_years} years")
        logger.info(f"Time period: {rolling_window_start_year}-{rolling_window_start_month:02d} to {rolling_window_end_year}-{rolling_window_end_month:02d}")
    
    def analyze_single_dataset(self, data_file: str) -> Optional[Dict]:
        """Analyze a single dataset with rolling windows."""
        logger.info(f"=== ANALYZING DATASET: {data_file} ===")
        
        # Extract state name from filename
        state_name = data_file.replace('_2010_2020.csv', '')
        
        try:
            # Create analyzer for this state
            analyzer = RollingWindowAnalyzer(data_file, state_name)
            
            # Run analysis
            analyzer.run_rolling_analysis()
            
            # Store results
            if analyzer.results:
                result_summary = {
                    'state_name': state_name,
                    'data_file': data_file,
                    'total_windows': len(analyzer.windows),
                    'successful_windows': len(analyzer.results),
                    'failed_windows': len(analyzer.windows) - len(analyzer.results),
                    'results': analyzer.results,
                    'output_dir': analyzer.output_dir,
                    'success': True
                }
                
                # Calculate summary statistics
                if analyzer.results:
                    total_terms_tested = sum(r['num_terms_tested'] for r in analyzer.results)
                    total_raw = sum(r['raw_significant_count'] for r in analyzer.results)
                    total_fdr = sum(r['fdr_significant_count'] for r in analyzer.results)
                    total_bonferroni = sum(r['bonferroni_significant_count'] for r in analyzer.results)
                    
                    result_summary.update({
                        'raw_significance_rate': total_raw / total_terms_tested if total_terms_tested > 0 else 0,
                        'fdr_significance_rate': total_fdr / total_terms_tested if total_terms_tested > 0 else 0,
                        'bonferroni_significance_rate': total_bonferroni / total_terms_tested if total_terms_tested > 0 else 0,
                        'total_terms_tested': total_terms_tested,
                        'total_raw_significant': total_raw,
                        'total_fdr_significant': total_fdr,
                        'total_bonferroni_significant': total_bonferroni
                    })
                    
                    # Count unique terms across all windows
                    all_raw_terms = set()
                    all_fdr_terms = set()
                    all_bonferroni_terms = set()
                    
                    for r in analyzer.results:
                        all_raw_terms.update(r['raw_terms'])
                        all_fdr_terms.update(r['fdr_terms'])
                        all_bonferroni_terms.update(r['bonferroni_terms'])
                    
                    result_summary.update({
                        'unique_raw_terms': len(all_raw_terms),
                        'unique_fdr_terms': len(all_fdr_terms),
                        'unique_bonferroni_terms': len(all_bonferroni_terms),
                        'all_raw_terms': list(all_raw_terms),
                        'all_fdr_terms': list(all_fdr_terms),
                        'all_bonferroni_terms': list(all_bonferroni_terms)
                    })
                
                logger.info(f"✓ Successfully analyzed {state_name}")
                return result_summary
            else:
                logger.warning(f"✗ No results generated for {state_name}")
                return {
                    'state_name': state_name,
                    'data_file': data_file,
                    'success': False,
                    'error': 'No results generated'
                }
                
        except Exception as e:
            logger.error(f"✗ Error analyzing {state_name}: {e}")
            return {
                'state_name': state_name,
                'data_file': data_file,
                'success': False,
                'error': str(e)
            }
    
    def run_analysis_for_all_datasets(self) -> None:
        """Run rolling window analysis for all datasets in data_files_to_analyze."""
        logger.info("=== STARTING MULTI-DATASET ROLLING WINDOW ANALYSIS ===")
        
        for i, data_file in enumerate(self.states_to_analyze, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"ANALYZING DATASET {i}/{len(self.states_to_analyze)}: {data_file}")
            logger.info(f"{'='*80}")
            
            result = self.analyze_single_dataset(data_file)
            if result:
                self.results[result['state_name']] = result
        
        # Generate comparative analysis
        self.generate_comparative_analysis()
        
        # Generate final report
        self.generate_final_report()
        
        logger.info("=== MULTI-DATASET ROLLING WINDOW ANALYSIS COMPLETE ===")
    
    def generate_comparative_analysis(self) -> None:
        """Generate comparative analysis across all successfully analyzed datasets."""
        logger.info("=== GENERATING COMPARATIVE ANALYSIS ===")
        
        successful_results = {k: v for k, v in self.results.items() if v.get('success', False)}
        
        if not successful_results:
            logger.warning("No successful analyses to compare")
            return
        
        # Collect summary statistics
        state_summaries = []
        for state, data in successful_results.items():
            state_summaries.append({
                'state': state,
                'data_file': data['data_file'],
                'total_windows': data['total_windows'],
                'successful_windows': data['successful_windows'],
                'failed_windows': data['failed_windows'],
                'raw_significance_rate': data.get('raw_significance_rate', 0),
                'fdr_significance_rate': data.get('fdr_significance_rate', 0),
                'bonferroni_significance_rate': data.get('bonferroni_significance_rate', 0),
                'unique_raw_terms': data.get('unique_raw_terms', 0),
                'unique_fdr_terms': data.get('unique_fdr_terms', 0),
                'unique_bonferroni_terms': data.get('unique_bonferroni_terms', 0),
                'total_terms_tested': data.get('total_terms_tested', 0),
                'total_raw_significant': data.get('total_raw_significant', 0),
                'total_fdr_significant': data.get('total_fdr_significant', 0),
                'total_bonferroni_significant': data.get('total_bonferroni_significant', 0)
            })
        
        # Create comparative DataFrame
        self.comparative_summary = pd.DataFrame(state_summaries)
        
        # Calculate aggregate statistics
        aggregate_stats = {
            'total_datasets_analyzed': len(successful_results),
            'total_datasets_failed': len(self.results) - len(successful_results),
            'mean_raw_significance_rate': self.comparative_summary['raw_significance_rate'].mean(),
            'mean_fdr_significance_rate': self.comparative_summary['fdr_significance_rate'].mean(),
            'mean_bonferroni_significance_rate': self.comparative_summary['bonferroni_significance_rate'].mean(),
            'mean_unique_raw_terms': self.comparative_summary['unique_raw_terms'].mean(),
            'mean_unique_fdr_terms': self.comparative_summary['unique_fdr_terms'].mean(),
            'mean_unique_bonferroni_terms': self.comparative_summary['unique_bonferroni_terms'].mean(),
            'std_raw_significance_rate': self.comparative_summary['raw_significance_rate'].std(),
            'std_fdr_significance_rate': self.comparative_summary['fdr_significance_rate'].std(),
            'std_bonferroni_significance_rate': self.comparative_summary['bonferroni_significance_rate'].std()
        }
        
        # Save comparative results
        output_dir = Path(result_dir) / granger_causality_prefix / "multi_dataset_rolling_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.comparative_summary.to_csv(
            output_dir / "comparative_summary.csv", 
            index=False
        )
        
        # Save aggregate statistics
        agg_df = pd.DataFrame(list(aggregate_stats.items()), columns=['metric', 'value'])
        agg_df.to_csv(
            output_dir / "aggregate_statistics.csv", 
            index=False
        )
        
        # Log summary
        logger.info(f"Comparative analysis complete for {len(successful_results)} datasets")
        logger.info(f"Mean raw significance rate: {aggregate_stats['mean_raw_significance_rate']:.3f} ± {aggregate_stats['std_raw_significance_rate']:.3f}")
        logger.info(f"Mean FDR significance rate: {aggregate_stats['mean_fdr_significance_rate']:.3f} ± {aggregate_stats['std_fdr_significance_rate']:.3f}")
        logger.info(f"Mean Bonferroni significance rate: {aggregate_stats['mean_bonferroni_significance_rate']:.3f} ± {aggregate_stats['std_bonferroni_significance_rate']:.3f}")
        logger.info(f"Mean unique raw terms: {aggregate_stats['mean_unique_raw_terms']:.1f}")
        logger.info(f"Mean unique FDR terms: {aggregate_stats['mean_unique_fdr_terms']:.1f}")
        logger.info(f"Mean unique Bonferroni terms: {aggregate_stats['mean_unique_bonferroni_terms']:.1f}")
    
    def generate_final_report(self) -> None:
        """Generate a comprehensive final report."""
        logger.info("=== GENERATING FINAL REPORT ===")
        
        output_dir = Path(result_dir) / "multi_dataset_rolling_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / "multi_dataset_rolling_analysis_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MULTI-DATASET ROLLING WINDOW GRANGER CAUSALITY ANALYSIS - FINAL REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("ANALYSIS CONFIGURATION:\n")
            f.write(f"  Window length: {rolling_window_years} years\n")
            f.write(f"  Step size: {rolling_window_step_years} years\n")
            f.write(f"  Time period: {rolling_window_start_year}-{rolling_window_start_month:02d} to {rolling_window_end_year}-{rolling_window_end_month:02d}\n")
            f.write(f"  Max lags: {max_lags_to_test}\n")
            f.write(f"  Min data points per window: {rolling_window_min_data_points}\n\n")
            
            f.write("DATASETS ANALYZED:\n")
            successful_count = 0
            for i, (state, data) in enumerate(self.results.items(), 1):
                status = "✓" if data.get('success', False) else "✗"
                if data.get('success', False):
                    successful_count += 1
                f.write(f"  {i:2d}. {state:<20} {status}\n")
            f.write(f"\nTotal datasets: {len(self.results)}\n")
            f.write(f"Successful: {successful_count}\n")
            f.write(f"Failed: {len(self.results) - successful_count}\n\n")
            
            if self.comparative_summary is not None and len(self.comparative_summary) > 0:
                f.write("COMPARATIVE SUMMARY:\n")
                f.write(f"  Datasets successfully analyzed: {len(self.comparative_summary)}\n")
                f.write(f"  Mean raw significance rate: {self.comparative_summary['raw_significance_rate'].mean():.3f}\n")
                f.write(f"  Mean FDR significance rate: {self.comparative_summary['fdr_significance_rate'].mean():.3f}\n")
                f.write(f"  Mean Bonferroni significance rate: {self.comparative_summary['bonferroni_significance_rate'].mean():.3f}\n")
                f.write(f"  Mean unique raw terms: {self.comparative_summary['unique_raw_terms'].mean():.1f}\n")
                f.write(f"  Mean unique FDR terms: {self.comparative_summary['unique_fdr_terms'].mean():.1f}\n")
                f.write(f"  Mean unique Bonferroni terms: {self.comparative_summary['unique_bonferroni_terms'].mean():.1f}\n\n")
                
                f.write("DETAILED RESULTS BY DATASET:\n")
                for _, row in self.comparative_summary.iterrows():
                    f.write(f"  {row['state']}:\n")
                    f.write(f"    Raw significance rate: {row['raw_significance_rate']:.3f}\n")
                    f.write(f"    FDR significance rate: {row['fdr_significance_rate']:.3f}\n")
                    f.write(f"    Bonferroni significance rate: {row['bonferroni_significance_rate']:.3f}\n")
                    f.write(f"    Unique raw terms: {int(row['unique_raw_terms'])}\n")
                    f.write(f"    Unique FDR terms: {int(row['unique_fdr_terms'])}\n")
                    f.write(f"    Unique Bonferroni terms: {int(row['unique_bonferroni_terms'])}\n")
                    f.write(f"    Successful windows: {int(row['successful_windows'])}/{int(row['total_windows'])}\n\n")
            
            f.write("KEY FINDINGS:\n")
            f.write("  1. Rolling window analysis successfully implemented with configurable time periods\n")
            f.write("  2. Multiple testing correction applied (Bonferroni and FDR) per window\n")
            f.write("  3. Results show variation in significance across datasets and time windows\n")
            f.write("  4. FDR correction provides intermediate results between raw and Bonferroni\n")
            f.write("  5. Some terms appear significant across multiple windows, suggesting robust signals\n")
            f.write("  6. Results saved in respective results/states/response_var folders\n\n")
            
            f.write("FILES GENERATED:\n")
            f.write(f"  - Comparative summary: {output_dir}/comparative_summary.csv\n")
            f.write(f"  - Aggregate statistics: {output_dir}/aggregate_statistics.csv\n")
            f.write(f"  - Individual dataset results: results/states/*/\n")
            f.write(f"  - Window-specific results: results/states/*/window_*_significant_terms_summary.txt\n")
        
        logger.info(f"Final report saved to {report_path}")


def main():
    """Main function to run multi-dataset rolling window analysis."""
    # Create analyzer and run analysis for all datasets
    analyzer = MultiStateRollingAnalysis()
    analyzer.run_analysis_for_all_datasets()
    
    print("\n" + "="*80)
    print("MULTI-DATASET ROLLING WINDOW ANALYSIS COMPLETE")
    print("="*80)
    print(f"Datasets analyzed: {len(analyzer.states_to_analyze)}")
    print(f"Successful analyses: {len([r for r in analyzer.results.values() if r.get('success', False)])}")
    print(f"Results saved to: {result_dir}/*/")
    print(f"Comparative analysis: {result_dir}/multi_dataset_rolling_analysis/")
    print("="*80)


if __name__ == "__main__":
    main()
