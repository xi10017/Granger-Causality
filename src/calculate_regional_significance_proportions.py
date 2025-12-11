"""
Calculate Regional Significance Proportions

This script analyzes the rolling window analysis results across all regions
to calculate the proportion of terms that were ever significant.

Author: Regional Analysis Implementation
Date: 2024
"""

import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

from confs import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class RegionalSignificanceAnalyzer:
    """Analyze significance proportions across all regions."""
    
    def __init__(self):
        """Initialize the regional analyzer."""
        self.results_dir = Path(result_dir) / granger_causality_prefix
        self.regions = []
        self.regional_data = {}
        self.summary_stats = {}
        
        logger.info("Initialized RegionalSignificanceAnalyzer")
    
    def load_regional_data(self) -> None:
        """Load data from all regions' matrix files."""
        logger.info("=== LOADING REGIONAL DATA ===")
        
        # Find all regions with matrix files
        for region_dir in self.results_dir.iterdir():
            if region_dir.is_dir():
                matrix_dir = region_dir / "rolling_window_analysis" / "matrices"
                if matrix_dir.exists():
                    self.regions.append(region_dir.name)
                    logger.info(f"Found region: {region_dir.name}")
        
        logger.info(f"Found {len(self.regions)} regions with matrix data")
        
        # Load matrix data for each region
        for region in self.regions:
            region_data = {}
            matrix_dir = self.results_dir / region / "rolling_window_analysis" / "matrices"
            
            for method in ['raw', 'fdr', 'bonferroni']:
                matrix_file = matrix_dir / f"pvalue_matrix_{method}.csv"
                if matrix_file.exists():
                    try:
                        df = pd.read_csv(matrix_file, index_col=0)
                        # Remove the proportion row for analysis
                        if 'Proportion_Significant' in df.index:
                            df = df.drop('Proportion_Significant')
                        
                        # Convert all columns to numeric, coercing errors to NaN
                        for col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        region_data[method] = df
                        logger.info(f"Loaded {method} matrix for {region}: {df.shape}")
                    except Exception as e:
                        logger.error(f"Error loading {method} matrix for {region}: {e}")
                        region_data[method] = None
                else:
                    logger.warning(f"Matrix file not found: {matrix_file}")
                    region_data[method] = None
            
            self.regional_data[region] = region_data
    
    def calculate_regional_proportions(self) -> None:
        """Calculate significance proportions across all regions."""
        logger.info("=== CALCULATING REGIONAL PROPORTIONS ===")
        
        # Initialize summary data
        summary_data = []
        
        for method in ['raw', 'fdr', 'bonferroni']:
            logger.info(f"Analyzing {method.upper()} results...")
            
            # Collect all terms across all regions
            all_terms = set()
            for region, region_data in self.regional_data.items():
                if region_data.get(method) is not None:
                    all_terms.update(region_data[method].columns)
            
            all_terms = sorted(list(all_terms))
            logger.info(f"Total unique terms across all regions: {len(all_terms)}")
            
            # Calculate which unique terms are significant in at least one region
            unique_terms_significant = set()
            for region, region_data in self.regional_data.items():
                if region_data.get(method) is not None:
                    df = region_data[method]
                    significant_mask = df < 0.05
                    # Get terms that are significant in this region
                    significant_terms = df.columns[significant_mask.any(axis=0)].tolist()
                    unique_terms_significant.update(significant_terms)
            
            # Calculate proportion of unique terms that are significant in at least one region
            unique_terms_proportion = len(unique_terms_significant) / len(all_terms) if len(all_terms) > 0 else 0
            logger.info(f"Unique terms significant in at least one region: {len(unique_terms_significant)}/{len(all_terms)} = {unique_terms_proportion:.3f}")
            
            # Calculate proportions for each region
            region_proportions = {}
            for region, region_data in self.regional_data.items():
                if region_data.get(method) is not None:
                    df = region_data[method]
                    
                    # Calculate proportion of terms ever significant in this region
                    significant_mask = df < 0.05
                    terms_ever_significant = significant_mask.any(axis=0).sum()
                    total_terms = len(df.columns)
                    proportion = terms_ever_significant / total_terms if total_terms > 0 else 0
                    
                    region_proportions[region] = {
                        'proportion': proportion,
                        'count': terms_ever_significant,
                        'total': total_terms,
                        'terms': df.columns.tolist()
                    }
                    
                    summary_data.append({
                        'region': region,
                        'method': method.upper(),
                        'proportion': proportion,
                        'count': terms_ever_significant,
                        'total': total_terms,
                        'formatted': f"{proportion:.3f} ({terms_ever_significant}/{total_terms})"
                    })
                    
                    logger.info(f"{region} ({method.upper()}): {proportion:.3f} ({terms_ever_significant}/{total_terms})")
                else:
                    logger.warning(f"No {method} data for {region}")
            
            # Calculate overall statistics
            if region_proportions:
                proportions = [data['proportion'] for data in region_proportions.values()]
                counts = [data['count'] for data in region_proportions.values()]
                totals = [data['total'] for data in region_proportions.values()]
                
                self.summary_stats[method] = {
                    'mean_proportion': np.mean(proportions),
                    'std_proportion': np.std(proportions),
                    'min_proportion': np.min(proportions),
                    'max_proportion': np.max(proportions),
                    'total_regions': len(region_proportions),
                    'mean_count': np.mean(counts),
                    'mean_total': np.mean(totals),
                    'overall_proportion': unique_terms_proportion,  # Proportion of unique terms significant in at least one region
                    'overall_fraction': f"{len(unique_terms_significant)}/{len(all_terms)}",  # Fraction of unique terms significant
                    'pooled_proportion': sum(counts) / sum(totals) if sum(totals) > 0 else 0,  # Old calculation for reference
                    'pooled_fraction': f"{sum(counts)}/{sum(totals)}",  # Fraction for pooled calculation
                    'unique_terms_total': len(all_terms),
                    'unique_terms_significant': len(unique_terms_significant)
                }
                
                logger.info(f"{method.upper()} Overall: {self.summary_stats[method]['overall_proportion']:.3f} (unique terms significant in at least one region)")
                logger.info(f"  Pooled proportion: {self.summary_stats[method]['pooled_proportion']:.3f} (all individual analyses)")
                logger.info(f"  Mean across regions: {self.summary_stats[method]['mean_proportion']:.3f} ± {self.summary_stats[method]['std_proportion']:.3f}")
                logger.info(f"  Range: {self.summary_stats[method]['min_proportion']:.3f} - {self.summary_stats[method]['max_proportion']:.3f}")
        
        # Create summary DataFrame
        self.summary_df = pd.DataFrame(summary_data)
        
        # Save detailed results
        self.save_results()
    
    def save_results(self) -> None:
        """Save analysis results to files."""
        logger.info("=== SAVING RESULTS ===")
        
        # Save results directly in granger_causality_results directory
        output_dir = Path(result_dir) / granger_causality_prefix
        
        # Save detailed summary
        summary_file = output_dir / "regional_significance_summary.csv"
        self.summary_df.to_csv(summary_file, index=False)
        logger.info(f"Detailed summary saved to {summary_file}")
        
        # Save overall statistics
        stats_data = []
        for method, stats in self.summary_stats.items():
            stats_data.append({
                'method': method.upper(),
                'overall_proportion': stats['overall_proportion'],  # Proportion of unique terms significant in at least one region
                'overall_fraction': stats['overall_fraction'],      # Fraction of unique terms significant (e.g., "52/63")
                'pooled_proportion': stats['pooled_proportion'],    # Old calculation for reference
                'pooled_fraction': stats['pooled_fraction'],        # Fraction for pooled calculation
                'unique_terms_total': stats['unique_terms_total'],
                'unique_terms_significant': stats['unique_terms_significant'],
                'mean_proportion': stats['mean_proportion'],
                'std_proportion': stats['std_proportion'],
                'min_proportion': stats['min_proportion'],
                'max_proportion': stats['max_proportion'],
                'total_regions': stats['total_regions'],
                'mean_count': stats['mean_count'],
                'mean_total': stats['mean_total']
            })
        
        stats_df = pd.DataFrame(stats_data)
        stats_file = output_dir / "overall_statistics.csv"
        stats_df.to_csv(stats_file, index=False)
        logger.info(f"Overall statistics saved to {stats_file}")
        
        # Create comprehensive report
        self.create_comprehensive_report(output_dir)
    
    def create_comprehensive_report(self, output_dir: Path) -> None:
        """Create a comprehensive text report."""
        report_file = output_dir / "regional_significance_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("REGIONAL SIGNIFICANCE ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("ANALYSIS OVERVIEW:\n")
            f.write(f"  Total regions analyzed: {len(self.regions)}\n")
            f.write(f"  Regions: {', '.join(sorted(self.regions))}\n")
            f.write(f"  Analysis methods: RAW, FDR, Bonferroni\n\n")
            
            f.write("METRIC EXPLANATIONS:\n")
            f.write("-" * 50 + "\n")
            f.write("OVERALL PROPORTION: The proportion of unique terms (out of 63 total) that are\n")
            f.write("  significant in at least one region.\n\n")
            f.write("POOLED PROPORTION: The pooled significance rate across all individual region\n")
            f.write("  analyses (the old calculation method).\n\n")
            f.write("MEAN ACROSS REGIONS: The average proportion of significant terms per region.\n\n")
            f.write("RANGE: The minimum and maximum proportions across all regions, showing\n")
            f.write("  regional variation in significance rates.\n\n")
            
            f.write("OVERALL STATISTICS BY METHOD:\n")
            f.write("-" * 50 + "\n")
            
            for method, stats in self.summary_stats.items():
                f.write(f"\n{method.upper()} CORRECTED:\n")
                f.write(f"  Overall proportion: {stats['overall_proportion']:.3f} ({stats['overall_fraction']})\n")
                f.write(f"    → Proportion of unique terms that are significant in at least one region\n")
                f.write(f"  Pooled proportion: {stats['pooled_proportion']:.3f} ({stats['pooled_fraction']})\n")
                f.write(f"    → Pooled significance rate across all individual region analyses\n")
                f.write(f"  Mean across regions: {stats['mean_proportion']:.3f} ± {stats['std_proportion']:.3f}\n")
                f.write(f"    → Average proportion of significant terms per region\n")
                f.write(f"  Range: {stats['min_proportion']:.3f} - {stats['max_proportion']:.3f}\n")
                f.write(f"    → Minimum and maximum proportions across all regions\n")
                f.write(f"  Regions analyzed: {stats['total_regions']}\n")
                f.write(f"  Mean terms per region: {stats['mean_count']:.1f}/{stats['mean_total']:.1f}\n")
                f.write(f"    → Average number of significant terms per region\n")
            
            f.write("\nDETAILED RESULTS BY REGION:\n")
            f.write("-" * 50 + "\n")
            
            for method in ['raw', 'fdr', 'bonferroni']:
                f.write(f"\n{method.upper()} CORRECTED:\n")
                method_data = self.summary_df[self.summary_df['method'] == method.upper()]
                
                for _, row in method_data.iterrows():
                    f.write(f"  {row['region']:<15}: {row['formatted']}\n")
            
            f.write("\nKEY FINDINGS:\n")
            f.write("-" * 50 + "\n")
            
            if self.summary_stats:
                raw_stats = self.summary_stats.get('raw', {})
                fdr_stats = self.summary_stats.get('fdr', {})
                bonferroni_stats = self.summary_stats.get('bonferroni', {})
                
                f.write("1. RAW (Uncorrected) Results:\n")
                f.write(f"   - Overall: {raw_stats.get('overall_proportion', 0):.3f} ({raw_stats.get('overall_fraction', 'N/A')}) of unique terms significant in at least one region\n")
                f.write(f"   - Regional variation: {raw_stats.get('min_proportion', 0):.3f} - {raw_stats.get('max_proportion', 0):.3f}\n")
                f.write(f"   - Pooled rate: {raw_stats.get('pooled_proportion', 0):.3f} ({raw_stats.get('pooled_fraction', 'N/A')}) across all individual analyses\n\n")
                
                f.write("2. FDR Corrected Results:\n")
                f.write(f"   - Overall: {fdr_stats.get('overall_proportion', 0):.3f} ({fdr_stats.get('overall_fraction', 'N/A')}) of unique terms significant in at least one region\n")
                f.write(f"   - Regional variation: {fdr_stats.get('min_proportion', 0):.3f} - {fdr_stats.get('max_proportion', 0):.3f}\n")
                f.write(f"   - Pooled rate: {fdr_stats.get('pooled_proportion', 0):.3f} ({fdr_stats.get('pooled_fraction', 'N/A')}) across all individual analyses\n\n")
                
                f.write("3. Bonferroni Corrected Results:\n")
                f.write(f"   - Overall: {bonferroni_stats.get('overall_proportion', 0):.3f} ({bonferroni_stats.get('overall_fraction', 'N/A')}) of unique terms significant in at least one region\n")
                f.write(f"   - Regional variation: {bonferroni_stats.get('min_proportion', 0):.3f} - {bonferroni_stats.get('max_proportion', 0):.3f}\n")
                f.write(f"   - Pooled rate: {bonferroni_stats.get('pooled_proportion', 0):.3f} ({bonferroni_stats.get('pooled_fraction', 'N/A')}) across all individual analyses\n\n")
                
                f.write("4. Correction Impact (Overall Proportions):\n")
                f.write(f"   - RAW to FDR: {raw_stats.get('overall_proportion', 0) - fdr_stats.get('overall_proportion', 0):.3f} reduction\n")
                f.write(f"   - FDR to Bonferroni: {fdr_stats.get('overall_proportion', 0) - bonferroni_stats.get('overall_proportion', 0):.3f} reduction\n")
                f.write(f"   - RAW to Bonferroni: {raw_stats.get('overall_proportion', 0) - bonferroni_stats.get('overall_proportion', 0):.3f} reduction\n")
            
            f.write("\nFILES GENERATED:\n")
            f.write("-" * 50 + "\n")
            f.write(f"  - Detailed summary: {output_dir}/regional_significance_summary.csv\n")
            f.write(f"  - Overall statistics: {output_dir}/overall_statistics.csv\n")
            f.write(f"  - This report: {output_dir}/regional_significance_report.txt\n")
        
        logger.info(f"Comprehensive report saved to {report_file}")
    
    def run_analysis(self) -> None:
        """Run the complete regional significance analysis."""
        logger.info("=== STARTING REGIONAL SIGNIFICANCE ANALYSIS ===")
        
        # Load data from all regions
        self.load_regional_data()
        
        if not self.regions:
            logger.error("No regions found with matrix data. Please run rolling window analysis first.")
            return
        
        # Calculate proportions
        self.calculate_regional_proportions()
        
        logger.info("=== REGIONAL SIGNIFICANCE ANALYSIS COMPLETE ===")


def main():
    """Main function to run regional significance analysis."""
    analyzer = RegionalSignificanceAnalyzer()
    analyzer.run_analysis()
    
    print("\n" + "="*80)
    print("REGIONAL SIGNIFICANCE ANALYSIS COMPLETE")
    print("="*80)
    print(f"Regions analyzed: {len(analyzer.regions)}")
    print(f"Results saved to: {result_dir}/granger_causality_results/")
    print("="*80)


if __name__ == "__main__":
    main()
