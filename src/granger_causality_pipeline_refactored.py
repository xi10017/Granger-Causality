"""
Refactored Granger Causality Analysis Pipeline

This module provides a comprehensive, well-structured implementation of Granger causality
testing with improved code quality, error handling, and maintainability.

Author: Refactored from original implementation
Date: 2024
"""

import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import f
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import InfeasibleTestError

# Import configuration
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
class AnalysisConfig:
    """Configuration class for Granger causality analysis."""
    data_dir: str
    result_dir: str
    max_terms: Optional[int]
    response_var: str
    max_lags_to_test: int
    low_variance_threshold: float
    zero_ratio_threshold: float
    alpha_level: float
    bonferroni_alpha: float
    fdr_alpha: float
    figure_dpi: int
    figure_bbox_inches: str
    file_name: str
    results_prefix: str
    visualization_prefix: str
    summary_prefix: str
    time_series_prefix: str
    granger_causality_prefix: str
    comprehensive_analysis_prefix: str
    
    @classmethod
    def from_confs(cls) -> 'AnalysisConfig':
        """Create configuration from confs.py variables."""
        return cls(
            data_dir=data_dir,
            result_dir=result_dir,
            max_terms=max_terms,
            response_var=response_var,
            max_lags_to_test=max_lags_to_test,
            low_variance_threshold=low_variance_threshold,
            zero_ratio_threshold=zero_ratio_threshold,
            alpha_level=alpha_level,
            bonferroni_alpha=bonferroni_alpha,
            fdr_alpha=fdr_alpha,
            figure_dpi=figure_dpi,
            figure_bbox_inches=figure_bbox_inches,
            file_name=file_name,
            results_prefix=results_prefix,
            visualization_prefix=visualization_prefix,
            summary_prefix=summary_prefix,
            time_series_prefix=time_series_prefix,
            granger_causality_prefix=granger_causality_prefix,
            comprehensive_analysis_prefix=comprehensive_analysis_prefix
        )


@dataclass
class GrangerResults:
    """Results from Granger causality analysis."""
    f_statistic: float
    p_value: float
    model_restricted: sm.regression.linear_model.RegressionResultsWrapper
    model_unrestricted: sm.regression.linear_model.RegressionResultsWrapper
    significant_uncorrected: List[str]
    significant_bonferroni: List[str]
    significant_fdr: List[str]
    term_significance: List[Tuple[str, float]]
    term_significance_by_lag: Dict[str, Dict[int, float]]
    bonferroni_threshold: float
    sample_size: int


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""
    base_width: int = 16
    base_height: int = 8
    small_font_size: int = 6
    medium_font_size: int = 8
    large_font_size: int = 10
    value_font_size: int = 8


class GrangerCausalityAnalyzer:
    """Main class for Granger causality analysis."""
    
    def __init__(self, config: AnalysisConfig):
        """Initialize analyzer with configuration."""
        self.config = config
        self.viz_config = VisualizationConfig()
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if not self.config.response_var:
            raise ValueError("response_var must be specified in configuration")
        
        if self.config.max_lags_to_test < 1:
            raise ValueError("max_lags_to_test must be >= 1")
        
        if not (0 < self.config.alpha_level < 1):
            raise ValueError("alpha_level must be between 0 and 1")
        
        # Validate paths
        data_path = Path(self.config.data_dir) / self.config.file_name
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
    
    def load_and_prepare_data(self) -> Tuple[Optional[pd.DataFrame], List[str], str]:
        """Load and prepare data from the specified file."""
        logger.info("=== LOADING AND PREPARING DATA ===")
        
        try:
            data_file_path = Path(self.config.data_dir) / self.config.file_name
            
            if not data_file_path.exists():
                logger.error(f"Data file {data_file_path} not found")
                return None, [], ""
            
            # Load the data
            data = pd.read_csv(data_file_path)
            logger.info(f"Data shape: {data.shape}")
            logger.info(f"Data columns: {list(data.columns[:5])}...")
            
            # Validate response variable
            if self.config.response_var not in data.columns:
                logger.error(f"Response variable '{self.config.response_var}' not found in data columns: {list(data.columns)}")
                return None, [], ""
            
            response_column = self.config.response_var
            logger.info(f"Response variable specified: {response_column}")
            
            # Get search terms (all columns except date and response variable)
            search_terms = [col for col in data.columns if col not in ['date', response_column]]
            
            return data, search_terms, response_column
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None, [], ""
    
    def perform_data_diagnostics(self, data: pd.DataFrame, search_terms: List[str]) -> List[str]:
        """Perform diagnostic analysis on data."""
        logger.info("=== DIAGNOSTIC ANALYSIS ===")
        
        if data is None:
            logger.warning("No data available for diagnostics")
            return []
        
        try:
            constant_columns = []
            low_variance_columns = []
            
            for col in search_terms:
                if col in data.columns:
                    if data[col].nunique() == 1:
                        constant_columns.append(col)
                    elif data[col].std() < self.config.low_variance_threshold:
                        low_variance_columns.append(col)
            
            logger.info(f"Constant columns: {len(constant_columns)}")
            logger.info(f"Low variance columns (std < {self.config.low_variance_threshold}): {len(low_variance_columns)}")
            
            # Filter out problematic columns
            filtered_columns = [col for col in search_terms 
                               if col not in constant_columns and col not in low_variance_columns]
            
            logger.info(f"After filtering: {len(filtered_columns)} terms remaining")
            return filtered_columns
            
        except Exception as e:
            logger.error(f"Error in data diagnostics: {e}")
            return search_terms
    
    def prepare_merged_data(self, data: pd.DataFrame, search_terms: List[str], 
                           response_column: str, max_lag: int) -> Tuple[Optional[pd.DataFrame], List[str], List[str], List[str]]:
        """Prepare dataset with lagged variables."""
        logger.info("=== PREPARING DATA WITH LAGGED VARIABLES ===")
        
        if data is None:
            logger.warning("No data available")
            return None, [], [], []
        
        try:
            # Parse date and add YEAR/WEEK columns
            data = data.copy()
            data['date'] = pd.to_datetime(data['date'])
            data['YEAR'] = data['date'].dt.isocalendar().year
            data['WEEK'] = data['date'].dt.isocalendar().week
            
            # Create lagged variables
            response_lags = []
            all_lags = []
            
            for lag in range(1, max_lag + 1):
                # Create lag for the response variable (target)
                lag_col = f'{response_column}_lag{lag}'
                data[lag_col] = data[response_column].shift(lag)
                response_lags.append(lag_col)
                
                # Create lags for search terms
                for term in search_terms:
                    if term in data.columns:
                        lag_col = f'{term}_lag{lag}'
                        data[lag_col] = data[term].shift(lag)
                        all_lags.append(lag_col)
            
            # Clean data - drop rows where we can't compute the target variable
            data = data.dropna(subset=[response_column])
            
            logger.info(f"Data points after cleaning: {len(data)}")
            logger.info(f"Response variable lag columns: {len(response_lags)}")
            logger.info(f"Search term lag columns: {len(all_lags)}")
            
            # Check how many complete cases we have
            complete_cases = data.dropna()
            logger.info(f"Complete cases (no NaN values): {len(complete_cases)}")
            
            return data, response_lags, all_lags, search_terms
            
        except Exception as e:
            logger.error(f"Error preparing merged data: {e}")
            return None, [], [], []
    
    def perform_granger_causality_test(self, df_processed: pd.DataFrame, response_lags: List[str], 
                                     all_lags: List[str], response_column: str) -> Optional[GrangerResults]:
        """Perform the main Granger causality test."""
        logger.info("=" * 60)
        logger.info("MULTIPLE LINEAR REGRESSION GRANGER CAUSALITY TEST")
        logger.info("=" * 60)
        
        try:
            # Create complete case dataset for regression
            regression_data = df_processed.dropna()
            
            if len(regression_data) == 0:
                logger.error("No complete cases available for regression after dropping NaN values")
                return None
            
            logger.info(f"Complete cases for regression: {len(regression_data)}")
            
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
                
                logger.info(f"Testing if ALL search terms together Granger-cause '{response_column}'")
                logger.info(f"Number of search terms included: {len(all_lags) // len(response_lags)}")
                logger.info(f"Number of lags: {len(response_lags)}")
                logger.info(f"Sample size: {len(regression_data)}")
                logger.info(f"F-statistic: {F:.4f}")
                logger.info(f"p-value: {p_value:.6f}")
                logger.info(f"Degrees of freedom (numerator): {df1}")
                logger.info(f"Degrees of freedom (denominator): {df2}")
                
                significance = self._get_significance_level(p_value)
                logger.info(f"Significance: {significance}")
                
                if p_value < self.config.alpha_level:
                    logger.info(f"CONCLUSION: Search terms collectively Granger-cause {response_column} (p < {self.config.alpha_level})")
                else:
                    logger.info(f"CONCLUSION: No evidence that search terms collectively Granger-cause {response_column}")
                
                # Model fit statistics
                logger.info(f"Model Fit Statistics:")
                logger.info(f"Restricted model R²: {model_restricted.rsquared:.4f}")
                logger.info(f"Unrestricted model R²: {model_unrestricted.rsquared:.4f}")
                logger.info(f"R² improvement: {model_unrestricted.rsquared - model_restricted.rsquared:.4f}")
                
                # Perform individual term analysis
                term_results = self._analyze_individual_terms(
                    model_unrestricted, all_lags, response_lags, len(response_lags)
                )
                
                return GrangerResults(
                    f_statistic=F,
                    p_value=p_value,
                    model_restricted=model_restricted,
                    model_unrestricted=model_unrestricted,
                    significant_uncorrected=term_results['significant_uncorrected'],
                    significant_bonferroni=term_results['significant_bonferroni'],
                    significant_fdr=term_results['significant_fdr'],
                    term_significance=term_results['term_significance'],
                    term_significance_by_lag=term_results['term_significance_by_lag'],
                    bonferroni_threshold=term_results['bonferroni_threshold'],
                    sample_size=len(regression_data)
                )
                
            else:
                logger.error("Cannot compute F-statistic (check degrees of freedom or RSS)")
                return None
                
        except InfeasibleTestError as e:
            logger.error(f"Infeasible test error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error in Granger causality test: {e}")
            return None
    
    def _get_significance_level(self, p_value: float) -> str:
        """Get significance level string based on p-value."""
        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < self.config.alpha_level:
            return "*"
        else:
            return ""
    
    def _analyze_individual_terms(self, model_unrestricted: sm.regression.linear_model.RegressionResultsWrapper,
                                 all_lags: List[str], response_lags: List[str], max_lag: int) -> Dict:
        """Analyze individual term significance from the multiple regression model."""
        logger.info("=" * 60)
        logger.info("INDIVIDUAL TERM SIGNIFICANCE FROM MULTIPLE REGRESSION")
        logger.info("=" * 60)
        
        try:
            # Extract individual term significance
            term_significance = []
            term_significance_by_lag = {}
            
            # Get unique search terms from lag names
            search_terms = set()
            for lag_name in all_lags:
                term = lag_name.rsplit('_lag', 1)[0]
                search_terms.add(term)
            
            # Extract coefficients and p-values for search term lags
            for term in search_terms:
                term_pvals = []
                term_pvals_by_lag = {}
                
                for lag in range(1, max_lag + 1):
                    lag_col = f'{term}_lag{lag}'
                    if lag_col in model_unrestricted.params.index:
                        coef = model_unrestricted.params[lag_col]
                        try:
                            pval = model_unrestricted.pvalues[lag_col]
                        except (IndexError, TypeError):
                            # Handle HAC results
                            if hasattr(model_unrestricted.pvalues, 'loc'):
                                pval = model_unrestricted.pvalues.loc[lag_col]
                            else:
                                param_index = list(model_unrestricted.params.index).index(lag_col)
                                pval = model_unrestricted.pvalues[param_index]
                        
                        term_pvals.append(pval)
                        term_pvals_by_lag[lag] = pval
                
                if term_pvals:
                    # Use the minimum p-value across all lags for this term
                    min_p = min(term_pvals)
                    term_significance.append((term, min_p))
                    term_significance_by_lag[term] = term_pvals_by_lag
            
            # Sort by significance
            term_significance.sort(key=lambda x: x[1])
            
            # Calculate Bonferroni-corrected significance threshold
            num_tests = len(term_significance)
            bonferroni_threshold = self.config.bonferroni_alpha / num_tests if num_tests > 0 else self.config.bonferroni_alpha
            
            # Perform FDR correction
            if num_tests > 0:
                search_term_pvalues = [pval for term, pval in term_significance]
                fdr_rejected, fdr_pvalues, _, _ = multipletests(
                    search_term_pvalues, method='fdr_bh', alpha=self.config.fdr_alpha
                )
                
                fdr_significant_terms = set()
                for i, (term, pval) in enumerate(term_significance):
                    if fdr_rejected[i]:
                        fdr_significant_terms.add(term)
            else:
                fdr_significant_terms = set()
            
            logger.info(f"Number of search terms tested: {num_tests}")
            logger.info(f"Bonferroni-corrected significance threshold: {bonferroni_threshold:.6f}")
            logger.info(f"FDR correction applied (Benjamini-Hochberg method, alpha={self.config.fdr_alpha})")
            
            # Identify significant terms
            significant_uncorrected = [term for term, pval in term_significance if pval < self.config.alpha_level]
            significant_bonferroni = [term for term, pval in term_significance if pval < bonferroni_threshold]
            
            logger.info(f"Significant terms (uncorrected p < {self.config.alpha_level}): {len(significant_uncorrected)}")
            logger.info(f"Significant terms (Bonferroni-corrected p < {bonferroni_threshold:.6f}): {len(significant_bonferroni)}")
            logger.info(f"Significant terms (FDR-corrected): {len(fdr_significant_terms)}")
            
            return {
                'term_significance': term_significance,
                'significant_uncorrected': significant_uncorrected,
                'significant_bonferroni': significant_bonferroni,
                'significant_fdr': list(fdr_significant_terms),
                'bonferroni_threshold': bonferroni_threshold,
                'term_significance_by_lag': term_significance_by_lag
            }
            
        except Exception as e:
            logger.error(f"Error in individual term analysis: {e}")
            return {
                'term_significance': [],
                'significant_uncorrected': [],
                'significant_bonferroni': [],
                'significant_fdr': [],
                'bonferroni_threshold': self.config.bonferroni_alpha,
                'term_significance_by_lag': {}
            }
    
    def create_visualization(self, results: GrangerResults, max_lag: int, response_column: str) -> None:
        """Create visualization of individual term significance."""
        logger.info("=== CREATING VISUALIZATION ===")
        
        try:
            # Check for valid p-values
            valid_pvals = [(term, pval) for term, pval in results.term_significance if not np.isnan(pval)]
            
            if not valid_pvals:
                logger.warning("All p-values are NaN. This indicates the multiple regression model failed to fit properly.")
                return
            
            # Setup figure layout
            fig, axes = self._setup_figure_layout(max_lag, len(valid_pvals))
            
            # Create subplots
            self._create_subplots(axes, results, max_lag, valid_pvals)
            
            # Add title and legend
            self._add_title_and_legend(fig, axes, max_lag, response_column)
            
            # Save visualization
            self._save_visualization(fig, max_lag, response_column)
            
            # Print summary
            self._print_visualization_summary(valid_pvals, results)
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
    
    def _setup_figure_layout(self, max_lag: int, num_terms: int) -> Tuple[plt.Figure, List[plt.Axes]]:
        """Setup figure layout for visualization."""
        # Calculate subplot layout
        rows, cols = max_lag, 1
        
        # Adjust figure size
        fig_width = self.viz_config.base_width
        fig_height = self.viz_config.base_height * rows
        
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        
        if max_lag == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        return fig, axes
    
    def _create_subplots(self, axes: List[plt.Axes], results: GrangerResults, 
                        max_lag: int, valid_pvals: List[Tuple[str, float]]) -> None:
        """Create individual subplots for each lag."""
        for lag in range(1, max_lag + 1):
            ax = axes[lag - 1]
            
            # Get p-values for this specific lag
            lag_data = self._prepare_lag_data(results, lag, valid_pvals)
            
            if not lag_data['lag_pvals']:
                ax.text(0.5, 0.5, f'No valid p-values for lag {lag}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Lag {lag}', fontsize=12)
                continue
            
            # Create bars with colors
            colors = self._get_bar_colors(lag_data, results.bonferroni_threshold)
            bars = ax.bar(lag_data['lag_terms'], lag_data['lag_pvals'], color=colors, alpha=0.7)
            
            # Configure subplot
            self._configure_subplot(ax, lag_data, bars, lag)
    
    def _prepare_lag_data(self, results: GrangerResults, lag: int, 
                         valid_pvals: List[Tuple[str, float]]) -> Dict:
        """Prepare data for a specific lag visualization."""
        lag_pvals = []
        lag_terms = []
        is_missing = []
        
        # Get all terms from the overall term_significance list
        all_terms = [term for term, pval in results.term_significance]
        
        for term in all_terms:
            lag_terms.append(term)
            if term in results.term_significance_by_lag and lag in results.term_significance_by_lag[term]:
                pval = results.term_significance_by_lag[term][lag]
                if not np.isnan(pval):
                    lag_pvals.append(pval)
                    is_missing.append(False)
                else:
                    lag_pvals.append(1.0)
                    is_missing.append(True)
            else:
                lag_pvals.append(1.0)
                is_missing.append(True)
        
        # Sort terms by p-value
        term_pval_missing_pairs = list(zip(lag_terms, lag_pvals, is_missing))
        term_pval_missing_pairs.sort(key=lambda x: x[1])
        lag_terms, lag_pvals, is_missing = zip(*term_pval_missing_pairs)
        
        return {
            'lag_terms': list(lag_terms),
            'lag_pvals': list(lag_pvals),
            'is_missing': list(is_missing)
        }
    
    def _get_bar_colors(self, lag_data: Dict, bonferroni_threshold: float) -> List[str]:
        """Get colors for bars based on significance levels."""
        colors = []
        for i, term in enumerate(lag_data['lag_terms']):
            pval = lag_data['lag_pvals'][i]
            if lag_data['is_missing'][i]:
                colors.append('lightgray')
            elif pval < bonferroni_threshold:
                colors.append('darkred')
            elif pval < self.config.alpha_level:
                colors.append('red')
            else:
                colors.append('orange')
        return colors
    
    def _configure_subplot(self, ax: plt.Axes, lag_data: Dict, bars: List, lag: int) -> None:
        """Configure individual subplot appearance."""
        # Set labels and title
        ax.set_ylabel('p-value (log scale)', fontsize=10)
        ax.set_title(f'Lag {lag}', fontsize=12)
        ax.axhline(self.config.alpha_level, color='red', linestyle='--', 
                  label=f'p={self.config.alpha_level}', linewidth=1)
        
        # Configure x-axis
        font_size = self.viz_config.large_font_size if len(lag_data['lag_terms']) <= 20 else self.viz_config.small_font_size
        ax.tick_params(axis='x', rotation=45, labelsize=font_size, pad=15)
        ax.set_xlim(-0.5, len(lag_data['lag_terms']) - 0.5)
        ax.set_xticklabels(lag_data['lag_terms'], rotation=45, ha='right', fontsize=font_size)
        
        # Add value labels on bars
        value_font_size = self.viz_config.value_font_size if len(lag_data['lag_terms']) <= 20 else self.viz_config.small_font_size
        for bar, pval in zip(bars, lag_data['lag_pvals']):
            height = bar.get_height()
            label_text = 'N/A' if pval == 1.0 else f'{pval:.3f}'
            
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label_text, ha='center', va='bottom', fontsize=value_font_size,
                   rotation=0, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        # Set y-axis to log scale and limits
        ax.set_yscale('log')
        if lag_data['lag_pvals']:
            # Filter out 1.0 values (missing data) for y-axis limits
            valid_pvals = [p for p in lag_data['lag_pvals'] if p != 1.0]
            if valid_pvals:
                min_pval = min(valid_pvals)
                max_pval = max(valid_pvals)
                ax.set_ylim(min_pval * 0.1, max_pval * 10)
            else:
                ax.set_ylim(0.001, 1.0)
        else:
            ax.set_ylim(0.001, 1.0)
        
        ax.grid(axis='y', alpha=0.3)
    
    def _add_title_and_legend(self, fig: plt.Figure, axes: List[plt.Axes], 
                             max_lag: int, response_column: str) -> None:
        """Add title and legend to the figure."""
        # Add overall title - positioned lower for better balance
        fig.suptitle(f'Individual Term Significance for {response_column} - Max Lag = {max_lag}', 
                    fontsize=14, y=0.92)
        
        # Create legend elements
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkred', alpha=0.7, label='Bonferroni significant'),
            Patch(facecolor='red', alpha=0.7, label='Uncorrected significant'),
            Patch(facecolor='orange', alpha=0.7, label='Not significant'),
            Patch(facecolor='lightgray', alpha=0.7, label='Missing/NaN values')
        ]
        
        # Add legend to the figure (not to individual subplots)
        # Position it in the upper right corner of the figure, above the subplots
        fig.legend(handles=legend_elements, fontsize=10, loc='upper right',
                  bbox_to_anchor=(0.98, 0.88), frameon=True, fancybox=True, shadow=True)
        
        # Adjust layout to make room for the legend and better title positioning
        plt.tight_layout(pad=5.0)
        plt.subplots_adjust(bottom=0.25, top=0.88, hspace=0.4, right=0.85)
    
    def _save_visualization(self, fig: plt.Figure, max_lag: int, response_column: str) -> None:
        """Save visualization to file."""
        # Create results directory
        granger_results_dir = Path(self.config.result_dir) / self.config.granger_causality_prefix / response_column
        granger_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save visualization
        viz_filename = granger_results_dir / f"{self.config.visualization_prefix}_{response_column}_lag{max_lag}.png"
        fig.savefig(viz_filename, dpi=self.config.figure_dpi, bbox_inches=self.config.figure_bbox_inches)
        plt.close(fig)
        
        logger.info(f"Visualization saved to {viz_filename}")
    
    def _print_visualization_summary(self, valid_pvals: List[Tuple[str, float]], 
                                   results: GrangerResults) -> None:
        """Print summary statistics for visualization."""
        significant_uncorrected_plot = [term for term, pval in valid_pvals if pval < self.config.alpha_level]
        significant_bonferroni_plot = [term for term, pval in valid_pvals if pval < results.bonferroni_threshold]
        significant_fdr_plot = [term for term, pval in valid_pvals if term in results.significant_fdr]
        
        logger.info(f"Summary:")
        logger.info(f"Uncorrected (p < {self.config.alpha_level}): {len(significant_uncorrected_plot)} terms")
        logger.info(f"Bonferroni-corrected (p < {results.bonferroni_threshold:.6f}): {len(significant_bonferroni_plot)} terms")
        logger.info(f"FDR-corrected: {len(significant_fdr_plot)} terms")
        
        if significant_fdr_plot:
            logger.info("Top 5 FDR-significant terms:")
            sorted_valid = sorted(valid_pvals, key=lambda x: x[1])
            for i, (term, pval) in enumerate(sorted_valid[:5]):
                significance = "***" if term in results.significant_fdr else "**" if pval < self.config.alpha_level else ""
                logger.info(f"{i+1}. {term}: p = {pval:.4f}{significance}")
    
    def save_results(self, results: GrangerResults, max_lag: int, response_column: str) -> None:
        """Save comprehensive results to CSV file."""
        logger.info("=== SAVING COMPREHENSIVE RESULTS ===")
        
        try:
            # Create results directory
            granger_results_dir = Path(self.config.result_dir) / self.config.granger_causality_prefix / response_column
            granger_results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save CSV results
            csv_filename = granger_results_dir / f"{self.config.results_prefix}_{response_column}_lag{max_lag}.csv"
            self._save_csv_results(csv_filename, max_lag, response_column, results)
            
            logger.info(f"CSV results saved to {csv_filename}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def _write_results_header(self, f, max_lag: int, response_column: str, results: GrangerResults) -> None:
        """Write results header to file."""
        f.write(f"=== COMPREHENSIVE GRANGER CAUSALITY ANALYSIS SUMMARY ===\n")
        f.write(f"Data file: {self.config.file_name}\n")
        f.write(f"Response variable: {response_column}\n")
        f.write(f"Max lag: {max_lag}\n")
        f.write(f"Number of tests: {len(results.term_significance)}\n")
        f.write(f"Bonferroni threshold: {results.bonferroni_threshold:.6f}\n")
        f.write(f"FDR correction applied (Benjamini-Hochberg method, alpha={self.config.fdr_alpha})\n")
        f.write(f"Overall Granger causality F-statistic: {results.f_statistic:.4f}\n")
        f.write(f"Overall Granger causality p-value: {results.p_value:.6f}\n")
        f.write(f"Model R-squared: {results.model_unrestricted.rsquared:.4f}\n\n")
    
    def _write_significance_summary(self, f, results: GrangerResults) -> None:
        """Write significance summary to file."""
        f.write(f"=== SIGNIFICANCE SUMMARY ===\n")
        f.write(f"Uncorrected significant (p < {self.config.alpha_level}): {len(results.significant_uncorrected)} terms\n")
        f.write(f"Bonferroni significant (p < {results.bonferroni_threshold:.6f}): {len(results.significant_bonferroni)} terms\n")
        f.write(f"FDR significant: {len(results.significant_fdr)} terms\n\n")
    
    def _write_significant_terms(self, f, results: GrangerResults) -> None:
        """Write significant terms to file with individual lag details."""
        # Collect all significant term-lag combinations
        significant_combinations = []
        
        for term, term_lags in results.term_significance_by_lag.items():
            for lag, pval in term_lags.items():
                if not np.isnan(pval) and pval < self.config.alpha_level:
                    # Determine significance level for this specific lag
                    if pval < results.bonferroni_threshold:
                        significance = "Bonferroni"
                    elif term in results.significant_fdr:
                        significance = "FDR"
                    else:
                        significance = "Uncorrected"
                    
                    significant_combinations.append({
                        'term': term,
                        'lag': lag,
                        'p_value': pval,
                        'significance': significance
                    })
        
        if significant_combinations:
            f.write(f"=== ALL SIGNIFICANT TERMS (n={len(significant_combinations)} combinations) ===\n")
            f.write(f"Term\tLag\tP_value\tSignificance\n")
            
            # Sort by term name, then by lag number
            significant_combinations.sort(key=lambda x: (x['term'], x['lag']))
            
            for combo in significant_combinations:
                f.write(f"{combo['term']}\tlag{combo['lag']}\t{combo['p_value']:.6f}\t{combo['significance']}\n")
        else:
            f.write("No terms were significant at any level.\n")
    
    def _save_csv_results(self, csv_filename: Path, max_lag: int, response_column: str, results: GrangerResults) -> None:
        """Save significant terms results to CSV file."""
        try:
            # Prepare significant terms data
            significant_combinations = []
            
            for term, term_lags in results.term_significance_by_lag.items():
                for lag, pval in term_lags.items():
                    if not np.isnan(pval) and pval < self.config.alpha_level:
                        # Determine significance level for this specific lag
                        if pval < results.bonferroni_threshold:
                            significance = "Bonferroni"
                        elif term in results.significant_fdr:
                            significance = "FDR"
                        else:
                            significance = "Uncorrected"
                        
                        significant_combinations.append({
                            'Term': term,
                            'Lag': lag,
                            'P_Value': pval,
                            'Significance': significance
                        })
            
            # Sort by term name, then by lag number
            significant_combinations.sort(key=lambda x: (x['Term'], x['Lag']))
            
            # Create DataFrame and save to CSV
            terms_df = pd.DataFrame(significant_combinations)
            terms_df.to_csv(csv_filename, index=False)
            
            logger.info(f"CSV saved to {csv_filename}")
            
        except Exception as e:
            logger.error(f"Error saving CSV results: {e}")
    
    def run_analysis(self) -> None:
        """Run the complete Granger causality analysis."""
        logger.info("=== GENERALIZED GRANGER CAUSALITY ANALYSIS ===")
        logger.info(f"Configuration loaded from confs.py")
        logger.info(f"Data directory: {self.config.data_dir}")
        logger.info(f"Results directory: {self.config.result_dir}")
        logger.info(f"Data file to analyze: {self.config.file_name}")
        logger.info(f"Max lags to test: {self.config.max_lags_to_test}")
        
        if not self.config.file_name:
            logger.error("Please specify file_name in confs.py")
            return
        
        try:
            # Load and prepare data
            data, search_terms, response_column = self.load_and_prepare_data()
            
            if data is None:
                logger.error(f"Failed to load data from {self.config.file_name}")
                return
            
            # Perform data diagnostics
            filtered_columns = self.perform_data_diagnostics(data, search_terms)
            
            # Use filtered columns for analysis
            search_terms = filtered_columns[:self.config.max_terms] if self.config.max_terms else filtered_columns
            logger.info(f"Final number of search terms to use: {len(search_terms)}")
            
            # Run analysis for configured max lags
            for max_lag in range(1, self.config.max_lags_to_test + 1):
                logger.info(f"\n{'='*80}")
                logger.info(f"RUNNING ANALYSIS FOR {self.config.file_name} WITH MAX LAG = {max_lag}")
                logger.info(f"{'='*80}")
                
                # Prepare data with lagged variables
                df_processed, response_lags, all_lags, search_terms_simple = self.prepare_merged_data(
                    data, search_terms, response_column, max_lag
                )
                
                if df_processed is None:
                    logger.warning("Failed to prepare data. Continuing to next lag.")
                    continue
                
                # Check for constant values in response variable
                if df_processed[response_column].nunique() <= 1:
                    logger.error(f"Response variable '{response_column}' has constant values")
                    continue
                
                # Perform Granger causality test
                results = self.perform_granger_causality_test(
                    df_processed, response_lags, all_lags, response_column
                )
                
                if results is None:
                    logger.warning("Granger causality test failed. Continuing to next lag.")
                    continue
                
                # Create visualization
                self.create_visualization(results, max_lag, response_column)
                
                # Save comprehensive results
                self.save_results(results, max_lag, response_column)
                
                logger.info(f"\n=== ANALYSIS COMPLETE FOR {self.config.file_name} WITH MAX LAG = {max_lag} ===")
            
            # Show final output location
            granger_results_dir = Path(self.config.result_dir) / self.config.granger_causality_prefix
            logger.info(f"\n=== ALL ANALYSES COMPLETE FOR {self.config.file_name} ===")
            logger.info(f"Results saved to: {granger_results_dir}")
            
        except Exception as e:
            logger.error(f"Error in main analysis: {e}")
            raise


def main():
    """Main function to run the complete Granger causality analysis."""
    try:
        config = AnalysisConfig.from_confs()
        analyzer = GrangerCausalityAnalyzer(config)
        analyzer.run_analysis()
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        raise


if __name__ == "__main__":
    main()
