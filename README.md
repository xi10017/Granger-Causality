# Comprehensive Granger Causality Analysis System

A complete, generalized system for performing Granger causality analysis on time series data with comprehensive visualization and statistical analysis capabilities. This system can analyze any response variable (states, regions, countries, etc.) to determine if search trends or other predictors Granger-cause changes in the target variable.

## Overview

This system performs **Granger causality analysis** to determine whether predictor variables (e.g., search trends) can predict changes in a response variable (e.g., state activity, disease incidence, economic indicators). The analysis tests whether past values of predictor variables help predict current values of the response variable.

**Direction**: Predictor Variables → Response Variable
- **X Variables**: Predictor variables (e.g., search term volumes, economic indicators)
- **Y Variable**: Response variable (e.g., state activity, disease cases, sales data)
- **Question**: Do predictor variables from X time periods ago predict current response variable values?

## Key Features

- **Configuration-driven**: All parameters controlled via `confs.py`
- **Complete Analysis Pipeline**: 4-step automated analysis process
- **Rich Visualizations**: Bar graphs, time series plots, and comparison charts
- **Multiple Testing Corrections**: Bonferroni and FDR corrections
- **Organized Output**: Structured folder hierarchy for easy navigation
- **Batch Processing**: Analyze multiple datasets automatically
- **Comprehensive Reports**: Detailed statistical summaries and CSV exports
- **Generalized**: Works with any response variable (states, regions, countries, etc.)
- **Rolling Window Analysis**: Time-varying Granger causality analysis
- **Regional Significance Analysis**: Cross-region statistical comparisons
- **Advanced Statistics**: Both proportions and fractions with clear explanations

## System Architecture

The system consists of multiple Python scripts that work together to provide comprehensive Granger causality analysis:

### Core Analysis Scripts
1. **`granger_causality_pipeline.py`** - Main Granger causality analysis
2. **`granger_causality_pipeline_refactored.py`** - Refactored version with improved structure
3. **`create_comprehensive_significant_terms_summary.py`** - Creates summary of significant terms
4. **`time_series_analysis.py`** - Generates time series visualizations
5. **`analyze_comprehensive_results.py`** - Creates comprehensive analysis with bar graphs
6. **`analyze_multiple_data_files.py`** - Orchestrates complete pipeline for multiple files

### Advanced Analysis Scripts
7. **`rolling_window_analysis.py`** - Rolling window Granger causality analysis
8. **`run_rolling_window_analysis.py`** - Orchestrates rolling window analysis for multiple regions
9. **`calculate_regional_significance_proportions.py`** - Calculates regional significance statistics

### Configuration & Documentation
- **`confs.py`** - Central configuration file
- **`requirements.txt`** - Python dependencies
- **`README.md`** - This documentation
- **`REFACTORING_SUMMARY.md`** - Details of system improvements

## Configuration (`confs.py`)

### Data Configuration
```python
data_dir = "data/"                    # Directory containing data files
result_dir = "results/"               # Directory for output files
file_name = "Example.csv"             # Data file to analyze
response_var = "ExampleState"         # Response variable column name
```

### Analysis Configuration
```python
max_terms = None                      # Max search terms to use (None = use all)
max_lags_to_test = 5                 # Maximum number of lags to test (1 to max_lags_to_test)
```

### Statistical Thresholds
```python
alpha_level = 0.05                    # Significance level for hypothesis testing
bonferroni_alpha = 0.05              # Alpha for Bonferroni correction
fdr_alpha = 0.05                     # Alpha for FDR correction
```

### Data Quality Filters
```python
low_variance_threshold = 0.1          # Standard deviation threshold for filtering
zero_ratio_threshold = 0.8            # Threshold for filtering columns with too many zeros
```

### Output Configuration
```python
# File naming prefixes
results_prefix = "granger_significant_terms_data"
visualization_prefix = "granger_pvalues_data"
summary_prefix = "summary_data"
time_series_prefix = "time_series_plots"
granger_causality_prefix = "granger_causality_results"
comprehensive_analysis_prefix = "comprehensive_analysis"

# Visualization settings
figure_dpi = 300
figure_bbox_inches = 'tight'
```

## Data Structure Requirements

### Data File Format
Each data file should have the following structure:
- **Column 1**: `date` in YYYY-MM-DD format
- **Column 2**: Response variable name (e.g., "ExampleState") - this is the target variable
- **Columns 3+**: Predictor variable columns with numerical values

### Example Data File: `Example.csv`
```csv
date,ExampleState,flu symptoms,the flu,rsv,flu how long,...
2010-10-09,2.13477,0.0,0.0,0.0,0.0,...
2010-10-16,2.23456,0.1,0.0,0.0,0.1,...
2010-10-23,2.34567,0.2,0.1,0.0,0.2,...
...
```

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Analysis
Edit `confs.py` to match your analysis requirements:
- Set `file_name` to your data file
- Set `response_var` to your response variable column name
- Adjust `max_lags_to_test` as needed
- Modify other parameters as required

### 3. Run Analysis

#### Single File Analysis
```bash
# Run complete 4-step pipeline for one file
python granger_causality_pipeline.py
python create_comprehensive_significant_terms_summary.py
python time_series_analysis.py
python analyze_comprehensive_results.py
```

#### Multiple Files Analysis
```bash
# Run complete pipeline for multiple files automatically
python analyze_multiple_data_files.py
```

#### Rolling Window Analysis
```bash
# Run rolling window analysis for a single region
python rolling_window_analysis.py

# Run rolling window analysis for multiple regions
python run_rolling_window_analysis.py
```

#### Regional Significance Analysis
```bash
# Calculate regional significance proportions across all regions
python calculate_regional_significance_proportions.py
```

## Complete Analysis Pipeline

The system runs a comprehensive 4-step analysis:

### Step 1: Granger Causality Analysis (`granger_causality_pipeline.py`)
- Loads and prepares data
- Creates lagged variables
- Performs Granger causality tests for multiple lags
- Applies multiple testing corrections (Bonferroni, FDR)
- Generates significance visualizations
- Saves detailed results

### Step 2: Summary Creation (`create_comprehensive_significant_terms_summary.py`)
- Parses Granger causality results
- Creates comprehensive summary of significant terms
- Organizes results by significance type
- Saves summary text file

### Step 3: Time Series Analysis (`time_series_analysis.py`)
- Generates time series plots for significant terms
- Shows lagged predictor variables vs. current response variable
- Creates dual-axis plots for easy comparison
- Saves organized time series visualizations

### Step 4: Comprehensive Results Analysis (`analyze_comprehensive_results.py`)
- Creates 9 bar graphs (3 significance types × 3 metrics)
- Generates comparison plots
- Creates CSV summary tables
- Provides comprehensive statistical analysis

## Advanced Analysis Features

### Rolling Window Analysis (`rolling_window_analysis.py`)
- **Time-varying Granger causality**: Analyzes how Granger causality changes over time
- **Sliding window approach**: Tests causality in overlapping time windows
- **Dynamic significance tracking**: Shows which terms are significant in each window
- **Comprehensive visualizations**: Heatmaps, time series plots, and trend analysis
- **Matrix outputs**: P-value matrices for further analysis

### Regional Significance Analysis (`calculate_regional_significance_proportions.py`)
- **Cross-region comparisons**: Analyzes significance patterns across multiple regions
- **Overall proportions**: Calculates proportion of unique terms significant in at least one region
- **Pooled statistics**: Provides both individual and pooled significance rates
- **Clear metric explanations**: Distinguishes between different types of proportions
- **Fraction reporting**: Shows both decimal proportions and exact fractions (e.g., "62/63")
- **Comprehensive reports**: Detailed text reports with statistical summaries

### Multi-Dataset Rolling Analysis (`run_rolling_window_analysis.py`)
- **Batch processing**: Runs rolling window analysis across multiple regions
- **Automated workflow**: Processes all data files in sequence
- **Consistent parameters**: Ensures uniform analysis across all regions
- **Progress tracking**: Shows analysis progress and completion status

## Output Structure

The system creates a well-organized folder structure:

### Standard Analysis Output
```
results/
└── granger_causality_results/
    └── {response_var}/
        ├── granger_pvalues_data_{response_var}_lag{1-5}.png
        ├── granger_significant_terms_data_{response_var}_lag{1-5}.txt
        ├── summary_data_{response_var}.txt
        ├── time_series_plots/
        │   └── {term_name}/
        │       └── {term_name}_{response_var}_analysis.png
        └── comprehensive_analysis/
            ├── bonferroni_count_bar_graph.png
            ├── bonferroni_mean_pval_bar_graph.png
            ├── bonferroni_median_pval_bar_graph.png
            ├── fdr_count_bar_graph.png
            ├── fdr_mean_pval_bar_graph.png
            ├── fdr_median_pval_bar_graph.png
            ├── uncorrected_count_bar_graph.png
            ├── uncorrected_mean_pval_bar_graph.png
            ├── uncorrected_median_pval_bar_graph.png
            ├── significance_category_comparison.png
            └── comprehensive_analysis_summary.csv
```

### Rolling Window Analysis Output
```
results/
└── granger_causality_results/
    └── {response_var}/
        └── rolling_window_analysis/
            ├── matrices/
            │   ├── pvalue_matrix_raw.csv
            │   ├── pvalue_matrix_fdr.csv
            │   ├── pvalue_matrix_bonferroni.csv
            │   ├── pvalue_heatmap_raw.png
            │   ├── pvalue_heatmap_fdr.png
            │   └── pvalue_heatmap_bonferroni.png
            ├── term_significance_across_windows.png
            ├── rolling_window_summary.txt
            └── rolling_window_analysis_report.txt
```

### Regional Significance Analysis Output
```
results/
└── granger_causality_results/
    ├── regional_significance_summary.csv
    ├── overall_statistics.csv
    └── regional_significance_report.txt
```

## Output Files Explained

### Granger Causality Results
- **Text files**: Detailed statistical results for each lag
- **PNG files**: Bar charts showing term significance with color coding

### Summary Files
- **Summary text**: Comprehensive list of all significant terms organized by significance type

### Time Series Plots
- **Dual-axis plots**: Show lagged predictors vs. current response variable
- **Organized by term**: Each significant term gets its own folder

### Comprehensive Analysis
- **9 Bar Graphs**: Count, mean p-value, and median p-value for each significance type
- **Comparison Plot**: Overview of significant terms by category
- **CSV Summary**: Detailed statistics table for further analysis

### Rolling Window Analysis Files
- **P-value Matrices**: CSV files with p-values for each term and time window
- **Heatmap Visualizations**: Color-coded heatmaps showing significance patterns over time
- **Term Significance Plots**: Line plots showing how term significance changes across windows
- **Summary Reports**: Text files with detailed analysis results

### Regional Significance Analysis Files
- **Overall Statistics CSV**: Summary statistics across all regions with both proportions and fractions
- **Regional Summary CSV**: Detailed results for each individual region
- **Comprehensive Report**: Text report with clear metric explanations and statistical summaries

## Visualization Features

### Granger Causality Plots
- Color-coded bars for different significance levels
- Dynamic figure sizing based on number of terms
- Clear legends and labels
- High-resolution output (300 DPI)

### Time Series Plots
- Dual y-axis for easy comparison
- Highlighted significant lags
- Professional formatting
- Organized by term for easy navigation

### Comprehensive Analysis Charts
- 9 different bar graphs for complete analysis
- Statistical comparison plots
- Publication-ready quality
- Detailed value labels

## Example Results

### Standard Analysis Results
For a typical analysis with max lag = 3:
- **Overall Granger causality**: F = 2.6456, p < 0.001 (highly significant)
- **R² improvement**: 0.0340 (3.4% improvement in prediction)
- **FDR-significant terms**: 3 terms including "symptoms of flu" and "how long does flu last"
- **Bonferroni-significant terms**: 1 term with very strong evidence
- **Uncorrected significant terms**: 15 terms (may include false positives)

### Regional Significance Analysis Results
For analysis across 50 US states with 63 unique search terms:
- **RAW (Uncorrected)**: 98.4% (62/63) of unique terms significant in at least one region
- **FDR Corrected**: 82.5% (52/63) of unique terms significant in at least one region
- **Bonferroni Corrected**: 69.8% (44/63) of unique terms significant in at least one region
- **Pooled significance rates**: 92.7% (RAW), 56.0% (FDR), 26.7% (Bonferroni)
- **Regional variation**: High variation across regions, with some showing 100% significance and others showing 0%

### Rolling Window Analysis Results
- **Time-varying patterns**: Shows how Granger causality changes over time
- **Window-by-window significance**: Tracks which terms are significant in each time window
- **Trend analysis**: Identifies periods of high and low predictive power
- **Matrix outputs**: P-value matrices for detailed statistical analysis

## Customization

### Changing Response Variable
1. Modify `response_var` in `confs.py`
2. Update `file_name` to point to your data file
3. Ensure the response variable column exists in your data
4. Run the analysis

### Batch Processing Multiple Files
1. Update the file list in `analyze_multiple_data_files.py`
2. Run the script to process all files automatically
3. Each file gets its own organized output folder

### Adjusting Statistical Parameters
1. Modify significance thresholds in `confs.py`
2. Change `max_lags_to_test` for different lag analysis
3. Adjust `max_terms` to limit the number of predictors
4. Update data quality filters as needed

## Statistical Methods

### Granger Causality Test
- **F-statistic**: Tests if predictors collectively improve prediction
- **p-value < α**: Suggests predictors collectively Granger-cause response variable
- **R² improvement**: Shows how much prediction improves with predictors

### Multiple Testing Corrections
- **Uncorrected**: Raw p-values (may have false positives)
- **Bonferroni**: Conservative correction (divides α by number of tests)
- **FDR (Benjamini-Hochberg)**: Controls proportion of false discoveries

### Data Quality Assessment
- **Constant columns**: Automatically filtered out
- **Low variance columns**: Filtered based on standard deviation threshold
- **Missing data**: Handled with appropriate warnings and diagnostics

## Troubleshooting

### Common Issues
1. **Data file not found**: Check `file_name` in `confs.py` and ensure file exists
2. **Response variable not found**: Verify `response_var` matches column name in data
3. **No complete cases**: Data may have too many missing values
4. **Memory errors**: Reduce `max_terms` or `max_lags_to_test` in `confs.py`

### Data Quality Issues
1. **Constant columns**: Automatically filtered out with warning
2. **Low variance columns**: Filtered based on `low_variance_threshold`
3. **Missing data**: Handled automatically with appropriate warnings
4. **Date parsing errors**: Ensure dates are in YYYY-MM-DD format

### Performance Optimization
1. **Large datasets**: Consider reducing `max_terms` or `max_lags_to_test`
2. **Memory usage**: Monitor memory when processing many terms or lags
3. **Computation time**: Higher lags and more terms increase analysis time
4. **Batch processing**: Use `analyze_multiple_data_files.py` for efficiency

## Advanced Usage

### Custom Analysis Workflows
1. **Single lag analysis**: Set `max_lags_to_test = 1`
2. **Extended lag analysis**: Increase `max_lags_to_test` for longer-term effects
3. **Subset analysis**: Use `max_terms` to focus on specific predictors
4. **Date filtering**: Modify time series analysis for specific time periods

### Integration with Other Tools
1. **R integration**: Export CSV files for further analysis in R
2. **Tableau/Power BI**: Use CSV exports for dashboard creation
3. **Publication**: High-resolution PNG files ready for papers
4. **Reproducible research**: All parameters saved in configuration

## Dependencies

The system requires the following Python packages:
- `numpy>=1.21.0` - Numerical computations
- `pandas>=1.3.0` - Data manipulation
- `statsmodels>=0.13.0` - Statistical modeling
- `scipy>=1.7.0` - Scientific computing
- `matplotlib>=3.5.0` - Visualization

## Contributing

To extend the system:
1. **New statistical tests**: Add functions to existing scripts
2. **New visualizations**: Create additional plotting functions
3. **New data sources**: Update data loading functions
4. **New output formats**: Add export capabilities

## Citation and References

This system implements:
- Granger causality testing via multiple linear regression
- Bonferroni correction for multiple testing
- Benjamini-Hochberg FDR correction
- Comprehensive diagnostic analysis
- Automated data quality assessment
- Professional visualization and reporting

## Recent Updates and Improvements

### Version 2.0 Features
- **Rolling Window Analysis**: Added time-varying Granger causality analysis capabilities
- **Regional Significance Analysis**: Cross-region statistical comparisons and reporting
- **Enhanced Statistics**: Both decimal proportions and exact fractions (e.g., "62/63")
- **Clear Metric Explanations**: Distinguishes between overall proportions and pooled statistics
- **Improved Reporting**: Professional text reports with comprehensive explanations
- **Refactored Code**: Better code organization and maintainability
- **Advanced Visualizations**: Heatmaps, time series plots, and trend analysis

### Key Improvements
- **Fixed Data Type Issues**: Resolved string/float comparison errors in matrix processing
- **Enhanced Error Handling**: Better error messages and debugging capabilities
- **Improved Documentation**: Clear explanations of all metrics and calculations
- **Professional Output**: Publication-ready reports and visualizations
- **Batch Processing**: Automated analysis across multiple regions and time periods

## Support

For issues or questions:
1. Check the configuration in `confs.py`
2. Verify data file formats and structure
3. Review error messages and warnings
4. Check output directory permissions
5. Ensure all dependencies are installed correctly
6. Check the `REFACTORING_SUMMARY.md` for recent changes

---

This system provides everything you need for comprehensive Granger causality analysis with professional-quality outputs, including advanced time-varying and cross-regional analysis capabilities.