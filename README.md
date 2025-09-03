# Generalized Granger Causality Analysis

This is a generalized system for performing Granger causality analysis on individual state data using configuration files. The system analyzes how search trends predict state-level activity for a specific state.

## Overview

This system performs **Granger causality analysis** to determine whether search trend data can predict state-level activity. The analysis tests whether past values of search terms (X variables) can help predict current values of state activity (Y variable).

**Direction**: Search Terms → State Activity
- **X Variables**: Search term volumes (e.g., "flu symptoms", "how long does flu last")
- **Y Variable**: State activity level (e.g., Alabama activity)
- **Question**: Do search trends from X weeks ago predict current state activity?

## What This Analysis Shows

The Granger causality test determines if including lagged values of search terms improves the prediction of current state activity compared to using only past values of state activity. A significant result means:

- **Search terms Granger-cause state activity**
- Past search behavior provides useful information for predicting current state activity
- There is a temporal relationship where search trends precede changes in state activity

The system consists of:
- **`confs.py`**: Configuration file containing all analysis parameters
- **`granger_causality_generalized.py`**: Main analysis script
- **`requirements.txt`**: Python dependencies

## Key Features

- **Configuration-driven**: All parameters are controlled via `confs.py`
- **Individual state analysis**: Works with one state at a time (no data merging)
- **Multiple testing corrections**: Bonferroni and FDR corrections
- **Comprehensive output**: Results, visualizations, and detailed analysis
- **Generalized approach**: Can be adapted to different states and research questions

## Configuration (`confs.py`)

### Data Configuration
```python
data_dir = "data/"                    # Directory containing data files
result_dir = "results/"               # Directory for output files
```

### Analysis Configuration
```python
max_terms = None                      # Max search terms to use (None = use all)
target_state = 'Alabama'              # Specific state to analyze
max_lags_to_test = 5                 # Maximum number of lags to test
```

### Statistical Thresholds
```python
alpha_level = 0.05                    # Significance level for hypothesis testing
bonferroni_alpha = 0.05              # Alpha for Bonferroni correction
fdr_alpha = 0.05                     # Alpha for FDR correction
```

### File Patterns
```python
state_file_pattern = "*_2010_2020.csv"  # Pattern for state data files
exclude_files = ["US_2010_2020.csv"]    # Files to exclude from analysis
```

## Data Structure Requirements

### State Data Files
Each state file should have the following structure:
- **Column 1**: `date` in YYYY-MM-DD format
- **Column 2**: State name (e.g., "Alabama") - this is the target variable
- **Columns 3+**: Search term columns with numerical values

Example: `Alabama_2010_2020.csv`
```csv
date,Alabama,flu symptoms,the flu,rsv,flu how long,...
2010-10-09,2.13477,0.0,0.0,0.0,0.0,...
...
```

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Analysis
Edit `confs.py` to match your analysis requirements:
- Set `target_state` to the state you want to analyze
- Adjust `max_lags_to_test` as needed
- Modify file patterns if needed

### 3. Run Analysis
```bash
python granger_causality_generalized.py
```

## Analysis Process

1. **Data Loading**: Loads data for the specified state
2. **Data Diagnostics**: Identifies and filters problematic columns
3. **Lag Creation**: Creates lagged variables for both state activity and search terms
4. **Granger Causality Testing**: Tests if search terms collectively predict state activity
5. **Individual Term Analysis**: Analyzes significance of individual terms
6. **Multiple Testing Correction**: Applies Bonferroni and FDR corrections
7. **Visualization**: Creates bar charts showing term significance
8. **Results Export**: Saves comprehensive results to text files

## Output Files

### Results Directory Structure
```
results/
├── granger_significant_terms_state_data_Alabama_lag1.txt
├── granger_significant_terms_state_data_Alabama_lag2.txt
├── granger_significant_terms_state_data_Alabama_lag3.txt
├── granger_significant_terms_state_data_Alabama_lag4.txt
├── granger_significant_terms_state_data_Alabama_lag5.txt
├── granger_pvalues_state_data_Alabama_lag1.png
├── granger_pvalues_state_data_Alabama_lag2.png
├── granger_pvalues_state_data_Alabama_lag3.png
├── granger_pvalues_state_data_Alabama_lag4.png
└── granger_pvalues_state_data_Alabama_lag5.png
```

### Result File Contents
- Overall Granger causality test results
- Individual term significance analysis
- Multiple testing correction results
- Model fit statistics
- Detailed significance information

### Visualization Features
- Color-coded bars for different significance levels
- Dynamic figure sizing based on number of terms
- Clear legend and labels
- High-resolution output

## Example Results

For Alabama with max lag = 3:
- **Overall Granger causality**: F = 2.6456, p < 0.001 (highly significant)
- **R² improvement**: 0.0340 (3.4% improvement in prediction)
- **FDR-significant terms**: 3 terms including "symptoms of flu" and "how long does flu last"

## Customization

### Changing Target State
1. Modify `target_state` in `confs.py`
2. Ensure the corresponding state file exists in `data_dir`
3. Run the script again

### Adding New Analysis Types
1. Modify `confs.py` to add new configuration parameters
2. Update the main script to use new parameters
3. Adjust data loading and processing functions as needed

### Changing Statistical Methods
1. Modify significance thresholds in `confs.py`
2. Update correction methods in the analysis functions
3. Adjust visualization parameters as needed

## Interpretation

### Granger Causality Test
- **F-statistic**: Tests if search terms collectively improve prediction
- **p-value < α**: Suggests search terms collectively Granger-cause state activity
- **R² improvement**: Shows how much prediction improves with search terms

### Individual Term Significance
- **Uncorrected**: Raw p-values (may have false positives)
- **Bonferroni**: Conservative correction for multiple testing
- **FDR**: Less conservative correction controlling false discovery rate

### Multiple Testing Correction
- **Bonferroni**: Divides α by number of tests (very conservative)
- **FDR (Benjamini-Hochberg)**: Controls proportion of false discoveries

## Troubleshooting

### Common Issues
1. **State file not found**: Check `target_state` in `confs.py` and ensure file exists
2. **No complete cases**: Data may have too many missing values
3. **Memory errors**: Reduce `max_terms` or `max_lags_to_test` in `confs.py`

### Data Quality Issues
1. **Constant columns**: Automatically filtered out
2. **Low variance columns**: Filtered based on `low_variance_threshold`
3. **Missing data**: Handled automatically with appropriate warnings

## Performance Considerations

- **Large datasets**: Consider reducing `max_terms` or `max_lags_to_test`
- **Memory usage**: Monitor memory when processing many terms or lags
- **Computation time**: Higher lags and more terms increase analysis time

## Extending the System

### Adding New Statistical Tests
1. Create new functions in the main script
2. Add configuration parameters to `confs.py`
3. Integrate with existing analysis pipeline

### Supporting New Data Sources
1. Update data loading functions
2. Modify file pattern matching
3. Adjust data processing logic

### Custom Visualizations
1. Add new plotting functions
2. Update configuration parameters
3. Integrate with results saving system

## Citation and References

This system implements:
- Granger causality testing via multiple linear regression
- Bonferroni correction for multiple testing
- Benjamini-Hochberg FDR correction
- Comprehensive diagnostic analysis
- Automated data quality assessment

## Support

For issues or questions:
1. Check the configuration in `confs.py`
2. Verify data file formats and structure
3. Review error messages and warnings
4. Check output directory permissions
