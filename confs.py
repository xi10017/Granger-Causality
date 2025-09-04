# Data configuration
data_dir = "data/"
result_dir = "results/"

# Analysis configuration
max_terms = None  # Maximum number of search terms to use (None = use all)
response_var = "Alabama"  # Response variable column name (must match column name in data file)
max_lags_to_test = 5  # Maximum number of lags to test (1 to max_lags_to_test)

# Data filtering thresholds
low_variance_threshold = 0.1  # Standard deviation threshold for filtering low variance columns
zero_ratio_threshold = 0.8  # Threshold for filtering columns with too many zeros

# Statistical significance thresholds
alpha_level = 0.05  # Significance level for hypothesis testing
bonferroni_alpha = 0.05  # Alpha level for Bonferroni correction
fdr_alpha = 0.05  # Alpha level for FDR correction

# Visualization settings
figure_dpi = 300
figure_bbox_inches = 'tight'

# Data file to analyze
file_name = "Alabama_2010_2020.csv"  # Specify which data file to analyze

# Output file naming
results_prefix = "granger_significant_terms_data"
visualization_prefix = "granger_pvalues_data"
summary_prefix = "summary_data"
time_series_prefix = "time_series_plots"
granger_causality_prefix = "granger_causality_results"