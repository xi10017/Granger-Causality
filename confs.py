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
comprehensive_analysis_prefix = "comprehensive_analysis"

# Multiple data files to analyze
data_files_to_analyze = [
    "Alabama_2010_2020.csv",
    "Alaska_2010_2020.csv",
    "Arizona_2010_2020.csv",
    "Arkansas_2010_2020.csv",
    "California_2010_2020.csv",
    "Colorado_2010_2020.csv",
    "Connecticut_2010_2020.csv",
    "Delaware_2010_2020.csv",
    "Florida_2010_2020.csv",
    "Georgia_2010_2020.csv",
    "Hawaii_2010_2020.csv",
    "Idaho_2010_2020.csv",
    "Illinois_2010_2020.csv",
    "Indiana_2010_2020.csv",
    "Iowa_2010_2020.csv",
    "Kansas_2010_2020.csv",
    "Kentucky_2010_2020.csv",
    "Louisiana_2010_2020.csv",                      
    "Maine_2010_2020.csv",
    "Maryland_2010_2020.csv",
    "Massachusetts_2010_2020.csv",      
    "Michigan_2010_2020.csv",
    "Minnesota_2010_2020.csv",
    "Mississippi_2010_2020.csv",
    "Missouri_2010_2020.csv",
    "Montana_2010_2020.csv",       
    "Nebraska_2010_2020.csv",
    "Nevada_2010_2020.csv",
    "New Hampshire_2010_2020.csv",
    "New Jersey_2010_2020.csv",
    "New Mexico_2010_2020.csv",
    "New York_2010_2020.csv",
    "North Carolina_2010_2020.csv",
    "North Dakota_2010_2020.csv",
    "Ohio_2010_2020.csv", 
    "Oklahoma_2010_2020.csv",
    "Oregon_2010_2020.csv",
    "Pennsylvania_2010_2020.csv",
    "Rhode Island_2010_2020.csv",   
    "South Carolina_2010_2020.csv",     
    "South Dakota_2010_2020.csv",
    "Tennessee_2010_2020.csv",
    "Texas_2010_2020.csv",
    "US_2010_2020.csv",
    "Utah_2010_2020.csv",
    "Vermont_2010_2020.csv",
    "Virginia_2010_2020.csv",
    "Washington_2010_2020.csv",
    "West Virginia_2010_2020.csv",
    "Wisconsin_2010_2020.csv",              
    "Wyoming_2010_2020.csv",
]