import os
import glob
import re
from confs import *

def parse_results_file(filepath):
    """Parse a Granger causality results file and extract significant terms"""
    significant_terms = []
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract response variable name and max lag from filename
        filename = os.path.basename(filepath)
        # Pattern: granger_significant_terms_data_{response_var}_lag{max_lag}.txt
        response_match = re.search(r'granger_significant_terms_data_(\w+)_lag(\d+)', filename)
        if not response_match:
            return []
        
        response_var_name = response_match.group(1)
        max_lag = response_match.group(2)
        
        # Look for significant terms in the results
        lines = content.split('\n')
        in_significant_section = False
        
        for line in lines:
            line = line.strip()
            
            # Start of significant terms section
            if line.startswith('=== ALL SIGNIFICANT TERMS'):
                in_significant_section = True
                continue
            
            # End of significant terms section
            if in_significant_section and line.startswith('==='):
                in_significant_section = False
                continue
            
            # Skip header lines
            if in_significant_section and line.startswith('Term\t'):
                continue
            
            # Parse significant term lines
            if in_significant_section and '\t' in line and not line.startswith('==='):
                parts = line.split('\t')
                if len(parts) >= 3:
                    term = parts[0].strip()
                    p_value = float(parts[1].strip())
                    significance_type = parts[2].strip()
                    
                    significant_terms.append({
                        'term': term,
                        'p_value': p_value,
                        'significance_type': significance_type,
                        'response_var': response_var_name,
                        'max_lag': max_lag
                    })
        
        return significant_terms
    
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return []

def create_comprehensive_significant_terms_summary():
    """Create a comprehensive summary of all significant terms from Granger causality analyses"""
    print("=== CREATING COMPREHENSIVE SIGNIFICANT TERMS SUMMARY ===")
    
    # Find all results files
    results_pattern = os.path.join(result_dir, f"{results_prefix}_*_lag*.txt")
    detailed_files = glob.glob(results_pattern)
    
    if not detailed_files:
        print(f"No results files found in {result_dir}!")
        print(f"Looking for pattern: {results_pattern}")
        return
    
    print(f"Found {len(detailed_files)} results files to process")
    
    # Parse all files and collect significant terms
    all_significant_terms = []
    for filepath in detailed_files:
        print(f"Processing: {os.path.basename(filepath)}")
        terms = parse_results_file(filepath)
        all_significant_terms.extend(terms)
    
    if not all_significant_terms:
        print("No significant terms found in any files!")
        return
    
    # Group terms by term name
    terms_dict = {}
    for item in all_significant_terms:
        term = item['term']
        if term not in terms_dict:
            terms_dict[term] = []
        terms_dict[term].append(item)
    
    # Create output file using configuration
    output_filename = os.path.join(result_dir, f"{summary_prefix}_{response_var}.txt")
    
    with open(output_filename, 'w') as f:
        f.write("=== GRANGER CAUSALITY SIGNIFICANT TERMS SUMMARY ===\n")
        f.write(f"Data file analyzed: {file_name}\n")
        f.write(f"Response variable: {response_var}\n")
        f.write(f"Total significant term combinations: {len(all_significant_terms)}\n")
        f.write(f"Unique terms: {len(terms_dict)}\n")
        f.write(f"Response variables analyzed: {', '.join(set([item['response_var'] for item in all_significant_terms]))}\n")
        f.write(f"Max lags tested: {', '.join(set([item['max_lag'] for item in all_significant_terms]))}\n\n")
        
        # Sort terms by minimum p-value
        sorted_terms = sorted(terms_dict.items(), 
                            key=lambda x: min([item['p_value'] for item in x[1]]))
        
        for term, term_data in sorted_terms:
            f.write(f"{term} significant combinations:\n")
            
            # Sort by response variable, then by max_lag, then by p_value
            sorted_data = sorted(term_data, key=lambda x: (x['response_var'], int(x['max_lag']), x['p_value']))
            
            for item in sorted_data:
                f.write(f"  {item['response_var']}_lag{item['max_lag']}: p={item['p_value']:.6f} ({item['significance_type']})\n")
            
            f.write("\n")
    
    print(f"Comprehensive summary saved to {output_filename}")
    return output_filename

if __name__ == "__main__":
    create_comprehensive_significant_terms_summary()