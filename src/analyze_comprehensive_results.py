#!/usr/bin/env python3
"""
Script to analyze comprehensive significant terms analysis results across all response variables
and generate bar graphs for individual terms across different significance categories.

This script works with any response variable (states, regions, etc.) and creates comprehensive
analysis visualizations for each response variable individually.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import glob
from confs import *

def parse_response_var_summary_file(filepath):
    """
    Parse a single response variable summary file and extract significant terms data.
    """
    response_var_name = os.path.basename(filepath).replace('summary_data_', '').replace('.txt', '')
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse the content to extract significant terms
    lines = content.split('\n')
    current_term = None
    term_data = defaultdict(list)
    
    for line in lines:
        original_line = line
        line = line.strip()
        
        # Check for term header (e.g., "how long does flu last significant combinations:")
        if ' significant combinations:' in line:
            current_term = line.split(' significant combinations:')[0].strip()
        elif line and current_term and original_line.startswith('  ') and ':' in line and 'p=' in line:
            # Parse: "  ExampleState_maxlag2_lag1: p=0.004558 (FDR)" (new format)
            # or: "  ExampleState_lag2: p=0.004558 (FDR)" (old format)
            try:
                # Extract lag and p-value - handle both new and old formats
                lag_match = re.search(r'_lag(\d+)', line)  # This will match both formats
                pval_match = re.search(r'p=([\d.]+)', line)
                sig_match = re.search(r'\(([^)]+)\)', line)
                
                if lag_match and pval_match and sig_match:
                    lag = int(lag_match.group(1))
                    pval = float(pval_match.group(1))
                    significance = sig_match.group(1)
                    
                    term_data[current_term].append({
                        'lag': lag,
                        'p_value': pval,
                        'significance': significance,
                        'response_var': response_var_name
                    })
            except Exception as e:
                print(f"Error parsing line '{line}': {e}")
                continue
    
    return response_var_name, term_data

def collect_all_response_var_data(results_dir):
    """
    Collect significant terms data from all response variable summary files.
    """
    all_data = {}
    
    # Find all response variable summary files
    pattern = os.path.join(results_dir, "*/summary_data_*.txt")
    summary_files = glob.glob(pattern)
    
    print(f"Found {len(summary_files)} response variable summary files")
    
    for filepath in summary_files:
        try:
            response_var_name, term_data = parse_response_var_summary_file(filepath)
            all_data[response_var_name] = term_data
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
    
    return all_data

def aggregate_term_statistics(all_data):
    """
    Aggregate statistics for each term across all response variables and significance categories.
    """
    term_stats = defaultdict(lambda: {
        'Bonferroni': {'count': 0, 'p_values': []},
        'FDR': {'count': 0, 'p_values': []},
        'Uncorrected': {'count': 0, 'p_values': []}
    })
    
    for response_var_name, response_var_data in all_data.items():
        for term, term_combinations in response_var_data.items():
            for combo in term_combinations:
                significance = combo['significance']
                p_value = combo['p_value']
                
                if significance in term_stats[term]:
                    term_stats[term][significance]['count'] += 1
                    term_stats[term][significance]['p_values'].append(p_value)
    
    # Calculate summary statistics
    summary_data = {}
    for term, categories in term_stats.items():
        summary_data[term] = {}
        for category, stats in categories.items():
            if stats['p_values']:
                summary_data[term][category] = {
                    'count': stats['count'],
                    'mean_pval': np.mean(stats['p_values']),
                    'median_pval': np.median(stats['p_values']),
                    'min_pval': np.min(stats['p_values']),
                    'max_pval': np.max(stats['p_values'])
                }
            else:
                summary_data[term][category] = {
                    'count': 0,
                    'mean_pval': 0,
                    'median_pval': 0,
                    'min_pval': 0,
                    'max_pval': 0
                }
    
    return summary_data

def create_term_bar_graphs(summary_data, output_dir):
    """
    Create bar graphs for terms across different significance categories.
    """
    categories = ['Bonferroni', 'FDR', 'Uncorrected']
    metrics = ['count', 'mean_pval', 'median_pval']
    metric_names = ['Count', 'Mean P-value', 'Median P-value']
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create graphs for each category and metric
    for category in categories:
        for metric, metric_name in zip(metrics, metric_names):
            # Collect data for this category and metric
            term_data = []
            for term, categories_data in summary_data.items():
                if category in categories_data and categories_data[category]['count'] > 0:
                    term_data.append({
                        'term': term,
                        'value': categories_data[category][metric],
                        'count': categories_data[category]['count']
                    })
            
            if not term_data:
                # Create empty plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.text(0.5, 0.5, f'No {category} significant terms found', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{category} - {metric_name}', fontweight='bold', fontsize=14)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
            else:
                # Sort data by value
                if metric == 'count':
                    term_data.sort(key=lambda x: x['value'], reverse=True)
                else:
                    term_data.sort(key=lambda x: x['value'])
                
                # Create the plot
                terms = [item['term'] for item in term_data]
                values = [item['value'] for item in term_data]
                
                fig, ax = plt.subplots(figsize=(max(12, len(terms) * 0.4), 8))
                
                # Create bars with colors based on metric
                colors = ['#FF6B6B' if metric == 'count' else '#4ECDC4' if 'mean' in metric else '#45B7D1']
                bars = ax.bar(range(len(terms)), values, color=colors[0], alpha=0.7, edgecolor='black', linewidth=1)
                
                # Customize the plot
                ax.set_title(f'{category} - {metric_name}', fontweight='bold', fontsize=14, pad=20)
                ax.set_ylabel(metric_name, fontsize=12)
                ax.set_xlabel('Terms', fontsize=12)
                
                # Set x-axis labels
                ax.set_xticks(range(len(terms)))
                ax.set_xticklabels(terms, rotation=45, ha='right', fontsize=8)
                
                # Add grid
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for i, (bar, value) in enumerate(zip(bars, values)):
                    height = bar.get_height()
                    if metric == 'count':
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value}', ha='center', va='bottom', fontsize=8)
                    else:
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                               f'{value:.4f}', ha='center', va='bottom', fontsize=8)
            
            # Adjust layout and save
            plt.tight_layout()
            filename = f'{category.lower()}_{metric}_bar_graph.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Created: {filename} ({len(term_data)} terms)")

def create_summary_table(summary_data, output_dir):
    """
    Create a summary table with statistics for each term and category.
    """
    # Flatten the data for the table
    table_data = []
    for term, categories_data in summary_data.items():
        for category, stats in categories_data.items():
            if stats['count'] > 0:
                table_data.append({
                    'Term': term,
                    'Category': category,
                    'Count': stats['count'],
                    'Mean_Pvalue': stats['mean_pval'],
                    'Median_Pvalue': stats['median_pval'],
                    'Min_Pvalue': stats['min_pval'],
                    'Max_Pvalue': stats['max_pval']
                })
    
    if table_data:
        df = pd.DataFrame(table_data)
        df = df.sort_values(['Category', 'Count'], ascending=[True, False])
        
        # Save to CSV
        csv_file = os.path.join(output_dir, 'comprehensive_analysis_summary.csv')
        df.to_csv(csv_file, index=False)
        
        print(f"\nSummary table saved to: {csv_file}")
        print(f"Total entries: {len(df)}")
        
        # Print summary statistics
        print("\nSummary by Category:")
        for category in ['Bonferroni', 'FDR', 'Uncorrected']:
            cat_data = df[df['Category'] == category]
            if len(cat_data) > 0:
                print(f"\n{category}:")
                print(f"  Number of terms: {len(cat_data)}")
                print(f"  Total significant combinations: {cat_data['Count'].sum()}")
                print(f"  Mean count per term: {cat_data['Count'].mean():.2f}")
                print(f"  Mean p-value: {cat_data['Mean_Pvalue'].mean():.6f}")
    else:
        print("No significant terms found to create summary table.")

def create_response_var_comparison_plot(summary_data, output_dir):
    """
    Create a plot comparing the number of significant terms across significance categories.
    """
    # Count significant terms by category
    category_counts = {'Bonferroni': 0, 'FDR': 0, 'Uncorrected': 0}
    
    for term, categories_data in summary_data.items():
        for category, stats in categories_data.items():
            if stats['count'] > 0:
                category_counts[category] += 1
    
    # Create comparison plot
    categories = ['Bonferroni', 'FDR', 'Uncorrected']
    counts = [category_counts[cat] for cat in categories]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(categories, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7, edgecolor='black')
    
    ax.set_title('Significant Terms by Significance Category', fontweight='bold', fontsize=14)
    ax.set_ylabel('Number of Terms', fontsize=12)
    ax.set_xlabel('Significance Category', fontsize=12)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'significance_category_comparison.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created: significance_category_comparison.png")

def analyze_single_response_var(response_var_name, response_var_data, results_dir):
    """
    Create comprehensive analysis for a single response variable.
    """
    print(f"\n=== ANALYZING {response_var_name.upper()} ===")
    
    # Set up output directory for this response variable
    response_var_output_dir = os.path.join(results_dir, response_var_name, comprehensive_analysis_prefix)
    os.makedirs(response_var_output_dir, exist_ok=True)
    
    # Convert response variable data to summary format
    summary_data = {}
    for term, term_combinations in response_var_data.items():
        summary_data[term] = {}
        
        # Group by significance type
        for combo in term_combinations:
            significance = combo['significance']
            p_value = combo['p_value']
            
            if significance not in summary_data[term]:
                summary_data[term][significance] = {
                    'count': 0,
                    'p_values': []
                }
            
            summary_data[term][significance]['count'] += 1
            summary_data[term][significance]['p_values'].append(p_value)
        
        # Calculate statistics for each significance type
        for significance, stats in summary_data[term].items():
            if stats['p_values']:
                summary_data[term][significance] = {
                    'count': stats['count'],
                    'mean_pval': np.mean(stats['p_values']),
                    'median_pval': np.median(stats['p_values']),
                    'min_pval': np.min(stats['p_values']),
                    'max_pval': np.max(stats['p_values'])
                }
            else:
                summary_data[term][significance] = {
                    'count': 0,
                    'mean_pval': 0,
                    'median_pval': 0,
                    'min_pval': 0,
                    'max_pval': 0
                }
    
    print(f"Found {len(summary_data)} unique terms with significant results for {response_var_name}")
    
    # Create visualizations
    create_term_bar_graphs(summary_data, response_var_output_dir)
    create_summary_table(summary_data, response_var_output_dir)
    create_response_var_comparison_plot(summary_data, response_var_output_dir)
    
    print(f"Analysis complete for {response_var_name}")
    return response_var_output_dir

def main():
    """
    Main function to run the comprehensive analysis for each response variable.
    """
    print("=== COMPREHENSIVE GRANGER CAUSALITY RESULTS ANALYSIS ===")
    
    # Set up paths
    results_dir = os.path.join(result_dir, granger_causality_prefix)
    
    print(f"Results directory: {results_dir}")
    
    # Collect data from all response variables
    print("\nCollecting data from all response variable summary files...")
    all_data = collect_all_response_var_data(results_dir)
    
    if not all_data:
        print("No response variable data found!")
        return
    
    print(f"Collected data from {len(all_data)} response variables")
    
    # Process each response variable individually
    processed_response_vars = []
    for response_var_name, response_var_data in all_data.items():
        try:
            output_dir = analyze_single_response_var(response_var_name, response_var_data, results_dir)
            processed_response_vars.append((response_var_name, output_dir))
        except Exception as e:
            print(f"Error processing {response_var_name}: {e}")
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Processed {len(processed_response_vars)} response variables successfully")
    
    print("\nGenerated files for each response variable:")
    print("- bonferroni_count_bar_graph.png")
    print("- bonferroni_mean_pval_bar_graph.png") 
    print("- bonferroni_median_pval_bar_graph.png")
    print("- fdr_count_bar_graph.png")
    print("- fdr_mean_pval_bar_graph.png")
    print("- fdr_median_pval_bar_graph.png")
    print("- uncorrected_count_bar_graph.png")
    print("- uncorrected_mean_pval_bar_graph.png")
    print("- uncorrected_median_pval_bar_graph.png")
    print("- significance_category_comparison.png")
    print("- comprehensive_analysis_summary.csv")
    
    print(f"\nResults saved in each response variable's folder under: {comprehensive_analysis_prefix}/")
    for response_var_name, output_dir in processed_response_vars:
        print(f"  {response_var_name}: {output_dir}")

if __name__ == "__main__":
    main()