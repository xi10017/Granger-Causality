# scripts/make_pvalue_heatmap.py
import os
import textwrap
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

# -----------------------------
# Configuration
# -----------------------------

# Root that contains state folders (each has rolling_window_analysis/matrices/pvalue_matrix_raw.csv)
ROOT_DIR = os.path.expanduser("~/Workspace/Granger-Causality/results/granger_causality_results")

# Where to save outputs
OUT_DIR = os.path.join(ROOT_DIR, "supporting_information")
OUT_PNG_RAW = os.path.join(OUT_DIR, "pvalue_heatmap_raw_50x20.png")
OUT_PNG_FDR = os.path.join(OUT_DIR, "pvalue_heatmap_fdr_50x20.png")
OUT_PNG_BONFERRONI = os.path.join(OUT_DIR, "pvalue_heatmap_bonferroni_50x20.png")
OUT_MIN_CSV_RAW = os.path.join(OUT_DIR, "pvalue_minima_raw_50x20.csv")
OUT_MIN_CSV_FDR = os.path.join(OUT_DIR, "pvalue_minima_fdr_50x20.csv")
OUT_MIN_CSV_BONFERRONI = os.path.join(OUT_DIR, "pvalue_minima_bonferroni_50x20.csv")
OUT_LABELS_CSV = os.path.join(OUT_DIR, "pvalue_keyword_labels_50x20.csv")

# The 50 U.S. states in alphabetical order (explicitly exclude aggregate "US")
US_STATES = [
    "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut",
    "Delaware","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa",
    "Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan",
    "Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada",
    "New Hampshire","New Jersey","New Mexico","New York","North Carolina",
    "North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island",
    "South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont",
    "Virginia","Washington","West Virginia","Wisconsin","Wyoming"
]

# Note: Each state will use its own top N keywords based on minimum p-values
# No need for a canonical keyword list since each state shows its own keywords
# The x-axis will be K1, K2, K3, etc. and each cell shows the actual keyword name

# Visualization knobs
P_SIG = 0.05                 # threshold for significance
ANNOTATE_KEYWORDS = True     # put keyword text inside each cell
ANNOTATE_SIGNIFICANT_ONLY = False  # set True if the full grid is too busy
WRAP_WIDTH = 10              # wrap long keywords in cells (will be calculated dynamically)
CELL_FONTSIZE = 6            # cell text size (will be calculated dynamically)
DPI = 300                    # figure resolution
MIN_FONT_SIZE = 6            # minimum font size (1.5x original)
MAX_FONT_SIZE = 12           # maximum font size (reasonable for smaller grid)

# Text fitting parameters - optimized to prevent overlapping
TEXT_FIT_RATIO = 0.7         # use 70% of cell height to prevent overlapping
TEXT_WIDTH_RATIO = 0.8       # use 80% of cell width to prevent overlapping
MAX_TEXT_LINES = 2           # limit to 2 lines per cell to prevent overlap

# -----------------------------
# Helpers
# -----------------------------

# Removed discover_common_keywords - no longer needed since each state uses its own keywords

def _csv_path_for_state(root_dir: str, state: str, correction_method: str = "raw") -> str:
    return os.path.join(
        root_dir,
        state,
        "rolling_window_analysis",
        "matrices",
        f"pvalue_matrix_{correction_method}.csv",
    )

def _read_state_df(csv_path: str) -> Optional[pd.DataFrame]:
    """
    Reads a state's p-value matrix. Rows are rolling 3-year windows; columns are keywords.
    Drops the 'Proportion_Significant' row if present. Returns None if file missing.
    """
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, index_col=0)
    # Drop any summary rows
    df = df[~df.index.fillna("").str.contains("Proportion", case=False, na=False)].copy()
    # Coerce to numeric (silently ignore any non-numeric remnants)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Strip whitespace in column names
    df.columns = [str(c).strip() for c in df.columns]
    return df

# Removed _map_keywords_for_state - no longer needed since each state uses its own keywords

def _collect_minima_and_labels(root_dir: str,
                               states: List[str],
                               correction_method: str = "raw",
                               num_keywords: int = 20
                              ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      df_min  : (states x num_keywords) float minima of p-values across all windows
      df_label: (states x num_keywords) string labels of the actual keyword used per cell
    Each state gets its own top N keywords based on minimum p-values.
    """
    # Create column names K1, K2, K3, etc.
    column_names = [f"K{i+1}" for i in range(num_keywords)]
    df_min = pd.DataFrame(index=states, columns=column_names, dtype=float)
    df_label = pd.DataFrame(index=states, columns=column_names, dtype=object)

    for state in states:
        csv_path = _csv_path_for_state(root_dir, state, correction_method)
        df = _read_state_df(csv_path)
        if df is None:
            print(f"[WARN] Missing file for {state}: {csv_path}")
            continue

        # Calculate minimum p-value for each keyword in this state
        keyword_mins = {}
        for col in df.columns:
            col_values = df[col].dropna()
            if len(col_values) > 0:
                keyword_mins[col] = col_values.min()
        
        # Sort keywords by their minimum p-value (most significant first)
        sorted_keywords = sorted(keyword_mins.items(), key=lambda x: x[1])
        
        # Take the top N keywords for this state
        for i, (keyword, min_pval) in enumerate(sorted_keywords[:num_keywords]):
            col_name = f"K{i+1}"
            df_min.loc[state, col_name] = min_pval
            df_label.loc[state, col_name] = keyword
        
        # Debug info for first few states
        if state in states[:3]:
            print(f"[DEBUG] {state} ({correction_method}): Using top {min(len(sorted_keywords), num_keywords)} keywords")
            print(f"  Top 5: {[kw for kw, _ in sorted_keywords[:5]]}")

    return df_min, df_label

def _make_colormap(max_val: float, p_sig: float) -> Tuple[LinearSegmentedColormap, Normalize]:
    """
    Build a piecewise linear colormap:
      - [0, p_sig] is solid red
      - (p_sig, 0.10] transitions linearly from red to blue
      - (0.10, max_val] is solid blue
    """
    # Guard: ensure max_val > 0.10 to allow the gradient
    eps = 1e-9
    if not np.isfinite(max_val) or max_val <= 0.10 + eps:
        max_val = 0.10 + 1e-3

    norm = Normalize(vmin=0.0, vmax=max_val)
    
    # Calculate normalized positions
    p_sig_pos = p_sig / max_val
    blue_start_pos = 0.10 / max_val
    
    # Create colormap with three regions
    cmap = LinearSegmentedColormap.from_list(
        "red_to_blue_gradient",
        [
            (0.0, "#d7301f"),           # deep red at 0
            (p_sig_pos, "#d7301f"),     # keep red through p_sig
            (blue_start_pos, "#2166ac"), # transition to blue at 0.10
            (1.0, "#2166ac"),           # solid blue from 0.10 onwards
        ],
    )
    # Make NaNs a light gray
    cmap.set_bad(color="#eeeeee")
    return cmap, norm

def _auto_ticks(max_val: float, p_sig: float) -> List[float]:
    """
    Produce a small, readable set of colorbar ticks, including p_sig and 0.10 thresholds.
    """
    candidates = [0.0, p_sig, 0.10, 0.2, 0.3, 0.5, max_val]
    ticks = sorted(set([round(x, 3) for x in candidates if 0.0 <= x <= max_val + 1e-9]))
    # Deduplicate tightly (e.g., when max_val ~ 0.2)
    deduped = []
    for t in ticks:
        if not deduped or abs(t - deduped[-1]) > 1e-3:
            deduped.append(t)
    return deduped

def _calculate_dynamic_text_params(fig, ax, n_rows: int, n_cols: int, 
                                 min_font_size: int = MIN_FONT_SIZE, 
                                 max_font_size: int = MAX_FONT_SIZE) -> Tuple[int, int]:
    """
    Calculate dynamic font size and wrap width based on cell dimensions.
    Returns (font_size, wrap_width)
    """
    # Get figure dimensions in inches
    fig_width, fig_height = fig.get_size_inches()
    
    # Calculate cell dimensions in inches
    cell_width = fig_width / n_cols
    cell_height = fig_height / n_rows
    
    # Calculate font size based on cell height - optimized for maximum readability
    # Use configurable ratio for text fitting
    target_text_height = cell_height * TEXT_FIT_RATIO
    
    # Convert to font size (approximate: 1 inch = 72 points, but we need to account for DPI)
    # Font size calculation: target_height * DPI / 72
    # Add a conservative multiplier to prevent overlapping
    calculated_font_size = int(target_text_height * DPI / 72 * 1.2)  # 20% larger than original, conservative
    
    # Clamp to our min/max range
    font_size = max(min_font_size, min(max_font_size, calculated_font_size))
    
    # Calculate wrap width based on cell width
    # Estimate characters per inch (rough approximation: 6-8 chars per inch for small fonts)
    chars_per_inch = 6 if font_size <= 6 else 8
    target_chars_per_line = int(cell_width * chars_per_inch * TEXT_WIDTH_RATIO)  # Use configurable ratio
    wrap_width = max(8, min(20, target_chars_per_line))  # Reasonable range
    
    return font_size, wrap_width

def plot_heatmap(df_min: pd.DataFrame,
                 df_label: pd.DataFrame,
                 out_png: str,
                 correction_method: str = "raw",
                 p_sig: float = P_SIG,
                 annotate_keywords: bool = ANNOTATE_KEYWORDS,
                 annotate_significant_only: bool = ANNOTATE_SIGNIFICANT_ONLY,
                 wrap_width: int = WRAP_WIDTH,
                 cell_fontsize: int = CELL_FONTSIZE,
                 dpi: int = DPI) -> None:
    """
    Renders and saves the 50x20 heatmap with keyword text inside cells.
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    data = df_min.values.astype(float)
    if np.all(np.isnan(data)):
        raise ValueError("All values are NaN; nothing to plot.")

    max_val = float(np.nanmax(data))
    cmap, norm = _make_colormap(max_val, p_sig)

    # Size heuristics: scale with matrix shape - optimized for large text
    n_rows, n_cols = df_min.shape
    fig_w = max(12, n_cols * 0.5)    # Back to original grid size
    fig_h = max(16, n_rows * 0.35)   # Back to original grid size

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi, constrained_layout=True)
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    # y ticks = state names
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(df_min.index)

    # x ticks = simple "K1..K20" (since keywords are printed inside cells)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([f"K{j+1}" for j in range(n_cols)], rotation=0)

    # Gridlines
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Title and labels
    correction_title = correction_method.upper() if correction_method != "raw" else "RAW"
    ax.set_title(
        f"P-Value Heatmap - {correction_title} Corrected (min over rolling 3-year windows)\n"
        f"Red: p < {p_sig:.2f}; gradient from {p_sig:.2f} to 0.10; Blue: p ≥ 0.10",
        fontsize=12,
        pad=10,
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    ticks = _auto_ticks(max_val, p_sig)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])
    cbar.set_label("Minimum p-value across 3-year windows", rotation=90)

    # Calculate dynamic text parameters based on cell size
    dynamic_font_size, dynamic_wrap_width = _calculate_dynamic_text_params(
        fig, ax, n_rows, n_cols, MIN_FONT_SIZE, MAX_FONT_SIZE
    )
    
    # Use the dynamic parameters (prioritize calculated values for maximum readability)
    final_font_size = dynamic_font_size
    final_wrap_width = dynamic_wrap_width
    
    # Debug information
    print(f"[DEBUG] Dynamic text parameters:")
    print(f"  Calculated font size: {dynamic_font_size}, Final: {final_font_size}")
    print(f"  Calculated wrap width: {dynamic_wrap_width}, Final: {final_wrap_width}")
    print(f"  Cell dimensions: {fig.get_size_inches()[0]/n_cols:.2f}\" x {fig.get_size_inches()[1]/n_rows:.2f}\"")
    
    # Cell annotations (keyword text only; no numbers)
    if annotate_keywords:
        for i in range(n_rows):
            for j in range(n_cols):
                val = df_min.iat[i, j]
                if np.isnan(val):
                    continue
                if annotate_significant_only and (val >= p_sig):
                    continue
                kw_text = df_label.iat[i, j]
                if not isinstance(kw_text, str) or not kw_text.strip():
                    continue
                
                # Use dynamic wrapping and font size
                txt = textwrap.fill(kw_text.strip(), width=final_wrap_width)
                
                # If text is still too long after wrapping, truncate it
                if len(txt) > final_wrap_width * MAX_TEXT_LINES:  # Use configurable max lines
                    # Take first part and add ellipsis
                    truncated = kw_text.strip()[:final_wrap_width*2] + "..."
                    txt = textwrap.fill(truncated, width=final_wrap_width)
                
                ax.text(j, i, txt, ha="center", va="center", fontsize=final_font_size)

    fig.savefig(out_png)
    plt.close(fig)

# -----------------------------
# Main
# -----------------------------

def main():
    # Each state will use its own top N most significant keywords (lowest p-values)
    num_keywords = 20  # Can be adjusted as needed
    
    print(f"[INFO] Creating heatmaps with top {num_keywords} keywords per state")
    print("[INFO] Each state will show its own most significant keywords")
    print("[INFO] Generating heatmaps for: RAW, FDR, and BONFERRONI corrected p-values")

    # Create output directory
    os.makedirs(OUT_DIR, exist_ok=True)

    # Generate heatmaps and CSV files for each correction method
    correction_methods = ["raw", "fdr", "bonferroni"]
    output_files = [OUT_PNG_RAW, OUT_PNG_FDR, OUT_PNG_BONFERRONI]
    csv_files = [OUT_MIN_CSV_RAW, OUT_MIN_CSV_FDR, OUT_MIN_CSV_BONFERRONI]
    
    for correction_method, output_file, csv_file in zip(correction_methods, output_files, csv_files):
        print(f"\n[INFO] Processing {correction_method.upper()} corrected p-values...")
        
        # Build the state x keyword table for this correction method
        df_min, df_label = _collect_minima_and_labels(
            root_dir=ROOT_DIR,
            states=US_STATES,
            correction_method=correction_method,
            num_keywords=num_keywords
        )

        # Plot & save the figure
        plot_heatmap(
            df_min=df_min,
            df_label=df_label,
            out_png=output_file,
            correction_method=correction_method,
            p_sig=P_SIG,
            annotate_keywords=ANNOTATE_KEYWORDS,
            annotate_significant_only=ANNOTATE_SIGNIFICANT_ONLY,
            wrap_width=WRAP_WIDTH,
            cell_fontsize=CELL_FONTSIZE,
            dpi=DPI
        )

        # Save CSV file for this correction method
        df_min.to_csv(csv_file, float_format="%.6f")
        
        print(f"[OK] Saved {correction_method.upper()} heatmap: {output_file}")
        print(f"[OK] Saved {correction_method.upper()} minima CSV: {csv_file}")

    # Save labels CSV (same for all correction methods, using raw data)
    print(f"\n[INFO] Saving labels CSV...")
    df_min_raw, df_label_raw = _collect_minima_and_labels(
        root_dir=ROOT_DIR,
        states=US_STATES,
        correction_method="raw",
        num_keywords=num_keywords
    )
    df_label_raw.to_csv(OUT_LABELS_CSV)
    
    print(f"[OK] Saved labels CSV: {OUT_LABELS_CSV}")
    print(f"\n[INFO] All heatmaps and CSV files generated successfully!")

if __name__ == "__main__":
    main()
    