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
OUT_PNG = os.path.join(OUT_DIR, "pvalue_heatmap_50x20.png")
OUT_MIN_CSV = os.path.join(OUT_DIR, "pvalue_minima_50x20.csv")
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
WRAP_WIDTH = 10              # wrap long keywords in cells
CELL_FONTSIZE = 6            # cell text size
DPI = 300                    # figure resolution

# -----------------------------
# Helpers
# -----------------------------

# Removed discover_common_keywords - no longer needed since each state uses its own keywords

def _csv_path_for_state(root_dir: str, state: str) -> str:
    return os.path.join(
        root_dir,
        state,
        "rolling_window_analysis",
        "matrices",
        "pvalue_matrix_raw.csv",
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
        csv_path = _csv_path_for_state(root_dir, state)
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
            print(f"[DEBUG] {state}: Using top {min(len(sorted_keywords), num_keywords)} keywords")
            print(f"  Top 5: {[kw for kw, _ in sorted_keywords[:5]]}")

    return df_min, df_label

def _make_colormap(max_val: float, p_sig: float) -> Tuple[LinearSegmentedColormap, Normalize]:
    """
    Build a piecewise linear colormap:
      - [0, p_sig] is solid red
      - (p_sig, max_val] transitions linearly from red to blue
    """
    # Guard: ensure max_val > p_sig to allow a gradient
    eps = 1e-9
    if not np.isfinite(max_val) or max_val <= p_sig + eps:
        max_val = p_sig + 1e-3

    norm = Normalize(vmin=0.0, vmax=max_val)
    # Normalized position of the threshold
    tpos = p_sig / max_val
    # Repeat red at tpos to create a 'flat' red region up to p_sig
    cmap = LinearSegmentedColormap.from_list(
        "red_to_blue_plateau",
        [
            (0.0, "#d7301f"),    # deep red at 0
            (tpos, "#d7301f"),   # keep red through the threshold
            (1.0, "#2166ac"),    # blue at the observed max
        ],
    )
    # Make NaNs a light gray
    cmap.set_bad(color="#eeeeee")
    return cmap, norm

def _auto_ticks(max_val: float, p_sig: float) -> List[float]:
    """
    Produce a small, readable set of colorbar ticks, always including the threshold.
    """
    candidates = [0.0, p_sig, 0.1, 0.2, 0.3, 0.5, max_val]
    ticks = sorted(set([round(x, 3) for x in candidates if 0.0 <= x <= max_val + 1e-9]))
    # Deduplicate tightly (e.g., when max_val ~ 0.2)
    deduped = []
    for t in ticks:
        if not deduped or abs(t - deduped[-1]) > 1e-3:
            deduped.append(t)
    return deduped

def plot_heatmap(df_min: pd.DataFrame,
                 df_label: pd.DataFrame,
                 out_png: str,
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

    # Size heuristics: scale with matrix shape
    n_rows, n_cols = df_min.shape
    fig_w = max(10, n_cols * 0.5)    # ~10–12 inches wide minimum
    fig_h = max(12, n_rows * 0.28)   # ~14 inches tall minimum for 50 rows

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
    ax.set_title(
        f"P-Value Heatmap (min over rolling 3-year windows)\n"
        f"Red: p < {p_sig:.2f}; gradient from {p_sig:.2f} to observed max ({max_val:.3f})",
        fontsize=12,
        pad=10,
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    ticks = _auto_ticks(max_val, p_sig)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])
    cbar.set_label("Minimum p-value across 3-year windows", rotation=90)

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
                txt = textwrap.fill(kw_text.strip(), width=wrap_width)
                ax.text(j, i, txt, ha="center", va="center", fontsize=cell_fontsize)

    fig.savefig(out_png)
    plt.close(fig)

# -----------------------------
# Main
# -----------------------------

def main():
    # Each state will use its own top N most significant keywords (lowest p-values)
    num_keywords = 20  # Can be adjusted as needed
    
    print(f"[INFO] Creating heatmap with top {num_keywords} keywords per state")
    print("[INFO] Each state will show its own most significant keywords")

    # Build the state x keyword table where each state gets its own top keywords
    df_min, df_label = _collect_minima_and_labels(
        root_dir=ROOT_DIR,
        states=US_STATES,
        num_keywords=num_keywords
    )

    # Save CSVs for auditing and SI materials
    os.makedirs(OUT_DIR, exist_ok=True)
    df_min.to_csv(OUT_MIN_CSV, float_format="%.6f")
    df_label.to_csv(OUT_LABELS_CSV)

    # Plot & save the figure
    plot_heatmap(
        df_min=df_min,
        df_label=df_label,
        out_png=OUT_PNG,
        p_sig=P_SIG,
        annotate_keywords=ANNOTATE_KEYWORDS,
        annotate_significant_only=ANNOTATE_SIGNIFICANT_ONLY,
        wrap_width=WRAP_WIDTH,
        cell_fontsize=CELL_FONTSIZE,
        dpi=DPI
    )

    print(f"[OK] Saved heatmap: {OUT_PNG}")
    print(f"[OK] Saved minima CSV: {OUT_MIN_CSV}")
    print(f"[OK] Saved labels CSV: {OUT_LABELS_CSV}")

if __name__ == "__main__":
    main()
    