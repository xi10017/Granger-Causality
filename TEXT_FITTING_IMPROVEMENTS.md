# Text Fitting Improvements for Heatmap

## Problem Solved
The original heatmap had text overflowing outside the grid cell boundaries, making it difficult to read and unprofessional looking.

## Solutions Implemented

### 1. Dynamic Text Sizing
- **Function**: `_calculate_dynamic_text_params()`
- **Purpose**: Calculates optimal font size and wrap width based on actual cell dimensions
- **Method**: Uses cell height and width to determine appropriate text parameters

### 2. Configurable Parameters
Added new configuration options for fine-tuning:

```python
# Text fitting parameters
TEXT_FIT_RATIO = 0.6         # fraction of cell height to use for text
TEXT_WIDTH_RATIO = 0.8       # fraction of cell width to use for text
MAX_TEXT_LINES = 3           # maximum number of text lines per cell
MIN_FONT_SIZE = 4            # minimum font size to ensure readability
MAX_FONT_SIZE = 10           # maximum font size to prevent overflow
```

### 3. Improved Figure Sizing
- **Before**: `fig_w = max(10, n_cols * 0.5)`, `fig_h = max(12, n_rows * 0.28)`
- **After**: `fig_w = max(12, n_cols * 0.6)`, `fig_h = max(16, n_rows * 0.35)`
- **Result**: Larger cells provide more space for text

### 4. Text Truncation Fallback
- **Feature**: Automatic truncation of overly long keywords
- **Method**: If text exceeds `MAX_TEXT_LINES` after wrapping, truncate and add "..."
- **Benefit**: Prevents text overflow while maintaining readability

### 5. Debug Information
Added debug output to monitor text fitting:
```
[DEBUG] Dynamic text parameters:
  Calculated font size: 4, Final: 4
  Calculated wrap width: 8, Final: 8
  Cell dimensions: 0.60" x 0.35"
```

## Results

### Before Improvements
- Text often overflowed cell boundaries
- Fixed font size (6) and wrap width (10) regardless of cell size
- Smaller figure dimensions led to cramped cells

### After Improvements
- **Cell dimensions**: 0.60" x 0.35" (increased from 0.50" x 0.28")
- **Font size**: 4 (calculated to fit within cells)
- **Wrap width**: 8 (calculated to fit within cell width)
- **Text truncation**: Long keywords are truncated with "..." if needed

## Technical Details

### Dynamic Calculation Algorithm
1. **Cell Size Calculation**: `cell_width = fig_width / n_cols`, `cell_height = fig_height / n_rows`
2. **Font Size**: `target_text_height = cell_height * TEXT_FIT_RATIO`
3. **Wrap Width**: `target_chars_per_line = cell_width * chars_per_inch * TEXT_WIDTH_RATIO`
4. **Bounds Checking**: Font size clamped between `MIN_FONT_SIZE` and `MAX_FONT_SIZE`

### Text Processing Pipeline
1. **Wrapping**: `textwrap.fill(keyword, width=final_wrap_width)`
2. **Length Check**: If wrapped text > `final_wrap_width * MAX_TEXT_LINES`
3. **Truncation**: Take first part + "..." and re-wrap
4. **Rendering**: `ax.text()` with calculated font size

## Customization Options

### Adjust Text Fitting
```python
TEXT_FIT_RATIO = 0.7        # Use more of cell height (default: 0.6)
TEXT_WIDTH_RATIO = 0.9      # Use more of cell width (default: 0.8)
```

### Adjust Font Limits
```python
MIN_FONT_SIZE = 3           # Allow smaller text (default: 4)
MAX_FONT_SIZE = 12          # Allow larger text (default: 10)
```

### Adjust Figure Size
```python
fig_w = max(15, n_cols * 0.7)    # Even larger cells
fig_h = max(20, n_rows * 0.4)    # Even taller cells
```

## Files Modified
- `plot.py` - Main script with dynamic text fitting
- `plot_backup.py` - Original version (backup)

## Output Files
- `pvalue_heatmap_50x20.png` - Updated heatmap with proper text fitting
- `pvalue_minima_50x20.csv` - Data file (unchanged)
- `pvalue_keyword_labels_50x20.csv` - Labels file (unchanged)

## Benefits
1. **Professional Appearance**: Text stays within cell boundaries
2. **Readability**: Appropriate font size for cell dimensions
3. **Flexibility**: Configurable parameters for different use cases
4. **Robustness**: Handles edge cases with text truncation
5. **Debugging**: Clear output showing calculated parameters













