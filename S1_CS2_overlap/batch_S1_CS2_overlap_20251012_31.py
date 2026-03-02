import os
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from pyproj import CRS
import warnings
warnings.filterwarnings('ignore')
import re

# -------------------- USER CONFIG --------------------
CS2_DIR = r"C:\Users\TJ002\Desktop\CS2_S1_result\filter\2023\gpkg"
S1_DIR = r"C:\Users\TJ002\Desktop\CS2_S1_result\filter\20213\tif"
MATCH_CSV = r"F:\NWP\CS2_S1_matched\time_match_2023_filter.csv"
OUTPUT_DIR = r"C:\Users\TJ002\Desktop\CS2_S1_result\overlap\2023_lead_refrozen"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ICE_MATCH_OPTION:
#   1: ice -> S1 value 1 only
#   2: ice -> S1 value 1 or 3 (including refrozen)
ICE_MATCH_OPTION = 1

# LEAD_MATCH_OPTION:
#   1: lead -> S1 value 2 only
#   2: lead -> S1 value 2 or 3 (including refrozen)
LEAD_MATCH_OPTION = 2

COLORS = {
    'ice': '#6BAED6', 'lead': '#FD8D3C', 'refrozen': '#BCBDDC',
    'background': '#F0F0F0', 'match_cmap': 'Greens', 'mismatch_cmap': 'Reds'
}

# -------------------- HELPER FUNCTIONS --------------------

# --- MODIFIED --- This function now accepts an optional 'extension' argument for more specific searches
def find_file_by_pattern(folder, pattern, extension=None):
    """Find file in folder that contains the pattern and optionally ends with a specific extension."""
    if not os.path.exists(folder):
        return None
    for file in os.listdir(folder):
        # Check if the main part of the name matches
        pattern_match = pattern in file
        
        # Check for the file extension if one is provided
        extension_match = True  # Assume it matches if no extension is specified
        if extension:
            extension_match = file.lower().endswith(extension.lower())
        
        if pattern_match and extension_match:
            return os.path.join(folder, file)
    return None

def check_and_reproject_crs(gdf, target_crs):
    """Check CRS and reproject if necessary."""
    if gdf.crs is None: raise ValueError("Input GeoDataFrame has no CRS defined.")
    if gdf.crs != target_crs:
        print(f"  Reprojecting from {gdf.crs} to {target_crs}")
        gdf = gdf.to_crs(target_crs)
    return gdf

def extract_cs2_values_at_points(cs2_gdf, s1_path):
    """Extract S1 raster values at CS2 point locations."""
    with rasterio.open(s1_path) as src:
        coords = [(x, y) for x, y in zip(cs2_gdf.geometry.x, cs2_gdf.geometry.y)]
        s1_values = [list(src.sample([coord]))[0][0] for coord in coords]
        return np.array(s1_values)

def classify_match(cs2_class, s1_value, ice_option=1, lead_option=1):
    """Determine if CS2 and S1 classifications match."""
    if pd.isna(s1_value) or s1_value == 0: return ('no_data', False)
    cs2_class_lower = str(cs2_class).lower().strip()
    
    if cs2_class_lower == 'ice':
        match_values = [1] if ice_option == 1 else [1, 3]
        is_correct = (s1_value in match_values)
        match_type = 'ice_match' if is_correct else 'ice_mismatch'
    elif cs2_class_lower == 'lead':
        match_values = [2] if lead_option == 1 else [2, 3]
        is_correct = (s1_value in match_values)
        match_type = 'lead_match' if is_correct else 'lead_mismatch'
    else: return ('unknown', False)
    return (match_type, is_correct)

# --- MODIFIED --- This version fixes the visualization KeyError
def analyze_pair(cs2_path, s1_path, ice_option=1, lead_option=1):
    """Analyze a single CS2-S1 pair."""
    try:
        cs2_gdf = gpd.read_file(cs2_path)
        class_col = next((c for c in ['class', 'Class', 'CLASS', 'classification'] if c in cs2_gdf.columns), None)
        if not class_col:
            raise ValueError("Classification column not found in CS2 file.")

        with rasterio.open(s1_path) as src:
            s1_crs, s1_bounds = src.crs, src.bounds

        cs2_gdf = check_and_reproject_crs(cs2_gdf, s1_crs)
        cs2_gdf = cs2_gdf.cx[s1_bounds.left:s1_bounds.right, s1_bounds.bottom:s1_bounds.top].reset_index(drop=True)

        if cs2_gdf.empty:
            print("  Warning: No CS2 points within S1 bounds")
            return None

        print(f"  CS2 points in S1 region: {len(cs2_gdf)}")

        cs2_gdf['s1_value'] = extract_cs2_values_at_points(cs2_gdf, s1_path)

        # --- FIX --- Unpack results into two lists before assignment
        # This is a more robust method that prevents pandas data type errors.
        match_types = []
        is_correct_list = []
        for _, row in cs2_gdf.iterrows():
            match_type, is_correct = classify_match(row[class_col], row['s1_value'], ice_option, lead_option)
            match_types.append(match_type)
            is_correct_list.append(is_correct)
        
        cs2_gdf['match_type'] = match_types
        cs2_gdf['is_correct'] = is_correct_list
        # --- END FIX ---

        valid_gdf = cs2_gdf[cs2_gdf['match_type'] != 'no_data'].copy()
        valid_gdf['class_lower'] = valid_gdf[class_col].str.lower()

        stats = {
            'total_points': len(cs2_gdf), 'valid_points': len(valid_gdf),
            'correct_points': int(valid_gdf['is_correct'].sum()),
        }
        stats['accuracy'] = stats['correct_points'] / stats['valid_points'] if stats['valid_points'] > 0 else 0

        # Calculate confusion matrix elements and metrics for both classes
        for class_name in ['lead', 'ice']:
            if class_name == 'lead':
                s1_positive_values = [2] if lead_option == 1 else [2, 3]
            else:  # class_name == 'ice'
                s1_positive_values = [1] if ice_option == 1 else [1, 3]

            tp = len(valid_gdf[(valid_gdf['class_lower'] == class_name) & (valid_gdf['is_correct'])])
            fp = len(valid_gdf[(valid_gdf['class_lower'] != class_name) & (valid_gdf['s1_value'].isin(s1_positive_values))])
            fn = len(valid_gdf[(valid_gdf['class_lower'] == class_name) & (~valid_gdf['is_correct'])])

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            stats[f'{class_name}_tp'] = tp
            stats[f'{class_name}_fp'] = fp
            stats[f'{class_name}_fn'] = fn
            stats[f'{class_name}_precision'] = precision
            stats[f'{class_name}_recall'] = recall
            stats[f'{class_name}_f1_score'] = f1_score

        stats['results_gdf'] = cs2_gdf
        return stats

    except Exception as e:
        print(f"  Error analyzing pair: {e}")
        return None
# --- MODIFIED --- Visualization text now shows stats for both ice and lead
def create_visualization(s1_path, stats, output_path):
    """Create visualization of CS2-S1 overlap using hexbin density plots."""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1, 1.2]})
        
        with rasterio.open(s1_path) as src:
            s1_data, s1_extent = src.read(1), [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        
        # Left subplot: S1 Classification
        cmap_s1 = plt.matplotlib.colors.ListedColormap([COLORS[c] for c in ['background', 'ice', 'lead', 'refrozen']])
        ax1.imshow(s1_data, extent=s1_extent, cmap=cmap_s1, vmin=0, vmax=3, interpolation='nearest')
        ax1.set_title('S1 RF Classification', fontsize=12, fontweight='bold')
        legend_s1 = [mpatches.Patch(color=c, label=l) for l, c in 
                     {'Ice (1)': COLORS['ice'], 'Lead (2)': COLORS['lead'], 'Refrozen (3)': COLORS['refrozen']}.items()]
        ax1.legend(handles=legend_s1, loc='upper right', fontsize=9)
        
        # Right subplot: Hexbin Overlay
        ax2.imshow(s1_data, extent=s1_extent, cmap=cmap_s1, vmin=0, vmax=3, interpolation='nearest', alpha=0.4)
        valid_gdf = stats['results_gdf'][stats['results_gdf']['match_type'] != 'no_data']
        correct_gdf = valid_gdf[valid_gdf['is_correct']]
        incorrect_gdf = valid_gdf[~valid_gdf['is_correct']]
        gridsize = 50 
        
        if not incorrect_gdf.empty:
            hb_mis = ax2.hexbin(incorrect_gdf.geometry.x, incorrect_gdf.geometry.y, gridsize=gridsize, cmap=COLORS['mismatch_cmap'], alpha=0.7, mincnt=1)
            cb1 = fig.colorbar(hb_mis, ax=ax2, shrink=0.6, pad=0.02); cb1.set_label('Mismatch Density', fontsize=9)
        if not correct_gdf.empty:
            hb_mat = ax2.hexbin(correct_gdf.geometry.x, correct_gdf.geometry.y, gridsize=gridsize, cmap=COLORS['match_cmap'], alpha=0.7, mincnt=1)
            cb2 = fig.colorbar(hb_mat, ax=ax2, shrink=0.6, pad=0.02); cb2.set_label('Match Density', fontsize=9)

        ax2.set_title(f"CS2-S1 Overlap Density (Overall Accuracy: {stats['accuracy']:.1%})", fontsize=12, fontweight='bold')
        
        # --- MODIFIED --- Stats text now includes both classes
        stats_text = (f"--- Lead Class ---\n"
                      f"  Precision: {stats['lead_precision']:.2f}\n"
                      f"  Recall:    : {stats['lead_recall']:.2f}\n"
                      f"  F1-Score  : {stats['lead_f1_score']:.2f}\n\n"
                      f"--- Ice Class ---\n"
                      f"  Precision: {stats['ice_precision']:.2f}\n"
                      f"  Recall     : {stats['ice_recall']:.2f}\n"
                      f"  F1-Score   : {stats['ice_f1_score']:.2f}")
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
        
        plt.tight_layout(); plt.savefig(output_path, dpi=200, bbox_inches='tight'); plt.close()
        print(f"  Visualization saved: {output_path}")
    except Exception as e: print(f"  Error creating visualization: {e}")

# -------------------- MAIN PROCESSING --------------------

def main():
    print("=" * 80 + "\nCS2-S1 Overlap Analysis\n" + "=" * 80)
    print(f"Ice match: {ICE_MATCH_OPTION} ({'1' if ICE_MATCH_OPTION==1 else '1,3'}) | Lead match: {LEAD_MATCH_OPTION} ({'2' if LEAD_MATCH_OPTION==1 else '2,3'})")
    print("=" * 80)
    
    match_df = pd.read_csv(MATCH_CSV)
    all_stats = []
    
    for idx, row in match_df.iterrows():
        print(f"\n[{idx+1}/{len(match_df)}] Processing pair: {row['sceneName']}")
        cs2_key_match = re.search(r'(\d{8}T\d{6}_\d{8}T\d{6})', Path(row['cs2_path']).name)
        if not cs2_key_match: continue
        
        # --- FIX --- Add extension=".tif" to ensure only the correct raster image is found
        s1_file = find_file_by_pattern(S1_DIR, row['sceneName'], extension=".tif")
        
        # The CS2 file search can be made more specific too, for example with ".gpkg"
        cs2_file = find_file_by_pattern(CS2_DIR, cs2_key_match.group(1), extension=".gpkg")
        
        if not all([s1_file, cs2_file]): 
            print("  ✗ Files not found."); 
            continue
        
        stats = analyze_pair(cs2_file, s1_file, ICE_MATCH_OPTION, LEAD_MATCH_OPTION)
        if not stats: continue
        
        create_visualization(s1_file, stats, os.path.join(OUTPUT_DIR, f"{Path(s1_file).stem}_overlap.png"))
        
        # --- MODIFIED --- Statistics file now includes both ice and lead performance
        stats_path = os.path.join(OUTPUT_DIR, f"{Path(s1_file).stem}_statistics.txt")
        with open(stats_path, 'w') as f:
            f.write(f"Pair Statistics: {row['sceneName']}\n" + "="*60 + "\n")
            f.write(f"Overall Accuracy: {stats['accuracy']:.2%}\n")
            f.write(f"Valid Points: {stats['valid_points']}/{stats['total_points']}\n\n")
            for class_name in ['ice', 'lead']:
                f.write(f"--- {class_name.capitalize()} Class Performance ---\n")
                f.write(f"Precision: {stats[f'{class_name}_precision']:.4f}\n")
                f.write(f"Recall (TPR): {stats[f'{class_name}_recall']:.4f}\n")
                f.write(f"F1-Score: {stats[f'{class_name}_f1_score']:.4f}\n")
                f.write(f"  TP: {stats[f'{class_name}_tp']} | FP: {stats[f'{class_name}_fp']} | FN: {stats[f'{class_name}_fn']}\n\n")
        print(f"  Statistics saved: {Path(stats_path).name}")

        stats['scene_name'] = row['sceneName']
        del stats['results_gdf']
        all_stats.append(stats)
        print(f"  ✓ Analysis complete - Accuracy: {stats['accuracy']:.1%}")

    if not all_stats: print("\nNo pairs were successfully processed."); return
        
    # --- MODIFIED --- Overall summary now calculates and displays stats for all classes
    summary_df = pd.DataFrame(all_stats)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'overall_summary.csv'), index=False)
    print("\n" + "="*80 + "\nOVERALL SUMMARY\n" + "="*80)
    
    total_valid = summary_df['valid_points'].sum()
    total_correct = summary_df['correct_points'].sum()
    overall_accuracy = total_correct / total_valid if total_valid > 0 else 0
    print(f"Overall Accuracy: {overall_accuracy:.2%}\n")

    for class_name in ['ice', 'lead']:
        total_tp = summary_df[f'{class_name}_tp'].sum()
        total_fp = summary_df[f'{class_name}_fp'].sum()
        total_fn = summary_df[f'{class_name}_fn'].sum()
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"--- Overall {class_name.capitalize()} Class Performance ({len(summary_df)} pairs) ---")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}\n")
        
    print("="*80 + "\nAnalysis complete!\n" + "="*80)

if __name__ == "__main__":
    main()