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
CS2_DIR = r"E:\NWP\CS2_S1_matched\CS2\CS2 class point"
S1_DIR = r"F:\NWP\Classification Result\2023_maskSIC"
MATCH_CSV = r"E:\NWP\CS2_S1_matched\time_match_2023_filter.csv"
OUTPUT_DIR = r"F:\NWP\S1_CS2_overlap\2023_new"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- MODIFIED --- Classification matching options are now separate for ice and lead
# ICE_MATCH_OPTION:
#   1: ice -> S1 value 1 only
#   2: ice -> S1 value 1 or 3 (including refrozen)
ICE_MATCH_OPTION = 1

# LEAD_MATCH_OPTION:
#   1: lead -> S1 value 2 only
#   2: lead -> S1 value 2 or 3 (including refrozen)
LEAD_MATCH_OPTION = 1

# --- MODIFIED --- Color scheme for visualization (added colors for hexbins)
COLORS = {
    'ice': '#6BAED6',           # Blue for ice
    'lead': '#FD8D3C',          # Orange for lead
    'refrozen': '#BCBDDC',      # Purple for refrozen
    'background': '#F0F0F0',    # Light gray for background
    'match_cmap': 'Greens',     # Colormap for correct matches hexbin
    'mismatch_cmap': 'Reds'     # Colormap for incorrect matches hexbin
}

# -------------------- HELPER FUNCTIONS --------------------

def find_file_by_pattern(folder, pattern):
    """Find file in folder that contains the pattern."""
    if not os.path.exists(folder):
        return None
    for file in os.listdir(folder):
        if pattern in file:
            return os.path.join(folder, file)
    return None

def check_and_reproject_crs(gdf, target_crs):
    """Check CRS and reproject if necessary."""
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS defined. Please set the correct CRS before proceeding.")
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

# --- MODIFIED --- Updated to handle separate ice and lead options
def classify_match(cs2_class, s1_value, ice_option=1, lead_option=1):
    """Determine if CS2 and S1 classifications match."""
    if pd.isna(s1_value) or s1_value == 0:
        return ('no_data', False)
    
    cs2_class_lower = str(cs2_class).lower().strip()
    
    if cs2_class_lower == 'ice':
        ice_match_values = [1] if ice_option == 1 else [1, 3]
        is_correct = (s1_value in ice_match_values)
        if is_correct:
            match_type = 'ice_to_ice' if s1_value == 1 else 'ice_to_refrozen'
        else:
            match_type = 'ice_mismatch'
            
    elif cs2_class_lower == 'lead':
        lead_match_values = [2] if lead_option == 1 else [2, 3]
        is_correct = (s1_value in lead_match_values)
        if is_correct:
            match_type = 'lead_to_lead' if s1_value == 2 else 'lead_to_refrozen'
        else:
            match_type = 'lead_mismatch'
            
    else:
        return ('unknown', False)
    
    return (match_type, is_correct)

def analyze_pair(cs2_path, s1_path, ice_option=1, lead_option=1):
    """Analyze a single CS2-S1 pair."""
    try:
        print(f"  Reading CS2: {os.path.basename(cs2_path)}")
        cs2_gdf = gpd.read_file(cs2_path)
        
        class_col = next((col for col in ['class', 'Class', 'CLASS', 'classification'] if col in cs2_gdf.columns), None)
        if class_col is None:
            print(f"  Error: No classification column found in CS2 file")
            return None
            
        with rasterio.open(s1_path) as src:
            s1_crs, s1_bounds = src.crs, src.bounds

        print(f"  S1 CRS: {s1_crs}")
        print(f"  CS2 CRS before check: {cs2_gdf.crs}")
        
        cs2_gdf = check_and_reproject_crs(cs2_gdf, s1_crs)
        cs2_gdf = cs2_gdf.cx[s1_bounds.left:s1_bounds.right, s1_bounds.bottom:s1_bounds.top]
        
        if len(cs2_gdf) == 0:
            print(f"  Warning: No CS2 points within S1 bounds")
            return None
        
        print(f"  CS2 points in S1 region: {len(cs2_gdf)}")
        
        s1_values = extract_cs2_values_at_points(cs2_gdf, s1_path)
        
        results = []
        for idx, row in cs2_gdf.iterrows():
            cs2_class = row[class_col]
            s1_val = s1_values[cs2_gdf.index.get_loc(idx)]
            match_type, is_correct = classify_match(cs2_class, s1_val, ice_option, lead_option)
            results.append({'s1_value': s1_val, 'match_type': match_type, 'is_correct': is_correct})
        
        results_df = pd.DataFrame(results, index=cs2_gdf.index)
        cs2_gdf = cs2_gdf.join(results_df)

        valid_gdf = cs2_gdf[cs2_gdf['match_type'] != 'no_data']
        stats = {
            'total_points': len(cs2_gdf),
            'valid_points': len(valid_gdf),
            'correct_points': int(valid_gdf['is_correct'].sum()),
            'no_data': len(cs2_gdf) - len(valid_gdf)
        }
        stats['correct_rate'] = stats['correct_points'] / stats['valid_points'] if stats['valid_points'] > 0 else 0
        
        # --- ADDED --- Professional statistics for 'lead' class
        # True Positive (TP): CS2 is lead, S1 agrees it's lead
        tp = len(valid_gdf[(valid_gdf[class_col].str.lower() == 'lead') & (valid_gdf['is_correct'])])
        
        # False Positive (FP): CS2 is ice, but S1 calls it lead
        fp_mask = (valid_gdf[class_col].str.lower() == 'ice')
        if lead_option == 1:
             fp_mask &= (valid_gdf['s1_value'] == 2)
        else: # lead_option == 2
             fp_mask &= (valid_gdf['s1_value'].isin([2, 3]))
        fp = fp_mask.sum()
        
        # False Negative (FN): CS2 is lead, but S1 calls it something else
        fn = len(valid_gdf[(valid_gdf[class_col].str.lower() == 'lead') & (~valid_gdf['is_correct'])])
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        stats['lead_tp'] = tp
        stats['lead_fp'] = fp
        stats['lead_fn'] = fn
        stats['lead_precision'] = precision
        stats['lead_recall'] = recall
        stats['lead_f1_score'] = f1_score
        
        stats['results_gdf'] = cs2_gdf
        return stats
    
    except Exception as e:
        print(f"  Error analyzing pair: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- MODIFIED --- Replaced scatter with hexbin for better density visualization
def create_visualization(s1_path, stats, output_path):
    """Create visualization of CS2-S1 overlap using hexbin density plots."""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1, 1.2]})
        
        with rasterio.open(s1_path) as src:
            s1_data = src.read(1)
            s1_extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        
        # Plot S1 classification (Left subplot)
        s1_colors = [COLORS['background'], COLORS['ice'], COLORS['lead'], COLORS['refrozen']]
        cmap_s1 = plt.matplotlib.colors.ListedColormap(s1_colors)
        ax1.imshow(s1_data, extent=s1_extent, cmap=cmap_s1, vmin=0, vmax=3, interpolation='nearest')
        ax1.set_title('S1 RF Classification', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        legend_s1 = [mpatches.Patch(color=c, label=l) for l, c in 
                     {'Ice (1)': COLORS['ice'], 'Lead (2)': COLORS['lead'], 'Refrozen (3)': COLORS['refrozen']}.items()]
        ax1.legend(handles=legend_s1, loc='upper right', fontsize=9)
        
        # Plot Hexbin Overlay (Right subplot)
        ax2.imshow(s1_data, extent=s1_extent, cmap=cmap_s1, vmin=0, vmax=3, interpolation='nearest', alpha=0.4)
        
        results_gdf = stats['results_gdf']
        valid_gdf = results_gdf[results_gdf['match_type'] != 'no_data']
        
        correct_gdf = valid_gdf[valid_gdf['is_correct']]
        incorrect_gdf = valid_gdf[~valid_gdf['is_correct']]
        
        # Use a common gridsize for both hexbins
        gridsize = 50 
        
        # Hexbin for Mismatches (plotted first, in red)
        if not incorrect_gdf.empty:
            hb_mis = ax2.hexbin(incorrect_gdf.geometry.x, incorrect_gdf.geometry.y,
                                gridsize=gridsize, cmap=COLORS['mismatch_cmap'],
                                alpha=0.7, mincnt=1) # mincnt=1 ensures empty hexes are not colored
        
        # Hexbin for Matches (plotted on top, in green)
        if not correct_gdf.empty:
            hb_mat = ax2.hexbin(correct_gdf.geometry.x, correct_gdf.geometry.y,
                                gridsize=gridsize, cmap=COLORS['match_cmap'],
                                alpha=0.7, mincnt=1)

        ax2.set_title(f"CS2-S1 Overlap Density (Overall Accuracy: {stats['correct_rate']:.1%})", fontsize=12, fontweight='bold')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        
        # Add colorbars
        if not incorrect_gdf.empty:
            cb1 = fig.colorbar(hb_mis, ax=ax2, shrink=0.6, pad=0.02)
            cb1.set_label('Mismatch Point Density', fontsize=9)
        if not correct_gdf.empty:
            cb2 = fig.colorbar(hb_mat, ax=ax2, shrink=0.6, pad=0.02)
            cb2.set_label('Match Point Density', fontsize=9)

        # Add statistics text
        stats_text = (f"Lead Class Performance:\n"
                      f"  Precision: {stats['lead_precision']:.2f}\n"
                      f"  Recall: {stats['lead_recall']:.2f}\n"
                      f"  F1-Score: {stats['lead_f1_score']:.2f}\n\n"
                      f"Confusion Matrix (Lead):\n"
                      f"  True Positives (TP): {stats['lead_tp']}\n"
                      f"  False Positives (FP): {stats['lead_fp']}\n"
                      f"  False Negatives (FN): {stats['lead_fn']}")
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"  Visualization saved: {output_path}")
        
    except Exception as e:
        print(f"  Error creating visualization: {e}")

# -------------------- MAIN PROCESSING --------------------

def main():
    print("=" * 80)
    print("CS2-S1 Overlap Analysis")
    print("=" * 80)
    # --- MODIFIED --- Displaying the new matching options
    print(f"Ice matching option: {ICE_MATCH_OPTION} ({'ice->1' if ICE_MATCH_OPTION==1 else 'ice->1 or 3'})")
    print(f"Lead matching option: {LEAD_MATCH_OPTION} ({'lead->2' if LEAD_MATCH_OPTION==1 else 'lead->2 or 3'})")
    print("=" * 80)
    
    match_df = pd.read_csv(MATCH_CSV)
    print(f"Total pairs in CSV: {len(match_df)}")
    
    all_stats = []
    
    for idx, row in match_df.iterrows():
        print(f"\n[{idx+1}/{len(match_df)}] Processing pair:")
        scene_name = row['sceneName']
        
        date_pattern = r'(\d{8}T\d{6}_\d{8}T\d{6})'
        match = re.search(date_pattern, os.path.basename(row['cs2_path']))
        if not match:
            print(f"  ✗ Could not extract date pattern from CS2 path")
            continue
        cs2_key = match.group(1)
        
        print(f"  Scene: {scene_name}, CS2 key: {cs2_key}")

        s1_file = find_file_by_pattern(S1_DIR, scene_name)
        cs2_file = find_file_by_pattern(CS2_DIR, cs2_key)
        
        if not all([s1_file, cs2_file]):
            print(f"  ✗ One or both files not found.")
            continue
        print(f"  ✓ Both files found")
        
        stats = analyze_pair(cs2_file, s1_file, ICE_MATCH_OPTION, LEAD_MATCH_OPTION)
        
        if stats is None: continue
        
        cs2_basename = Path(cs2_file).stem
        
        vis_filename = f"{scene_name}_{cs2_basename}_overlap.png"
        create_visualization(s1_file, stats, os.path.join(OUTPUT_DIR, vis_filename))
        
        stats['scene_name'] = scene_name
        
        # --- MODIFIED --- Writing new professional statistics to file
        pair_stats_filename = f"{scene_name}_{cs2_basename}_statistics.txt"
        with open(os.path.join(OUTPUT_DIR, pair_stats_filename), 'w') as f:
            f.write(f"Pair Statistics: {scene_name}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Overall Accuracy: {stats['correct_rate']:.2%}\n")
            f.write(f"Valid Points: {stats['valid_points']}/{stats['total_points']}\n\n")
            f.write("--- Lead Class Performance ---\n")
            f.write(f"Precision: {stats['lead_precision']:.4f}\n")
            f.write(f"Recall (TPR): {stats['lead_recall']:.4f}\n")
            f.write(f"F1-Score: {stats['lead_f1_score']:.4f}\n\n")
            f.write("--- Lead Confusion Matrix ---\n")
            f.write(f"True Positives (TP): {stats['lead_tp']}\n")
            f.write(f"False Positives (FP): {stats['lead_fp']}\n")
            f.write(f"False Negatives (FN): {stats['lead_fn']}\n")
        print(f"  Statistics saved: {pair_stats_filename}")
        
        del stats['results_gdf']
        all_stats.append(stats)
        
        print(f"  ✓ Analysis complete - Lead F1-Score: {stats['lead_f1_score']:.2f}")

    if not all_stats:
        print("\nNo pairs were successfully processed.")
        return
        
    # --- MODIFIED --- Overall summary now includes professional stats
    print("\n" + "=" * 80 + "\nOVERALL SUMMARY\n" + "=" * 80)
    summary_df = pd.DataFrame(all_stats)
    summary_path = os.path.join(OUTPUT_DIR, 'overall_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary CSV saved to: {summary_path}")

    # Calculate overall metrics
    total_valid = summary_df['valid_points'].sum()
    total_correct = summary_df['correct_points'].sum()
    total_tp = summary_df['lead_tp'].sum()
    total_fp = summary_df['lead_fp'].sum()
    total_fn = summary_df['lead_fn'].sum()
    
    overall_accuracy = total_correct / total_valid if total_valid > 0 else 0
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    print(f"\n--- Overall Performance ({len(summary_df)} pairs) ---")
    print(f"Overall Accuracy: {overall_accuracy:.2%}")
    print("\n--- Overall Lead Class Performance ---")
    print(f"Precision: {overall_precision:.4f}")
    print(f"Recall: {overall_recall:.4f}")
    print(f"F1-Score: {overall_f1:.4f}")
    print("\n" + "=" * 80 + "\nAnalysis complete!\n" + "=" * 80)

if __name__ == "__main__":
    main()