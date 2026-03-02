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
OUTPUT_DIR = r"F:\NWP\S1_CS2_overlap\2023"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Classification matching options
# Option 1: ice -> S1 value 1 only
# Option 2: ice -> S1 value 1 or 3 (including refrozen)
ICE_MATCH_OPTION = 1  # Change to 2 if you want ice to match both 1 and 3

# Color scheme for visualization
COLORS = {
    'ice': '#6BAED6',           # Blue for ice
    'lead': '#FD8D3C',          # Orange for lead
    'refrozen': '#BCBDDC',      # Purple for refrozen
    'background': '#F0F0F0'     # Light gray for background
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
        print("  Warning: GeoDataFrame has no CRS, assuming EPSG:4326")
        gdf.set_crs("EPSG:4326", inplace=True)
    
    if gdf.crs != target_crs:
        print(f"  Reprojecting from {gdf.crs} to {target_crs}")
        gdf = gdf.to_crs(target_crs)
    
    return gdf

def extract_cs2_values_at_points(cs2_gdf, s1_path):
    """Extract S1 raster values at CS2 point locations."""
    with rasterio.open(s1_path) as src:
        # Get coordinates
        coords = [(x, y) for x, y in zip(cs2_gdf.geometry.x, cs2_gdf.geometry.y)]
        
        # Sample raster at point locations
        s1_values = []
        for coord in coords:
            try:
                # Sample returns a generator, get first value
                val = list(src.sample([coord]))[0][0]
                s1_values.append(val)
            except:
                s1_values.append(np.nan)
        
        return np.array(s1_values)

def classify_match(cs2_class, s1_value, ice_option=1):
    """
    Determine if CS2 and S1 classifications match.
    
    Parameters:
    -----------
    cs2_class : str
        CS2 classification ('ice' or 'lead')
    s1_value : int
        S1 classification value (1=ice, 2=lead, 3=refrozen)
    ice_option : int
        1: ice matches only value 1
        2: ice matches value 1 or 3
    
    Returns:
    --------
    tuple: (match_type, is_correct)
    """
    if pd.isna(s1_value) or s1_value == 0:
        return ('no_data', False)
    
    cs2_class_lower = str(cs2_class).lower().strip()
    
    if cs2_class_lower == 'ice':
        if ice_option == 1:
            is_correct = (s1_value == 1)
            match_type = 'ice_to_ice' if is_correct else 'ice_mismatch'
        else:  # ice_option == 2
            is_correct = (s1_value in [1, 3])
            if s1_value == 1:
                match_type = 'ice_to_ice'
            elif s1_value == 3:
                match_type = 'ice_to_refrozen'
            else:
                match_type = 'ice_mismatch'
    
    elif cs2_class_lower == 'lead':
        is_correct = (s1_value == 2)
        match_type = 'lead_to_lead' if is_correct else 'lead_mismatch'
    
    else:
        return ('unknown', False)
    
    return (match_type, is_correct)

def analyze_pair(cs2_path, s1_path, ice_option=1):
    """
    Analyze a single CS2-S1 pair.
    
    Returns:
    --------
    dict: Statistics dictionary or None if analysis fails
    """
    try:
        # Read CS2 GPKG
        print(f"  Reading CS2: {os.path.basename(cs2_path)}")
        cs2_gdf = gpd.read_file(cs2_path)
        
        # Check for class attribute with error tolerance
        class_col = None
        for col in ['class', 'Class', 'CLASS', 'classification']:
            if col in cs2_gdf.columns:
                class_col = col
                break
        
        if class_col is None:
            print(f"  Error: No classification column found in CS2 file")
            return None
        
        # Read S1 raster CRS
        with rasterio.open(s1_path) as src:
            s1_crs = src.crs
            s1_bounds = src.bounds
        
        print(f"  S1 CRS: {s1_crs}")
        print(f"  CS2 CRS: {cs2_gdf.crs}")
        
        # Reproject CS2 if necessary
        cs2_gdf = check_and_reproject_crs(cs2_gdf, s1_crs)
        
        # Filter CS2 points to S1 bounds
        cs2_gdf = cs2_gdf.cx[s1_bounds.left:s1_bounds.right, 
                              s1_bounds.bottom:s1_bounds.top]
        
        if len(cs2_gdf) == 0:
            print(f"  Warning: No CS2 points within S1 bounds")
            return None
        
        print(f"  CS2 points in S1 region: {len(cs2_gdf)}")
        
        # Extract S1 values at CS2 points
        s1_values = extract_cs2_values_at_points(cs2_gdf, s1_path)
        
        # Classify matches
        results = []
        for idx, row in cs2_gdf.iterrows():
            cs2_class = row[class_col]
            s1_val = s1_values[cs2_gdf.index.get_loc(idx)]
            match_type, is_correct = classify_match(cs2_class, s1_val, ice_option)
            results.append({
                'cs2_class': cs2_class,
                's1_value': s1_val,
                'match_type': match_type,
                'is_correct': is_correct
            })
        
        results_df = pd.DataFrame(results)
        
        # Calculate statistics
        total_points = len(results_df)
        valid_points = len(results_df[results_df['match_type'] != 'no_data'])
        correct_points = results_df['is_correct'].sum()
        
        stats = {
            'total_points': total_points,
            'valid_points': valid_points,
            'correct_points': int(correct_points),
            'correct_rate': correct_points / valid_points if valid_points > 0 else 0,
            'ice_to_ice': len(results_df[results_df['match_type'] == 'ice_to_ice']),
            'lead_to_lead': len(results_df[results_df['match_type'] == 'lead_to_lead']),
            'ice_mismatch': len(results_df[results_df['match_type'] == 'ice_mismatch']),
            'lead_mismatch': len(results_df[results_df['match_type'] == 'lead_mismatch']),
            'no_data': len(results_df[results_df['match_type'] == 'no_data'])
        }
        
        if ice_option == 2:
            stats['ice_to_refrozen'] = len(results_df[results_df['match_type'] == 'ice_to_refrozen'])
        
        # Add detailed results for visualization
        stats['cs2_gdf'] = cs2_gdf
        stats['s1_values'] = s1_values
        stats['results_df'] = results_df
        
        return stats
    
    except Exception as e:
        print(f"  Error analyzing pair: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_visualization(cs2_path, s1_path, stats, output_path, ice_option=1):
    """Create visualization of CS2-S1 overlap."""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Read S1 raster
        with rasterio.open(s1_path) as src:
            s1_data = src.read(1)
            s1_extent = [src.bounds.left, src.bounds.right, 
                        src.bounds.bottom, src.bounds.top]
        
        # Plot S1 classification
        s1_colors = [COLORS['background'], COLORS['ice'], COLORS['lead'], COLORS['refrozen']]
        cmap_s1 = plt.matplotlib.colors.ListedColormap(s1_colors)
        
        im1 = ax1.imshow(s1_data, extent=s1_extent, cmap=cmap_s1, 
                        vmin=0, vmax=3, interpolation='nearest')
        ax1.set_title('S1 RF Classification', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        
        # Add S1 legend
        legend_s1 = [
            mpatches.Patch(color=COLORS['ice'], label='Ice (1)'),
            mpatches.Patch(color=COLORS['lead'], label='Lead (2)'),
            mpatches.Patch(color=COLORS['refrozen'], label='Refrozen (3)')
        ]
        ax1.legend(handles=legend_s1, loc='upper right', fontsize=9)
        
        # Plot overlay with CS2 points
        ax2.imshow(s1_data, extent=s1_extent, cmap=cmap_s1, 
                  vmin=0, vmax=3, interpolation='nearest', alpha=0.6)
        
        cs2_gdf = stats['cs2_gdf']
        results_df = stats['results_df']
        
        # Plot CS2 points colored by match/mismatch
        correct_mask = results_df['is_correct']
        incorrect_mask = ~correct_mask & (results_df['match_type'] != 'no_data')
        
        if correct_mask.sum() > 0:
            ax2.scatter(cs2_gdf[correct_mask].geometry.x, 
                       cs2_gdf[correct_mask].geometry.y,
                       c='green', s=20, alpha=0.7, edgecolors='darkgreen', 
                       linewidth=0.5, label='Match')
        
        if incorrect_mask.sum() > 0:
            ax2.scatter(cs2_gdf[incorrect_mask].geometry.x,
                       cs2_gdf[incorrect_mask].geometry.y,
                       c='red', s=20, alpha=0.7, edgecolors='darkred',
                       linewidth=0.5, label='Mismatch')
        
        ax2.set_title('CS2-S1 Overlap (Correct Rate: {:.1f}%)'.format(
            stats['correct_rate'] * 100), fontsize=12, fontweight='bold')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.legend(loc='upper right', fontsize=9)
        
        # Add statistics text
        stats_text = f"Total Points: {stats['total_points']}\n"
        stats_text += f"Valid Points: {stats['valid_points']}\n"
        stats_text += f"Correct: {stats['correct_points']}\n"
        stats_text += f"Ice→Ice: {stats['ice_to_ice']}\n"
        stats_text += f"Lead→Lead: {stats['lead_to_lead']}\n"
        if ice_option == 2 and 'ice_to_refrozen' in stats:
            stats_text += f"Ice→Refrozen: {stats['ice_to_refrozen']}\n"
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Visualization saved: {output_path}")
        
    except Exception as e:
        print(f"  Error creating visualization: {e}")
        import traceback
        traceback.print_exc()

# -------------------- MAIN PROCESSING --------------------

def main():
    print("=" * 80)
    print("CS2-S1 Overlap Analysis")
    print("=" * 80)
    print(f"Ice matching option: {ICE_MATCH_OPTION}")
    print(f"  1: ice -> S1 value 1 only")
    print(f"  2: ice -> S1 value 1 or 3 (including refrozen)")
    print("=" * 80)
    
    # Read matching CSV
    print(f"\nReading match file: {MATCH_CSV}")
    match_df = pd.read_csv(MATCH_CSV)
    print(f"Total pairs in CSV: {len(match_df)}")
    
    # Process each pair
    all_stats = []
    successful_pairs = 0
    failed_pairs = 0
    
    for idx, row in match_df.iterrows():
        print(f"\n[{idx+1}/{len(match_df)}] Processing pair:")
        
        scene_name = row['sceneName']
        # Extract date-time part from CS2 path
        # e.g., from "CS_OFFL_SIR_SIN_2__20230630T102149_20230630T102404_E001.nc"
        # extract "20230630T102149_20230630T102404"
        cs2_path_full = row['cs2_path']
        cs2_basename = os.path.basename(cs2_path_full).replace('.nc', '')

        # Match pattern: 8 digits + T + 6 digits + _ + 8 digits + T + 6 digits
        date_pattern = r'(\d{8}T\d{6}_\d{8}T\d{6})'
        match = re.search(date_pattern, cs2_basename)
        cs2_key = match.group(1) if match else cs2_basename

        print(f"  Scene: {scene_name}")
        print(f"  CS2 date key: {cs2_key}")

        # Find files
        s1_file = find_file_by_pattern(S1_DIR, scene_name)
        cs2_file = find_file_by_pattern(CS2_DIR, cs2_key)
            
        if s1_file is None:
            print(f"  ✗ S1 file not found")
            failed_pairs += 1
            continue
        
        if cs2_file is None:
            print(f"  ✗ CS2 file not found")
            failed_pairs += 1
            continue
        
        print(f"  ✓ Both files found")
        
        # Analyze pair
        stats = analyze_pair(cs2_file, s1_file, ICE_MATCH_OPTION)
        
        if stats is None:
            failed_pairs += 1
            continue
        
        # Create visualization
        vis_filename = f"{scene_name}_{cs2_basename}_overlap.png"
        vis_path = os.path.join(OUTPUT_DIR, vis_filename)
        create_visualization(cs2_file, s1_file, stats, vis_path, ICE_MATCH_OPTION)
        
        # Save detailed results
        detail_filename = f"{scene_name}_{cs2_basename}_details.csv"
        detail_path = os.path.join(OUTPUT_DIR, detail_filename)
        stats['results_df'].to_csv(detail_path, index=False)
        
        # Add pair info to stats
        stats['scene_name'] = scene_name
        stats['cs2_basename'] = cs2_basename
        stats['s1_file'] = os.path.basename(s1_file)
        stats['cs2_file'] = os.path.basename(cs2_file)
        
        # Save pair statistics
        pair_stats_filename = f"{scene_name}_{cs2_basename}_statistics.txt"
        pair_stats_path = os.path.join(OUTPUT_DIR, pair_stats_filename)
        with open(pair_stats_path, 'w') as f:
            f.write(f"Pair Statistics: {scene_name} - {cs2_basename}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Total points: {stats['total_points']}\n")
            f.write(f"Valid points: {stats['valid_points']}\n")
            f.write(f"Correct points: {stats['correct_points']}\n")
            f.write(f"Overall correct rate: {stats['correct_rate']*100:.2f}%\n\n")
            f.write(f"Ice → Ice: {stats['ice_to_ice']}\n")
            f.write(f"Lead → Lead: {stats['lead_to_lead']}\n")
            f.write(f"Total lead points: {stats['total_lead_points']}\n")
            f.write(f"Lead match rate: {stats['lead_correct_rate']*100:.2f}%\n\n")
            if ICE_MATCH_OPTION == 2 and 'ice_to_refrozen' in stats:
                f.write(f"Ice → Refrozen: {stats['ice_to_refrozen']}\n")
            f.write(f"Ice mismatches: {stats['ice_mismatch']}\n")
            f.write(f"Lead mismatches: {stats['lead_mismatch']}\n")
            f.write(f"No data points: {stats['no_data']}\n")
        print(f"  Statistics saved: {pair_stats_path}")
        
        # Remove large objects before storing
        del stats['cs2_gdf']
        del stats['s1_values']
        del stats['results_df']
        
        all_stats.append(stats)
        successful_pairs += 1
        
        print(f"  ✓ Analysis complete - Correct rate: {stats['correct_rate']*100:.1f}%")
    
    # Generate overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total pairs in CSV: {len(match_df)}")
    print(f"Successfully processed: {successful_pairs}")
    print(f"Failed: {failed_pairs}")
    
    if successful_pairs > 0:
        summary_df = pd.DataFrame(all_stats)
        
        # Calculate totals
        total_points = summary_df['total_points'].sum()
        total_valid = summary_df['valid_points'].sum()
        total_correct = summary_df['correct_points'].sum()
        overall_correct_rate = total_correct / total_valid if total_valid > 0 else 0
        
        print(f"\nTotal CS2 points analyzed: {total_points}")
        print(f"Valid points (with S1 data): {total_valid}")
        print(f"Correct matches: {total_correct}")
        print(f"Overall correct rate: {overall_correct_rate*100:.2f}%")
        
        print(f"\nDetailed statistics:")
        print(f"  Ice → Ice: {summary_df['ice_to_ice'].sum()}")
        print(f"  Lead → Lead: {summary_df['lead_to_lead'].sum()}")
        if ICE_MATCH_OPTION == 2:
            print(f"  Ice → Refrozen: {summary_df['ice_to_refrozen'].sum()}")
        print(f"  Ice mismatches: {summary_df['ice_mismatch'].sum()}")
        print(f"  Lead mismatches: {summary_df['lead_mismatch'].sum()}")
        print(f"  No data: {summary_df['no_data'].sum()}")
        
        # Save summary CSV
        summary_path = os.path.join(OUTPUT_DIR, 'overall_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()