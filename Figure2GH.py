#!/usr/bin/env python3
"""
Standalone script for Figure Panels G and H
G: KDE (density plot) of demixing index values
H: Bar graph of demixing index vs composition
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Publication-quality style
plt.rcParams.update({
    'font.size': 20,
    'axes.linewidth': 2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 8,
    'xtick.major.width': 2,
    'ytick.major.size': 8,
    'ytick.major.width': 2,
    'legend.frameon': False,
    'figure.dpi': 300
})


def assign_composition_bin(comp):
    """Assign composition to bins."""
    if 85 <= comp <= 100:
        return '85-95'
    elif 65 <= comp < 85:
        return '65-85'
    elif 35 <= comp < 65:
        return '50'
    elif 15 <= comp < 35:
        return '15-35'
    elif 0 <= comp < 15:
        return '5-15'
    return None


def load_and_prepare_data():
    """Load and prepare datasets."""
    print("Loading datasets...")
    df_dem = pd.read_csv("../DATASETS/demixing.csv")
    df_tie = pd.read_csv("../DATASETS/tie_lines.csv")
    df_prot = pd.read_csv("../DATASETS/gin_samples.csv")
    
    print(f"Loaded {len(df_dem)} demixing entries")
    print(f"Loaded {len(df_tie)} tie line entries")
    
    # Merge demixing with tie lines
    df_merged = pd.merge(
        df_dem,
        df_tie[['protein1', 'protein2', 'composition1', 'composition2', 'phase_behavior']],
        on=['protein1', 'protein2', 'composition1', 'composition2'],
        how='left'
    )
    
    # Merge with protein properties for additional features
    cols = ['seq_name', 'faro', 'ah_ij', 'N', 'shd', 'mean_lambda', 'SPR_svr']
    for p, suffix in [('protein1', '_p1'), ('protein2', '_p2')]:
        df_merged = pd.merge(
            df_merged, df_prot[cols], 
            left_on=p, right_on='seq_name', 
            how='left', suffixes=('', '_temp')
        )
        df_merged = df_merged.drop(columns=['seq_name']).rename(
            columns={c: c+suffix for c in cols[1:]}
        )
    
    return df_merged


def prepare_panel_G_data(df_merged):
    """Prepare data for Panel G (KDE of demixing index)."""
    print("\nPreparing data for Panel G (KDE)...")
    
    df = df_merged.copy()
    df_filtered = df[
        (df['dGij'] < -3)
    ].copy()
    
    print(f"Rows after filtering: {len(df_filtered)}")
    print(f"Demixing index range: {df_filtered['demixing_composition'].min():.3f} to {df_filtered['demixing_composition'].max():.3f}")
    
    return df_filtered


def prepare_panel_H_data(df_merged):
    """Prepare data for Panel H (demixing vs composition bars)."""
    print("\nPreparing data for Panel H (Composition bars)...")
    
    df = df_merged.copy()
    df_filtered = df[
        (df['dGij'] < -3)
    ].copy()
    
    df_filtered['composition_bin'] = df_filtered['composition1'].apply(assign_composition_bin)
    df_filtered = df_filtered[df_filtered['composition_bin'].notna()].copy()
    
    print(f"Rows after filtering: {len(df_filtered)}")
    
    return df_filtered


def create_panels_GH():
    """Create Panels G and H."""
    # Load data
    df_merged = load_and_prepare_data()
    
    # Prepare data for each panel
    df_G = prepare_panel_G_data(df_merged)
    df_H = prepare_panel_H_data(df_merged)
    
    # Create figure with 1x2 layout (aspect ratio 1:3 for each panel)
    fig = plt.figure(figsize=(16, 3))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)
    
    # Bar colors
    bar_colors = ['#cad2c5', '#84a98c', '#52796f', '#354f52', '#2f3e46']
    
    # Panel G: Horizontal box plot of demixing index
    ax = fig.add_subplot(gs[0, 0])
    ax.text(-0.1, 1.12, 'G', transform=ax.transAxes, fontsize=24, 
            fontweight='bold', va='top', ha='right')
    
    demixing_values = df_G['demixing_composition'].dropna()
    
    # Create horizontal box plot
    bp = ax.boxplot([demixing_values], vert=False, widths=0.5, patch_artist=True,
                     boxprops=dict(facecolor='lightgrey', alpha=0.8, linewidth=2),
                     medianprops=dict(color='black', linewidth=2),
                     whiskerprops=dict(linewidth=2),
                     capprops=dict(linewidth=2))
    
    mean_val = demixing_values.mean()
    median_val = demixing_values.median()
    
    ax.set_xlabel('Demixing Index', fontsize=18.0)
    ax.set_yticks([])
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add statistics text box
    stats_text = f'n = {len(demixing_values)}\nMean = {mean_val:.3f}\nMedian = {median_val:.3f}\nStd = {demixing_values.std():.3f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=14,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
    
    # Panel H: Demixing vs composition bar graph
    ax = fig.add_subplot(gs[0, 1])
    ax.text(-0.1, 1.12, 'H', transform=ax.transAxes, fontsize=24, 
            fontweight='bold', va='top', ha='right')
    
    bin_order = ['5-15', '15-35', '50', '65-85', '85-95']
    composition_stats = df_H.groupby('composition_bin')['demixing_composition'].agg(
        ['mean', 'std', 'count']
    ).reindex(bin_order)
    composition_stats['sem'] = composition_stats['std'] / np.sqrt(composition_stats['count'])
    
    x_positions = np.arange(len(bin_order))
    
    # Create bars with different colors
    for i, (pos, color) in enumerate(zip(x_positions, bar_colors)):
        ax.bar(pos, composition_stats['mean'].iloc[i], 
               yerr=composition_stats['sem'].iloc[i],
               capsize=8, alpha=0.8, color=color,
               edgecolor='black', linewidth=2)
    
    # Add sample counts above bars
    for i, (pos, count) in enumerate(zip(x_positions, composition_stats['count'])):
        ax.text(pos, composition_stats['mean'].iloc[i] + composition_stats['sem'].iloc[i] + 0.003,
                f'n={int(count)}', ha='center', va='bottom', fontsize=14)
    
    ax.set_xlabel('IDR-1 Composition Range (%)', fontsize=18.0)
    ax.set_ylabel('Demixing Index', fontsize=18.0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(bin_order, rotation=0)
    ax.set_xlim(-0.6, len(bin_order) - 0.4)
    ax.set_ylim(0, 0.35)
    ax.grid(True, alpha=0.3, axis='y')
    
    print(f"\nPanel H statistics by composition bin:")
    for bin_name, row in composition_stats.iterrows():
        print(f"  {bin_name}: mean={row['mean']:.3f}, sem={row['sem']:.3f}, n={int(row['count'])}")
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('Figure2GH.pdf', bbox_inches='tight')
    
    return fig


if __name__ == '__main__':
    create_panels_GH()
