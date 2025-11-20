#!/usr/bin/env python3
"""
Extended Data Figure 1: Supplementary analysis
Row 1: A (R²), B (c_dil scatter), C (c_dil zoomed), D (dG scatter)
Row 2: E (violin plot spanning full width)
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr

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

# Color scheme
COLORS = {
    'dense': '#e02b35',
    'mid': '#EBECED',
    'dilute': '#2066a8',
    'dG': "#a559aa",
    'dataset1': "#feebe2",
    'dataset2': "#969696"
}


def save_data_to_csv(data_dict, filename):
    """Save data dictionary to CSV in extended_fig1_data directory."""
    output_dir = Path("supp_fig1_data")
    output_dir.mkdir(exist_ok=True)
    
    max_len = max(len(v) if hasattr(v, '__len__') and not isinstance(v, str) else 1 
                  for v in data_dict.values())
    
    for key, val in data_dict.items():
        if not hasattr(val, '__len__') or isinstance(val, str):
            data_dict[key] = [val] * max_len
        elif len(val) < max_len:
            data_dict[key] = list(val) + [np.nan] * (max_len - len(val))
    
    pd.DataFrame(data_dict).to_csv(output_dir / filename, index=False)


def load_calculated_data():
    """Load all calculated density data from results_35 directory."""
    results_dir = Path("../Figure0/results_35")
    data = {}
    for npz_file in results_dir.glob("chunk_density_data_*.npz"):
        dir_name = npz_file.stem.replace("chunk_density_data_", "")
        try:
            with np.load(npz_file) as f:
                data[dir_name] = {
                    'hist_den': float(f['hist_den']),
                    'hist_dil': float(f['hist_dil'])
                }
        except Exception:
            continue
    return data


def r2_for_cube(folder: Path, df_ref: pd.DataFrame):
    """Return (R²_dense, R²_dilute) for folder or None if <2 valid rows."""
    rows = []
    for npz_file in folder.glob('chunk_density_data_*.npz'):
        seq = npz_file.stem.replace('chunk_density_data_', '')
        row = df_ref[df_ref['seq_name'] == seq]
        if row.empty:
            continue
        with np.load(npz_file) as f:
            if {'hist_den', 'hist_dil'} <= set(f.files):
                rows.append({
                    'known_cden': float(row['cden'].iloc[0]),
                    'known_cdil': float(row['cdil'].iloc[0]),
                    'pred_den': float(f['hist_den']),
                    'pred_dil': float(f['hist_dil'])
                })
    
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()
    df_valid = df[df['known_cden'] != 0].copy()
    
    if len(df_valid) < 2:
        return None
    
    r_dense, _ = pearsonr(df_valid['known_cden'], df_valid['pred_den'])
    r_dil, _ = pearsonr(df_valid['known_cdil'], df_valid['pred_dil'])
    return r_dense**2, r_dil**2


def create_extended_figure1():
    """Create Extended Data Figure 1."""
    # Load reference data
    df_training = pd.read_csv('../DATASETS/df_training.csv')
    gin_samples_df = pd.read_csv('../DATASETS/gin_samples.csv')
    
    # Collect R² data
    cube_sizes_list, r2_dense, r2_dil = [], [], []
    result_folders = sorted([p for p in Path('.').glob('../Figure0/results_*') if p.is_dir()])
    
    for folder in result_folders:
        m = re.fullmatch(r'results_(\d+(?:\.\d+)?)', folder.name)
        if not m:
            continue
        size = float(m.group(1))
        r_pair = r2_for_cube(folder, df_training)
        if r_pair is None:
            continue
        cube_sizes_list.append(size)
        r2_dense.append(r_pair[0])
        r2_dil.append(r_pair[1])
    
    cube_sizes = np.array(cube_sizes_list)
    if cube_sizes.size > 0:
        idx = np.argsort(cube_sizes)
        cube_sizes = cube_sizes[idx]
        r2_dense = np.array(r2_dense)[idx]
        r2_dil = np.array(r2_dil)[idx]
    
    # Load scatter plot data
    calc_data = load_calculated_data()
    rows = []
    for name, d in calc_data.items():
        match = df_training[df_training['seq_name'] == name]
        if not match.empty:
            original_cden_val = float(match['cden'].iloc[0])
            cden_was_nan = np.isnan(original_cden_val)
            known_cden_val = 0.0 if cden_was_nan else original_cden_val
            
            rows.append({
                'dir_name': name,
                'known_cden': known_cden_val,
                'known_cdil': match['cdil'].iloc[0],
                'known_dG': match['dG'].iloc[0],
                'hist_den': d['hist_den'],
                'hist_dil': d['hist_dil'],
                'cden_was_nan': cden_was_nan
            })
    
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna(subset=['known_cdil', 'hist_den', 'hist_dil'])
    df['pred_dG'] = np.log(df['hist_dil'] / df['hist_den'])
    df_valid = df[df['cden_was_nan'] == False]
    df_grey = df[df['cden_was_nan'] == True]
    
    df_dG = df[['known_dG', 'pred_dG', 'cden_was_nan']].replace([np.inf, -np.inf], np.nan).dropna()
    df_dG_valid = df_dG[df_dG['cden_was_nan'] == False]
    df_dG_grey = df_dG[df_dG['cden_was_nan'] == True]
    
    # Create figure
    fig = plt.figure(figsize=(28, 14))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.4, height_ratios=[1, 1.2])
    
    # Panel A: R² vs cube size
    ax = fig.add_subplot(gs[0, 0])
    ax.text(-0.1, 1.2, 'A', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    if cube_sizes.size > 0:
        ax.plot(cube_sizes, r2_dense, marker='o', linewidth=3, color=COLORS['dense'],
               label='$c_{den}$', markersize=8)
        ax.plot(cube_sizes, r2_dil, marker='s', linewidth=3, color=COLORS['dilute'],
               label='$c_{dil}$', markersize=8)
        ax.axvline(x=35, color='grey', linestyle='--', linewidth=2, label='35 Å')
        ax.legend()
        save_data_to_csv({'cube_sizes': cube_sizes, 'r2_dense': r2_dense, 'r2_dil': r2_dil},
                        'panelA_r2_data.csv')
    ax.set_xlabel('Voxel Side Length - S (Å)')
    ax.set_ylabel('Comp. w/ Established Methods ($R^{2}$)')
    ax.set_ylim(0.75, 1.05)
    
    # Panel B: Dilute phase scatter
    ax = fig.add_subplot(gs[0, 1])
    ax.text(-0.1, 1.2, 'B', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    ax.scatter(df_grey['known_cdil'], df_grey['hist_dil'], color='grey', s=150, alpha=0.5,
              edgecolors='darkgrey', linewidth=1)
    ax.scatter(df_valid['known_cdil'], df_valid['hist_dil'], color=COLORS['dilute'], s=150,
              alpha=0.7, edgecolors='black', linewidth=1)
    x_all, y_all = df['known_cdil'], df['hist_dil']
    lo, hi = min(x_all.min(), y_all.min()), max(x_all.max(), y_all.max())
    padding = (hi - lo) * 0.05
    lo, hi = lo - padding, hi + padding
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, linewidth=2)
    x_valid, y_valid = df_valid['known_cdil'], df_valid['hist_dil']
    if len(x_valid) >= 2:
        r, _ = stats.pearsonr(x_valid, y_valid)
        ax.text(0.05, 0.95, f'$R^{{2}}$ = {r**2:.3f}', transform=ax.transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel('$c_{dil}$ (Gibbs Dividing Surface) (mM)')
    ax.set_ylabel('$c_{dil}$ (Domain Decomposition) (mM)')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    save_data_to_csv({'known_cdil': df['known_cdil'], 'hist_dil': df['hist_dil'],
                     'is_grey': df['cden_was_nan']}, 'panelB_dilute_scatter.csv')
    
    # Panel C: Dilute phase scatter (zoomed 0-1 mM)
    ax = fig.add_subplot(gs[0, 2])
    ax.text(-0.1, 1.2, 'C', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    df_zoom = df[(df['known_cdil'] <= 1) & (df['hist_dil'] <= 1)]
    df_zoom_valid = df_zoom[df_zoom['cden_was_nan'] == False]
    df_zoom_grey = df_zoom[df_zoom['cden_was_nan'] == True]
    ax.scatter(df_zoom_grey['known_cdil'], df_zoom_grey['hist_dil'], color='grey', s=150,
              alpha=0.5, edgecolors='darkgrey', linewidth=1)
    ax.scatter(df_zoom_valid['known_cdil'], df_zoom_valid['hist_dil'], color=COLORS['dilute'],
              s=150, alpha=0.7, edgecolors='black', linewidth=1)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2)
    if len(df_zoom_valid) >= 2:
        r, _ = stats.pearsonr(df_zoom_valid['known_cdil'], df_zoom_valid['hist_dil'])
        ax.text(0.05, 0.95, f'$R^{{2}}$ = {r**2:.3f}', transform=ax.transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel('$c_{dil}$ (Gibbs Dividing Surface) (mM)')
    ax.set_ylabel('$c_{dil}$ (Domain Decomposition) (mM)')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
    save_data_to_csv({'known_cdil_zoom': df_zoom['known_cdil'], 'hist_dil_zoom': df_zoom['hist_dil'],
                     'is_grey': df_zoom['cden_was_nan']}, 'panelC_dilute_zoom.csv')
    
    # Panel D: dG scatter
    ax = fig.add_subplot(gs[0, 3])
    ax.text(-0.1, 1.2, 'D', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    ax.scatter(df_dG_grey['known_dG'], df_dG_grey['pred_dG'], color='grey', s=150, alpha=0.5,
              edgecolors='darkgrey', linewidth=1)
    ax.scatter(df_dG_valid['known_dG'], df_dG_valid['pred_dG'], color=COLORS['dG'], s=150,
              alpha=0.7, edgecolors='black', linewidth=1)
    ax.plot([0, -10], [0, -10], 'k--', alpha=0.5, linewidth=2)
    if len(df_dG_valid) >= 2:
        r, _ = stats.pearsonr(df_dG_valid['known_dG'], df_dG_valid['pred_dG'])
        ax.text(0.05, 0.95, f'$R^{{2}}$ = {r**2:.3f}', transform=ax.transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel(r'$\Delta G$ (Gibbs Dividing Surface)')
    ax.set_ylabel(r'$\Delta G$ (Domain Decomposition)')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, -10.2)
    ax.set_ylim(0, -10.2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))
    save_data_to_csv({'known_dG': df_dG['known_dG'], 'pred_dG': df_dG['pred_dG'],
                     'is_grey': df_dG['cden_was_nan']}, 'panelD_dG_scatter.csv')
    
    # Panel E: Violin plot (spanning full width of row 2)
    ax = fig.add_subplot(gs[1, :])
    ax.text(-0.01, 1.2, 'E', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    
    # Parameters to plot
    params = ['N', 'mean_lambda', 'dG', 'faro', 'fcr', 'ncpr']
    param_ylabels = {
        'N': 'N',
        'mean_lambda': r'$\langle\lambda\rangle$',
        'dG': r'$\Delta G$',
        'faro': r'$f_{\mathrm{aro}}$',
        'fcr': 'FCR',
        'ncpr': 'NCPR',
    }
    param_titles = {
        'N': 'Sequence Length',
        'mean_lambda': 'Mean Residue\nHydrophobicity',
        'faro': 'Fraction Aromatic\nResidues',
        'dG': 'Transfer Free Energy',
        'fcr': 'Fraction Charged\nResidues',
        'ncpr': 'Net Charge Per\nResidue',
    }
    
    palette = {
        'Our Dataset': COLORS['dataset1'],
        'von Bülow Dataset': COLORS['dataset2']
    }
    
    # Create subplots for violin plots
    ax.set_axis_off()
    gs_violin = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=ax.get_subplotspec(), wspace=0.5)
    
    for idx, param in enumerate(params):
        ax_violin = fig.add_subplot(gs_violin[0, idx])
        
        # Prepare data
        our_data = pd.DataFrame({'value': gin_samples_df[param], 'source': 'Our Dataset'})
        larger_data = pd.DataFrame({'value': df_training[param], 'source': 'von Bülow Dataset'})
        combined_data = pd.concat([our_data, larger_data], ignore_index=True)
        
        # Create violin plot
        sns.violinplot(x='source', y='value', data=combined_data, ax=ax_violin, palette=palette,
                      inner='quartile', linewidth=1.5, hue='source', split=True, legend=False,
                      density_norm='width', linecolor='black')
        
        # Add range lines
        param_min = df_training[param].min()
        param_max = df_training[param].max()
        ax_violin.axhline(y=param_min, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
        ax_violin.axhline(y=param_max, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
        
        ax_violin.set_ylabel(param_ylabels[param], fontsize=18)
        ax_violin.set_xlabel(param_titles[param], fontsize=18)
        ax_violin.set_xticks([])
        
        for spine in ax_violin.spines.values():
            spine.set_linewidth(0.8)
            spine.set_visible(True)
        ax_violin.spines['top'].set_visible(False)
        ax_violin.spines['right'].set_visible(False)
        
        # Save data for this parameter
        save_data_to_csv({
            'our_dataset': gin_samples_df[param].values,
            'von bulow_dataset': df_training[param].values
        }, f'panelE_violin_{param}.csv')
    
    # Create legend
    import matplotlib.patches as mpatches
    legend_handles = [mpatches.Patch(color=color, label=label) for label, color in palette.items()]
    fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.52),
              ncol=2, frameon=False, fontsize=18)
    
    plt.tight_layout()
    plt.savefig('Supplementary_Figure1.pdf', bbox_inches='tight')
    print("Extended Data Figure 1 saved as Supplementary_Figure1.pdf")
    print("All data saved in extended_fig1_data/ directory")


if __name__ == '__main__':
    create_extended_figure1()