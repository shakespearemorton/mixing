#!/usr/bin/env python3
"""
Figure 1: Main manuscript figure with 2x4 layout (A-H)
A: Composite plot (spatial map + KDE)
B: R² vs voxel side length
C: c_den scatter
D: c_dil scatter
E: Pair 1 probability densities
F: Pair 1 scatter + marginal KDEs
G: Pair 2 probability densities
H: Pair 2 scatter + marginal KDEs
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr, gaussian_kde
from scipy.signal import find_peaks, peak_widths
from matplotlib.colors import LinearSegmentedColormap
from glob import glob
import os

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
    'comp1': '#a559aa',
    'comp2': '#59a89c',
    'compalt': '#A89759'
}

try:
    import MDAnalysis as mda
    MDANALYSIS_AVAILABLE = True
except ImportError:
    MDANALYSIS_AVAILABLE = False
    print("Warning: MDAnalysis not found. Panel A will be skipped.")


def save_data_to_csv(data_dict, filename):
    """Save data dictionary to CSV in figure1_data directory."""
    output_dir = Path("figure1_data")
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


def plot_composite_subplot_A(fig, main_ax):
    """Panel A: Composite spatial map and KDE plot."""
    DATA_FILE = '../Figure0/results_35/chunk_density_data_Q3LI60_1_44.npz'
    DIRECTORY = '../Figure0/Q3LI60_1_44'
    CHUNK_SIZE = 35
    ATOMS_A3_TO_mM = 1.660539e6

    root = Path.cwd()
    direc = root / DIRECTORY
    pdb = direc / "top.pdb"
    dcd_files = list(direc.glob("*.dcd"))

    if not dcd_files:
        raise FileNotFoundError(f"No DCD files found in {direc}")

    u = mda.Universe(str(pdb), str(dcd_files[0]))
    u.trajectory[-1]
    
    box = u.dimensions[:3]
    n_chunks = np.maximum(np.round(box / CHUNK_SIZE).astype(int), 1)
    chunk_dims_calc = box / n_chunks
    nx, ny, nz = n_chunks

    pos = u.atoms.positions
    idx = np.clip(np.floor_divide(pos, chunk_dims_calc).astype(int), 0, n_chunks - 1)

    counts_3d = np.zeros(n_chunks, dtype=np.float64)
    for atom_idx in range(len(pos)):
        i, j, k = idx[atom_idx]
        counts_3d[i, j, k] += 1

    chunk_volume = np.prod(chunk_dims_calc)
    concentration_zx = np.mean(counts_3d, axis=1).T

    data = np.load(DATA_FILE)
    x_range = data['x_range']
    den_data = (data['active_data'] * chunk_volume * len(u.segments[0].atoms)) / ATOMS_A3_TO_mM
    den_data = den_data.flatten()
    kde = gaussian_kde(den_data)
    kde_y = kde(x_range)
    peaks, _ = find_peaks(kde_y, height=1e-8)

    peak_xs = x_range[peaks]
    c_dilute = peak_xs[0]
    c_dense = peak_xs[-1] if len(peaks) > 1 else peak_xs[0]

    cmap = LinearSegmentedColormap.from_list('custom', [
        (0.0, COLORS['dilute']), (0.15, COLORS['mid']), (1.0, COLORS['dense'])
    ])

    main_ax.set_axis_off()
    gs_sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=main_ax.get_subplotspec(),
                                              height_ratios=[3, 1], hspace=0.0)
    
    # Spatial map
    ax_map = fig.add_subplot(gs_sub[0])
    extent = [0, box[2], 0, box[0]]
    ax_map.imshow(concentration_zx.T, origin='lower', extent=extent, aspect='equal',
                  cmap=cmap, vmin=x_range.min(), vmax=x_range.max())

    for i in range(nz + 1):
        ax_map.axvline(i * chunk_dims_calc[2], color='black', linestyle=':', linewidth=0.5, alpha=0.5)
    for i in range(nx + 1):
        ax_map.axhline(i * chunk_dims_calc[0], color='black', linestyle=':', linewidth=0.5, alpha=0.5)

    ax_map.set_xlabel('Z (Å)')
    ax_map.set_ylabel('X (Å)')

    # KDE
    ax_kde = fig.add_subplot(gs_sub[1])
    for i in range(len(x_range)-1):
        x_segment = x_range[i:i+2]
        y_segment = kde_y[i:i+2]
        norm_color = (x_range[i] - x_range.min()) / (x_range.max() - x_range.min())
        ax_kde.fill_between(x_segment, 0, y_segment, color=cmap(norm_color), alpha=0.6)

    ax_kde.plot(x_range, kde_y, 'k-', linewidth=1.5)
    ax_kde.axvline(c_dilute, color=COLORS['dilute'], linestyle='--', linewidth=1.5, alpha=0.8)
    if len(peaks) > 1:
        ax_kde.axvline(c_dense, color=COLORS['dense'], linestyle='--', linewidth=1.5, alpha=0.8)

    y_max = kde_y.max()
    ax_kde.text(c_dilute, y_max * 1.05, f'{c_dilute:.1f}', ha='center', va='bottom',
                color=COLORS['dilute'], fontweight='bold')
    if len(peaks) > 1:
        ax_kde.text(c_dense, y_max * 1.05, f'{c_dense:.1f}', ha='center', va='bottom',
                    color=COLORS['dense'], fontweight='bold')

    ax_kde.set_xlabel('Concentration (Res./Voxel)')
    ax_kde.set_ylabel('Prob. Density')
    
    # Save data
    save_data_to_csv({
        'concentration_zx_flat': concentration_zx.flatten(),
        'kde_x': x_range,
        'kde_y': kde_y,
        'c_dilute': c_dilute,
        'c_dense': c_dense
    }, 'panelA_composite_data.csv')


def plot_pair_composite(fig, main_ax, pair_id, comp_label, panel_colors):
    """Plot probability densities for a pair (panels E, G)."""
    DATA_FILE = f"../Figure1/active_data_final/{pair_id}_{comp_label}_active_data.npy"
    ATOMS_A3_TO_mM = 1.660539e6
    CHUNK_SIZE = 35
    box = np.asarray([200, 200, 2000])
    n_chunks = np.maximum(np.round(box / CHUNK_SIZE).astype(int), 1)
    chunk_volume = np.prod(box / n_chunks)
    
    active_data = np.load(DATA_FILE)
    fraction_s1 = active_data[:, 4]
    densities = chunk_volume * active_data[:, 0] / ATOMS_A3_TO_mM
    
    densities_s1 = densities * fraction_s1
    densities_s2 = densities * (1 - fraction_s1)
    
    x_range = np.linspace(0, densities.max(), 500)
    
    kde_combined = gaussian_kde(densities, bw_method='scott')
    kde_s1 = gaussian_kde(densities_s1, bw_method='scott')
    kde_s2 = gaussian_kde(densities_s2, bw_method='scott')
    
    kde_y_combined = kde_combined(x_range)
    kde_y_s1 = kde_s1(x_range)
    kde_y_s2 = kde_s2(x_range)
    
    peaks, _ = find_peaks(kde_y_combined, height=6e-8)
    peaks = sorted(peaks, key=lambda p: kde_y_combined[p], reverse=True)
    main_ax.set_axis_off()
    gs_sub = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=main_ax.get_subplotspec(), hspace=0.4)

    kde_data = [
        (kde_y_combined, None, 'k', 'Combined'),
        (kde_y_s1, panel_colors[0], panel_colors[0], 'IDR-1'),
        (kde_y_s2, panel_colors[1], panel_colors[1], 'IDR-2')
    ]
    
    for i, (kde_y, fill_color, line_color, label) in enumerate(kde_data):
        ax_kde = fig.add_subplot(gs_sub[i])
        
        if fill_color:
            ax_kde.fill_between(x_range, 0, kde_y, color=fill_color, alpha=0.6)
        ax_kde.plot(x_range, kde_y, color=line_color, linewidth=1.5)
        
        if i == 0:
            y_max = kde_y.max()
            for peak in peaks:
                if abs(x_range[peak] - 37) < 1:
                    continue
                ax_kde.axvline(x_range[peak], color='k', linestyle='--', linewidth=1.5, alpha=0.8)
                ax_kde.text(x_range[peak], y_max * 1.05, f'{x_range[peak]:.1f}',
                           ha='center', va='bottom', color='k', fontweight='bold')
        ax_kde.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
        ax_kde.set_ylabel(label, fontsize=15, rotation=90, labelpad=20, va='center')
        if i == 2:
            ax_kde.set_xlabel('Concentration (Res./Voxel)')

    return densities, densities_s1, densities_s2


def plot_pair_scatter(fig, main_ax, d, d1, d2, s1, s2, panel_colors):
    """Plot scatter with marginal KDEs (panels F, H)."""
    xlabel = f'{s1} (Res./Voxel)'
    ylabel = f'{s2} (Res./Voxel)'
    maxi = np.max(d)
    
    main_ax.set_axis_off()
    gs_sub = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=main_ax.get_subplotspec(),
                                             hspace=0.05, wspace=0.05,
                                             height_ratios=[0.5, 1, 1, 1],
                                             width_ratios=[1, 1, 1, 0.5])
    
    ax_main = fig.add_subplot(gs_sub[1:, :-1])
    ax_main.scatter(d1, d2, alpha=0.4, s=40, color='grey', edgecolors='black',
                   linewidth=0.5, rasterized=True)
    ax_main.set_aspect('equal', adjustable='box')
    ax_main.set_xlim(0, maxi)
    ax_main.set_ylim(0, maxi)
    ax_main.set_yticks(ax_main.get_xticks())
    
    for x in ax_main.get_xticks():
        if 0 <= x <= maxi:
            ax_main.plot([x, x], [0, max(0, maxi - x)], color='gray', linestyle='--',
                        alpha=0.3, linewidth=0.8)
    for y in ax_main.get_yticks():
        if 0 <= y <= maxi:
            ax_main.plot([0, max(0, maxi - y)], [y, y], color='gray', linestyle='--',
                        alpha=0.3, linewidth=0.8)
    
    ax_main.plot([0, maxi], [maxi, 0], color='black', linestyle='--', alpha=0.5,
                linewidth=2, zorder=10)
    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)
    
    # Top KDE
    ax_top = fig.add_subplot(gs_sub[0, :-1], sharex=ax_main)
    kde1 = gaussian_kde(d1)
    x_range = np.linspace(0, maxi, 200)
    kde1_values = kde1(x_range)
    ax_top.fill_between(x_range, kde1_values, alpha=0.5, color=panel_colors[0])
    ax_top.plot(x_range, kde1_values, color=panel_colors[0], linewidth=2)
    ax_top.set_xlim(0, maxi)
    ax_top.axis('off')
    
    # Right KDE
    ax_right = fig.add_subplot(gs_sub[1:, -1], sharey=ax_main)
    kde2 = gaussian_kde(d2)
    y_range = np.linspace(0, maxi, 200)
    kde2_values = kde2(y_range)
    ax_right.fill_betweenx(y_range, kde2_values, alpha=0.5, color=panel_colors[1])
    ax_right.plot(kde2_values, y_range, color=panel_colors[1], linewidth=2)
    ax_right.set_ylim(0, maxi)
    ax_right.axis('off')


def create_figure1():
    """Create main Figure 1 with 2x4 layout."""
    # Load reference data
    df_training = pd.read_csv('../DATASETS/df_training.csv')
    
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
                'hist_den': d['hist_den'],
                'hist_dil': d['hist_dil'],
                'cden_was_nan': cden_was_nan
            })
    
    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna(subset=['known_cdil', 'hist_den', 'hist_dil'])
    df_valid = df[df['cden_was_nan'] == False]
    df_grey = df[df['cden_was_nan'] == True]
    
    # Create figure
    fig = plt.figure(figsize=(30, 14))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.4)
    
    # Panel A: Composite plot
    ax = fig.add_subplot(gs[0, 0])
    ax.text(-0.1, 1.2, 'A', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    if MDANALYSIS_AVAILABLE:
        plot_composite_subplot_A(fig, ax)
    
    # Panel B: R² vs cube size
    ax = fig.add_subplot(gs[0, 1])
    ax.text(-0.1, 1.2, 'B', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    if cube_sizes.size > 0:
        ax.plot(cube_sizes, r2_dense, marker='o', linewidth=3, color=COLORS['dense'],
               label='$c_{den}$', markersize=8)
        ax.plot(cube_sizes, r2_dil, marker='s', linewidth=3, color=COLORS['dilute'],
               label='$c_{dil}$', markersize=8)
        ax.axvline(x=35, color='grey', linestyle='--', linewidth=2, label='35 Å')
        ax.legend()
        save_data_to_csv({'cube_sizes': cube_sizes, 'r2_dense': r2_dense, 'r2_dil': r2_dil},
                        'panelB_r2_data.csv')
    ax.set_xlabel('Voxel Side Length - S (Å)')
    ax.set_ylabel('Comp. w/ Established Methods ($R^{2}$)')
    ax.set_ylim(0.75, 1.05)
    
    # Panel C: Dense phase scatter
    ax = fig.add_subplot(gs[0, 2])
    ax.text(-0.1, 1.2, 'C', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    ax.scatter(df_grey['known_cden'], df_grey['hist_den'], color='grey', s=150, alpha=0.5,
              edgecolors='darkgrey', linewidth=1)
    ax.scatter(df_valid['known_cden'], df_valid['hist_den'], color=COLORS['dense'], s=150,
              alpha=0.7, edgecolors='black', linewidth=1)
    x_all, y_all = df['known_cden'], df['hist_den']
    lo, hi = min(x_all.min(), y_all.min()), max(x_all.max(), y_all.max())
    padding = (hi - lo) * 0.05
    lo, hi = lo - padding, hi + padding
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, linewidth=2)
    x_valid, y_valid = df_valid['known_cden'], df_valid['hist_den']
    if len(x_valid) >= 2:
        r, _ = stats.pearsonr(x_valid, y_valid)
        ax.text(0.05, 0.95, f'$R^{{2}}$ = {r**2:.3f}', transform=ax.transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel('$c_{den}$ (Gibbs Dividing Surface) (mM)')
    ax.set_ylabel('$c_{den}$ (Domain Decomposition) (mM)')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    save_data_to_csv({'known_cden': df['known_cden'], 'hist_den': df['hist_den'],
                     'is_grey': df['cden_was_nan']}, 'panelC_dense_scatter.csv')
    
    # Panel D: Dilute phase scatter
    ax = fig.add_subplot(gs[0, 3])
    ax.text(-0.1, 1.2, 'D', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
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
                     'is_grey': df['cden_was_nan']}, 'panelD_dilute_scatter.csv')
    
    # Panels E-F: Pair 1 (Q96K19_200_258_Q3LI60_1_44)
    pair1_id = 'Q96K19_200_258_Q3LI60_1_44'
    df_prep = pd.read_csv("../DATASETS/gin_prep.csv")
    pair_row = df_prep[df_prep['seq_name1'] + '_' + df_prep['seq_name2'] == pair1_id].iloc[0]
    s1_p1, s2_p1 = pair_row['seq_name1'], pair_row['seq_name2']
    
    # Find 50/50 composition
    files = glob(f"../Figure1/active_data_final/{pair1_id}_*_active_data.npy")
    compositions = []
    for file_path in files:
        basename = os.path.basename(file_path)
        comp_str = basename.replace(f"{pair1_id}_", "").replace("_active_data.npy", "")
        comp1, comp2 = map(int, comp_str.split('_'))
        compositions.append({'comp1': comp1, 'comp2': comp2})
    closest_idx = min(range(len(compositions)), 
                     key=lambda i: abs(compositions[i]['comp1'] / (compositions[i]['comp1'] + compositions[i]['comp2']) - 0.5))
    closest_comp = compositions[closest_idx]
    comp_label_p1 = f"{closest_comp['comp1']}_{closest_comp['comp2']}"
    
    # Panel E
    ax = fig.add_subplot(gs[1, 0])
    ax.text(-0.1, 1.2, 'E', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    d_p1, d1_p1, d2_p1 = plot_pair_composite(fig, ax, pair1_id, comp_label_p1,
                                             [COLORS['compalt'], COLORS['comp2']])
    
    # Panel F
    ax = fig.add_subplot(gs[1, 1])
    ax.text(-0.1, 1.2, 'F', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    plot_pair_scatter(fig, ax, d_p1, d1_p1, d2_p1, s1_p1, s2_p1,
                     [COLORS['compalt'], COLORS['comp2']])
    save_data_to_csv({'densities_total': d_p1, 'densities_s1': d1_p1, 'densities_s2': d2_p1},
                    f'panelEF_{pair1_id}_data.csv')
    
    # Panels G-H: Pair 2 (Q8WTT2_1_202_Q3LI60_1_44)
    pair2_id = 'Q8WTT2_1_202_Q3LI60_1_44'
    pair_row = df_prep[df_prep['seq_name1'] + '_' + df_prep['seq_name2'] == pair2_id].iloc[0]
    s1_p2, s2_p2 = pair_row['seq_name1'], pair_row['seq_name2']
    
    files = glob(f"../Figure1/active_data_final/{pair2_id}_*_active_data.npy")
    compositions = []
    for file_path in files:
        basename = os.path.basename(file_path)
        comp_str = basename.replace(f"{pair2_id}_", "").replace("_active_data.npy", "")
        comp1, comp2 = map(int, comp_str.split('_'))
        compositions.append({'comp1': comp1, 'comp2': comp2})
    closest_idx = min(range(len(compositions)), 
                     key=lambda i: abs(compositions[i]['comp1'] / (compositions[i]['comp1'] + compositions[i]['comp2']) - 0.5))
    closest_comp = compositions[closest_idx]
    comp_label_p2 = f"{closest_comp['comp1']}_{closest_comp['comp2']}"
    
    # Panel G
    ax = fig.add_subplot(gs[1, 2])
    ax.text(-0.1, 1.2, 'G', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    d_p2, d1_p2, d2_p2 = plot_pair_composite(fig, ax, pair2_id, comp_label_p2,
                                             [COLORS['comp1'], COLORS['comp2']])
    
    # Panel H
    ax = fig.add_subplot(gs[1, 3])
    ax.text(-0.1, 1.2, 'H', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    plot_pair_scatter(fig, ax, d_p2, d1_p2, d2_p2, s1_p2, s2_p2,
                     [COLORS['comp1'], COLORS['comp2']])
    save_data_to_csv({'densities_total': d_p2, 'densities_s1': d1_p2, 'densities_s2': d2_p2},
                    f'panelGH_{pair2_id}_data.csv')
    
    plt.tight_layout()
    plt.savefig('Figure1.pdf', bbox_inches='tight')
    print("Figure 1 saved as Figure1.pdf")
    print("All data saved in figure1_data/ directory")


if __name__ == '__main__':
    create_figure1()