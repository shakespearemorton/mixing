#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from glob import glob
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks, peak_widths
from scipy.spatial.distance import cdist

plt.rcParams.update({
    'font.size': 16, 'axes.linewidth': 1.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.size': 6, 'xtick.major.width': 1.5, 'ytick.major.size': 6, 'ytick.major.width': 1.5,
    'legend.frameon': False, 'figure.dpi': 300, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.edgecolor': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black',
    'grid.color': 'black', 'text.color': 'black'
})

cmap = plt.get_cmap('tab20b')
all_tab20b_colors = [mcolors.to_hex(cmap(i)) for i in range(cmap.N)]
COLORS = {
    'ratio_0': '#2c0703',  
    'ratio_1': '#890620',  
    'ratio_2': '#b6465f',  
    'ratio_3': '#da9f93',  
    'ratio_4': '#ebd4cb',
    'ratio_5': '#090F1A',  
    'ratio_6': '#2A3A59',  
    'ratio_7': '#5A708C',  
    'ratio_8': '#D9D4BF',  
    'ratio_9': '#E5E3D9'    
}

def get_volume_fractions(active_data):
    """Calculate volume fractions from active data."""
    ATOMS_A3_TO_mM = 1.660539e6
    CHUNK_SIZE = 35
    box = np.asarray([200, 200, 2000])
    n_chunks = np.maximum(np.round(box / CHUNK_SIZE).astype(int), 1)
    chunk_dims = box / n_chunks
    chunk_volume = np.prod(chunk_dims)
    
    fraction_s1 = active_data[:, 4]
    d = chunk_volume * active_data[:, 0] / ATOMS_A3_TO_mM
    
    d1 = d * fraction_s1
    d2 = d * (1 - fraction_s1)
    
    return d,d1,d2

def analyze_single_pair(pair_id, swap, ax_main, color_offset=0):
    """Analyze a single protein pair showing all compositions on one plot."""
    df_samples = pd.read_csv("../DATASETS/demixing.csv")
    df_prep = pd.read_csv("../DATASETS/gin_prep.csv")
    
    pair_row = df_prep[df_prep['seq_name1'] + '_' + df_prep['seq_name2'] == pair_id]
    if pair_row.empty:
        print(f"Pair {pair_id} not found")
        return
    
    s1, s2 = pair_row.iloc[0]['seq_name1'], pair_row.iloc[0]['seq_name2']
    
    xlabel = fr'{s1} (Res. / Voxel)'
    ylabel = fr'{s2} (Res. / Voxel)'
    
    files = glob(f"../Figure1/active_data_final/{pair_id}_*_active_data.npy")
    
    compositions = []
    for file_path in files:
        basename = os.path.basename(file_path)
        comp_str = basename.replace(f"{pair_id}_", "").replace("_active_data.npy", "")
        comp1, comp2 = map(int, comp_str.split('_'))
        if comp1 == 0 or comp2 == 0:
            continue
        
        # Get demixing value for this specific composition
        comp_key = f"{comp1}_{comp2}"
        demix_row = df_samples[(df_samples['protein1'] == s1) & (df_samples['composition1'] == comp1)]
        demix = demix_row.iloc[0]['demixing_composition'] if not demix_row.empty else 'N/A'
        
        data = np.load(file_path)
        d,phi1, phi2 = get_volume_fractions(data)
        
        compositions.append({
            'comp1': comp1,
            'comp2': comp2,
            'phi1': phi1,
            'phi2': phi2,
            'demixing': demix
        })
    
    compositions = sorted(compositions, key=lambda x: x['comp1'])[:5]
    
    if len(compositions) == 0:
        print(f"No valid compositions found for {pair_id}")
        return
    
    # Find global max for axes
    all_phi1 = np.concatenate([comp['phi1'] for comp in compositions])
    all_phi2 = np.concatenate([comp['phi2'] for comp in compositions])
    maxi = max(all_phi1.max(), all_phi2.max()) * 1.1
    
    for idx, comp in enumerate(compositions):
        color = COLORS[f'ratio_{idx + color_offset}']
        
        phi1, phi2 = comp['phi1'], comp['phi2']
        label = f"{comp['comp1']}:{comp['comp2']} (D = {comp['demixing']:.3f})"
        
        # Plot scatter
        if comp['comp1'] != 50:
            a = 0.8
            l = 0.7
        else:
            a = 0.8
            l = .7
        a = 0.7
        l = 0.2
        ax_main.scatter(phi1, phi2, alpha=a, s=40, color=color, 
                       edgecolors='black', linewidth=l, rasterized=True,
                       label=label)
    
    ax_main.set_xlim(0, maxi)
    ax_main.set_ylim(0, maxi)
    ax_main.set_aspect('equal', adjustable='box') 
    ax_main.yaxis.set_major_locator(ax_main.xaxis.get_major_locator())    
    ax_main.plot([0, maxi], [maxi, 0], color='gray', 
                linestyle='--', alpha=0.5, linewidth=2)
    
    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)
    ax_main.legend(loc='best', fontsize=15)
    for x in ax_main.get_xticks():
        if 0 <= x <= maxi:
            ax_main.plot([x, x], [0, max(0, maxi - x)], color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    for y in ax_main.get_yticks():
        if 0 <= y <= maxi:
            ax_main.plot([0, max(0, maxi - y)], [y, y], color='gray', linestyle='--', alpha=0.3, linewidth=0.8)
    
    print(f"\nAnalysis complete for {pair_id}")
    print(f"Analyzed {len(compositions)} compositions")

if __name__ == "__main__":

    pair_id_1 = 'Q3LI60_1_44_Q92581_1_59'
    pair_id_2 = 'O60229_1_34_Q96RR1_1_152'
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 16))
    analyze_single_pair(pair_id_1, 0, ax2, color_offset=0)
    analyze_single_pair(pair_id_2, 0, ax1, color_offset=5)
    
    #plt.tight_layout()
    plt.savefig('Figure2AC.pdf', facecolor='white', dpi=300)
    print(f"\nCombined figure saved as: Figure2AC.pdf")