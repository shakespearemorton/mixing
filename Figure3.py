#!/usr/bin/env python3
"""
Figure 3: Chemical grammar and demixing analysis (2x3 layout A-F)
A: Logistic regression model (scatter + logistic curve)
B: ROC curve
C: Demixing index by interaction type (vertical boxplot)
D: Heterotypic dG vs |dG_ii - dG_jj| (difference)
E: Heterotypic dG vs (dG_ii + dG_jj)/2 (average)
F: Heterotypic dG vs |Net Charge| (colored by charged residues)
"""
import re
import numpy as np
import pandas as pd
import colormaps as cmaps
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr, linregress
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve
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

# Color scheme
COLORS = {
    'dense': '#e02b35',
    'dilute': '#2066a8',
    'model': 'darkred',
    'scatter': 'grey'
}

# Charged residues
POSITIVE_AA = set('KR')
NEGATIVE_AA = set('DE')

# GIN Category Definitions for Panel C
GIN_CATEGORY_MAP = {
    # Strongly Charged (tracts, blocks, high net charge)
    7: 'C', 9: 'C', 18: 'C',
    19: 'C', 23: 'C', 25: 'C',
    26: 'C', 3: 'C', 8: 'C',
    17: 'C', 24: 'C', 29: 'C',
    # Weakly Charged / Polar (patches, weak net charge, polar residues)
    0: 'P', 1: 'P', 2: 'P',
    4: 'P', 5: 'P', 6: 'P',
    11: 'P', 12: 'P', 13: 'P',
    14: 'P', 15: 'P', 16: 'P',
    21: 'P', 22: 'P', 27: 'P',
    # Hydrophobic / Aromatic (enriched in M, A, Y)
    10: 'H', 20: 'H', 28: 'H'
}


def save_filtering_criteria(panel_name, criteria_dict, filename):
    """Save filtering criteria to text file."""
    output_dir = Path("figure3_data")
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / filename
    with open(filepath, 'w') as f:
        f.write(f"Filtering Criteria for Panel {panel_name}\n")
        f.write("=" * 60 + "\n\n")
        for key, value in criteria_dict.items():
            f.write(f"{key}: {value}\n")


def save_data_to_csv(data_dict, filename):
    """Save data dictionary to CSV in figure3_data directory."""
    output_dir = Path("figure3_data")
    output_dir.mkdir(exist_ok=True)
    
    max_len = max(len(v) if hasattr(v, '__len__') and not isinstance(v, str) else 1 
                  for v in data_dict.values())
    
    for key, val in data_dict.items():
        if not hasattr(val, '__len__') or isinstance(val, str):
            data_dict[key] = [val] * max_len
        elif len(val) < max_len:
            data_dict[key] = list(val) + [np.nan] * (max_len - len(val))
    
    pd.DataFrame(data_dict).to_csv(output_dir / filename, index=False)


def calculate_net_charge(sequence):
    """Calculate net charge of a protein sequence."""
    if not isinstance(sequence, str):
        return 0
    seq_upper = sequence.upper()
    positive = sum(1 for aa in seq_upper if aa in POSITIVE_AA)
    negative = sum(1 for aa in seq_upper if aa in NEGATIVE_AA)
    return positive - negative


def calculate_total_charged_residues(sequence):
    """Calculate total number of charged residues."""
    if not isinstance(sequence, str):
        return 0
    seq_upper = sequence.upper()
    positive = sum(1 for aa in seq_upper if aa in POSITIVE_AA)
    negative = sum(1 for aa in seq_upper if aa in NEGATIVE_AA)
    return positive + negative


def get_gin_category(gin_value):
    """Map a GIN number to its category."""
    if pd.isna(gin_value):
        return 'Other'
    return GIN_CATEGORY_MAP.get(int(gin_value), 'Other')


def get_interaction_type(gin1, gin2):
    """Get the interaction type from two GIN values."""
    if pd.isna(gin1) or pd.isna(gin2):
        return 'Other'
    cat1 = get_gin_category(gin1)
    cat2 = get_gin_category(gin2)
    if cat1 == 'Other' or cat2 == 'Other':
        return 'Other'
    # Sort to make (A,B) the same as (B,A)
    interaction = '-'.join(sorted([cat1, cat2]))
    return interaction


def load_and_prepare_data():
    """Load all required datasets and merge."""
    print("Loading datasets...")
    df_dem = pd.read_csv("../DATASETS/demixing.csv")
    df_tie = pd.read_csv("../DATASETS/tie_lines.csv")
    df_prot = pd.read_csv("../DATASETS/gin_samples.csv")
    df_gin = pd.read_csv("../DATASETS/gin_samples.csv", usecols=['seq_name', 'fasta'])
    
    print(f"Loaded {len(df_dem)} demixing entries")
    print(f"Loaded {len(df_tie)} tie line entries")
    
    # Calculate additional protein features
    df_prot['n_charged'] = df_prot['fasta'].apply(
        lambda s: sum(1 for aa in s if aa in 'KRED') if pd.notna(s) else np.nan
    )
    df_prot['net_charge_unweighted'] = df_prot['fasta'].apply(
        lambda s: sum(1 for aa in s if aa in 'KR') - sum(1 for aa in s if aa in 'DE') 
        if pd.notna(s) else np.nan
    )
    
    # Merge demixing with tie lines
    df_merged = pd.merge(
        df_dem,
        df_tie[['protein1', 'protein2', 'composition1', 'composition2', 'phase_behavior']],
        on=['protein1', 'protein2', 'composition1', 'composition2'],
        how='left'
    )
    
    # Merge with protein properties
    cols = ['seq_name', 'faro', 'ah_ij', 'net_charge_unweighted', 'N', 'shd','mean_lambda','SPR_svr']
    for p, suffix in [('protein1', '_p1'), ('protein2', '_p2')]:
        df_merged = pd.merge(
            df_merged, df_prot[cols], 
            left_on=p, right_on='seq_name', 
            how='left', suffixes=('', '_temp')
        )
        df_merged = df_merged.drop(columns=['seq_name']).rename(
            columns={c: c+suffix for c in cols[1:]}
        )
    
    # Add sequences for charge calculation
    seq_lookup = dict(zip(df_gin['seq_name'], df_gin['fasta']))
    df_merged['seq1'] = df_merged['protein1'].map(seq_lookup)
    df_merged['seq2'] = df_merged['protein2'].map(seq_lookup)
    
    return df_merged


def prepare_panel_AB_data(df_merged):
    """Prepare data for Panels A & B (logistic regression)."""
    print("\nPreparing data for Panels A & B (Logistic Regression)...")
    
    df = df_merged.copy()
    df['demixed'] = (df['demixing_composition'] > 0.5).astype(int)
    df['demixing_composition_color'] = df['demixing_composition']
    df['net_charge_weighted_diff'] = abs(
        df['nuse1'] * df['net_charge_unweighted_p1'] - 
        df['nuse2'] * df['net_charge_unweighted_p2']
    )
    df['faro_abs_diff'] = abs(df['faro_p1'] - df['faro_p2'])
    df['mean_lambda_sum'] = (df['mean_lambda_p1'] + df['mean_lambda_p2']) / 2
    df['SPR_svr_sum'] = (df['SPR_svr_p1'] + df['SPR_svr_p2'])
    
    # Apply filters
    df_filtered = df[
        (df['dGij'] < -3)
    ].copy()
    
    X = df_filtered[['net_charge_weighted_diff', 'faro_abs_diff', 'mean_lambda_sum', 'SPR_svr_sum']].dropna()
    y = df_filtered['demixed'].loc[X.index]
    colors = df_filtered['demixing_composition_color'].loc[X.index]
    
    # Save filtering criteria
    save_filtering_criteria('A & B', {
        'Dataset': 'demixing.csv merged with tie_lines.csv and gin_samples.csv',
        'dGij filter': '< -3 kBT',
        'Features used':'net_charge_weighted_diff, faro_abs_diff, mean_lambda_sum, SPR_svr',
        'Binary threshold': 'demixing_index > 0.5',
        'Rows after filtering': len(X),
        'Phase behavior filter': 'None (all phases included)'
    }, 'panelAB_filtering_criteria.txt')
    
    print(f"Rows after filtering: {len(X)}")
    print(f"Max demixing index: {df_filtered['demixing_composition'].max():.3f}")
    
    return X, y, colors, df_filtered


def prepare_panel_C_data(df_merged):
    """Prepare data for Panel C (demixing by interaction type)."""
    print("\nPreparing data for Panel C (Demixing by Interaction Type)...")
    
    df = df_merged.copy()
    
    # Filter for mixed phase behavior only
    df_filtered = df[
        (df['composition1'] < 95) & 
        (df['composition2'] < 95) & 
        (df['dGij'] < -3)
    ].copy()
    
    # Categorize interactions
    df_filtered['interaction_type'] = df_filtered.apply(
        lambda row: get_interaction_type(row['gin1'], row['gin2']), 
        axis=1
    )
    
    # Remove 'Other' category
    df_filtered = df_filtered[df_filtered['interaction_type'] != 'Other']
    
    # Remove duplicate protein pairs
    df_filtered.drop_duplicates(
        subset=['protein1', 'protein2', 'composition1'], 
        keep='first', 
        inplace=True
    )
    
    # Save filtering criteria
    save_filtering_criteria('C', {
        'Dataset': 'demixing.csv merged with tie_lines.csv',
        'Composition1 filter': '< 95%',
        'Composition2 filter': '< 95%',
        'dGij filter': '< -3 kBT',
        'Phase behavior filter': 'mixed only',
        'Interaction types': 'Based on GIN categories (C=Charged, P=Polar, H=Hydrophobic)',
        'Rows after filtering': len(df_filtered),
        'Unique interaction types': df_filtered['interaction_type'].nunique()
    }, 'panelC_filtering_criteria.txt')
    
    print(f"Rows after filtering: {len(df_filtered)}")
    print(f"Interaction type counts:\n{df_filtered['interaction_type'].value_counts().sort_index()}")
    
    return df_filtered


def prepare_panel_DE_data(df_merged):
    """Prepare data for Panels D & E (dG relationships)."""
    print("\nPreparing data for Panels D & E (dG relationships)...")
    
    df = df_merged.copy()
    df_filtered = df[
        (df['composition1'] < 95) & 
        (df['composition2'] < 95) &
        (df['demixing_composition'] < 0.5)
    ].copy()
    
    df_filtered['dG_deviation'] = abs(df_filtered['dG1'] - df_filtered['dG2'])
    df_filtered['average_dG'] = (df_filtered['dG1'] + df_filtered['dG2']) / 2
    df_filtered = df_filtered.dropna(subset=['dGij', 'dG_deviation', 'average_dG'])
    
    # Save filtering criteria
    save_filtering_criteria('D & E', {
        'Dataset': 'demixing.csv merged with tie_lines.csv',
        'Composition1 filter': '< 95%',
        'Composition2 filter': '< 95%',
        'Phase behavior filter': 'mixed only',
        'dG_deviation': '|dG1 - dG2|',
        'average_dG': '(dG1 + dG2) / 2',
        'Rows after filtering': len(df_filtered)
    }, 'panelDE_filtering_criteria.txt')
    
    print(f"Rows after filtering: {len(df_filtered)}")
    
    return df_filtered


def prepare_panel_F_data(df_merged):
    """Prepare data for Panel F (charge vs dGij)."""
    print("\nPreparing data for Panel F (Charge vs dGij)...")
    
    df = df_merged.copy()
    df_filtered = df[
        (df['composition1'] < 95) & 
        (df['composition2'] < 95) &
        (df['demixing_composition'] < 0.5)
    ].copy()
    
    # Remove rows with missing sequences
    df_filtered = df_filtered.dropna(subset=['seq1', 'seq2', 'nuse1', 'nuse2', 'dGij'])
    
    # Calculate charges
    df_filtered['net_charge1'] = df_filtered['seq1'].apply(calculate_net_charge) * df_filtered['nuse1']
    df_filtered['net_charge2'] = df_filtered['seq2'].apply(calculate_net_charge) * df_filtered['nuse2']
    df_filtered['charged_residues1'] = df_filtered['seq1'].apply(calculate_total_charged_residues) * df_filtered['nuse1']
    df_filtered['charged_residues2'] = df_filtered['seq2'].apply(calculate_total_charged_residues) * df_filtered['nuse2']
    
    df_filtered['pair_id'] = df_filtered['protein1'] + '_' + df_filtered['protein2']
    df_filtered['total_net_charge'] = df_filtered['net_charge1'] + df_filtered['net_charge2']
    df_filtered['abs_total_net_charge'] = df_filtered['total_net_charge'].abs()
    df_filtered['total_charged_residues'] = df_filtered['charged_residues1'] + df_filtered['charged_residues2']
    
    # Filter pairs with at least one dGij < -3
    pairs_to_keep = df_filtered[df_filtered['dGij'] < -3]['pair_id'].unique()
    df_filtered = df_filtered[df_filtered['pair_id'].isin(pairs_to_keep)]
    
    # Remove single-point pairs
    pair_counts = df_filtered['pair_id'].value_counts()
    single_point_pairs = pair_counts[pair_counts == 1].index
    df_filtered = df_filtered[~df_filtered['pair_id'].isin(single_point_pairs)]
    
    # Save filtering criteria
    save_filtering_criteria('F', {
        'Dataset': 'demixing.csv merged with tie_lines.csv and gin_samples.csv',
        'Composition1 filter': '< 95%',
        'Composition2 filter': '< 95%',
        'Phase behavior filter': 'mixed only',
        'Pair filter': 'at least one dGij < -3 kBT',
        'Single-point pairs': 'removed',
        'Unique pairs': df_filtered['pair_id'].nunique(),
        'Rows after filtering': len(df_filtered)
    }, 'panelF_filtering_criteria.txt')
    
    print(f"Rows after filtering: {len(df_filtered)}")
    print(f"Unique pairs: {df_filtered['pair_id'].nunique()}")
    
    return df_filtered


def train_logistic_model(X, y):
    """Train logistic regression model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(class_weight='balanced', random_state=42))
    ])
    
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    print(f"Model trained: AUC = {auc:.3f}")
    return model, X_train, X_test, y_train, y_test, auc


def create_figure3():
    """Create Figure 3 with all panels."""
    # Load data
    df_merged = load_and_prepare_data()
    
    # Prepare data for each panel set
    X, y, colors, df_AB = prepare_panel_AB_data(df_merged)
    df_C = prepare_panel_C_data(df_merged)
    df_DE = prepare_panel_DE_data(df_merged)
    df_F = prepare_panel_F_data(df_merged)
    
    # Train model for panels A & B
    model, X_train, X_test, y_train, y_test, auc = train_logistic_model(X, y)
    
    # Create figure with equal row heights
    fig = plt.figure(figsize=(33, 22))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.0, wspace=0.15, height_ratios=[0.5, 1])    
    # Panel A: Logistic regression scatter
    ax = fig.add_subplot(gs[0, 0])
    # Shade incorrect prediction regions

    ax.text(-0.1, 1.12, 'A', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    scaler = model.named_steps['scaler']
    clf = model.named_steps['model']
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])
    y_obs_continuous = colors.loc[X_all.index]
    X_scaled = scaler.transform(X_all)
    raw_score = X_scaled.dot(clf.coef_[0]) + clf.intercept_[0]
    y_pred_continuous = model.predict_proba(X_all)[:, 1]
    sort_idx = np.argsort(raw_score)

    # Create secondary y-axis first
    ax2 = ax.twinx()
    
    # Add spine to secondary y-axis
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_linewidth(2)
    
    # Plot scatter on secondary axis with lower zorder
    scatter = ax.scatter(raw_score, y_obs_continuous, c='grey', alpha=0.6, s=80, zorder=2)
    ax.set_ylabel('Demixing Index', fontsize=20.0)
    ax.set_ylim(0, 1.05)
    
    # Plot reference lines on primary axis with low zorder
    ax2.axhline(0.5, color='black', linestyle='--', linewidth=2.5, alpha=0.75, zorder=1)
    ax2.axvline(0.0, color='grey', linestyle=':', linewidth=2.5, alpha=0.75, zorder=1)
    
    # Plot model prediction on primary axis with highest zorder
    ax2.plot(raw_score[sort_idx], y_pred_continuous[sort_idx], color='darkred', linewidth=5, 
            linestyle='--', label='Model', zorder=3)
    
    ax.set_xlabel('Model raw score (logit)', fontsize=20.0)
    #ax2.set_ylabel('Model prediction', fontsize=20.0)
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_yticklabels(['Mixed (0)', '0.5', 'Demixed (1)'])
    ax2.legend(fontsize=14, loc='upper left')
    
    save_data_to_csv({
        'raw_score': raw_score,
        'predicted_probability': y_pred_continuous,
        'observed_demixing': y_obs_continuous
    }, 'panelA_logistic_data.csv')
    
    # Panel B: ROC curve - make it square
    ax = fig.add_subplot(gs[0, 1])
    ax.text(-0.1, 1.12, 'B', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
    ax.plot(fpr, tpr, color='k', linewidth=4, label=f'ROC (AUC: {auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')
    ax.set_xlabel('False positive rate', fontsize=20.0)
    ax.set_ylabel('True positive rate', fontsize=20.0)
    ax.legend(fontsize=14, loc='lower right')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal', adjustable='box')
    
    save_data_to_csv({
        'fpr': fpr,
        'tpr': tpr,
        'auc': auc
    }, 'panelB_roc_data.csv')
    
    # Panel C: Vertical boxplot of demixing index by interaction type
    ax = fig.add_subplot(gs[0, 2])
    ax.text(-0.1, 1.12, 'C', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    
    # Order interaction types by median demixing index
    order_by_median = df_C.groupby('interaction_type')['demixing_composition'].median().sort_values(ascending=True)
    order = order_by_median.index.tolist()
    
    # Define a professional, colorblind-friendly palette
    palette = {
        'C-C': '#2E86AB',        # Blue
        'C-H': '#A23B72',        # Purple-pink
        'C-P': '#F18F01',        # Orange
        'H-H': '#C73E1D',        # Red
        'H-P': '#708238',        # Olive green
        'P-P': '#1B998B'         # Teal green
    }
    
    # Create vertical boxplot
    bp = ax.boxplot(
        [df_C[df_C['interaction_type'] == it]['demixing_composition'].values for it in order],
        positions=range(len(order)),
        widths=0.6,
        patch_artist=True,
        flierprops=dict(marker='o', markerfacecolor='gray', alpha=0.5, markersize=6),
        boxprops=dict(linewidth=2),
        medianprops=dict(linewidth=2.5, color='white'),
        whiskerprops=dict(linewidth=2),
        capprops=dict(linewidth=2)
    )
    
    # Color the boxes
    for patch, interaction in zip(bp['boxes'], order):
        patch.set_facecolor(palette.get(interaction, 'lightgrey'))
        patch.set_alpha(0.8)
    
    ax.set_ylabel('Demixing Index', fontsize=20.0)
    ax.set_xlabel('Interaction Type', fontsize=20.0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=0)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add sample counts above boxes
    for i, interaction in enumerate(order):
        count = len(df_C[df_C['interaction_type'] == interaction])
        y_pos = df_C[df_C['interaction_type'] == interaction]['demixing_composition'].max()
        ax.text(i, y_pos + 0.05, f'n={count}', ha='center', va='bottom', fontsize=16)
    
    save_data_to_csv({
        'interaction_type': df_C['interaction_type'].values,
        'demixing_composition': df_C['demixing_composition'].values
    }, 'panelC_interaction_data.csv')
    
    # Panel D: dGij vs |dG1 - dG2|
    ax = fig.add_subplot(gs[1, 0])
    ax.text(-0.1, 1.12, 'D', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    
    h = sns.histplot(data=df_DE, x='dG_deviation', y='dGij', binwidth=(0.2, 0.2), 
                cmap=cmaps.reds_light_r, cbar=True, ax=ax, cbar_kws={'label': 'Count', 'shrink': 0.5})
    
    # Format colorbar to show integers
    cbar = h.collections[0].colorbar
    from matplotlib.ticker import MaxNLocator
    cbar.locator = MaxNLocator(integer=True)
    cbar.update_ticks()
    
    ax.set_xlabel('|ΔG$_{ii}$ - ΔG$_{jj}$| ($k_BT$)', fontsize=20.0)
    ax.set_ylabel('Heterotypic ΔG$_{ij}$ ($k_BT$)', fontsize=20.0)
    ax.set_xlim(0, 10)
    ax.set_ylim(0.5, -10)
    ax.set_aspect('equal', adjustable='box')
    
    # Set matching tick positions for x and y
    tick_positions = [0, -2, -4, -6, -8, -10]
    ax.set_yticks(tick_positions)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    
    ax.grid(True, alpha=0.3)
    
    save_data_to_csv({
        'dG_deviation': df_DE['dG_deviation'].values,
        'dGij': df_DE['dGij'].values
    }, 'panelD_dG_difference.csv')
    
    # Panel E: dGij vs average dG
    ax = fig.add_subplot(gs[1, 1])
    ax.text(-0.1, 1.12, 'E', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    
    sns.histplot(data=df_DE, x='average_dG', y='dGij', binwidth=(0.2, 0.2), 
                cmap=cmaps.blues_light_r, cbar=True, ax=ax, cbar_kws={'label': 'Count', 'shrink': 0.5})
    
    # Add line of best fit
    slope, intercept, r_value, p_value, std_err = linregress(df_DE['average_dG'], df_DE['dGij'])
    r_squared = r_value**2
    x_fit = np.array([-10, 0])
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, color='black', linestyle='--', linewidth=3, label='Best Fit')
    
    text_label = f'Slope = {slope:.2f}\n$R^2$ = {r_squared:.2f}'
    ax.text(0.95, 0.05, text_label, transform=ax.transAxes, fontsize=14, 
            va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
    
    ax.set_xlabel('(ΔG$_{ii}$ + ΔG$_{jj}$)/2 ($k_BT$)', fontsize=20.0)
    ax.set_ylabel('Heterotypic ΔG$_{ij}$ ($k_BT$)', fontsize=20.0)
    ax.plot([-10, 0], [-10, 0], color='grey', linestyle='--', linewidth=2)
    ax.set_xlim(0.5, -10)
    ax.set_ylim(0.5, -10)
    ax.set_aspect('equal', adjustable='box')
    
    # Set matching tick positions for x and y
    tick_positions = [0, -2, -4, -6, -8, -10]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    
    ax.grid(True, alpha=0.3)
    
    save_data_to_csv({
        'average_dG': df_DE['average_dG'].values,
        'dGij': df_DE['dGij'].values,
        'slope': slope,
        'r_squared': r_squared
    }, 'panelE_dG_average.csv')
    
    # Panel F: dGij vs |Net Charge|
    ax = fig.add_subplot(gs[1, 2])
    ax.text(-0.1, 1.12, 'F', transform=ax.transAxes, fontsize=24, fontweight='bold', va='top', ha='right')
    
    # Plot lines connecting points from same pair
    for pair_id in df_F['pair_id'].unique():
        pair_data = df_F[df_F['pair_id'] == pair_id].sort_values('abs_total_net_charge')
        ax.plot(pair_data['abs_total_net_charge'], pair_data['dGij'],
                color='gray', alpha=0.3, linewidth=1.5, zorder=1)
    
    # Plot scatter points
    scatter = ax.scatter(df_F['abs_total_net_charge'], df_F['dGij'],
                        c=df_F['total_charged_residues'], cmap='YlOrBr',
                        s=100, alpha=0.7, edgecolors='black', linewidths=1, zorder=2)
    
    ax.set_xlabel('|Net Charge in Simulation|', fontsize=20.0)
    ax.set_ylabel('Heterotypic ΔG$_{ij}$ ($k_BT$)', fontsize=20.0)
    ax.set_ylim(0.5, -10)
    ax.set_box_aspect(1)
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02, shrink=0.5)
    cbar.set_label('Num. of Charged Residues', rotation=90, labelpad=20, va='center')
    
    # Format colorbar in scientific notation
    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    cbar.ax.yaxis.set_major_formatter(formatter)
    cbar.ax.yaxis.get_offset_text().set_fontsize(12)
    
    ax.grid(True, alpha=0.3)
    
    save_data_to_csv({
        'abs_total_net_charge': df_F['abs_total_net_charge'].values,
        'dGij': df_F['dGij'].values,
        'total_charged_residues': df_F['total_charged_residues'].values,
        'pair_id': df_F['pair_id'].values
    }, 'panelF_charge_data.csv')
    
    plt.tight_layout()
    plt.savefig('Figure3.pdf', bbox_inches='tight')
    print("\nFigure 3 saved as Figure3.pdf")
    print("Panels: A (logistic), B (ROC), C (interaction types), D-F (dG plots)")
    print("All data and filtering criteria saved in figure3_data/ directory")


if __name__ == '__main__':
    create_figure3()