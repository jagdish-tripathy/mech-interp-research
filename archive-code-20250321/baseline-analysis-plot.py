"""
baseline-analysis-plot.py - Analysis and Visualisation from Saved Results

Loads confidence_results_<label>.csv produced by baseline-analysis-run.py
and generates all statistical analyses and figures.

Edit this file freely to iterate on plots without re-running the model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import binomtest
import statsmodels.api as sm
from statsmodels.formula.api import ols
from pathlib import Path
import json
import argparse

# ---------------------------------------------------------------------------
# Global plot style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

COLORS = {"White": "#2563EB", "Black": "#DC2626"}


# =============================================================================
# 1.  SUMMARY STATISTICS
# =============================================================================

def print_summary(df):
    white = df[df['race'] == 'White']
    black = df[df['race'] == 'Black']

    print("\n" + "="*70)
    print("CONFIDENCE BIAS ANALYSIS")
    print("="*70)
    print(f"\nSample sizes:  White={len(white)},  Black={len(black)}")

    # Approval rates
    for method, col in [("Grammar-constrained", "decision_grammar"),
                         ("Logit argmax",        "decision_logit")]:
        w = (white[col] == 'A').mean() * 100
        b = (black[col] == 'A').mean() * 100
        print(f"\n{method} approval rates:")
        print(f"  White: {w:.2f}%   Black: {b:.2f}%   Gap: {w-b:+.2f}pp")

    # Margins
    t, p = stats.ttest_ind(white['margin'], black['margin'])
    print(f"\nLogit margins:  White={white['margin'].mean():.3f} ± {white['margin'].std():.3f}"
          f"   Black={black['margin'].mean():.3f} ± {black['margin'].std():.3f}")
    print(f"  T-test: t={t:.3f}, p={p:.4f}")

    # Probabilities
    t2, p2 = stats.ttest_ind(white['prob_A'], black['prob_A'])
    print(f"\nApproval probabilities:  White={white['prob_A'].mean():.3f}"
          f"   Black={black['prob_A'].mean():.3f}")
    print(f"  T-test: t={t2:.3f}, p={p2:.4f}")


# =============================================================================
# 2.  FIGURE: Confidence distributions
# =============================================================================

def plot_confidence_distributions(df, save_dir):
    white = df[df['race'] == 'White']
    black = df[df['race'] == 'Black']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of margins
    ax = axes[0]
    ax.hist(white['margin'], bins=40, alpha=0.6, label='White', color=COLORS['White'])
    ax.hist(black['margin'], bins=40, alpha=0.6, label='Black', color=COLORS['Black'])
    ax.axvline(white['margin'].mean(), color=COLORS['White'], linestyle='--',
               label=f"White mean: {white['margin'].mean():.2f}")
    ax.axvline(black['margin'].mean(), color=COLORS['Black'], linestyle='--',
               label=f"Black mean: {black['margin'].mean():.2f}")
    ax.set_xlabel("Logit Margin (A − B)")
    ax.set_ylabel("Count")
    ax.set_title("Decision Confidence Distribution", fontweight='bold')
    ax.legend()

    # Box plot
    ax = axes[1]
    bp = ax.boxplot([white['margin'], black['margin']], labels=['White', 'Black'],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor(COLORS['White'] + '99')
    bp['boxes'][1].set_facecolor(COLORS['Black'] + '99')
    ax.set_ylabel("Logit Margin (A − B)")
    ax.set_title("Decision Confidence Comparison", fontweight='bold')

    plt.tight_layout()
    _save(fig, save_dir, "confidence_distributions.png")


# =============================================================================
# 3.  FIGURE: Approval probability distributions
# =============================================================================

def plot_probability_distributions(df, save_dir):
    white = df[df['race'] == 'White']
    black = df[df['race'] == 'Black']

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(white['prob_A'], bins=40, alpha=0.6, label='White', color=COLORS['White'])
    ax.hist(black['prob_A'], bins=40, alpha=0.6, label='Black', color=COLORS['Black'])
    ax.set_xlabel("P(Approve)")
    ax.set_ylabel("Count")
    ax.set_title("Approval Probability Distribution", fontweight='bold')
    ax.legend()
    plt.tight_layout()
    _save(fig, save_dir, "probability_distributions.png")


# =============================================================================
# 4.  FIGURE: Approval rates by credit score
# =============================================================================

def plot_approval_by_credit_score(df, save_dir):
    agg = (df.groupby(['race', 'credit_score_bucket'])
             ['decision_grammar']
             .apply(lambda x: (x == 'A').mean() * 100)
             .reset_index(name='approval_rate'))

    fig, ax = plt.subplots(figsize=(12, 5))
    for race in ['White', 'Black']:
        d = agg[agg['race'] == race].sort_values('credit_score_bucket')
        ax.plot(d['credit_score_bucket'], d['approval_rate'],
                marker='o' if race == 'White' else 's',
                linewidth=2.5, markersize=8, label=race, color=COLORS[race])

    ax.set_xlabel("Credit Score Bucket")
    ax.set_ylabel("Approval Rate (%)")
    ax.set_title("Approval Rate by Credit Score and Race", fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    _save(fig, save_dir, "approval_by_credit_score.png")

    # Console summary
    print("\n" + "="*70)
    print("APPROVAL RATES BY CREDIT SCORE")
    print("="*70)
    white_d = agg[agg['race'] == 'White'].sort_values('credit_score_bucket')
    black_d = agg[agg['race'] == 'Black'].sort_values('credit_score_bucket')
    for bucket in sorted(white_d['credit_score_bucket']):
        w = white_d.loc[white_d['credit_score_bucket'] == bucket, 'approval_rate'].values[0]
        b = black_d.loc[black_d['credit_score_bucket'] == bucket, 'approval_rate'].values[0]
        print(f"  {bucket:20s}  White={w:.1f}%  Black={b:.1f}%  Gap={w-b:+.1f}pp")

    return agg


# =============================================================================
# 5.  FIGURE: Margin by credit score
# =============================================================================

def plot_margin_by_credit_score(df, save_dir):
    agg = (df.groupby(['race', 'credit_score_bucket'])['margin']
             .mean()
             .reset_index(name='mean_margin'))

    fig, ax = plt.subplots(figsize=(12, 5))
    for race in ['White', 'Black']:
        d = agg[agg['race'] == race].sort_values('credit_score_bucket')
        ax.plot(d['credit_score_bucket'], d['mean_margin'],
                marker='o' if race == 'White' else 's',
                linewidth=2.5, markersize=8, label=race, color=COLORS[race])

    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel("Credit Score Bucket")
    ax.set_ylabel("Mean Logit Margin (A − B)")
    ax.set_title("Decision Confidence by Credit Score and Race", fontweight='bold')
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    _save(fig, save_dir, "margin_by_credit_score.png")


# =============================================================================
# 6.  FIGURE: Paired disagreement rate by credit score
# =============================================================================

def plot_paired_disagreement_by_credit(df, save_dir):
    pair_counts = df.groupby('pair_id').size()
    valid_pairs = pair_counts[pair_counts == 2].index
    paired = df[df['pair_id'].isin(valid_pairs)].copy()

    rows = []
    for pid in valid_pairs:
        p = paired[paired['pair_id'] == pid]
        w = p[p['race'] == 'White'].iloc[0]
        b = p[p['race'] == 'Black'].iloc[0]
        rows.append({
            'pair_id': pid,
            'credit_score_bucket': w['credit_score_bucket'],
            'disagree': w['decision_grammar'] != b['decision_grammar'],
        })
    dis_df = pd.DataFrame(rows)

    by_credit = (dis_df.groupby('credit_score_bucket')
                       .agg(disagreement_rate=('disagree', lambda x: x.mean() * 100),
                            n_pairs=('disagree', 'count'))
                       .reset_index()
                       .sort_values('credit_score_bucket'))

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(by_credit)), by_credit['disagreement_rate'],
                  color='#DC2626', alpha=0.7, edgecolor='black', linewidth=1.2)
    for bar, n in zip(bars, by_credit['n_pairs']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f"n={n}", ha='center', va='bottom', fontsize=9)

    ax.set_xticks(range(len(by_credit)))
    ax.set_xticklabels(by_credit['credit_score_bucket'], rotation=45, ha='right')
    ax.set_xlabel("Credit Score Bucket")
    ax.set_ylabel("Disagreement Rate (%)")
    ax.set_title("Paired Prompt Disagreement Rate by Credit Score\n"
                 "(% of matched pairs with different decisions)", fontweight='bold')
    ax.set_ylim(0, by_credit['disagreement_rate'].max() * 1.25)
    plt.tight_layout()
    _save(fig, save_dir, "paired_disagreement_by_credit.png")

    # Binomial test
    total_dis = dis_df['disagree'].sum()
    total_n   = len(dis_df)
    res = binomtest(total_dis, total_n, p=0, alternative='greater')
    print(f"\nPaired disagreement (overall): {total_dis}/{total_n} = "
          f"{total_dis/total_n*100:.1f}%  p={res.pvalue:.4f}")

    return dis_df, by_credit


# =============================================================================
# 7.  LINEAR PROBABILITY MODELS + coefficient plots
# =============================================================================

def run_linear_probability_models(df, save_dir):
    """
    Fit two OLS models:
      approved ~ race_dummies + credit_dummies + race×credit interactions + controls
      margin   ~ same

    Plots:
      - Coefficient forest plots with 95 % CI bands
      - Interaction coefficient plots with shaded 95 % CI
    """
    print("\n" + "="*70)
    print("LINEAR PROBABILITY MODELS")
    print("="*70)

    df = df.copy()
    df['credit_score_bucket'] = df['credit_score_bucket'].str.replace('-', '_')
    df['approved']  = (df['decision_grammar'] == 'A').astype(int)
    df['is_black']  = (df['race'] == 'Black').astype(int)
    df['is_white']  = (df['race'] == 'White').astype(int)

    credit_buckets = sorted(df['credit_score_bucket'].unique())
    print(f"Credit buckets: {credit_buckets}")

    for b in credit_buckets:
        df[f'credit_{b}']         = (df['credit_score_bucket'] == b).astype(int)
        df[f'black_x_credit_{b}'] = df['is_black'] * df[f'credit_{b}']
        df[f'white_x_credit_{b}'] = df['is_white'] * df[f'credit_{b}']

    credit_dummies     = [f'credit_{b}'         for b in credit_buckets[1:]]
    black_interactions = [f'black_x_credit_{b}' for b in credit_buckets]
    white_interactions = [f'white_x_credit_{b}' for b in credit_buckets]

    controls = []
    for col in ['income', 'loan_amount', 'ltv_ratio']:
        if col in df.columns and not df[col].isna().all():
            controls.append(col)
    if 'county' in df.columns:
        controls.append('C(county)')

    rhs = ['is_black', 'is_white'] + credit_dummies + black_interactions + white_interactions + controls
    formula_approved = 'approved ~ ' + ' + '.join(rhs)
    formula_margin   = 'margin ~ '   + ' + '.join(rhs)

    print(f"N = {len(df)},  predictors = {len(rhs)}")

    results = {}
    for name, formula, outcome_col in [
        ("Approval (LPM)",  formula_approved, "approved"),
        ("Margin (OLS)",    formula_margin,   "margin"),
    ]:
        print(f"\n{'─'*70}\n{name}")
        try:
            m = ols(formula, data=df).fit()
            print(m.summary())
            results[name] = m
        except Exception as e:
            print(f"ERROR: {e}")
            results[name] = None

    # -------------------------------------------------------------------------
    # Interaction coefficient plots with 95 % CI error bars
    # -------------------------------------------------------------------------
    for label, model in results.items():
        if model is None:
            continue
        _interaction_plot(model, credit_buckets, label, save_dir)

    # Save text summaries
    for label, model in results.items():
        if model is None:
            continue
        slug = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
        path = Path(save_dir) / f"lpm_{slug}_summary.txt"
        with open(path, "w") as f:
            f.write(model.summary().as_text())
        print(f"Saved: {path}")

    return results


# ---------------------------------------------------------------------------
# Interaction plot helper — grouped bar chart with 95 % CI error bars
# ---------------------------------------------------------------------------

def _interaction_plot(model, credit_buckets, title, save_dir):
    """
    Bar chart of Black×Credit and White×Credit interaction coefficients
    with 95 % CI error bars (i.e. 1.96 × SE).
    """
    params = model.params
    se     = model.bse

    black_coef, black_err = [], []
    white_coef, white_err = [], []

    for b in credit_buckets:
        for coef_list, err_list, prefix in [
            (black_coef, black_err, f'black_x_credit_{b}'),
            (white_coef, white_err, f'white_x_credit_{b}'),
        ]:
            if prefix in params.index:
                coef_list.append(params[prefix])
                err_list.append(1.96 * se[prefix])
            else:
                coef_list.append(0.0)
                err_list.append(0.0)

    display_labels = [b.replace('_', '-') for b in credit_buckets]
    x     = np.arange(len(credit_buckets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(x - width/2, black_coef, width,
           yerr=black_err, capsize=5,
           label='Black × Credit', color=COLORS['Black'], alpha=0.7,
           error_kw=dict(elinewidth=1.5, ecolor='black'))
    ax.bar(x + width/2, white_coef, width,
           yerr=white_err, capsize=5,
           label='White × Credit', color=COLORS['White'], alpha=0.7,
           error_kw=dict(elinewidth=1.5, ecolor='black'))

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=45, ha='right')
    ax.set_xlabel("Credit Score Bucket")
    ax.set_ylabel("Coefficient (95 % CI error bars)")
    ax.set_title(f"Race × Credit Interaction Effects — {title}\n"
                 "(Error bars = 95 % confidence interval)", fontweight='bold')
    ax.legend(fontsize=10)

    plt.tight_layout()
    slug = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
    _save(fig, save_dir, f"interactions_{slug}.png")


# =============================================================================
# UTILS
# =============================================================================

def _save(fig, save_dir, filename):
    if save_dir:
        p = Path(save_dir) / filename
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f"Saved: {p}")
    plt.close(fig)


def _clean_label(name):
    """Make regression term names human-readable."""
    return (name
            .replace('black_x_credit_', 'Black × ')
            .replace('white_x_credit_', 'White × ')
            .replace('credit_', 'Credit: ')
            .replace('_', '-')
            .replace('is_black', 'Black')
            .replace('is_white', 'White'))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot and analyse saved inference results")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to confidence_results_<label>.csv from baseline-analysis-run.py")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for figures (default: same dir as results CSV)")
    args = parser.parse_args()

    results_path = Path(args.results)

    if results_path.is_dir():
        candidates = sorted(results_path.glob("confidence_results_*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No confidence_results_*.csv found in {results_path}")
        results_path = candidates[-1]
    
    save_dir = Path(args.output) if args.output else results_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} rows from {results_path}")
    print(f"Race distribution: {df['race'].value_counts().to_dict()}")

    # 1. Summary statistics (console)
    print_summary(df)

    # 2. Distribution figures
    plot_confidence_distributions(df, save_dir)
    plot_probability_distributions(df, save_dir)

    # 3. Credit score breakdowns
    plot_approval_by_credit_score(df, save_dir)
    plot_margin_by_credit_score(df, save_dir)

    # 4. Paired disagreement
    plot_paired_disagreement_by_credit(df, save_dir)

    # 5. Regression models + coefficient plots
    run_linear_probability_models(df, save_dir)

    print("\n" + "="*70)
    print(f"All figures saved to: {save_dir}")
    print("="*70)