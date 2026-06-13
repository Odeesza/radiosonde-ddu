import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils import _sounding_diagnostics
from feature_engineering import _DIAG_GROUPS, _DEFAULT_FEATURES, _FEATURE_UNITS


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level helpers shared by both ClusterProfiles and FeatureClusterProfiles
# ─────────────────────────────────────────────────────────────────────────────

_MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def _color_family(base_rgba, n):
    """n shades from dark to light around the hue of base_rgba."""
    import colorsys
    r, g, b = base_rgba[:3]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    ls = np.linspace(0.28, 0.72, max(n, 1))
    return [colorsys.hls_to_rgb(h, li, min(s, 0.95)) for li in ls]


def _print_cluster_summary(mean_df, std_df):
    clusters = mean_df.index.tolist()
    col_w = 14
    for group, feats in _DIAG_GROUPS.items():
        avail = [f for f in feats if f in mean_df.columns]
        if not avail:
            continue
        print(f"\n── {group} ──")
        header = f"{'feature':<26}" + "".join(f"{'C'+str(k):>{col_w}}" for k in clusters)
        print(header)
        print("─" * len(header))
        for feat in avail:
            vals = ""
            for k in clusters:
                m = mean_df.loc[k, feat]
                s = std_df.loc[k, feat]
                if np.isnan(m):
                    cell = "—"
                elif "lapse" in feat or "N2" in feat:
                    cell = f"{m:.5f}±{s:.5f}"
                else:
                    cell = f"{m:.2f}±{s:.2f}"
                vals += f"{cell:>{col_w}}"
            print(f"{feat:<26}{vals}")


def _build_date_df(soundings, labels, clean_mask=None):
    """Build a DataFrame of (date, cluster, month, year, doy) for date plots."""
    used = soundings
    if clean_mask is not None:
        used = [s for s, keep in zip(soundings, clean_mask) if keep]
    if len(used) != len(labels):
        raise ValueError(
            f"len(soundings after mask)={len(used)} != len(labels)={len(labels)}. "
            "Pass the same soundings list used for build_X / build_features_2."
        )
    dates = pd.to_datetime([s['header']['date'].item() for s in used])
    return pd.DataFrame({
        'date':    dates,
        'cluster': labels,
        'month':   dates.month,
        'year':    dates.year,
        'doy':     dates.day_of_year,
    })


def _plot_date_distribution_impl(df, figsize=(13, 5), save_path=None):
    """
    Core implementation for plot_date_distribution, shared by both cluster classes.

    Panel (a): absolute-count stacked bars — bar height = total soundings per month,
               each segment is one cluster. Shows both composition and monthly volume.
    Panel (b): per-cluster count line chart — shows if a cluster increases or
               decreases over the course of the season.

    The legend is placed below the figure so it never overlaps bar annotations,
    regardless of which season the data covers.

    Parameters
    ----------
    df        : pd.DataFrame with columns date, cluster, month, year, doy
    figsize   : tuple
    save_path : str or None — if provided, saves the figure at 300 dpi

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    month_abbr = _MONTH_ABBR

    clusters = sorted(df['cluster'].unique())
    n_cl     = len(clusters)
    cmap     = plt.cm.tab10
    colors   = {k: cmap(i / 10) for i, k in enumerate(clusters)}
    leg_lbl  = {k: f'C{k}  (n={(df.cluster == k).sum()})' for k in clusters}
    markers  = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']

    # Only months that actually have soundings, in calendar order
    active_months = sorted(df['month'].unique())
    m_labels = [month_abbr[m - 1] for m in active_months]
    x_pos    = list(range(len(active_months)))

    counts = (df.groupby(['month', 'cluster'])
                .size()
                .unstack(fill_value=0)
                .reindex(index=active_months, fill_value=0)
                .reindex(columns=clusters, fill_value=0))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    bar_handles = []

    # ── Panel (a): absolute count stacked bars ────────────────────────────
    ax = axes[0]
    bottom = np.zeros(len(active_months))

    for k in clusters:
        v = counts[k].values.astype(float)
        b = ax.bar(x_pos, v, bottom=bottom,
                   color=colors[k], alpha=0.9,
                   width=0.72, edgecolor='white', linewidth=0.4)
        bar_handles.append(b[0])
        for mi, (val, bot) in enumerate(zip(v, bottom)):
            if val >= 2:
                ax.text(mi, bot + val / 2, str(int(val)),
                        ha='center', va='center',
                        fontsize=7, color='white', fontweight='bold')
        bottom += v

    # Total count above each bar
    for mi, tot in enumerate(bottom):
        if tot > 0:
            ax.text(mi, tot + 0.3, str(int(tot)),
                    ha='center', va='bottom', fontsize=8, color='0.35')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(m_labels, fontsize=9)
    ax.set_ylabel('Number of soundings', fontsize=11)
    ax.set_xlabel('Month', fontsize=11)
    ax.set_xlim(-0.6, len(active_months) - 0.4)
    ax.set_title('(a) Cluster composition per month', fontsize=12)
    ax.grid(axis='y', alpha=0.3, lw=0.5)
    ax.spines[['top', 'right']].set_visible(False)

    # ── Panel (b): per-cluster count line chart ───────────────────────────
    ax = axes[1]

    for i, k in enumerate(clusters):
        v = counts[k].values.astype(float)
        ax.plot(x_pos, v, color=colors[k],
                marker=markers[i % len(markers)], lw=2, ms=7, zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(m_labels, fontsize=9)
    ax.set_ylabel('Number of soundings', fontsize=11)
    ax.set_xlabel('Month', fontsize=11)
    ax.set_xlim(-0.4, len(active_months) - 0.6)
    ax.set_title('(b) Cluster count evolution over season', fontsize=12)
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[['top', 'right']].set_visible(False)

    # Shared legend below both panels (never overlaps bars)
    fig.legend(bar_handles, [leg_lbl[k] for k in clusters],
               loc='lower center', ncol=min(n_cl, 5), fontsize=9,
               bbox_to_anchor=(0.5, -0.04), framealpha=0.9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig


def _plot_seasonal_evolution_impl(df, season_months=None, figsize=(12, 4), save_path=None):
    """
    Dedicated temporal-evolution figure: how each cluster's proportion and count
    change over the course of a season.

    Panel (a): proportion (%) line chart with grey bars showing total soundings
               per month (twin y-axis) — shows if a cluster gains/loses share.
    Panel (b): absolute count line chart — shows raw occurrence trends.

    Parameters
    ----------
    df            : pd.DataFrame  with columns date, cluster, month, year, doy
    season_months : list of int, optional
        Months in *chronological* order for the target season.
        Austral winter : [5, 6, 7, 8, 9]
        Austral summer : [10, 11, 12, 1, 2, 3]
        If None, all months present in the data are used in calendar order.
    figsize       : tuple
    save_path     : str or None

    Returns
    -------
    fig     : matplotlib.figure.Figure
    summary : pd.DataFrame  — count per cluster per month (index = months)
    """
    month_abbr = _MONTH_ABBR

    present = set(df['month'].unique())
    if season_months is None:
        season_months = sorted(present)
    else:
        season_months = [m for m in season_months if m in present]
    if not season_months:
        raise ValueError("None of the provided season_months are present in the data.")

    clusters = sorted(df['cluster'].unique())
    n_cl     = len(clusters)
    cmap     = plt.cm.tab10
    colors   = {k: cmap(i / 10) for i, k in enumerate(clusters)}
    leg_lbl  = {k: f'C{k}  (n={(df.cluster == k).sum()})' for k in clusters}
    markers  = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']

    counts = (df.groupby(['month', 'cluster'])
                .size()
                .unstack(fill_value=0)
                .reindex(index=season_months, fill_value=0)
                .reindex(columns=clusters, fill_value=0))

    total_per_month = counts.sum(axis=1).replace(0, np.nan)
    props = counts.div(total_per_month, axis=0).fillna(0) * 100

    x_pos    = list(range(len(season_months)))
    m_labels = [month_abbr[m - 1] for m in season_months]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    line_handles = []

    # ── Panel (a): proportion (%) + total count as background bars ────────
    ax = axes[0]
    ax2 = ax.twinx()
    totals = total_per_month.reindex(season_months).fillna(0).values
    ax2.bar(x_pos, totals, color='grey', alpha=0.15, width=0.65, zorder=0)
    ax2.set_ylabel('Total soundings\n(grey bars)', fontsize=9, color='0.5')
    ax2.tick_params(axis='y', colors='0.5', labelsize=8)
    ax2.spines[['top']].set_visible(False)
    ax2.set_ylim(0, totals.max() * 3.5 if totals.max() > 0 else 1)

    for i, k in enumerate(clusters):
        ln, = ax.plot(x_pos, props[k].values, color=colors[k],
                      marker=markers[i % len(markers)], lw=2, ms=7, zorder=3)
        line_handles.append(ln)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(m_labels, fontsize=9)
    ax.set_ylabel('Cluster proportion (%)', fontsize=11)
    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylim(0, 50)
    ax.set_xlim(-0.4, len(season_months) - 0.6)
    ax.set_title('(a) Cluster proportion over season', fontsize=12)
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    # ── Panel (b): absolute count lines ───────────────────────────────────
    ax = axes[1]

    for i, k in enumerate(clusters):
        ax.plot(x_pos, counts[k].values.astype(float),
                color=colors[k], marker=markers[i % len(markers)],
                lw=2, ms=7, zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(m_labels, fontsize=9)
    ax.set_ylabel('Number of soundings', fontsize=11)
    ax.set_xlabel('Month', fontsize=11)
    ax.set_xlim(-0.4, len(season_months) - 0.6)
    ax.set_title('(b) Cluster count over season', fontsize=12)
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[['top', 'right']].set_visible(False)

    # Shared legend below both panels
    fig.legend(line_handles, [leg_lbl[k] for k in clusters],
               loc='lower center', ncol=min(n_cl, 5), fontsize=9,
               bbox_to_anchor=(0.5, -0.04), framealpha=0.9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig, counts




def _plot_feature_seasonal_impl(df_diag, season_months=None, features=None,
                                 stat='median', n_cols=4,
                                 figsize=None, save_path=None):
    """
    Plot the seasonal evolution of physical diagnostics, one subplot per feature,
    one line per cluster (hue).

    The central line shows the per-cluster median (or mean) and the shaded ribbon
    shows the interquartile range (or ± 1 std), all computed per month.

    Parameters
    ----------
    df_diag       : pd.DataFrame
        Must contain one row per sounding with columns: all diagnostic features,
        'cluster', 'month'.
    season_months : list of int, optional
        Months in *chronological* season order (e.g. [5,6,7,8,9] for austral
        winter, [10,11,12,1,2,3] for summer). If None, uses all months present.
    features      : list of str or None
        Feature names to plot. If None, uses a default publication-ready subset.
        Pass features='all' to plot every available diagnostic.
    stat          : 'median' or 'mean'
        Central statistic. 'median' uses Q25–Q75 ribbon; 'mean' uses ±1 std.
    n_cols        : int
        Number of subplot columns.
    figsize       : tuple or None
        If None, inferred from the number of features and n_cols.
    save_path     : str or None

    Returns
    -------
    fig    : matplotlib.figure.Figure
    df_agg : pd.DataFrame
        Aggregated table with MultiIndex (cluster, month) and columns
        [feature]_center, [feature]_lo, [feature]_hi for every plotted feature.
    """
    month_abbr = _MONTH_ABBR

    # ── resolve season months ─────────────────────────────────────────────
    present = set(df_diag['month'].unique())
    if season_months is None:
        season_months = sorted(present)
    else:
        season_months = [m for m in season_months if m in present]
    if not season_months:
        raise ValueError("None of the provided season_months are present in the data.")

    df_diag = df_diag[df_diag['month'].isin(season_months)].copy()

    # ── resolve features ──────────────────────────────────────────────────
    all_diag_cols = [c for c in df_diag.columns if c not in ('cluster', 'month', 'year', 'date', 'doy')]
    if features is None:
        features = [f for f in _DEFAULT_FEATURES if f in all_diag_cols]
    elif features == 'all':
        features = all_diag_cols
    else:
        features = [f for f in features if f in all_diag_cols]
    if not features:
        raise ValueError("None of the requested features are present in the diagnostics DataFrame.")

    # ── colour / style setup ──────────────────────────────────────────────
    clusters  = sorted(df_diag['cluster'].unique())
    n_cl      = len(clusters)
    cmap      = plt.cm.tab10
    colors    = {k: cmap(i / 10) for i, k in enumerate(clusters)}
    leg_lbl   = {k: f'C{k}  (n={(df_diag.cluster == k).sum()})' for k in clusters}
    markers   = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']

    x_pos    = list(range(len(season_months)))
    m_labels = [month_abbr[m - 1] for m in season_months]

    # ── compute aggregates ────────────────────────────────────────────────
    grp = df_diag.groupby(['cluster', 'month'])

    if stat == 'median':
        center_fn = lambda g: g.median()
        lo_fn     = lambda g: g.quantile(0.25)
        hi_fn     = lambda g: g.quantile(0.75)
    else:
        center_fn = lambda g: g.mean()
        lo_fn     = lambda g: g.mean() - g.std()
        hi_fn     = lambda g: g.mean() + g.std()

    df_center = grp[features].apply(center_fn).reindex(
        pd.MultiIndex.from_product([clusters, season_months], names=['cluster', 'month']),
        fill_value=np.nan
    )
    df_lo = grp[features].apply(lo_fn).reindex(df_center.index, fill_value=np.nan)
    df_hi = grp[features].apply(hi_fn).reindex(df_center.index, fill_value=np.nan)

    # ── layout ────────────────────────────────────────────────────────────
    n_feat  = len(features)
    n_cols  = min(n_cols, n_feat)
    n_rows  = int(np.ceil(n_feat / n_cols))
    if figsize is None:
        figsize = (n_cols * 3.5, n_rows * 3.0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             squeeze=False, sharey=False)
    axes_flat = axes.flatten()

    line_handles = []

    for fi, feat in enumerate(features):
        ax = axes_flat[fi]
        unit = _FEATURE_UNITS.get(feat, '')

        for i, k in enumerate(clusters):
            try:
                ctr = df_center.loc[(k,), feat].reindex(season_months).values.astype(float)
                lo  = df_lo.loc[(k,), feat].reindex(season_months).values.astype(float)
                hi  = df_hi.loc[(k,), feat].reindex(season_months).values.astype(float)
            except KeyError:
                continue

            ln, = ax.plot(x_pos, ctr, color=colors[k],
                          marker=markers[i % len(markers)], lw=2, ms=6, zorder=3)
            ax.fill_between(x_pos, lo, hi, color=colors[k], alpha=0.18, zorder=2)

            if fi == 0:
                line_handles.append(ln)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(m_labels, fontsize=8)
        ylabel = f'{unit}' if unit else ''
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(feat, fontsize=9, pad=3)
        ax.set_xlim(-0.4, len(season_months) - 0.6)
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)

    # hide unused axes
    for ax in axes_flat[n_feat:]:
        ax.set_visible(False)

    # shared legend below figure
    stat_label = 'Median ± IQR' if stat == 'median' else 'Mean ± 1 std'
    fig.legend(line_handles, [leg_lbl[k] for k in clusters],
               loc='lower center', ncol=min(n_cl, 5), fontsize=9,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.9,
               title=stat_label, title_fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10 + 0.02 * (n_cl > 4))
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # build aggregated return table
    records = []
    for k in clusters:
        for m in season_months:
            row = {'cluster': k, 'month': m}
            try:
                for feat in features:
                    row[f'{feat}_center'] = df_center.loc[(k, m), feat]
                    row[f'{feat}_lo']     = df_lo.loc[(k, m), feat]
                    row[f'{feat}_hi']     = df_hi.loc[(k, m), feat]
            except KeyError:
                pass
            records.append(row)
    df_agg = pd.DataFrame(records).set_index(['cluster', 'month'])

    return fig, df_agg


def _plot_cluster_carpet_impl(df, season_months=None, years=None,
                               sounding_width=0.5, figsize=None,
                               save_path=None, title=None):
    """
    Carpet plot: each row = one winter, x = day in the season,
    each sounding = a bar coloured by cluster.

    Allows visualising:
    - Temporal coherence: blocks of the same colour = stable regime over several days
    - Inter-annual repetitions: patterns aligned vertically across rows
    - Sounding density and gaps

    Parameters
    ----------
    df            : pd.DataFrame  columns: date (datetime), cluster, month, year
    season_months : list of int, optional
        Chronological season order (e.g. [5,6,7,8,9] for austral winter).
        If None, all months present in the data, sorted.
    years         : int, list of int, or None
        Filter on one or several winters. None = all winters.
    sounding_width : float
        Width of each sounding in days (default 0.5 — suited to 2 soundings/day).
        Reduce to 0.2-0.3 for 4 soundings/day to avoid overlap.
    figsize       : tuple or None
    save_path     : str or None
    title         : str or None
    """
    import calendar
    from matplotlib.patches import Patch

    month_abbr = _MONTH_ABBR

    # ── Resolve season ──────────────────────────────────────────────────────
    if season_months is None:
        season_months = sorted(df['month'].unique())

    # ── Filter years ────────────────────────────────────────────────────────
    all_years = sorted(df['year'].unique())
    if years is not None:
        years_req = [years] if isinstance(years, int) else list(years)
        all_years = [y for y in all_years if y in years_req]
    if not all_years:
        raise ValueError("No data for the requested years.")

    n_years   = len(all_years)
    year_to_row = {y: i for i, y in enumerate(all_years)}

    # ── Clusters and colours ────────────────────────────────────────────────
    clusters = sorted(df['cluster'].unique())
    n_cl     = len(clusters)
    cmap     = plt.cm.tab20 if n_cl > 10 else plt.cm.tab10
    colors   = {k: cmap(i / 20) for i, k in enumerate(clusters)} if n_cl > 10 else {k: cmap(i / 10) for i, k in enumerate(clusters)}

    # ── Day offsets per month in the season ─────────────────────────────────
    day_offsets = {}
    offset = 0
    for m in season_months:
        day_offsets[m] = offset
        offset += calendar.monthrange(2000, m)[1]
    total_days = offset

    def _dos(dt):
        """Day-of-season (fractional, to position 00h vs 12h)."""
        if dt.month not in day_offsets:
            return None
        return day_offsets[dt.month] + dt.day - 1 + dt.hour / 24.0

    # ── Figure ──────────────────────────────────────────────────────────────
    if figsize is None:
        figsize = (15, max(3.0, n_years * 0.6 + 2.0))

    fig, ax = plt.subplots(figsize=figsize)

    # Light grey background per season row
    for yi in range(n_years):
        ax.barh(yi, total_days, left=0, height=0.88,
                color='0.93', linewidth=0, zorder=0)

    # ── Draw soundings ──────────────────────────────────────────────────────
    df_filt = df[df['year'].isin(all_years) & df['month'].isin(season_months)].copy()
    df_filt['_yi']  = df_filt['year'].map(year_to_row)
    df_filt['_dos'] = df_filt['date'].apply(_dos)
    df_filt = df_filt.dropna(subset=['_dos'])

    for (k, yi), grp in df_filt.groupby(['cluster', '_yi']):
        ax.barh(
            grp['_yi'].values,
            sounding_width,
            left=grp['_dos'].values - sounding_width / 2,
            height=0.80,
            color=colors[k],
            linewidth=0,
            zorder=2,
        )

    # ── Month separators (white vertical lines) ─────────────────────────────
    for m in season_months:
        ax.axvline(day_offsets[m], color='white', lw=1.5, zorder=3)

    # ── Month labels on x-axis (centred in each month) ──────────────────────
    month_mids   = [day_offsets[m] + calendar.monthrange(2000, m)[1] / 2
                    for m in season_months]
    month_labels = [month_abbr[m - 1] for m in season_months]
    ax.set_xticks(month_mids)
    ax.set_xticklabels(month_labels, fontsize=10)
    ax.tick_params(axis='x', length=0)

    # ── Y-axis: years ────────────────────────────────────────────────────────
    ax.set_yticks(range(n_years))
    ax.set_yticklabels(all_years, fontsize=9)
    ax.invert_yaxis()   # oldest season at the top -> chronological reading
    ax.set_xlim(0, total_days)

    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.grid(axis='x', color='white', lw=0.5, zorder=1)

    # ── Legend ──────────────────────────────────────────────────────────────
    n_total = len(df_filt)
    handles = [
        Patch(color=colors[k],
              label=f'C{k}  ({(df_filt["cluster"] == k).sum()} / {n_total})')
        for k in clusters
    ]
    ax.legend(handles=handles, loc='lower center',
              ncol=min(n_cl, 6), fontsize=8, framealpha=0.9,
              bbox_to_anchor=(0.5, -0.12))

    ax.set_title(title or 'Clusters temporal partitions — winter',
                 fontsize=12, pad=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.14)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig


def _plot_interannual_consistency_impl(df, season_months=None, years=None,
                                       window=7, min_soundings=3,
                                       figsize=None, save_path=None, title=None):
    """
    Evaluate and visualise the inter-annual consistency of clusters.

    3 panels:
    (a) Normalised Shannon entropy per time window — shows periods where
        the classification is stable from one year to the next.
    (b) Heatmap of the dominant cluster per window x year — synthetic view
        allowing repeated patterns to be spotted visually.
    (c) ARI (Adjusted Rand Index) matrix between pairs of years — global
        similarity measure: ARI=1 -> identical winters, ARI=0 -> no better
        than chance.

    Parameters
    ----------
    df            : pd.DataFrame  columns: date, cluster, month, year
    season_months : list of int, optional
    years         : int, list of int, or None  filter years
    window        : int  temporal window size in days (default 7)
    min_soundings : int  minimum soundings per bin to consider it valid (default 3)
    figsize       : tuple or None
    save_path     : str or None
    title         : str or None

    Returns
    -------
    fig    : matplotlib.figure.Figure
    stats  : dict  keys 'dominant', 'entropy', 'ari', 'bin_centers', 'years'
    """
    import calendar
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    from sklearn.metrics import adjusted_rand_score

    month_abbr = _MONTH_ABBR

    # ── Resolve season and years ────────────────────────────────────────────
    if season_months is None:
        season_months = sorted(df['month'].unique())

    all_years = sorted(df['year'].unique())
    if years is not None:
        years_req = [years] if isinstance(years, int) else list(years)
        all_years = [y for y in all_years if y in years_req]
    if not all_years:
        raise ValueError("No data for the requested years.")

    n_years = len(all_years)

    clusters = sorted(df['cluster'].unique())
    n_cl     = len(clusters)
    cl_to_idx = {k: i for i, k in enumerate(clusters)}
    cmap_tab  = plt.cm.tab10 if n_cl <= 10 else plt.cm.tab20
    cl_colors = {k: cmap_tab(i / 10) for i, k in enumerate(clusters)} if n_cl <= 10 else {k: cmap_tab(i / 20) for i, k in enumerate(clusters)}

    # ── Day offsets in the season ────────────────────────────────────────────
    day_offsets = {}
    offset = 0
    for m in season_months:
        day_offsets[m] = offset
        offset += calendar.monthrange(2000, m)[1]
    total_days = offset
    n_bins     = (total_days + window - 1) // window
    bin_centers = np.array([b * window + window / 2 for b in range(n_bins)])

    # ── Pre-compute the bin for each sounding ───────────────────────────────
    df_filt = df[df['year'].isin(all_years) & df['month'].isin(season_months)].copy()

    def _dos(dt):
        if dt.month not in day_offsets:
            return np.nan
        return float(day_offsets[dt.month] + dt.day - 1)

    df_filt['_dos'] = df_filt['date'].apply(_dos)
    df_filt['_bin'] = (df_filt['_dos'] / window).apply(
        lambda x: int(x) if not np.isnan(x) else np.nan)
    df_filt = df_filt.dropna(subset=['_bin'])
    df_filt['_bin'] = df_filt['_bin'].astype(int).clip(0, n_bins - 1)

    # ── Dominant cluster per bin x year ─────────────────────────────────────
    dominant = np.full((n_years, n_bins), np.nan)
    for yi, y in enumerate(all_years):
        df_y = df_filt[df_filt['year'] == y]
        for bi in range(n_bins):
            sub = df_y[df_y['_bin'] == bi]
            if len(sub) >= min_soundings:
                mode_val = sub['cluster'].mode()
                if len(mode_val) > 0:
                    dominant[yi, bi] = mode_val.iloc[0]

    # ── Normalised entropy per bin (across years) ───────────────────────────
    H_norm = np.full(n_bins, np.nan)
    for bi in range(n_bins):
        valid = dominant[:, bi][~np.isnan(dominant[:, bi])].astype(int)
        if len(valid) >= 2:
            counts = np.array([(valid == k).sum() for k in clusters], dtype=float)
            p = counts / counts.sum()
            p_nz = p[p > 0]
            H = -np.sum(p_nz * np.log(p_nz))
            H_norm[bi] = H / np.log(n_cl) if n_cl > 1 else 0.0

    # ── ARI matrix between pairs of years ───────────────────────────────────
    ari_mat = np.full((n_years, n_years), np.nan)
    np.fill_diagonal(ari_mat, 1.0)
    for i in range(n_years):
        for j in range(i + 1, n_years):
            valid = ~np.isnan(dominant[i]) & ~np.isnan(dominant[j])
            if valid.sum() >= 2:
                score = adjusted_rand_score(
                    dominant[i][valid].astype(int),
                    dominant[j][valid].astype(int)
                )
                ari_mat[i, j] = score
                ari_mat[j, i] = score

    # ── Figure ──────────────────────────────────────────────────────────────
    heat_h = max(2.5, n_years * 0.45)
    ari_h  = max(2.5, n_years * 0.40)
    if figsize is None:
        figsize = (16, 2.5 + max(heat_h, ari_h))

    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(2, 2,
                           height_ratios=[1.8, max(heat_h, ari_h)],
                           width_ratios=[2.8, 1.0],
                           hspace=0.45, wspace=0.30)

    ax_ent  = fig.add_subplot(gs[0, :])
    ax_heat = fig.add_subplot(gs[1, 0])
    ax_ari  = fig.add_subplot(gs[1, 1])

    month_mids   = [day_offsets[m] + calendar.monthrange(2000, m)[1] / 2
                    for m in season_months]
    month_strs   = [month_abbr[m - 1] for m in season_months]
    month_vlines = [day_offsets[m] for m in season_months[1:]]

    # ── (a) Entropy ──────────────────────────────────────────────────────────
    valid_e = ~np.isnan(H_norm)
    ax_ent.fill_between(bin_centers[valid_e], 0, H_norm[valid_e],
                        alpha=0.25, color='steelblue')
    ax_ent.plot(bin_centers[valid_e], H_norm[valid_e],
                color='steelblue', lw=2)
    ax_ent.axhline(0.5, color='0.5', ls='--', lw=1,
                   label='H = 0.5 (reference)')
    for x in month_vlines:
        ax_ent.axvline(x, color='0.75', lw=0.8, ls=':')
    ax_ent.set_xticks(month_mids)
    ax_ent.set_xticklabels(month_strs, fontsize=9)
    ax_ent.tick_params(axis='x', length=0)
    ax_ent.set_xlim(0, total_days)
    ax_ent.set_ylim(0, 1.08)
    ax_ent.set_ylabel('Normalized Entropy', fontsize=9)
    ax_ent.set_title('(a)  0 = Same cluster every year   |   1 = Uniform distribution',
                     fontsize=10)
    ax_ent.legend(fontsize=8, framealpha=0)
    ax_ent.spines[['top', 'right']].set_visible(False)

    # ── (b) Dominant cluster heatmap ─────────────────────────────────────────
    cmap_disc = ListedColormap([cl_colors[k] for k in clusters])
    norm_disc = BoundaryNorm(np.arange(-0.5, n_cl + 0.5), n_cl)

    dom_idx = np.full_like(dominant, np.nan)
    for i in range(n_years):
        for bi in range(n_bins):
            if not np.isnan(dominant[i, bi]):
                dom_idx[i, bi] = cl_to_idx[int(dominant[i, bi])]

    im = ax_heat.imshow(
        np.ma.masked_invalid(dom_idx),
        aspect='auto', cmap=cmap_disc, norm=norm_disc,
        extent=[0, total_days, n_years - 0.5, -0.5],
        interpolation='none',
    )
    for x in month_vlines:
        ax_heat.axvline(x, color='white', lw=1.2)
    ax_heat.set_xticks(month_mids)
    ax_heat.set_xticklabels(month_strs, fontsize=9)
    ax_heat.tick_params(axis='x', length=0)
    ax_heat.set_yticks(range(n_years))
    ax_heat.set_yticklabels(all_years, fontsize=9)
    ax_heat.set_xlim(0, total_days)
    ax_heat.set_title(f'(b)  Dominant cluster per window {window}d × year', fontsize=10)
    for sp in ax_heat.spines.values():
        sp.set_visible(False)

    cbar = fig.colorbar(im, ax=ax_heat, orientation='horizontal',
                        pad=0.12, shrink=0.55, aspect=22,
                        ticks=list(range(n_cl)))
    cbar.set_ticklabels([f'C{k}' for k in clusters], fontsize=8)

    # ── (c) ARI matrix ───────────────────────────────────────────────────────
    masked_ari = np.ma.masked_invalid(ari_mat)
    im2 = ax_ari.imshow(masked_ari, vmin=-0.2, vmax=1.0,
                        cmap='RdYlGn', aspect='auto')
    for i in range(n_years):
        for j in range(n_years):
            if not np.isnan(ari_mat[i, j]):
                txt_col = 'white' if abs(ari_mat[i, j]) > 0.7 else 'black'
                ax_ari.text(j, i, f'{ari_mat[i, j]:.2f}',
                            ha='center', va='center',
                            fontsize=max(6, 9 - n_years // 3),
                            color=txt_col)
    ax_ari.set_xticks(range(n_years))
    ax_ari.set_xticklabels(all_years, rotation=45, ha='right', fontsize=8)
    ax_ari.set_yticks(range(n_years))
    ax_ari.set_yticklabels(all_years, fontsize=8)
    ax_ari.set_title('(c) Inter-annual ARI', fontsize=10)
    for sp in ax_ari.spines.values():
        sp.set_visible(False)

    cbar2 = fig.colorbar(im2, ax=ax_ari, orientation='horizontal',
                         pad=0.12, shrink=0.9, aspect=15)
    cbar2.set_label('ARI', fontsize=8)
    cbar2.ax.tick_params(labelsize=7)

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return fig, {
        'dominant':    dominant,
        'entropy':     H_norm,
        'ari':         ari_mat,
        'bin_centers': bin_centers,
        'years':       all_years,
    }


class ClusterProfiles:

    def __init__(self, variables=['t', 'td', 't-td','u', 'v']):
        self.variables = variables
        self.X = None
        self.X_clean = None
        self.X_norm = None
        self.labels = None
        self.kmeans = None
        self.clean_mask = None
        self.X_red = None
        self.Z_linkage = None
        self.pca = None
        self.labels_wind  = None
        self.labels_therm = None
        self._twostage_map = None
        self.X_wind_red        = None
        self._k_wind           = None
        self._k_therm          = None
        self._X_therm_red_bywind = None

    # Build matrix
    def build_X(self, soundings, T_anomaly=False):
        n_soundings = len(soundings)
        n_levels = len(soundings[0]['data'])

        X = np.full((n_soundings, n_levels * len(self.variables)), np.nan)
        GAMMA_D = 9.8 / 1000  # K/m
        for i, s in enumerate(soundings):
            df = s['data'].copy()
            df['t-td'] = df['t'] - df['td']
            z = df['altitude'].values
            T = df['t'].values
            Td = df['td'].values
            
            z_rel = z - z[0]
            T_adiabat = T[0] - GAMMA_D * z_rel
            T_anom = T - T_adiabat
            df['t-adiabat'] = T_anom
            df['td-adiabat']= Td-T_adiabat
            # Reconstruction u/v 
            if 'ff' in df.columns and 'dd' in df.columns:
                ff = df['ff'].values
                dd = np.deg2rad(df['dd'].values)

                u = -ff * np.sin(dd)
                v = -ff * np.cos(dd)

                df['u'] = u
                df['v'] = v

            else:
                print(f"Warning: ff or dd missing in sounding {i}")

            
            # Fill matrix
            for j, var in enumerate(self.variables):
                if var in df.columns:
                    if var =='td' and T_anomaly: var == 'td-adiabat'
                    if var =='t' and T_anomaly: var == 't-adiabat' #can optionally cluster on the temperature anomaly relative to the dry adiabat
                    X[i, j*n_levels:(j+1)*n_levels] = df[var].values
                else:
                    print(f"Warning: {var} missing in sounding {i}")
                    X[i, j*n_levels:(j+1)*n_levels] = np.nan

        self.n_levels = n_levels
        self.X = X

        return X
    
    def treat_nan(self, strategy=("mean", "median", "most_frequent","drop")):
        if strategy == "drop":
            mask = ~np.isnan(self.X).any(axis=1)
            self.clean_mask = mask
            self.X_clean = self.X[mask]
        else:
            self.clean_mask = np.ones(len(self.X), dtype=bool)
            imputer = SimpleImputer(strategy=strategy)
            self.X_clean = imputer.fit_transform(self.X)

        return self.X_clean

    # NORMALIZATION
    def normalize(self):
        mean = np.nanmean(self.X_clean, axis=0)
        std = np.nanstd(self.X_clean, axis=0)

        # avoid division by zero
        std[std == 0] = 1

        self.X_norm = (self.X_clean - mean) / std
        return self.X_norm

    def normalize_shape(self, shape_vars=None, global_scale=False):
        """
        Min-max normalisation of vertical profiles.

        Parameters
        ----------
        shape_vars   : list of str, optional
            Variables to normalise. Default: all variables.
            Variables not in the list keep their raw values.
        global_scale : bool, default False
            False (default) — per-profile normalisation: each sounding is
                mapped independently to [0, 1]. Removes the absolute level
                and amplitude; only the shape is preserved.
                Unstable for u/v when wind is calm (range ≈ 0).
            True — global normalisation: min and max are computed across
                all soundings for each variable. Absolute levels are
                preserved relative to other soundings.
                Preserves regime differences (cold katabatic vs. warm marine)
                while bounding values in [0, 1].
        """
        if shape_vars is None:
            shape_vars = self.variables

        X = self.X_clean.copy().astype(float)

        if global_scale:
            for j, var in enumerate(self.variables):
                if var not in shape_vars:
                    continue
                sl = slice(j * self.n_levels, (j + 1) * self.n_levels)
                block = X[:, sl]
                gmin = np.nanmin(block)
                gmax = np.nanmax(block)
                rng  = gmax - gmin
                if rng > 1e-6:
                    X[:, sl] = (block - gmin) / rng
                else:
                    X[:, sl] = 0.5
        else:
            for j, var in enumerate(self.variables):
                if var not in shape_vars:
                    continue
                sl    = slice(j * self.n_levels, (j + 1) * self.n_levels)
                block = X[:, sl]                                         # (n_soundings, n_levels)
                vmin  = np.nanmin(block, axis=1, keepdims=True)
                vmax  = np.nanmax(block, axis=1, keepdims=True)
                rng   = vmax - vmin
                safe  = rng > 1e-6
                X[:, sl] = np.where(safe, (block - vmin) / np.where(safe, rng, 1.0), 0.5)

        self.X_norm = X
        return self.X_norm

    def normalize_block(self, variables=None,method="std"):
        """
        Block scaling applied after normalize() or normalize_shape().
        Each specified group of variables is rescaled to contribute
        equally to the total variance, independently of the variable's
        natural amplitude and the number of levels.

        Without this rescaling, u/v (large amplitude, vertically coherent)
        dominate the first PCs and drown out the T/Td variability.

        Call after normalize() or normalize_shape(), before apply_pca().

        Parameters
        ----------
        variables : list of str or None
            Variables to rescale by block scaling.
            E.g. ['u', 'v'] to rescale only the wind components.
            None (default): all variables are rescaled.
        """
        if method not in ["std", "z", "max"]:
            raise ValueError(f"Method '{method}' not recognized. "
                             "Choose from 'std', 'z', or 'max'.")
        if self.X_norm is None:
            X = self.X_clean
        else:
            X = self.X_norm.copy()
            
        if variables is None:
            variables = self.variables

        # Check that all requested variables exist
        unknown = [v for v in variables if v not in self.variables]
        if unknown:
            raise ValueError(f"Unknown variables: {unknown}. "
                            f"Available variables: {self.variables}")

        
        for var in variables:
            j = self.variables.index(var)
            sl = slice(j * self.n_levels, (j + 1) * self.n_levels)
            block_std = np.sqrt(np.mean(np.var(X[:, sl], axis=0)))
            block_mean = np.mean(X[:, sl])
            block_max = np.max(X[:, sl])
            if method == "std":
                if block_std > 1e-8:
                    X[:, sl] /= block_std
            elif method == "z":
                if block_std > 1e-8:
                    X[:, sl] -= block_mean
                    X[:, sl] /= block_std
            elif method == "max":
                if block_max > 1e-8:
                    X[:, sl] /= block_max
        self.X_norm = X
        return X

    # PCA ANALYSIS
    def find_nb_of_components(self):
        pca = PCA()
        pca.fit(self.X_norm)

        cumvar = np.cumsum(pca.explained_variance_ratio_)

        plt.plot(cumvar)
        plt.xlabel("Number of components")
        plt.ylabel("Cumulative explained variance")
        plt.grid()
        plt.xlim((0,15))
        plt.show()

    def apply_pca(self, n_components):
        pca = PCA(n_components=n_components)
        self.X_red = pca.fit_transform(self.X_norm)
        self.pca = pca
        return self.X_red

    def plot_pca_loadings(self, z_grid, n_components=None, figsize=None, save_path=None):
        """
        Visualize PCA component loadings as vertical profiles.

        For each PC, the loadings are reshaped into the original variable × altitude
        format and plotted as vertical profiles (one subplot per variable, one line
        per PC). A positive loading at altitude z means that higher values of that
        variable at that level push a sounding toward the positive end of the PC axis.

        Requires apply_pca to have been called first.

        Parameters
        ----------
        z_grid       : array-like  altitude grid (m), same length as used in build_X
        n_components : int or None  number of PCs to plot (default: all stored)
        figsize      : tuple or None
        save_path    : str or None

        Returns
        -------
        fig         : matplotlib.figure.Figure
        df_loadings : pd.DataFrame  (index = feature labels, columns = PC1, PC2, ...)
        """
        if self.pca is None:
            raise ValueError("Run apply_pca before plot_pca_loadings.")

        z_grid = np.asarray(z_grid)
        components = self.pca.components_
        n_comp_total = components.shape[0]

        if n_components is None:
            n_components = n_comp_total
        else:
            n_components = min(n_components, n_comp_total)

        n_vars   = len(self.variables)
        n_levels = self.n_levels

        cmap   = plt.cm.tab10
        colors = [cmap(i / 10) for i in range(n_components)]

        _var_labels = {
            't': 'Temperature', 'td': 'Dew Point', 't-td': 'T−Td',
            'u': 'u wind', 'v': 'v wind',
        }

        if figsize is None:
            figsize = (n_vars * 3.5, 6)

        fig, axes = plt.subplots(1, n_vars, figsize=figsize, sharey=True)
        if n_vars == 1:
            axes = [axes]

        for pc_idx in range(n_components):
            loading = components[pc_idx]
            ev      = self.pca.explained_variance_ratio_[pc_idx] * 100
            label   = f'PC{pc_idx + 1}  ({ev:.1f}%)'

            for j, var in enumerate(self.variables):
                var_loading = loading[j * n_levels: (j + 1) * n_levels]
                axes[j].plot(var_loading, z_grid, color=colors[pc_idx], lw=2,
                             label=label if j == 0 else None)

        for j, var in enumerate(self.variables):
            ax = axes[j]
            ax.axvline(0, color='k', lw=0.7, ls='--', alpha=0.5)
            ax.set_xlabel('Loading', fontsize=10)
            ax.set_title(_var_labels.get(var, var), fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.spines[['top', 'right']].set_visible(False)

        axes[0].set_ylabel('Altitude (m)', fontsize=10)

        handles, labels_leg = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_leg, loc='lower center',
                   ncol=min(n_components, 5), fontsize=9,
                   bbox_to_anchor=(0.5, -0.04), framealpha=0.9)

        fig.suptitle('PCA — Loadings as vertical profiles', fontsize=13)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.14)

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        feature_names = [f'{var}_z{int(z)}' for var in self.variables for z in z_grid]
        df_loadings = pd.DataFrame(
            components[:n_components].T,
            index=feature_names,
            columns=[f'PC{i + 1}' for i in range(n_components)],
        )
        return fig, df_loadings

    # CLUSTER NUMBER
    def find_nb_of_clusters(self, X, max_k=10, seed=0, plot=True, n_init='auto'):
        inertia = []
        silhouettes = []
        K = range(2, max_k + 1)

        for k in K:
            kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
            labels = kmeans.fit_predict(X)
            inertia.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X, labels))

        if plot:
            plt.plot(K, inertia, marker='o')
            plt.xlabel("Number of clusters")
            plt.ylabel("Inertia")
            plt.grid()
            plt.show()

            plt.plot(K, silhouettes, marker='o')
            plt.xlabel("Number of clusters")
            plt.ylabel("Silhouette Score")
            plt.grid()
            plt.show()

        return silhouettes

    # FIT CLUSTERING
    def fit_kmeans(self, k, seed=0,n_init=20):
        self.kmeans = KMeans(n_clusters=k, n_init=n_init ,random_state=seed)
        self.labels = self.kmeans.fit_predict(self.X_red)
        return self.labels

    # ── HIERARCHICAL CLUSTERING ──────────────────────────────────────────────

    def plot_dendrogram(self, linkage_method='ward', truncate_mode='lastp',
                        p=20, show_cut_at_k=None, figsize=(14, 5)):
        """
        Plot a dendrogram on self.X_red and store the linkage matrix in
        self.Z_linkage for reuse by fit_hierarchical.

        Parameters
        ----------
        linkage_method : str   'ward' | 'complete' | 'average' | 'single'
        truncate_mode  : str   'lastp' shows only the last p merges (cleaner
                               for large datasets). Set to None for the full tree.
        p              : int   number of leaf nodes / merge nodes to show
        show_cut_at_k  : int   draws a dashed line at the height that yields k
                               clusters, and colours the dendrogram accordingly
        """
        from scipy.cluster.hierarchy import dendrogram, linkage as sp_linkage

        if self.X_red is None:
            raise ValueError("Run apply_pca before plot_dendrogram.")

        Z = sp_linkage(self.X_red, method=linkage_method)
        self.Z_linkage = Z

        cut_h = None
        if show_cut_at_k is not None:
            k, n = show_cut_at_k, len(Z) + 1
            if 2 <= k < n:
                cut_h = (Z[-k, 2] + Z[-k + 1, 2]) / 2

        fig, ax = plt.subplots(figsize=figsize)
        dendrogram(Z, truncate_mode=truncate_mode, p=p, ax=ax,
                   color_threshold=cut_h if cut_h is not None else 0,
                   above_threshold_color='grey')

        if cut_h is not None:
            ax.axhline(cut_h, color='red', ls='--', lw=1.5,
                       label=f'{show_cut_at_k} clusters  (h={cut_h:.3f})')
            ax.legend()

        ax.set_title(f"Dendrogram  —  {linkage_method} linkage")
        ax.set_xlabel("Soundings (merged nodes)")
        ax.set_ylabel("Distance")
        plt.tight_layout()
        plt.show()
        return Z

    def fit_hierarchical(self, k, linkage_method='ward'):
        """
        Fit agglomerative clustering on self.X_red.
        Sets self.labels — compatible with all plot_* and cluster_summary_full
        methods, exactly like fit_kmeans.

        Parameters
        ----------
        k              : int  number of clusters
        linkage_method : str  'ward' | 'complete' | 'average' | 'single'
        """
        from sklearn.cluster import AgglomerativeClustering

        if self.X_red is None:
            raise ValueError("Run apply_pca before fit_hierarchical.")

        model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
        self.labels = model.fit_predict(self.X_red)
        return self.labels

    def fit_twostage(self, k_wind,
                     wind_vars=None, therm_vars=None,
                     n_pca_wind=3,
                     k_therm=None,
                     n_pca_therm=2,
                     max_k_therm=5,
                     cumvar_therm=0.90,
                     min_silhouette=0.05,
                     min_cluster_size=10,
                     n_init='auto',
                     seed=0):
        """
        Two-stage clustering: dynamic regimes (wind) then thermal
        subtypes within each regime.

        Stage 1 — cluster on wind_vars → k_wind wind regimes.
        Stage 2 — for each wind regime, sub-cluster on therm_vars.

        Parameters
        ----------
        k_wind         : int    number of wind clusters
        wind_vars      : list   wind variables (default: ['u','v'])
        therm_vars     : list   thermal variables (default: ['t','td','t-td'])
        n_pca_wind     : int    wind PCA components (default 3)
        k_therm        : int or None
            int  → fixed number of thermal sub-clusters per wind regime;
                   n_pca_therm is used for reduction.
            None → automatic selection per wind regime:
                   n_pca determined by cumvar >= cumvar_therm,
                   k chosen by best silhouette in 2..max_k_therm
                   (stays k=1 if no k>=2 exceeds min_silhouette).
        n_pca_therm    : int    thermal PCA components in fixed k_therm mode (default 2)
        max_k_therm      : int    maximum k tested in auto mode (default 5)
        cumvar_therm     : float  cumulative variance threshold for auto n_pca (default 0.90)
        min_silhouette   : float  minimum silhouette to accept k>1 (default 0.05)
        min_cluster_size : int    minimum size of a sub-cluster to validate k
            If after KMeans(k) a sub-cluster has fewer than min_cluster_size profiles,
            that k is rejected and the next k (by decreasing silhouette) is tested.
            If no valid k exists, falls back to k=1. (default 10)

        Returns
        -------
        labels_wind  : array  wind labels (0..k_wind-1)
        labels_therm : array  thermal labels within each wind cluster frame
        labels       : array  combined sequential labels 0..N-1 (= self.labels)
        """
        if self.X_norm is None:
            raise ValueError("Run normalize before fit_twostage.")

        if wind_vars is None:
            wind_vars = [v for v in self.variables if v in ('u', 'v')]
        if therm_vars is None:
            therm_vars = [v for v in self.variables if v in ('t', 'td', 't-td')]
        if not wind_vars:
            raise ValueError("No wind variable found. Specify wind_vars.")
        if not therm_vars:
            raise ValueError("No thermal variable found. Specify therm_vars.")

        def extract_blocks(var_list):
            blocks = []
            for v in var_list:
                if v not in self.variables:
                    raise ValueError(f"Variable '{v}' not found in self.variables.")
                j = self.variables.index(v)
                blocks.append(self.X_norm[:, j * self.n_levels:(j + 1) * self.n_levels])
            return np.concatenate(blocks, axis=1)

        X_wind  = extract_blocks(wind_vars)
        X_therm = extract_blocks(therm_vars)
        n       = X_wind.shape[0]
        auto    = (k_therm is None)

        # ── Stage 1: clustering on wind ─────────────────────────────────────
        n_pc_w  = min(n_pca_wind, X_wind.shape[1], n - 1)
        X_w_red = PCA(n_components=n_pc_w).fit_transform(X_wind)
        labels_wind = KMeans(n_clusters=k_wind, n_init=n_init, random_state=seed).fit_predict(X_w_red)
        self.labels_wind = labels_wind
        self.X_wind_red  = X_w_red
        self._k_wind     = k_wind

        # ── Stage 2: thermal sub-clustering per wind regime ─────────────────
        print(f"\n── Thermal stage {'(auto)' if auto else f'(k_therm={k_therm} fixed)'} ──")
        labels_therm       = np.zeros(n, dtype=int)
        X_therm_red_bywind = {}
        k_therm_per_wind   = {}

        for w in range(k_wind):
            mask = labels_wind == w
            n_w  = mask.sum()
            X_t  = X_therm[mask]

            if auto:
                # n_pca by cumvar >= cumvar_therm
                pca_full = PCA().fit(X_t)
                cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
                n_pc_t   = int(np.searchsorted(cumvar, cumvar_therm) + 1)
                n_pc_t   = max(1, min(n_pc_t, X_t.shape[1], n_w - 1))
                X_t_red  = PCA(n_components=n_pc_t).fit_transform(X_t)

                # Compute silhouette + labels for all candidate k values
                sil_scores = {}
                km_labels  = {}
                for k_try in range(2, min(max_k_therm + 1, n_w)):
                    labs = KMeans(n_clusters=k_try, n_init=n_init,
                                  random_state=seed).fit_predict(X_t_red)
                    if len(np.unique(labs)) >= 2:
                        sil_scores[k_try] = silhouette_score(X_t_red, labs)
                        km_labels[k_try]  = labs

                # Cascade: k sorted by decreasing silhouette,
                # accepted only if all sub-clusters >= min_cluster_size
                best_k      = 1
                reject_log  = []
                candidates  = sorted(
                    [(k, s) for k, s in sil_scores.items() if s >= min_silhouette],
                    key=lambda x: x[1], reverse=True
                )
                for k_try, sil in candidates:
                    labs       = km_labels[k_try]
                    sizes      = [int((labs == c).sum()) for c in np.unique(labs)]
                    min_size   = min(sizes)
                    if min_size >= min_cluster_size:
                        best_k = k_try
                        break
                    reject_log.append(f'k{k_try}(min={min_size}<{min_cluster_size})')

                sil_str    = ', '.join(f'k{k}:{v:.3f}' for k, v in sil_scores.items())
                reject_str = f'  rejected=[{", ".join(reject_log)}]' if reject_log else ''
                print(f"  Wind {w} (n={n_w}) : n_pca={n_pc_t} "
                      f"(cumvar={cumvar[n_pc_t-1]:.1%})  "
                      f"sil=[{sil_str}]{reject_str}  -> k_therm={best_k}")
            else:
                n_pc_t  = min(n_pca_therm, X_t.shape[1], n_w - 1)
                X_t_red = PCA(n_components=n_pc_t).fit_transform(X_t)
                best_k  = k_therm if n_w >= k_therm else 1

            k_therm_per_wind[w]   = best_k
            X_therm_red_bywind[w] = X_t_red

            if best_k >= 2:
                if auto and best_k in km_labels:
                    labels_therm[mask] = km_labels[best_k]
                else:
                    labels_therm[mask] = KMeans(n_clusters=best_k, n_init='auto',
                                                random_state=seed).fit_predict(X_t_red)
            # else labels_therm[mask] stays 0

        self.labels_therm        = labels_therm
        self._X_therm_red_bywind = X_therm_red_bywind
        self._k_therm            = k_therm_per_wind   # dict {w: k_t}

        # ── Combined sequential labels ───────────────────────────────────────
        pair_to_combined = {}
        counter = 0
        for w in range(k_wind):
            for t in range(k_therm_per_wind[w]):
                pair_to_combined[(w, t)] = counter
                counter += 1

        self.labels = np.array([pair_to_combined[(w, t)]
                                 for w, t in zip(labels_wind, labels_therm)])
        self._twostage_map = {c: wt for wt, c in pair_to_combined.items()}

        n_total = len(pair_to_combined)
        print(f"\nResult: {k_wind} wind regimes -> {n_total} combined clusters")
        for c, (w, t) in sorted(self._twostage_map.items()):
            print(f"  C{c}  [wind={w}, therm={t}]  n={(self.labels == c).sum()}")

        return self.labels_wind, self.labels_therm, self.labels

    def plot_twostage_scatter(self, figsize=(14, 6), save_path=None):
        """
        Two PCA scatter panels (PC1 vs PC2 in wind space) after fit_twostage.

        Panel (a) — wind regimes only: one colour per wind cluster.
        Panel (b) — thermal subtypes: same colour family = same wind regime
                    (dark to light), different marker per thermal subtype.
                    Each point's edge colour repeats the base wind-regime
                    colour to reinforce membership.

        Parameters
        ----------
        figsize   : tuple
        save_path : str or None
        """
        if self._twostage_map is None:
            raise ValueError("Run fit_twostage before plot_twostage_scatter.")
        if self.X_wind_red is None:
            raise ValueError("Rerun fit_twostage to regenerate wind PCA coordinates.")

        k_wind       = self._k_wind
        k_therm_dict = self._k_therm   # {w: k_t}
        max_kt       = max(k_therm_dict.values())

        base_cmap   = plt.cm.tab10
        base_colors = [base_cmap(i / 10) for i in range(k_wind)]
        families    = {w: _color_family(base_colors[w], k_therm_dict[w])
                       for w in range(k_wind)}
        markers     = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
        reverse_map = {v: k for k, v in self._twostage_map.items()}

        pc1 = self.X_wind_red[:, 0]
        pc2 = self.X_wind_red[:, 1]

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # ── Panel (a): wind regimes ──────────────────────────────────────────
        ax = axes[0]
        for w in range(k_wind):
            mask = self.labels_wind == w
            ax.scatter(pc1[mask], pc2[mask],
                       color=base_colors[w], alpha=0.7, s=22,
                       label=f'Wind {w}  (n={mask.sum()})')

        ax.set_xlabel('PC1 (wind)', fontsize=10)
        ax.set_ylabel('PC2 (wind)', fontsize=10)
        ax.set_title('(a) Wind regimes', fontsize=12)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)

        # ── Panel (b): thermal subtypes ──────────────────────────────────────
        ax = axes[1]
        handles = []

        for w in range(k_wind):
            for t in range(k_therm_dict[w]):
                if (w, t) not in reverse_map:
                    continue
                c    = reverse_map[(w, t)]
                mask = self.labels == c
                sc = ax.scatter(
                    pc1[mask], pc2[mask],
                    color=families[w][t],
                    edgecolors=base_colors[w],
                    linewidths=0.6,
                    marker=markers[t % len(markers)],
                    alpha=0.80, s=30,
                    label=f'C{c} [wind={w}, therm={t}]  n={mask.sum()}'
                )
                handles.append(sc)

        ax.set_xlabel('PC1 (wind)', fontsize=10)
        ax.set_ylabel('PC2 (wind)', fontsize=10)
        ax.set_title('(b) Thermal subclusters', fontsize=12)
        ax.grid(True, alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)

        ax.legend(handles=handles, fontsize=8, framealpha=0.9,
                  ncol=max(max_kt, 1), loc='best',
                  title='color = wind   |   marker = therm', title_fontsize=7)

        plt.suptitle('Two stages clustering — PCA wind space', fontsize=13)
        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    def plot_twostage_thermal_scatter(self, figsize=None, save_path=None):
        """
        Thermal PCA space of sub-clusters, one subplot per wind regime.

        Each panel shows soundings from one wind regime projected onto the
        first two components of the thermal PCA specific to that regime.
        Points are coloured by thermal subtype.

        PCA spaces are independent across panels (each wind regime has its
        own thermal decomposition), making it possible to see how subtypes
        separate within each regime.

        Parameters
        ----------
        figsize   : tuple or None  (default: (4.5*k_wind, 4.5))
        save_path : str or None
        """
        if self._twostage_map is None:
            raise ValueError("Run fit_twostage before plot_twostage_thermal_scatter.")
        if self._X_therm_red_bywind is None:
            raise ValueError("Rerun fit_twostage to regenerate thermal PCA coordinates.")

        k_wind       = self._k_wind
        k_therm_dict = self._k_therm   # {w: k_t}

        base_cmap   = plt.cm.tab10
        base_colors = [base_cmap(i / 10) for i in range(k_wind)]
        families    = {w: _color_family(base_colors[w], k_therm_dict[w])
                       for w in range(k_wind)}
        markers     = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
        reverse_map = {v: k for k, v in self._twostage_map.items()}

        if figsize is None:
            figsize = (4.5 * k_wind, 4.5)

        fig, axes = plt.subplots(1, k_wind, figsize=figsize)
        if k_wind == 1:
            axes = [axes]

        for w, ax in enumerate(axes):
            X_t    = self._X_therm_red_bywind.get(w)
            mask_w = self.labels_wind == w
            k_t    = k_therm_dict[w]

            if X_t is None or X_t.shape[1] < 2:
                ax.text(0.5, 0.5, f'Wind {w}\nk_therm=1\n(no subdivision)',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=10, color=base_colors[w])
                ax.set_title(f'Wind regime {w}  (n={mask_w.sum()})', fontsize=11,
                             color=base_colors[w], fontweight='bold')
                ax.axis('off')
                continue

            local_therm = self.labels_therm[mask_w]

            for t in range(k_t):
                local_mask = local_therm == t
                if local_mask.sum() == 0:
                    continue
                c = reverse_map.get((w, t))
                label = (f'C{c} [therm={t}]  n={local_mask.sum()}'
                         if c is not None else f'therm={t}  n={local_mask.sum()}')
                ax.scatter(
                    X_t[local_mask, 0], X_t[local_mask, 1],
                    color=families[w][t],
                    edgecolors=base_colors[w],
                    linewidths=0.6,
                    marker=markers[t % len(markers)],
                    alpha=0.80, s=35,
                    label=label,
                )

            n_w = mask_w.sum()
            ax.set_title(f'Wind regime {w}  (n={n_w}, k_therm={k_t})', fontsize=11,
                         color=base_colors[w], fontweight='bold')
            ax.set_xlabel('PC1 (therm)', fontsize=9)
            ax.set_ylabel('PC2 (therm)', fontsize=9)
            ax.legend(fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.25)
            ax.spines[['top', 'right']].set_visible(False)

        plt.suptitle('Thermal sub clusters — Thermal PCA space per wind regime',
                     fontsize=12)
        plt.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return fig

    def silhouette_hierarchical(self, max_k=10, linkage_method='ward', figsize=(8, 4)):
        """
        Compute silhouette scores for agglomerative clustering over k = 2…max_k.

        Reuses self.Z_linkage if already computed by plot_dendrogram (no
        recomputation). The dendrogram is cut at each k via fcluster, which is
        much faster than re-running AgglomerativeClustering for each k.

        Parameters
        ----------
        max_k          : int   maximum k to evaluate (default 10)
        linkage_method : str   used only if Z_linkage is not yet stored
        figsize        : tuple

        Returns
        -------
        scores : dict  {k: silhouette_score}
        """
        from scipy.cluster.hierarchy import linkage as sp_linkage, fcluster

        if self.X_red is None:
            raise ValueError("Run apply_pca before silhouette_hierarchical.")

        if self.Z_linkage is None:
            self.Z_linkage = sp_linkage(self.X_red, method=linkage_method)

        scores = {}
        for k in range(2, max_k + 1):
            labels = fcluster(self.Z_linkage, t=k, criterion='maxclust') - 1
            scores[k] = silhouette_score(self.X_red, labels)

        best_k = max(scores, key=scores.get)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(list(scores.keys()), list(scores.values()), marker='o')
        ax.axvline(best_k, color='red', ls='--', lw=1,
                   label=f'best k={best_k}  ({scores[best_k]:.3f})')
        ax.set_xlabel("Nb clusters k")
        ax.set_ylabel("Silhouette score")
        ax.set_title(f"Silhouette — hierarchical clustering  ({linkage_method})")
        ax.legend()
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.show()

        return scores

    def plot_pca_scatter(self, title=''):
        """
        Scatter plot of self.X_red coloured by self.labels.
        Works identically after fit_kmeans or fit_hierarchical.
        """
        if self.labels is None:
            raise ValueError("Run fit_kmeans or fit_hierarchical first.")

        clusters = np.unique(self.labels)
        colors   = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
        fig, ax  = plt.subplots(figsize=(8, 6))

        for k, c in zip(clusters, colors):
            m = self.labels == k
            ax.scatter(self.X_red[m, 0], self.X_red[m, 1],
                       label=f'C{k}  (n={m.sum()})', color=c, alpha=0.7, s=20)

        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_title(title or 'Clusters in PCA space')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # PLOT MEAN PROFILES
    def plot_cluster_mean_profiles(self, z_grid):

        if self.labels is None:
            raise ValueError("Run clustering before plotting")

        n_clusters = len(np.unique(self.labels))
        n_levels = len(z_grid)

        fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
        colors = plt.cm.tab10(np.arange(n_clusters))
        for k in range(n_clusters):
        
            cluster_profiles = self.X_clean[self.labels == k]
            mean_profile = np.nanmean(cluster_profiles, axis=0)

            T  = mean_profile[0:n_levels]
            Td = mean_profile[n_levels:2*n_levels]#or t-td
            u  = mean_profile[2*n_levels:3*n_levels]
            
            if "v" in self.variables: #sometimes v is not available
                v  = mean_profile[3*n_levels:4*n_levels]
            if "u" in self.variables and "v" in self.variables:
                ff = np.sqrt(u**2 + v**2)

            axes[0].plot(T - 273.15, z_grid, label=f"C{k}", color=colors[k])
            axes[0].plot(Td - 273.15, z_grid, linestyle='--', color=colors[k]) if "td" in self.variables else axes[0].plot(-(Td-T)-273.15, z_grid, linestyle='--', color=colors[k])

            if "u" in self.variables and "v" in self.variables:
                axes[1].plot(ff, z_grid, label=f"C{k}", color=colors[k])

                axes[2].plot(u, z_grid, label=f"C{k}", color=colors[k])
                axes[2].plot(v, z_grid, linestyle='--', color=colors[k])
            elif "u" in self.variables:
                axes[2].plot(u, z_grid, label=f"C{k}", color=colors[k])
            elif "v" in self.variables:
                axes[2].plot(v, z_grid, label=f"C{k}", color=colors[k])

        axes[0].set_xlabel("Temp (°C)")
        axes[0].set_title("T / Td")

        axes[1].set_xlabel("Wind speed (m/s)")
        axes[1].set_title("Wind speed")

        axes[2].set_xlabel("u / v (m/s)")
        axes[2].set_title("Wind components")

        axes[0].set_ylabel("Altitude (m)")

        for ax in axes:
            ax.grid(True)

        axes[0].legend()
        axes[1].legend()
        axes[2].legend()

        plt.tight_layout()
        plt.show()

    def plot_cluster_minipages(self, soundings, z_grid,
                          quantiles=(0.1, 0.9),
                          ylim=(0, 15000)):

        if self.labels is None:
            raise ValueError("Run clustering first")

        import numpy as np
        import matplotlib.pyplot as plt

        clusters = np.unique(self.labels)
        K = len(clusters)

        # COLLECT DATA PER CLUSTER
        data = {}

        for k in clusters:

            idx_k = np.where(self.labels == k)[0]

            T_all, Td_all = [], []
            u_all, v_all, ff_all = [], [], []

            for i in idx_k:
                df = soundings[i]["data"]

                z = df["altitude"].values
                mask = (z >= ylim[0]) & (z <= ylim[1])

                T = df["t"].values[mask] - 273.15
                Td = df["td"].values[mask] - 273.15

                if "ff" in df.columns and "dd" in df.columns:
                    theta = np.deg2rad(df["dd"].values[mask])
                    ff = df["ff"].values[mask]
                    u = -ff * np.sin(theta)
                    v = -ff * np.cos(theta)
                else:
                    u = v = ff = np.full_like(T, np.nan)

                T_all.append(T)
                Td_all.append(Td)
                u_all.append(u)
                v_all.append(v)
                ff_all.append(ff)

            def stats(arr):
                arr = np.array(arr)
                mean = np.nanmean(arr, axis=0)
                q1 = np.nanquantile(arr, quantiles[0], axis=0)
                q2 = np.nanquantile(arr, quantiles[1], axis=0)
                return mean, q1, q2

            data[k] = {
                "T": stats(T_all),
                "Td": stats(Td_all),
                "u": stats(u_all),
                "v": stats(v_all),
                "ff": stats(ff_all),
            }

        # 1. TEMPERATURE
        fig, axes = plt.subplots(1, K, figsize=(4*K, 6), sharey=True)

        for i, k in enumerate(clusters):
            ax = axes[i] if K > 1 else axes

            T_mean, T_q1, T_q2 = data[k]["T"]
            Td_mean, Td_q1, Td_q2 = data[k]["Td"]

            ax.plot(T_mean, z_grid, color='red')
            ax.fill_betweenx(z_grid, T_q1, T_q2, color='red', alpha=0.3)

            ax.plot(Td_mean, z_grid, color='orange')
            ax.fill_betweenx(z_grid, Td_q1, Td_q2, color='orange', alpha=0.3)

            ax.set_title(f"C{k}")
            ax.grid()
            ax.set_ylim(ylim)

        axes[0].set_ylabel("Altitude (m)")
        fig.suptitle("Temperature / Dew Point")
        plt.tight_layout()
        plt.show()

        # 2. WIND SPEED
        fig, axes = plt.subplots(1, K, figsize=(4*K, 6), sharey=True)

        for i, k in enumerate(clusters):
            ax = axes[i] if K > 1 else axes

            ff_mean, ff_q1, ff_q2 = data[k]["ff"]

            ax.plot(ff_mean, z_grid, color='black')
            ax.fill_betweenx(z_grid, ff_q1, ff_q2, color='black', alpha=0.3)

            ax.set_title(f"C{k}")
            ax.grid()
            ax.set_ylim(ylim)

        axes[0].set_ylabel("Altitude (m)")
        fig.suptitle("Wind Speed")
        plt.tight_layout()
        plt.show()

        # 3. U / V
        fig, axes = plt.subplots(1, K, figsize=(4*K, 6), sharey=True)

        for i, k in enumerate(clusters):
            ax = axes[i] if K > 1 else axes

            u_mean, u_q1, u_q2 = data[k]["u"]
            v_mean, v_q1, v_q2 = data[k]["v"]

            ax.plot(u_mean, z_grid, color='blue', label='u')
            ax.fill_betweenx(z_grid, u_q1, u_q2, color='blue', alpha=0.3)

            ax.plot(v_mean, z_grid, color='green', label='v')
            ax.fill_betweenx(z_grid, v_q1, v_q2, color='green', alpha=0.3)

            ax.set_title(f"C{k}")
            ax.grid()
            ax.set_ylim(ylim)

        axes[0].set_ylabel("Altitude (m)")
        axes[0].legend()
        fig.suptitle("Wind Components")
        plt.tight_layout()
        plt.show()

    def plot_cluster_minipages(self, soundings, z_grid,
                          quantiles=(0.1, 0.9),
                          ylim=(0, 15000)):
        from metpy.calc import relative_humidity_from_dewpoint, saturation_vapor_pressure
        from metpy.units import units

        if self.labels is None:
            raise ValueError("Run clustering first")

        import numpy as np
        import matplotlib.pyplot as plt

        clusters = np.unique(self.labels)
        K = len(clusters)

        # COLLECT DATA PER CLUSTER
        data = {}
        nb_samples_per_cluster = {}

        for k in clusters:

            idx_k = np.where(self.labels == k)[0]
            nb_samples_per_cluster[k] = len(idx_k)

            T_all, Td_all, RH_all, RHi_all = [], [], [], []
            u_all, v_all, ff_all, dd_all = [], [], [], []

            for i in idx_k:
                df = soundings[i]["data"]

                z = df["altitude"].values
                mask = (z >= ylim[0]) & (z <= ylim[1])

                T = df["t"].values[mask] - 273.15
                Td = df["td"].values[mask] - 273.15

                #RH
                t_k = (T + 273.15) * units.kelvin
                td_k = (Td + 273.15) * units.kelvin
                RH = relative_humidity_from_dewpoint(t_k, td_k).to('percent').magnitude
                rhi = RH * (
                    saturation_vapor_pressure(t_k, phase='liquid') /
                    saturation_vapor_pressure(t_k, phase='solid')
                ).magnitude

                if "ff" in df.columns and "dd" in df.columns:
                    ff = df["ff"].values[mask]
                    dd = df["dd"].values[mask]
                    theta = np.deg2rad(dd)
                    u = -ff * np.sin(theta)
                    v = -ff * np.cos(theta)
                else:
                    u = v = ff = np.full_like(T, np.nan)
                    dd = np.full_like(T, np.nan)

                T_all.append(T)
                Td_all.append(Td)
                RH_all.append(RH)
                RHi_all.append(rhi)
                u_all.append(u)
                v_all.append(v)
                ff_all.append(ff)
                dd_all.append(dd)

            def stats(arr):
                arr = np.array(arr)
                mean = np.nanmean(arr, axis=0)
                q1 = np.nanquantile(arr, quantiles[0], axis=0)
                q2 = np.nanquantile(arr, quantiles[1], axis=0)
                return mean, q1, q2

            def circ_mean_profile(arr):
                """Circular mean per level (avoids the 359°+1°=180° artefact)."""
                arr_rad = np.deg2rad(np.array(arr))
                m_sin = np.nanmean(np.sin(arr_rad), axis=0)
                m_cos = np.nanmean(np.cos(arr_rad), axis=0)
                return np.degrees(np.arctan2(m_sin, m_cos)) % 360

            data[k] = {
                "T":   stats(T_all),
                "Td":  stats(Td_all),
                "u":   stats(u_all),
                "v":   stats(v_all),
                "ff":  stats(ff_all),
                "RH":  stats(RH_all),
                "RHi": stats(RHi_all),
                "dd":  circ_mean_profile(dd_all),
            }

        # 1. TEMPERATURE
        fig, axes = plt.subplots(1, K, figsize=(4*K, 6), sharey=True)

        for i, k in enumerate(clusters):
            ax = axes[i] if K > 1 else axes

            T_mean, T_q1, T_q2 = data[k]["T"]
            Td_mean, Td_q1, Td_q2 = data[k]["Td"]

            ax.plot(T_mean, z_grid, color='red')
            ax.fill_betweenx(z_grid, T_q1, T_q2, color='red', alpha=0.3)

            ax.plot(Td_mean, z_grid, color='orange')
            ax.fill_betweenx(z_grid, Td_q1, Td_q2, color='orange', alpha=0.3)

            ax.set_title(f"C{k}")
            ax.grid()
            ax.set_ylim(ylim)

        axes[0].set_ylabel("Altitude (m)")
        fig.suptitle("Temperature / Dew Point")
        plt.tight_layout()
        plt.show()

        # 2. WIND SPEED
        fig, axes = plt.subplots(1, K, figsize=(4*K, 6), sharey=True)

        for i, k in enumerate(clusters):
            ax = axes[i] if K > 1 else axes

            ff_mean, ff_q1, ff_q2 = data[k]["ff"]

            ax.plot(ff_mean, z_grid, color='black')
            ax.fill_betweenx(z_grid, ff_q1, ff_q2, color='black', alpha=0.3)

            ax.set_title(f"C{k}")
            ax.grid()
            ax.set_ylim(ylim)

        axes[0].set_ylabel("Altitude (m)")
        fig.suptitle("Wind Speed")
        plt.tight_layout()
        plt.show()

        # 3. U / V
        fig, axes = plt.subplots(1, K, figsize=(4*K, 6), sharey=True)

        for i, k in enumerate(clusters):
            ax = axes[i] if K > 1 else axes

            u_mean, u_q1, u_q2 = data[k]["u"]
            v_mean, v_q1, v_q2 = data[k]["v"]

            ax.plot(u_mean, z_grid, color='blue', label='u')
            ax.fill_betweenx(z_grid, u_q1, u_q2, color='blue', alpha=0.3)

            ax.plot(v_mean, z_grid, color='green', label='v')
            ax.fill_betweenx(z_grid, v_q1, v_q2, color='green', alpha=0.3)

            ax.set_title(f"C{k}")
            ax.grid()
            ax.set_ylim(ylim)
        
        axes[0].set_ylabel("Altitude (m)")
        axes[0].legend()
        fig.suptitle("Wind Components")
        plt.tight_layout()
        plt.show()

        # RH
        fig, axes = plt.subplots(1, K, figsize=(4*K, 6), sharey=True)

        for i, k in enumerate(clusters):
            ax = axes[i] if K > 1 else axes

            RH_mean, RH_q1, RH_q2 = data[k]["RH"]
            RHi_mean, RHi_q1, RHi_q2 = data[k]["RHi"]

            ax.plot(RH_mean, z_grid, color='steelblue')
            ax.fill_betweenx(z_grid, RH_q1, RH_q2, color='steelblue', alpha=0.3, label='RH')
            ax.plot(RHi_mean, z_grid, color='navy')
            ax.fill_betweenx(z_grid, RHi_q1, RHi_q2, color='navy', alpha=0.3, label='RHi')
            ax.legend()
            ax.axvline(100, color='grey', lw=0.5)

            ax.set_title(f"C{k}")
            ax.set_xlabel("RH (%)")
            ax.grid()
            ax.set_ylim(ylim)

        axes[0].set_ylabel("Altitude (m)")
        fig.suptitle("Relative Humidity")
        plt.tight_layout()
        plt.show()

        # 5. WIND DIRECTION
        # Axis offset per cluster: the cut falls opposite the cluster's mean
        # direction, so it is never crossed by its own curve.
        compass = {0: 'N', 90: 'E', 180: 'S', 270: 'W'}

        def dd_axis_lo(dd_profile):
            s = np.nanmean(np.sin(np.deg2rad(dd_profile)))
            c = np.nanmean(np.cos(np.deg2rad(dd_profile)))
            mean_deg = np.degrees(np.arctan2(s, c)) % 360
            return (mean_deg + 180) % 360

        fig, axes = plt.subplots(1, K, figsize=(4*K, 6), sharey=True)

        for i, k in enumerate(clusters):
            ax = axes[i] if K > 1 else axes

            dd_mean = data[k]["dd"]
            lo = dd_axis_lo(dd_mean)
            dd_shifted = (np.asarray(dd_mean) - lo) % 360 + lo

            tick_pairs = sorted(
                [((d - lo) % 360 + lo, lbl) for d, lbl in compass.items()]
            )
            dd_ticks  = [t for t, _ in tick_pairs]
            dd_labels = [l for _, l in tick_pairs]

            ax.plot(dd_shifted, z_grid, color='purple', lw=2)
            ax.set_xlim(lo, lo + 360)
            ax.set_xticks(dd_ticks)
            ax.set_xticklabels(dd_labels, fontsize=9)
            ax.set_title(f"C{k}")
            ax.set_xlabel("Wind Direction")
            ax.grid(axis='y')
            ax.set_ylim(ylim)

        axes[0].set_ylabel("Altitude (m)")
        fig.suptitle("Wind Direction (circular mean)")
        plt.tight_layout()
        plt.show()

        for k in clusters: print(f"{k}: {nb_samples_per_cluster[k]} soundings")

    def plot_x_first_m_profiles(self, soundings,x):
        """Plot the first 1000 m of all profiles in each cluster to visualize variability"""
        if self.labels is None:
            raise ValueError("Run clustering first")

        n_clusters = len(np.unique(self.labels))

        fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
        colors = plt.cm.tab10(np.arange(n_clusters))

        for k in range(n_clusters):

            idx_k = np.where(self.labels == k)[0]
            T_list, Td_list = [], []
            u_list, v_list = [], []

            for i in idx_k:
                df = soundings[i]["data"]

                z = df["altitude"].values
                mask = z <= x

                T = df["t"].values[mask]
                Td = df["td"].values[mask]

                if "ff" in df.columns and "dd" in df.columns:
                    theta = np.deg2rad(df["dd"].values[mask])
                    u = -df["ff"].values[mask] * np.sin(theta)
                    v = -df["ff"].values[mask] * np.cos(theta)
                    ws = np.sqrt(u**2 + v**2)
                else:
                    u = v = ws = np.full_like(T, np.nan)

                T_list.append(T)
                Td_list.append(Td)
                u_list.append(u)
                v_list.append(v)

            # mean profiles
            T_mean = np.nanmean(T_list, axis=0)
            Td_mean = np.nanmean(Td_list, axis=0)
            u_mean = np.nanmean(u_list, axis=0)
            v_mean = np.nanmean(v_list, axis=0)

            ff_mean = np.sqrt(u_mean**2 + v_mean**2)   
            
            axes[0].plot(T_mean - 273.15, z[mask], color=colors[k], alpha=1.0, label=f"C{k}")
            axes[0].plot(Td_mean - 273.15, z[mask], linestyle='--', color=colors[k], alpha=1.0, label=f"C{k}")

            axes[1].plot(ff_mean, z[mask], color=colors[k], alpha=1.0, label=f"C{k}")

            axes[2].plot(u_mean, z[mask], color=colors[k], alpha=1.0, label=f"C{k}")
            axes[2].plot(v_mean, z[mask], linestyle='--', color=colors[k], alpha=1.0, label=f"C{k}")

        axes[0].set_xlabel("Temp (°C)")
        axes[0].set_title("T / Td")

        axes[1].set_xlabel("Wind speed (m/s)")
        axes[1].set_title("Wind speed")

        axes[2].set_xlabel("u / v (m/s)")
        axes[2].set_title("Wind components")

        axes[0].set_ylabel("Altitude (m)")

        for ax in axes:
            ax.grid(True)

        plt.tight_layout()
        plt.show()
        
    def diagnose_clusters(self,cp,z):
        import numpy as np

        for k in np.unique(cp.labels):
            Xk = cp.X_clean[cp.labels == k]
            mean = np.nanmean(Xk, axis=0)

            n = len(z)

            T  = mean[0:n]
            Td = mean[n:2*n] #or t-td
            u  = mean[2*n:3*n]
            v  = mean[3*n:4*n]

            # diagnostics
            lapse_rate = np.gradient(T, z)
            if "u" in self.variables and "v" in self.variables:
                shear = np.sqrt(np.gradient(u, z)**2 + np.gradient(v, z)**2)
                wind_speed = np.sqrt(u**2 + v**2)

            print(f"\nCluster {k}")
            print(f"Surface temp: {T[0]-273.15:.2f} °C")
            if 'td' in self.variables:
                print(f"Mean dew point depression: {np.nanmean(T-Td):.2f} K")
            else:
                print(f"Mean dew point depression: {np.nanmean(-1*(Td-T)):.2f} K")
            print(f"Mean temperature variance: {np.nanmean((T - np.nanmean(T))**2):.4f} K^2")
            print(f"Mean lapse rate: {np.nanmean(lapse_rate):.4f} K/m")
            if "u" in self.variables and "v" in self.variables:
                print(f"Max wind speed: {np.nanmax(wind_speed):.2f} m/s")
                print(f"Max shear: {np.nanmax(shear):.4f}")

    def cluster_summary_full(self, soundings, print_summary=True):
        """
        Compute physical diagnostics per sounding and aggregate by cluster.

        Parameters
        ----------
        soundings : list
            Original list passed to build_X. If treat_nan was called with
            strategy='drop', self.clean_mask is applied automatically.
        print_summary : bool

        Returns
        -------
        mean_df, std_df : pd.DataFrame  (index = cluster id, columns = features)
        """
        if self.labels is None:
            raise ValueError("Run fit_kmeans before cluster_summary_full.")

        used = soundings
        if self.clean_mask is not None:
            used = [s for s, keep in zip(soundings, self.clean_mask) if keep]

        if len(used) != len(self.labels):
            raise ValueError(
                f"len(soundings after mask)={len(used)} != len(labels)={len(self.labels)}. "
                "Pass the same soundings list used for build_X."
            )

        rows    = [_sounding_diagnostics(s) for s in used]
        df      = pd.DataFrame(rows)
        df["cluster"] = self.labels
        mean_df = df.groupby("cluster").mean()
        std_df  = df.groupby("cluster").std()
        # Std across all profiles (no groupby)
        all_std = df.drop(columns="cluster").std()
        all_mean = df.drop(columns="cluster").mean()

        if print_summary:
            n_cl = len(np.unique(self.labels))
            print(f"\n{'='*60}")
            print(f"  CLUSTER SUMMARY  —  {len(self.labels)} soundings, {n_cl} clusters")
            print(f"{'='*60}")
            _print_cluster_summary(mean_df, std_df)

        return mean_df, std_df, all_mean, all_std, df

    def plot_date_distribution(self, soundings, figsize=(13, 5), save_path=None):
        """
        Visualise the temporal distribution of sounding dates across clusters.

        Panel (a): absolute-count stacked bars per month.
        Panel (b): per-cluster count line chart — shows seasonal evolution.
        Legend is placed below the figure to avoid overlap.

        Parameters
        ----------
        soundings : list
            Same list passed to build_X. clean_mask is applied automatically.
        figsize   : tuple
        save_path : str or None

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        if self.labels is None:
            raise ValueError("Run fit_kmeans or fit_hierarchical before plot_date_distribution.")
        df = _build_date_df(soundings, self.labels, self.clean_mask)
        return _plot_date_distribution_impl(df, figsize=figsize, save_path=save_path)

    def plot_cluster_seasonal_evolution(self, soundings, season_months=None,
                                        figsize=(12, 4), save_path=None):
        """
        Show how each cluster's proportion and absolute count evolve over a season.

        Panel (a): proportion (%) lines with grey bars for total soundings per month.
        Panel (b): absolute count lines per cluster.

        Parameters
        ----------
        soundings     : list  same list passed to build_X. clean_mask applied automatically.
        season_months : list of int, optional
            Months in *chronological* season order.
            Austral winter : [5, 6, 7, 8, 9]
            Austral summer : [10, 11, 12, 1, 2, 3]
            If None, uses all months present in the data.
        figsize       : tuple
        save_path     : str or None

        Returns
        -------
        fig     : matplotlib.figure.Figure
        summary : pd.DataFrame  — count per cluster per month
        """
        if self.labels is None:
            raise ValueError("Run fit_kmeans or fit_hierarchical before plot_cluster_seasonal_evolution.")
        df = _build_date_df(soundings, self.labels, self.clean_mask)
        return _plot_seasonal_evolution_impl(df, season_months=season_months,
                                             figsize=figsize, save_path=save_path)

    def plot_feature_seasonal_evolution(self, soundings, season_months=None,
                                        features=None, stat='median',
                                        n_cols=4, figsize=None, save_path=None):
        """
        Plot the seasonal evolution of physical diagnostics per cluster.

        Each subplot shows one diagnostic feature (from cluster_summary_full) over
        the months of the season. Lines are coloured by cluster (hue); the ribbon
        shows the interquartile range (stat='median') or ±1 std (stat='mean').

        Parameters
        ----------
        soundings     : list  same list passed to build_X. clean_mask applied automatically.
        season_months : list of int, optional
            Months in *chronological* season order.
            Austral winter : [5, 6, 7, 8, 9]
            Austral summer : [10, 11, 12, 1, 2, 3]
            If None, uses all months present in the data.
        features      : list of str, 'all', or None
            Feature names from _sounding_diagnostics to plot.
            None  → default publication subset (12 key features).
            'all' → every available diagnostic.
        stat          : 'median' or 'mean'
        n_cols        : int  number of subplot columns (default 4)
        figsize       : tuple or None
        save_path     : str or None

        Returns
        -------
        fig    : matplotlib.figure.Figure
        df_agg : pd.DataFrame  — aggregated statistics (center / lo / hi) per
                 (cluster, month) for every plotted feature
        """
        if self.labels is None:
            raise ValueError("Run fit_kmeans or fit_hierarchical before plot_feature_seasonal_evolution.")

        used = soundings
        if self.clean_mask is not None:
            used = [s for s, keep in zip(soundings, self.clean_mask) if keep]
        if len(used) != len(self.labels):
            raise ValueError(
                f"len(soundings after mask)={len(used)} != len(labels)={len(self.labels)}."
            )

        dates = pd.to_datetime([s['header']['date'].item() for s in used])
        rows  = [_sounding_diagnostics(s) for s in used]
        df_diag = pd.DataFrame(rows)
        df_diag['cluster'] = self.labels
        df_diag['month']   = dates.month

        return _plot_feature_seasonal_impl(
            df_diag, season_months=season_months, features=features,
            stat=stat, n_cols=n_cols, figsize=figsize, save_path=save_path
        )

    def plot_cluster_carpet(self, soundings, season_months=None, years=None,
                            sounding_width=0.5, figsize=None,
                            save_path=None, title=None):
        """
        Temporal carpet plot: one row per winter, each sounding coloured by cluster.

        Allows visualising temporal coherence (blocks of the same colour = stable regime)
        and inter-annual patterns (same structure from one year to the next).

        Parameters
        ----------
        soundings      : list  same list passed to build_X. clean_mask applied.
        season_months  : list of int, optional
            Chronological season order. Austral winter: [5,6,7,8,9].
            If None, all months present in the data.
        years          : int, list of int, or None
            Filter on one or several winters (e.g. years=2016 or years=[2015,2016]).
            None = all available winters.
        sounding_width : float
            Width of each bar in days (default 0.5).
            Reduce for 4 soundings/day (~0.2) to avoid overlap.
        figsize        : tuple or None
        save_path      : str or None
        title          : str or None
        """
        if self.labels is None:
            raise ValueError("Run fit_kmeans or fit_hierarchical before plot_cluster_carpet.")
        df = _build_date_df(soundings, self.labels, self.clean_mask)
        return _plot_cluster_carpet_impl(
            df, season_months=season_months, years=years,
            sounding_width=sounding_width, figsize=figsize,
            save_path=save_path, title=title
        )

    def plot_interannual_consistency(self, soundings, season_months=None, years=None,
                                     window=7, min_soundings=3,
                                     figsize=None, save_path=None, title=None):
        """
        Evaluate the inter-annual consistency of clusters over the season.

        Parameters
        ----------
        soundings      : list  same list passed to build_X. clean_mask applied.
        season_months  : list of int, optional  e.g. [5,6,7,8,9] austral winter
        years          : int, list of int, or None  filter years
        window         : int  temporal window in days (default 7)
        min_soundings  : int  minimum soundings per bin to consider it (default 3)
        figsize        : tuple or None
        save_path      : str or None
        title          : str or None

        Returns
        -------
        fig   : matplotlib.figure.Figure
        stats : dict  'dominant', 'entropy', 'ari', 'bin_centers', 'years'
        """
        if self.labels is None:
            raise ValueError("Run fit_kmeans or fit_hierarchical before plot_interannual_consistency.")
        df = _build_date_df(soundings, self.labels, self.clean_mask)
        return _plot_interannual_consistency_impl(
            df, season_months=season_months, years=years,
            window=window, min_soundings=min_soundings,
            figsize=figsize, save_path=save_path, title=title
        )


class FeatureClusterProfiles:

    def __init__(self):
        self.X = None
        self.df_features = None
        self.X_clean = None
        self.X_norm = None
        self.labels = None
        self.kmeans = None
        self.X_red = None
        self.Z_linkage = None

    def filter_soundings(self, soundings, max_start_alt=500, z_required=3000,
                         min_levels=30, max_nan_ratio=0.3):
        """
        Secondary filter applied after utils.clean_and_interpolate.

        Parameters
        ----------
        soundings     : list  already-interpolated sounding dicts
        max_start_alt : float  discard if sounding starts above this altitude (m).
                               Should match the value used in clean_and_interpolate.
        z_required    : float  sounding must reach at least this altitude (m).
                               Default 3000 m — needed for free-troposphere features.
        min_levels    : int    minimum number of valid levels (default 30)
        max_nan_ratio : float  maximum fraction of NaN in temperature (default 0.3)
        """
        clean = []

        for s in soundings:
            df = s["data"]

            if not all(col in df.columns for col in ["t", "td", "altitude"]):
                continue

            z = df["altitude"].dropna().values
            T = df["t"].values

            if len(z) < min_levels:
                continue

            if np.nanmin(z) > max_start_alt or np.nanmax(z) < z_required:
                continue

            if not np.all(np.diff(z) > 0):
                continue

            if np.isnan(T).sum() / len(T) > max_nan_ratio:
                continue


            clean.append(s)

        return clean

    def build_features_2(self, soundings):
        """Build the physics-based feature matrix from a list of sounding dicts."""
        import feature_engineering as fe
        self.df_features = fe.build_features_full(soundings)
        self.X = self.df_features.values
        return self.df_features

    def build_features(self, soundings):
        features = []

        for s in soundings:
            df = s["data"].copy()

            if not all(col in df.columns for col in ["t", "td", "altitude"]):
                continue

            z = df["altitude"].values
            T = df["t"].values
            Td = df["td"].values

            # WIND
            if "ff" in df.columns and "dd" in df.columns:
                theta = np.deg2rad(df["dd"].values)
                u = -df["ff"].values * np.sin(theta)
                v = -df["ff"].values * np.cos(theta)
                ws = np.sqrt(u**2 + v**2)
            else:
                u = v = ws = np.full_like(T, np.nan)

            # FULL COLUMN FEATURES (0–6000 m)
            lapse_full = np.nanmean(np.gradient(T, z))
            T_mean_full = np.nanmean(T)
            T_var_full = np.nanvar(T)

            ws_mean_full = np.nanmean(ws)

            # shear global
            shear_full = ws[-1] - ws[0] if len(ws) > 2 else np.nan

            
            # MULTI-LAYER STRUCTURE
            
            def layer_stats(mask):
                if np.sum(mask) < 5:
                    return [np.nan]*3

                Tm = T[mask]
                zm = z[mask]

                lapse = np.nanmean(np.gradient(Tm, zm))
                T_mean = np.nanmean(Tm)
                T_var = np.nanvar(Tm)

                return [lapse, T_mean, T_var]

            m1 = z <= 1000
            m2 = (z > 1000) & (z <= 3000)
            m3 = (z > 3000) & (z <= 6000)

            lapse_0_1, T_0_1, var_0_1 = layer_stats(m1)
            lapse_1_3, T_1_3, var_1_3 = layer_stats(m2)
            lapse_3_6, T_3_6, var_3_6 = layer_stats(m3)

            # HUMIDITY (global proxy)
            RH_proxy = np.nanmean(T - Td)

            # FINAL FEATURE VECTOR
            features.append([
                # global structure
                lapse_full,
                T_mean_full,
                T_var_full,

                # wind
                #ws_mean_full,
                #shear_full,

                # humidity
                RH_proxy,

                # vertical structure
                lapse_0_1, T_0_1, var_0_1,
                lapse_1_3, T_1_3, var_1_3,
                lapse_3_6, T_3_6, var_3_6
            ])

        cols = [
            # global
            "lapse_full",
            "T_mean_full",
            "T_var_full",

            # wind
            #"wind_speed_mean",
            #"shear_full",

            # humidity
            "RH_proxy",

            # layers 0–1 km
            "lapse_0_1", "T_0_1", "var_0_1",

            # layers 1–3 km
            "lapse_1_3", "T_1_3", "var_1_3",

            # layers 3–6 km
            "lapse_3_6", "T_3_6", "var_3_6"
        ]

        self.df_features = pd.DataFrame(features, columns=cols)
        self.X = self.df_features.values

        return self.df_features
    def handle_nan_2(self):
        import numpy as np

        X = self.df_features.copy()

        # PHYSICAL FEATURES → 0
        physical_zero = [
            "inversion_strength",
            "inversion_height",
            "n_inversions",
            "delta_T_0_500",
            "delta_ws_0_500"
        ]

        for col in physical_zero:
            if col in X.columns:
                X[col] = X[col].fillna(0)

        # REMAINING → MEDIAN
        for col in X.columns:
            if X[col].isna().sum() > 0:
                median = X[col].median()
                X[col] = X[col].fillna(median)

        self.X_clean = X.values
        return self.X_clean
    
    def handle_nan(self, strategy="median"):
        imputer = SimpleImputer(strategy=strategy)
        self.X_clean = imputer.fit_transform(self.X)
        return self.X_clean
    
    def normalize(self):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        self.X_norm = scaler.fit_transform(self.X_clean)
        return self.X_norm
    
    def find_nb_of_components(self):
        pca = PCA()
        pca.fit(self.X_norm)

        cumvar = np.cumsum(pca.explained_variance_ratio_)

        plt.plot(cumvar)
        plt.xlabel("Number of components")
        plt.ylabel("Cumulative explained variance")
        plt.grid()
        plt.show()


    def apply_pca(self, n_components):
        pca = PCA(n_components=n_components)
        self.X_red = pca.fit_transform(self.X_norm)
        return self.X_red
    
    def find_k(self, X ,max_k=10):
        inertia = []
        sil = []
        K = range(2, max_k+1)

        for k in K:
            km = KMeans(n_clusters=k, n_init=20, random_state=0)
            labels = km.fit_predict(X)

            inertia.append(km.inertia_)
            sil.append(silhouette_score(X, labels))

        plt.figure()
        plt.plot(K, inertia, marker="o")
        plt.title("Elbow")
        plt.grid()
        plt.show()

        plt.figure()
        plt.plot(K, sil, marker="o")
        plt.title("Silhouette")
        plt.grid()
        plt.show()

    def fit(self, X, k):
        self.kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
        self.labels = self.kmeans.fit_predict(X)
        return self.labels

    # ── HIERARCHICAL CLUSTERING ──────────────────────────────────────────────

    def plot_dendrogram(self, X, linkage_method='ward', truncate_mode='lastp',
                        p=20, show_cut_at_k=None, figsize=(14, 5)):
        """
        Plot a dendrogram on X and store the linkage matrix in self.Z_linkage.

        Parameters
        ----------
        X              : array  same X passed to fit / fit_hierarchical
        linkage_method : str    'ward' | 'complete' | 'average' | 'single'
        truncate_mode  : str    'lastp' shows only the last p merges
        p              : int    nodes shown when truncate_mode='lastp'
        show_cut_at_k  : int    draws a dashed line for k clusters
        """
        from scipy.cluster.hierarchy import dendrogram, linkage as sp_linkage

        Z = sp_linkage(X, method=linkage_method)
        self.Z_linkage = Z

        cut_h = None
        if show_cut_at_k is not None:
            k, n = show_cut_at_k, len(Z) + 1
            if 2 <= k < n:
                cut_h = (Z[-k, 2] + Z[-k + 1, 2]) / 2

        fig, ax = plt.subplots(figsize=figsize)
        dendrogram(Z, truncate_mode=truncate_mode, p=p, ax=ax,
                   color_threshold=cut_h if cut_h is not None else 0,
                   above_threshold_color='grey')

        if cut_h is not None:
            ax.axhline(cut_h, color='red', ls='--', lw=1.5,
                       label=f'{show_cut_at_k} clusters  (h={cut_h:.3f})')
            ax.legend()

        ax.set_title(f"Dendrogram  —  {linkage_method} linkage")
        ax.set_xlabel("Soundings (merged nodes)")
        ax.set_ylabel("Distance")
        plt.tight_layout()
        plt.show()
        return Z

    def silhouette_hierarchical(self, X, max_k=10, linkage_method='ward', figsize=(8, 4)):
        """
        Compute silhouette scores for agglomerative clustering over k = 2…max_k.

        Reuses self.Z_linkage if already computed by plot_dendrogram.

        Parameters
        ----------
        X              : array  same X passed to fit / plot_dendrogram
        max_k          : int    maximum k to evaluate (default 10)
        linkage_method : str    used only if Z_linkage is not yet stored
        figsize        : tuple

        Returns
        -------
        scores : dict  {k: silhouette_score}
        """
        from scipy.cluster.hierarchy import linkage as sp_linkage, fcluster

        if self.Z_linkage is None:
            self.Z_linkage = sp_linkage(X, method=linkage_method)

        scores = {}
        for k in range(2, max_k + 1):
            labels = fcluster(self.Z_linkage, t=k, criterion='maxclust') - 1
            scores[k] = silhouette_score(X, labels)

        best_k = max(scores, key=scores.get)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(list(scores.keys()), list(scores.values()), marker='o')
        ax.axvline(best_k, color='red', ls='--', lw=1,
                   label=f'best k={best_k}  ({scores[best_k]:.3f})')
        ax.set_xlabel("Nb clusters k")
        ax.set_ylabel("Silhouette score")
        ax.set_title(f"Silhouette — hierarchical clustering  ({linkage_method})")
        ax.legend()
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.show()

        return scores

    def fit_hierarchical(self, X, k, linkage_method='ward'):
        """
        Fit agglomerative clustering on X.
        Sets self.labels — compatible with all plot_* and cluster_summary_full
        methods, exactly like fit.

        Parameters
        ----------
        X              : array  same X passed to fit / plot_dendrogram
        k              : int    number of clusters
        linkage_method : str    'ward' | 'complete' | 'average' | 'single'
        """
        from sklearn.cluster import AgglomerativeClustering

        model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
        self.labels = model.fit_predict(X)
        return self.labels

    def plot_pca_scatter(self, title=''):
        """
        Scatter plot of self.X_red coloured by self.labels.
        Works identically after fit or fit_hierarchical.
        Requires apply_pca to have been called.
        """
        if self.labels is None:
            raise ValueError("Run fit or fit_hierarchical first.")
        if self.X_red is None:
            raise ValueError("Run apply_pca first.")

        clusters = np.unique(self.labels)
        colors   = plt.cm.tab10(np.linspace(0, 1, len(clusters)))
        fig, ax  = plt.subplots(figsize=(8, 6))

        for k, c in zip(clusters, colors):
            m = self.labels == k
            ax.scatter(self.X_red[m, 0], self.X_red[m, 1],
                       label=f'C{k}  (n={m.sum()})', color=c, alpha=0.7, s=20)

        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_title(title or 'Clusters in PCA space')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def cluster_summary(self):
        df = self.df_features.copy()
        df["cluster"] = self.labels

        return df.groupby("cluster").mean()
    
    def plot_features(self):
        df = self.df_features.copy()
        df["cluster"] = self.labels

        cols = df.columns[:-1]

        for c in cols:
            plt.figure()
            for k in sorted(df["cluster"].unique()):
                plt.hist(df[df["cluster"]==k][c], alpha=0.5, label=f"C{k}")
            plt.title(c)
            plt.legend()
            plt.grid()
            plt.show()
            
    def plot_cluster_mean_profiles(self, soundings, z_grid):

        if self.labels is None:
            raise ValueError("Run clustering before plotting")

        n_clusters = len(np.unique(self.labels))

        fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
        colors = plt.cm.tab10(np.arange(n_clusters))

        for k in range(n_clusters):

            idx_k = np.where(self.labels == k)[0]

            T_list, Td_list = [], []
            u_list, v_list = [], []

            for i in idx_k:
                df = soundings[i]["data"]

                z = df["altitude"].values
                T = df["t"].values
                Td = df["td"].values

                # wind
                if "ff" in df.columns and "dd" in df.columns:
                    theta = np.deg2rad(df["dd"].values)
                    u = -df["ff"].values * np.sin(theta)
                    v = -df["ff"].values * np.cos(theta)
                    ws = np.sqrt(u**2 + v**2)
                else:
                    u = v = ws = np.full_like(T, np.nan)

                T_list.append(T)
                Td_list.append(Td)
                u_list.append(u)
                v_list.append(v)

            # mean profiles
            T_mean = np.nanmean(T_list, axis=0)
            Td_mean = np.nanmean(Td_list, axis=0)
            u_mean = np.nanmean(u_list, axis=0)
            v_mean = np.nanmean(v_list, axis=0)

            ff_mean = np.sqrt(u_mean**2 + v_mean**2)

            # plot
            axes[0].plot(T_mean - 273.15, z_grid, label=f"C{k}", color=colors[k])
            axes[0].plot(Td_mean - 273.15, z_grid, linestyle='--', color=colors[k])

            axes[1].plot(ff_mean, z_grid, label=f"C{k}", color=colors[k])

            axes[2].plot(u_mean, z_grid, label=f"C{k}", color=colors[k])
            axes[2].plot(v_mean, z_grid, linestyle='--', color=colors[k])

        axes[0].set_xlabel("Temp (°C)")
        axes[0].set_title("T / Td")

        axes[1].set_xlabel("Wind speed (m/s)")
        axes[1].set_title("Wind speed")

        axes[2].set_xlabel("u / v (m/s)")
        axes[2].set_title("Wind components")

        axes[0].set_ylabel("Altitude (m)")

        for ax in axes:
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()

    def plot_500_first_m_profiles(self, soundings):
        """Plot the first 1000 m of all profiles in each cluster to visualize variability"""
        if self.labels is None:
            raise ValueError("Run clustering first")

        n_clusters = len(np.unique(self.labels))

        fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True)
        colors = plt.cm.tab10(np.arange(n_clusters))

        for k in range(n_clusters):

            idx_k = np.where(self.labels == k)[0]
            T_list, Td_list = [], []
            u_list, v_list = [], []

            for i in idx_k:
                df = soundings[i]["data"]

                z = df["altitude"].values
                mask = z <= 500

                T = df["t"].values[mask]
                Td = df["td"].values[mask]

                if "ff" in df.columns and "dd" in df.columns:
                    theta = np.deg2rad(df["dd"].values[mask])
                    u = -df["ff"].values[mask] * np.sin(theta)
                    v = -df["ff"].values[mask] * np.cos(theta)
                    ws = np.sqrt(u**2 + v**2)
                else:
                    u = v = ws = np.full_like(T, np.nan)

                T_list.append(T)
                Td_list.append(Td)
                u_list.append(u)
                v_list.append(v)

            # mean profiles
            T_mean = np.nanmean(T_list, axis=0)
            Td_mean = np.nanmean(Td_list, axis=0)
            u_mean = np.nanmean(u_list, axis=0)
            v_mean = np.nanmean(v_list, axis=0)

            ff_mean = np.sqrt(u_mean**2 + v_mean**2)   
            
            axes[0].plot(T_mean - 273.15, z[mask], color=colors[k], alpha=1.0, label=f"C{k}")
            axes[0].plot(Td_mean - 273.15, z[mask], linestyle='--', color=colors[k], alpha=1.0, label=f"C{k}")

            axes[1].plot(ff_mean, z[mask], color=colors[k], alpha=1.0, label=f"C{k}")

            axes[2].plot(u_mean, z[mask], color=colors[k], alpha=1.0, label=f"C{k}")
            axes[2].plot(v_mean, z[mask], linestyle='--', color=colors[k], alpha=1.0, label=f"C{k}")

        axes[0].set_xlabel("Temp (°C)")
        axes[0].set_title("T / Td")

        axes[1].set_xlabel("Wind speed (m/s)")
        axes[1].set_title("Wind speed")

        axes[2].set_xlabel("u / v (m/s)")
        axes[2].set_title("Wind components")

        axes[0].set_ylabel("Altitude (m)")

        for ax in axes:
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_cluster_minipages(self, soundings, z_grid,
                          quantiles=(0.1, 0.9),
                          ylim=(0, 15000)):
        from metpy.calc import relative_humidity_from_dewpoint, saturation_vapor_pressure
        from metpy.units import units

        if self.labels is None:
            raise ValueError("Run clustering first")

        import numpy as np
        import matplotlib.pyplot as plt

        clusters = np.unique(self.labels)
        K = len(clusters)

        # COLLECT DATA PER CLUSTER
        data = {}
        nb_samples_per_cluster = {}

        for k in clusters:

            idx_k = np.where(self.labels == k)[0]
            nb_samples_per_cluster[k] = len(idx_k)

            T_all, Td_all, RH_all,RHi_all = [], [], [], []
            u_all, v_all, ff_all = [], [], []

            for i in idx_k:
                df = soundings[i]["data"]

                z = df["altitude"].values
                mask = (z >= ylim[0]) & (z <= ylim[1])

                T = df["t"].values[mask] - 273.15
                Td = df["td"].values[mask] - 273.15

                # --- RH (CORRECT PHYSICS)
                t_k = (T + 273.15) * units.kelvin
                td_k = (Td + 273.15) * units.kelvin
                RH = relative_humidity_from_dewpoint(t_k, td_k).to('percent').magnitude
                rhi = RH * (
                    saturation_vapor_pressure(t_k, phase='liquid') /
                    saturation_vapor_pressure(t_k, phase='solid')
                ).magnitude

                if "ff" in df.columns and "dd" in df.columns:
                    theta = np.deg2rad(df["dd"].values[mask])
                    ff = df["ff"].values[mask]
                    u = -ff * np.sin(theta)
                    v = -ff * np.cos(theta)
                else:
                    u = v = ff = np.full_like(T, np.nan)

                T_all.append(T)
                Td_all.append(Td)
                RH_all.append(RH)
                RHi_all.append(rhi)
                u_all.append(u)
                v_all.append(v)
                ff_all.append(ff)

            def stats(arr):
                arr = np.array(arr)
                mean = np.nanmean(arr, axis=0)
                q1 = np.nanquantile(arr, quantiles[0], axis=0)
                q2 = np.nanquantile(arr, quantiles[1], axis=0)
                return mean, q1, q2

            data[k] = {
                "T": stats(T_all),
                "Td": stats(Td_all),
                "u": stats(u_all),
                "v": stats(v_all),
                "ff": stats(ff_all),
                "RH": stats(RH_all),
                "RHi": stats(RHi_all)
            }

        # 1. TEMPERATURE
        fig, axes = plt.subplots(1, K, figsize=(4*K, 6), sharey=True)

        for i, k in enumerate(clusters):
            ax = axes[i] if K > 1 else axes

            T_mean, T_q1, T_q2 = data[k]["T"]
            Td_mean, Td_q1, Td_q2 = data[k]["Td"]

            ax.plot(T_mean, z_grid, color='red')
            ax.fill_betweenx(z_grid, T_q1, T_q2, color='red', alpha=0.3)

            ax.plot(Td_mean, z_grid, color='orange')
            ax.fill_betweenx(z_grid, Td_q1, Td_q2, color='orange', alpha=0.3)

            ax.set_title(f"C{k}")
            ax.grid()
            ax.set_ylim(ylim)

        axes[0].set_ylabel("Altitude (m)")
        fig.suptitle("Temperature / Dew Point")
        plt.tight_layout()
        plt.show()

        # 2. WIND SPEED
        fig, axes = plt.subplots(1, K, figsize=(4*K, 6), sharey=True)

        for i, k in enumerate(clusters):
            ax = axes[i] if K > 1 else axes

            ff_mean, ff_q1, ff_q2 = data[k]["ff"]

            ax.plot(ff_mean, z_grid, color='black')
            ax.fill_betweenx(z_grid, ff_q1, ff_q2, color='black', alpha=0.3)

            ax.set_title(f"C{k}")
            ax.grid()
            ax.set_ylim(ylim)

        axes[0].set_ylabel("Altitude (m)")
        fig.suptitle("Wind Speed")
        plt.tight_layout()
        plt.show()

        # 3. U / V
        fig, axes = plt.subplots(1, K, figsize=(4*K, 6), sharey=True)

        for i, k in enumerate(clusters):
            ax = axes[i] if K > 1 else axes

            u_mean, u_q1, u_q2 = data[k]["u"]
            v_mean, v_q1, v_q2 = data[k]["v"]

            ax.plot(u_mean, z_grid, color='blue', label='u')
            ax.fill_betweenx(z_grid, u_q1, u_q2, color='blue', alpha=0.3)

            ax.plot(v_mean, z_grid, color='green', label='v')
            ax.fill_betweenx(z_grid, v_q1, v_q2, color='green', alpha=0.3)

            ax.set_title(f"C{k}")
            ax.grid()
            ax.set_ylim(ylim)
        
        axes[0].set_ylabel("Altitude (m)")
        axes[0].legend()
        fig.suptitle("Wind Components")
        plt.tight_layout()
        plt.show()

        # RH
        fig, axes = plt.subplots(1, K, figsize=(4*K, 6), sharey=True)

        for i, k in enumerate(clusters):
            ax = axes[i] if K > 1 else axes

            RH_mean, RH_q1, RH_q2 = data[k]["RH"]
            RHi_mean, RHi_q1, RHi_q2 = data[k]["RHi"]

            ax.plot(RH_mean, z_grid, color='steelblue')
            ax.fill_betweenx(z_grid, RH_q1, RH_q2, color='steelblue', alpha=0.3, label='RH')
            ax.plot(RHi_mean, z_grid, color='navy')
            ax.fill_betweenx(z_grid, RHi_q1, RHi_q2, color='navy', alpha=0.3, label='RHi')
            ax.legend()
            ax.axvline(100, color='grey', lw=0.5)
            
            ax.set_title(f"C{k}")
            ax.set_xlabel("RH (%)")
            ax.grid()
            ax.set_ylim(ylim)

        axes[0].set_ylabel("Altitude (m)")
        fig.suptitle("Relative Humidity")
        plt.tight_layout()
        plt.show()
        for k in clusters: print(f"{k}: {nb_samples_per_cluster[k]} soundings") 

    def diagnose_clusters(self, soundings):

        import numpy as np

        if self.labels is None:
            raise ValueError("Run clustering first")

        for k in np.unique(self.labels):

            idx_k = np.where(self.labels == k)[0]

            T_all, Td_all = [], []
            u_all, v_all = [], []

            for i in idx_k:
                df = soundings[i]["data"]

                T = df["t"].values
                Td = df["td"].values
                z = df["altitude"].values
                
                if "ff" in df.columns and "dd" in df.columns:
                    theta = np.deg2rad(df["dd"].values)
                    u = -df["ff"].values * np.sin(theta)
                    v = -df["ff"].values * np.cos(theta)
                else:
                    u = v = np.full_like(T, np.nan)

                T_all.append(T)
                Td_all.append(Td)
                u_all.append(u)
                v_all.append(v)

            T_mean = np.nanmean(T_all, axis=0)
            Td_mean = np.nanmean(Td_all, axis=0)
            u_mean = np.nanmean(u_all, axis=0)
            v_mean = np.nanmean(v_all, axis=0)

            lapse_rate = np.gradient(T_mean, z)
            wind_speed = np.sqrt(u_mean**2 + v_mean**2)
            shear = np.sqrt(np.gradient(u_mean, z)**2 + np.gradient(v_mean, z)**2)

            print(f"\nCluster {k}")
            print(f"Surface temp: {T_mean[0]-273.15:.2f} °C")
            print(f"Mean dew point depression: {np.nanmean(T_mean - Td_mean):.2f} K")
            print(f"Mean lapse rate: {np.nanmean(lapse_rate):.4f} K/m")
            print(f"Max wind speed: {np.nanmax(wind_speed):.2f} m/s")
            print(f"Max shear: {np.nanmax(shear):.4f}")

    def cluster_summary_full(self, soundings, print_summary=True):
        """
        Compute physical diagnostics per sounding and aggregate by cluster.

        soundings must be the same list (already filtered by filter_soundings
        and matched to build_features_2 output — same length as self.labels).

        Returns
        -------
        mean_df, std_df : pd.DataFrame  (index = cluster id, columns = features)
        """
        if self.labels is None:
            raise ValueError("Run fit before cluster_summary_full.")
        if len(soundings) != len(self.labels):
            raise ValueError(
                f"len(soundings)={len(soundings)} != len(labels)={len(self.labels)}."
            )

        rows    = [_sounding_diagnostics(s) for s in soundings]
        df      = pd.DataFrame(rows)
        df["cluster"] = self.labels
        mean_df = df.groupby("cluster").mean()
        std_df  = df.groupby("cluster").std()

        if print_summary:
            n_cl = len(np.unique(self.labels))
            print(f"\n{'='*60}")
            print(f"  CLUSTER SUMMARY [FeatureClusterProfiles]")
            print(f"  {len(self.labels)} soundings, {n_cl} clusters")
            print(f"{'='*60}")
            _print_cluster_summary(mean_df, std_df)

        return mean_df, std_df

    def plot_date_distribution(self, soundings, figsize=(13, 5), save_path=None):
        """
        Visualise the temporal distribution of sounding dates across clusters.

        Panel (a): absolute-count stacked bars per month.
        Panel (b): per-cluster count line chart — shows seasonal evolution.
        Legend is placed below the figure to avoid overlap.

        Parameters
        ----------
        soundings : list
            Same list passed to build_features_2 (already filtered, matches labels 1-to-1).
        figsize   : tuple
        save_path : str or None

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        if self.labels is None:
            raise ValueError("Run fit or fit_hierarchical before plot_date_distribution.")
        df = _build_date_df(soundings, self.labels, clean_mask=None)
        return _plot_date_distribution_impl(df, figsize=figsize, save_path=save_path)

    def plot_cluster_seasonal_evolution(self, soundings, season_months=None,
                                        figsize=(12, 4), save_path=None):
        """
        Show how each cluster's proportion and absolute count evolve over a season.

        Panel (a): proportion (%) lines with grey bars for total soundings per month.
        Panel (b): absolute count lines per cluster.

        Parameters
        ----------
        soundings     : list  same list passed to build_features_2.
        season_months : list of int, optional
            Months in *chronological* season order.
            Austral winter : [5, 6, 7, 8, 9]
            Austral summer : [10, 11, 12, 1, 2, 3]
            If None, uses all months present in the data.
        figsize       : tuple
        save_path     : str or None

        Returns
        -------
        fig     : matplotlib.figure.Figure
        summary : pd.DataFrame  — count per cluster per month
        """
        if self.labels is None:
            raise ValueError("Run fit or fit_hierarchical before plot_cluster_seasonal_evolution.")
        df = _build_date_df(soundings, self.labels, clean_mask=None)
        return _plot_seasonal_evolution_impl(df, season_months=season_months,
                                             figsize=figsize, save_path=save_path)

    def plot_feature_seasonal_evolution(self, soundings, season_months=None,
                                        features=None, stat='median',
                                        n_cols=4, figsize=None, save_path=None):
        """
        Plot the seasonal evolution of physical diagnostics per cluster.

        Each subplot shows one diagnostic feature (from cluster_summary_full) over
        the months of the season. Lines are coloured by cluster (hue); the ribbon
        shows the interquartile range (stat='median') or ±1 std (stat='mean').

        Parameters
        ----------
        soundings     : list  same list passed to build_features_2.
        season_months : list of int, optional
            Months in *chronological* season order.
            Austral winter : [5, 6, 7, 8, 9]
            Austral summer : [10, 11, 12, 1, 2, 3]
            If None, uses all months present in the data.
        features      : list of str, 'all', or None
            Feature names from _sounding_diagnostics to plot.
            None  → default publication subset (12 key features).
            'all' → every available diagnostic.
        stat          : 'median' or 'mean'
        n_cols        : int  number of subplot columns (default 4)
        figsize       : tuple or None
        save_path     : str or None

        Returns
        -------
        fig    : matplotlib.figure.Figure
        df_agg : pd.DataFrame  — aggregated statistics (center / lo / hi) per
                 (cluster, month) for every plotted feature
        """
        if self.labels is None:
            raise ValueError("Run fit or fit_hierarchical before plot_feature_seasonal_evolution.")

        if len(soundings) != len(self.labels):
            raise ValueError(
                f"len(soundings)={len(soundings)} != len(labels)={len(self.labels)}."
            )

        dates = pd.to_datetime([s['header']['date'].item() for s in soundings])
        rows  = [_sounding_diagnostics(s) for s in soundings]
        df_diag = pd.DataFrame(rows)
        df_diag['cluster'] = self.labels
        df_diag['month']   = dates.month

        return _plot_feature_seasonal_impl(
            df_diag, season_months=season_months, features=features,
            stat=stat, n_cols=n_cols, figsize=figsize, save_path=save_path
        )

    def plot_cluster_carpet(self, soundings, season_months=None, years=None,
                            sounding_width=0.5, figsize=None,
                            save_path=None, title=None):
        """
        Temporal carpet plot: one row per winter, each sounding coloured by cluster.

        Parameters
        ----------
        soundings      : list  same list passed to build_features_2.
        season_months  : list of int, optional
            Chronological season order. Austral winter: [5,6,7,8,9].
        years          : int, list of int, or None
            Filter on one or several winters. None = all.
        sounding_width : float  width of each bar in days (default 0.5)
        figsize        : tuple or None
        save_path      : str or None
        title          : str or None
        """
        if self.labels is None:
            raise ValueError("Run fit or fit_hierarchical before plot_cluster_carpet.")
        df = _build_date_df(soundings, self.labels, clean_mask=None)
        return _plot_cluster_carpet_impl(
            df, season_months=season_months, years=years,
            sounding_width=sounding_width, figsize=figsize,
            save_path=save_path, title=title
        )

    def plot_interannual_consistency(self, soundings, season_months=None, years=None,
                                     window=7, min_soundings=3,
                                     figsize=None, save_path=None, title=None):
        """
        Evaluate the inter-annual consistency of clusters over the season.

        Parameters
        ----------
        soundings      : list  same list passed to build_features_2.
        season_months  : list of int, optional  e.g. [5,6,7,8,9] austral winter
        years          : int, list of int, or None  filter years
        window         : int  temporal window in days (default 7)
        min_soundings  : int  minimum soundings per bin to consider it (default 3)
        figsize        : tuple or None
        save_path      : str or None
        title          : str or None

        Returns
        -------
        fig   : matplotlib.figure.Figure
        stats : dict  'dominant', 'entropy', 'ari', 'bin_centers', 'years'
        """
        if self.labels is None:
            raise ValueError("Run fit or fit_hierarchical before plot_interannual_consistency.")
        df = _build_date_df(soundings, self.labels, clean_mask=None)
        return _plot_interannual_consistency_impl(
            df, season_months=season_months, years=years,
            window=window, min_soundings=min_soundings,
            figsize=figsize, save_path=save_path, title=title
        )