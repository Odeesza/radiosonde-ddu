import numpy as np
import pandas as pd


def clean_soundings(soundings, min_levels=20):


    cleaned = []

    total_stats = []


    thresholds = {
        "t": (150, 350),          # temperature K
        "td": (150, 350),         # dewpoint K
        "p_niv": (500, 110000),  # pressure Pa
        "altitude": (-300, 40000),
        "dd": (0, 360),           # wind direction
        "ff": (0, 200)            # wind speed m/s
    }

    for s in soundings:
        #Rename geop or 0 to altitude if present
        if "geop" in s["data"].columns:
            s["data"]["altitude"] = s["data"]["geop"]
            s["data"] = s["data"].drop(columns=["geop"])
        if "0" in s["data"].columns:
            s["data"]["altitude"] = s["data"]["0"]
            s["data"] = s["data"].drop(columns=["0"])

        stats = {
            "threshold": 0,
            "dewpoint": 0,
            "temp_spikes": 0,
            "wind_spikes": 0,
            "pressure_nan": 0,
            "largest_gap_pressure_removed": 0,
            "non_monotonic_pressure": 0,
            "discarded_b_of_pressure": 0,
            "discarded_b_of_len": 0,
            "len_before": len(s["data"]),
            "len_after": len(s["data"]),
            "monoticity": None
        }

        df = s['data'].copy()

        # detect and remove outliers based on thresholds
        for col, (vmin, vmax) in thresholds.items():
            if col in df.columns:
                mask = (df[col] < vmin) | (df[col] > vmax)
                stats["threshold"] += mask.sum()
                df.loc[mask, col] = np.nan

        # td <= t
        if "t" in df and "td" in df:
            mask = df["td"] > df["t"]
            stats["dewpoint"] += mask.sum()
            df.loc[mask, "td"] = np.nan

        #drop duplicates (same altitude)
        df = df.drop_duplicates(subset="altitude")

        #drop levels with missing pressure
        n_nan, max_consecutive, pressure_ok = analyze_sounding_pressure(df)
        stats["pressure_nan"] += n_nan
        stats["largest_gap_pressure_removed"] = max_consecutive

        if not pressure_ok:
            stats["discarded_b_of_pressure"] += 1
            continue

        df = df.dropna(subset=["p_niv"])

        #sort by altitude
        df = df.sort_values("altitude").reset_index(drop=True)

        #pressure monotonicity
        if "p_niv" in df:

            before = len(df)
            df,stats_mono_p = check_pressure_monotonicity(df)

            stats["non_monotonic_pressure"] += before - len(df)
            stats['monoticity']= stats_mono_p


        # temperature spikes
        if {"t", "altitude"}.issubset(df.columns):

            valid = df["t"].notna() & df["altitude"].notna()

            z = df.loc[valid, "altitude"].values
            T = df.loc[valid, "t"].values

            dz = np.diff(z)
            dT = np.diff(T)

            # avoid very close levels which can create huge gradients due to noise
            mask_dz = dz > 50   # ignore levels < 50 m

            grad = np.full_like(dT, np.nan, dtype=float)

            with np.errstate(divide='ignore', invalid='ignore'):
                grad[mask_dz] = dT[mask_dz] / dz[mask_dz]

            LAPSE_RATE_DRY_ADIABATIC = 9.8e-3  # K/m
            SPIKE_THRESHOLD_NEG = -3 * LAPSE_RATE_DRY_ADIABATIC  # ~-29 K/km, very permissive
            SPIKE_THRESHOLD_POS = +0.1  # K/m = +100 K/km, only eliminates gross errors

            spike = np.where(
                (grad < SPIKE_THRESHOLD_NEG) | (grad > SPIKE_THRESHOLD_POS)
            )[0]

            stats["temp_spikes"] += len(spike)

            idx = df.loc[valid].index
            df.loc[idx[spike + 1], "t"] = np.nan


        # wind spikes
        if {"ff", "altitude"}.issubset(df.columns):

            valid = df["ff"].notna() & df["altitude"].notna()

            z = df.loc[valid, "altitude"].values
            v = df.loc[valid, "ff"].values

            dz = np.diff(z)
            dv = np.diff(v)

            mask_dz = dz > 50

            shear = np.full_like(dv, np.nan, dtype=float)

            with np.errstate(divide='ignore', invalid='ignore'):
                shear[mask_dz] = dv[mask_dz] / dz[mask_dz]

            spike = np.where(np.abs(shear) > 0.15)[0]   # ~150 m/s/km

            stats["wind_spikes"] += len(spike)

            idx = df.loc[valid].index
            df.loc[idx[spike + 1], "ff"] = np.nan

        #don't use soundings with too few levels after cleaning
        if len(df) >= min_levels:
            stats['len_after']= len(df)
            new_s = s.copy()
            new_s["data"] = df
            cleaned.append(new_s)
            total_stats.append(stats)
        else:
            stats["discarded_b_of_len"] += 1
            total_stats.append(stats)


    return cleaned, total_stats


def check_pressure_monotonic(sounding):

    df = sounding["data"]
    p = df["p_niv"].values
    return np.all(np.diff(p) < 0)

def interpolate_soundings(soundings, z_grid=None):
    if z_grid is None:
        z_grid = np.arange(0, 15000, 100)

    interpolated = []
    variables_direct = ["t", "td", "p_niv"]  # direct interpolation

    for s in soundings:
        df = s["data"].copy().sort_values("altitude")
        z = df["altitude"].values

        # Decompose wind direction into components before interpolation
        if "ff" in df.columns and "dd" in df.columns:
            df["u"] = df["ff"] * np.sin(np.deg2rad(df["dd"]))
            df["v"] = df["ff"] * np.cos(np.deg2rad(df["dd"]))

        new_df = pd.DataFrame({"altitude": z_grid})

        for var in variables_direct + ["u", "v"]:
            if var in df.columns:
                y = df[var].values
                valid = ~np.isnan(y)
                if valid.sum() > 3:
                    new_df[var] = np.interp(
                        z_grid, z[valid], y[valid],
                        left=np.nan, right=np.nan
                    )
                else:
                    new_df[var] = np.nan
            else:
                new_df[var] = np.nan


        # Reconstruct ff and dd from u, v
        new_df["ff"] = np.sqrt(new_df["u"]**2 + new_df["v"]**2)
        new_df["dd"] = np.rad2deg(np.arctan2(new_df["u"], new_df["v"])) % 360

        # Verify pressure monotonicity
        if np.any(np.diff(new_df["p_niv"].dropna()) > 0):
            print(f"WARNING: non-monotonic pressure after interpolation ({s.get('date', '?')})")

        new_s = s.copy()
        new_s["data"] = new_df
        interpolated.append(new_s)

    return interpolated

def mean_soundings(soundings):
    """Average of soundings interpolated on the same altitude grid."""

    if len(soundings) == 0:
        raise ValueError("Empty list")

    n_levels = len(soundings[0]['data'])

    for i, s in enumerate(soundings):
        if len(s['data']) != n_levels:
            raise ValueError(f"Sounding {i} has different size")

    # Stack variables
    t_stack = np.array([s['data']["t"].values for s in soundings])
    td_stack = np.array([s['data']["td"].values for s in soundings])
    p_stack = np.array([s['data']["p_niv"].values for s in soundings])

    ff_stack = np.array([s['data']["ff"].values for s in soundings])
    dd_stack = np.deg2rad(np.array([s['data']["dd"].values for s in soundings]))

    # ---- Wind ----
    u_stack = -ff_stack * np.sin(dd_stack)
    v_stack = -ff_stack * np.cos(dd_stack)

    u_mean = np.nanmean(u_stack, axis=0)
    v_mean = np.nanmean(v_stack, axis=0)

    ff_mean = np.sqrt(u_mean**2 + v_mean**2)
    dd_mean = (np.rad2deg(np.arctan2(-u_mean, -v_mean))) % 360

    # ---- Mean ----
    t_mean = np.nanmean(t_stack, axis=0)
    td_mean = np.nanmean(td_stack, axis=0)
    p_mean = np.nanmean(p_stack, axis=0)

    altitude = soundings[0]['data']["altitude"].values

    df_mean = pd.DataFrame({
        "altitude": altitude,
        "t": t_mean,
        "td": td_mean,
        "p_niv": p_mean,
        "ff": ff_mean,
        "dd": dd_mean
    })

    header_mean = soundings[0]['header'].copy()
    header_mean["date"] = "mean"
    header_mean["n_soundings"] = len(soundings)
    header_mean["nb_niv"] = n_levels

    dict_mean = {
        "header": header_mean,
        "data": df_mean
    }
    return dict_mean

def inversion_statistics(soundings):

    strengths = []
    count = 0

    for df in soundings:
        df = df["data"]
        z = df.altitude.values
        T = df.t.values

        dT = np.diff(T)
        dz = np.diff(z)

        lapse = dT/dz

        inv = lapse > 0

        if np.any(inv) and np.max(dT)>=2:
            count += 1
            strengths.append(np.max(dT))

    freq = count / len(soundings)

    return {
        "inversion_frequency": freq,
        "mean_inversion_strength": np.nanmean(strengths)
    }


def pbl_height_distribution(soundings):
    #PBL = Planetary Boundary Layer
    pbl_heights = []

    for df in soundings:

        df = df["data"]
        z = df.altitude.values
        T = df.t.values

        dTdz = np.gradient(T, z)

        idx = np.argmax(dTdz)

        pbl_heights.append(z[idx])

    return np.array(pbl_heights)


def _get_date(s):
    """Safely extract the launch datetime from a sounding dict's header."""
    try:
        d = s["header"].iloc[0]["date"]
        return d if pd.notna(d) else None
    except (KeyError, IndexError, TypeError, AttributeError):
        return None


def stitch_profiles(soundings, z_tol=500):
    """
    Merge consecutive soundings that are fragments of the same balloon ascent.

    At DDU, the Meteofrance HR format can split one ascent into several records
    (e.g. 0-200 m in record A, 200-25 000 m in record B). Two soundings are
    merged when they share the same launch timestamp AND sounding B starts within
    z_tol metres of where sounding A ends.

    Stitching is done BEFORE quality control so that spike detection and pressure
    monotonicity checks act on the full merged profile.

    Parameters
    ----------
    soundings : list of sounding dicts (output of parse_rs / get_data)
    z_tol     : float  max altitude gap allowed between two fragments (m).
                       500 m is permissive enough to absorb small logging gaps.

    Returns
    -------
    stitched   : list of sounding dicts
    n_stitched : int   number of fragment pairs that were merged
    """
    if not soundings:
        return soundings, 0

    result     = []
    n_stitched = 0
    i          = 0

    while i < len(soundings):
        current  = dict(soundings[i])
        df_cur   = current["data"].sort_values("altitude").reset_index(drop=True)
        date_cur = _get_date(current)

        # Greedily absorb any immediate continuation
        z_top_cur = df_cur["altitude"].dropna().max()
        while i + 1 < len(soundings):
            nxt      = soundings[i + 1]
            df_nxt   = nxt["data"].sort_values("altitude").reset_index(drop=True)
            date_nxt = _get_date(nxt)

            z_bot_nxt = df_nxt["altitude"].dropna().min()

            if np.isnan(z_top_cur) or np.isnan(z_bot_nxt):
                break

            # Same launch timestamp (year-month-day-hour level - DDU does 00h and 12h)
            if date_cur is not None and date_nxt is not None:
                same_launch = (
                    date_cur.year  == date_nxt.year  and
                    date_cur.month == date_nxt.month and
                    date_cur.day   == date_nxt.day   and
                    date_cur.hour  == date_nxt.hour
                )
            else:
                same_launch = True  # no date info -> fall back to altitude check only

            # B must start where A ends, and must itself start above the surface
            alt_continuous = (abs(z_bot_nxt - z_top_cur) <= z_tol and z_bot_nxt > 50)

            if same_launch and alt_continuous:
                merged = (
                    pd.concat([df_cur, df_nxt], ignore_index=True)
                    .sort_values("altitude")
                    .drop_duplicates(subset="altitude")
                    .reset_index(drop=True)
                )
                current["data"] = merged
                df_cur          = merged
                z_top_cur       = df_cur["altitude"].dropna().max()
                n_stitched     += 1
                i              += 1
            else:
                break

        result.append(current)
        i += 1

    return result, n_stitched


def filter_by_coverage(soundings, z_grid, max_start_alt=500, min_coverage=0.8):
    """
    Discard soundings that don't adequately cover the target altitude range.

    Two independent criteria:
    1. Start altitude  - the sounding must begin at or below max_start_alt.
       For katabatic and inversion analysis the 0-500 m layer is critical;
       a profile missing this layer is scientifically unusable.
    2. Vertical reach  - the sounding must reach at least
       min(z_grid) + min_coverage x (max(z_grid) - min(z_grid)).
       A profile truncated at mid-altitude biases cluster shapes.

    Parameters
    ----------
    soundings     : list of cleaned sounding dicts (pre-interpolation)
    z_grid        : array  target altitude grid used for interpolation
    max_start_alt : float  max acceptable starting altitude in m (default 500 m)
    min_coverage  : float  fraction of z_grid range the sounding must reach
                           (default 0.8 -> must reach 80 % of the way up z_grid)

    Returns
    -------
    kept      : list
    discarded : list
    """
    z_min_grid  = float(np.min(z_grid))
    z_max_grid  = float(np.max(z_grid))
    z_reach_req = z_min_grid + min_coverage * (z_max_grid - z_min_grid)

    kept      = []
    discarded = []

    for s in soundings:
        z = s["data"]["altitude"].dropna().values
        if len(z) == 0:
            discarded.append(s)
            continue

        z_start = float(z.min())
        z_reach = float(z.max())

        if z_start > max_start_alt or z_reach < z_reach_req:
            discarded.append(s)
        else:
            kept.append(s)

    return kept, discarded


def clean_and_interpolate(soundings, z_grid=None, min_levels=20,
                          stitch=True, z_stitch_tol=500,
                          max_start_alt=500, min_coverage=0.8):
    """
    Full preprocessing pipeline for radiosonde data.

    Steps (in order):
      1. Stitch   - merge consecutive fragments of the same ascent
      2. Clean    - threshold QC, spike removal, pressure monotonicity
      3. Coverage - discard profiles that start too high or don't reach far enough
      4. Interpolate - resample onto z_grid

    Parameters
    ----------
    soundings     : list of raw sounding dicts
    z_grid        : array  target altitude grid (default 0-15 000 m every 100 m)
    min_levels    : int    min level count after QC (default 20)
    stitch        : bool   merge split soundings (default True)
    z_stitch_tol  : float  altitude gap tolerance for stitching in m (default 500)
    max_start_alt : float  discard profiles that start above this altitude (default 500 m).
                           Pass np.inf to disable.
    min_coverage  : float  fraction of z_grid range the profile must reach (default 0.8).
                           Pass 0 to disable.
    """
    if z_grid is None:
        z_grid = np.arange(0, 15000, 100)

    # 1. Stitch split fragments
    if stitch:
        soundings, n_stitched = stitch_profiles(soundings, z_tol=z_stitch_tol)
        if n_stitched:
            print(f"[stitch]   {n_stitched} fragment pair(s) merged")

    # 2. Quality control
    cleaned, total_stats = clean_soundings(soundings, min_levels=min_levels)

    # 3. Coverage filter
    kept, discarded = filter_by_coverage(cleaned, z_grid,
                                         max_start_alt=max_start_alt,
                                         min_coverage=min_coverage)
    if discarded:
        print(f"[coverage] {len(discarded)} profile(s) discarded "
              f"(start > {max_start_alt} m  or  reach < {min_coverage*100:.0f} % of z_grid)")

    # 4. Interpolate
    interpolated = interpolate_soundings(kept, z_grid)

    return interpolated, total_stats


def analyze_sounding_pressure(df, max_gap_allowed=5):
    """
    Analyse NaN values in the p_niv column.

    Returns
    -------
    n_nan           : int   total number of NaN pressure values
    max_consecutive : int   longest consecutive NaN gap
    is_valid        : bool  True if max gap is within the allowed limit
    """
    nan_pressure = df["p_niv"].isna().values
    n_nan = int(nan_pressure.sum())

    if n_nan == 0:
        max_consecutive = 0
    else:
        padded = np.concatenate(([False], nan_pressure, [False]))
        changes = np.diff(padded.astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        max_consecutive = int((ends - starts).max())

    is_valid = max_consecutive <= max_gap_allowed
    return n_nan, max_consecutive, is_valid

def check_pressure_monotonicity(df, col="p_niv", dense_threshold=0.2):
    """
    Check for decreasing pressure monotonicity and classify violations.

    Parameters
    ----------
    df : pandas.DataFrame
    col : str
        Name of the pressure column.
    dense_threshold : float
        Violation density above which violations are classified as "dense".

    Returns
    -------
    df_filtered : DataFrame
    stats : dict
    """

    stats = {
        "total_points": len(df),
        "violations": 0,
        "violation_ratio": 0.0,
        "violation_type": "none"
    }

    if col not in df or len(df) <= 1:
        return df, stats

    keep_idx = [0]
    violation_idx = []

    for i in range(1, len(df)):
        if df.iloc[i][col] <= df.iloc[keep_idx[-1]][col]:
            keep_idx.append(i)
        else:
            violation_idx.append(i)

    # stats
    stats['violation_idx']=violation_idx
    stats["violations"] = len(violation_idx)
    stats["violation_ratio"] = len(violation_idx) / len(df)

    if stats["violations"] == 0:
        stats["violation_type"] = "none"
    elif stats["violation_ratio"] < dense_threshold:
        stats["violation_type"] = "sparse"
    else:
        stats["violation_type"] = "dense"

    # filter non-monotonic levels
    df_filtered = df.loc[keep_idx].reset_index(drop=True)

    return df_filtered, stats

def violation_clusters(indices):
    clusters = []
    current = []

    for idx in indices:
        if not current or idx == current[-1] + 1:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]

    if current:
        clusters.append(current)

    return clusters

def plot_vertical_coverage(sounding_list, var="t", z_key="altitude"):
    """
    Compute and plot vertical coverage (fraction of valid profiles per level).

    Parameters
    ----------
    sounding_list : list of dict
        List of soundings (with key "data").
    var : str
        Variable used to define validity (e.g. "t", "td").
    z_key : str
        Name of the altitude column.

    Returns
    -------
    z_grid, coverage_frac
    """

    import matplotlib.pyplot as plt

    # stack all profiles
    profiles = []

    for s in sounding_list:
        df = s["data"]

        if var in df.columns:
            profiles.append(df[var].values)

    profiles = np.array(profiles)  # shape (n_profiles, n_levels)

    # altitude (assumed identical for all profiles)
    z_grid = sounding_list[0]["data"][z_key].values

    # coverage
    coverage = np.sum(~np.isnan(profiles), axis=0)
    coverage_frac = coverage / profiles.shape[0]

    # plot
    fig, ax = plt.subplots(figsize=(4,6))

    ax.plot(coverage_frac, z_grid, color="black")
    ax.set_xlabel("Fraction of valid profiles")
    ax.set_ylabel("Altitude [m]")
    ax.set_title(f"Vertical coverage ({var})")

    ax.set_xlim(0, 1)
    ax.grid(True, linestyle="--", linewidth=0.5)
    plt.show()
    return z_grid, coverage_frac


def wind_profile_statistics(soundings):

    profiles = []

    for df in soundings:

        f = interp1d(df.altitude, df.ff, bounds_error=False, fill_value=np.nan)
        profiles.append(f(Z_GRID))

    profiles = np.array(profiles)

    return {
        "mean_wind_profile": np.nanmean(profiles, axis=0),
        "std_wind_profile": np.nanstd(profiles, axis=0)
    }


def _sounding_diagnostics(s):
    """Return a dict of physical diagnostics for one sounding dict."""
    from scipy.signal import argrelmax
    from scipy.ndimage import uniform_filter1d

    df = s["data"].copy()
    if not all(c in df.columns for c in ["t", "altitude"]):
        return None

    z  = df["altitude"].values
    T  = df["t"].values
    Td = df["td"].values if "td" in df.columns else np.full_like(T, np.nan)
    p  = df["p_niv"].values if "p_niv" in df.columns else np.full_like(T, np.nan)

    if len(z) < 5:
        return None

    idx = np.argsort(z)
    z, T, Td, p = z[idx], T[idx], Td[idx], p[idx]
    z_rel = z - z[0]

    if "ff" in df.columns and "dd" in df.columns:
        ff = df["ff"].values[idx]
        dd = np.deg2rad(df["dd"].values[idx])
        u  = -ff * np.sin(dd)
        v  = -ff * np.cos(dd)
        ws = np.sqrt(u**2 + v**2)
    else:
        u = v = ws = np.full_like(T, np.nan)

    # ── Layer masks ───────────────────────────────────────────────────────
    m0_200 = z_rel <= 200
    m0_500 = z_rel <= 500
    m0_1km = z_rel <= 1000
    m0_2km = z_rel <= 2000
    m1_3km = (z_rel > 1000) & (z_rel <= 3000)
    m3_6km = (z_rel > 3000) & (z_rel <= 6000)

    # ── Helper functions ──────────────────────────────────────────────────

    def smean(x, m=None):
        a = x[m] if m is not None else x
        return np.nan if np.sum(~np.isnan(a)) < 3 else float(np.nanmean(a))

    def sstd(x, m=None):
        a = x[m] if m is not None else x
        return np.nan if np.sum(~np.isnan(a)) < 3 else float(np.nanstd(a))

    def interp1(zt, zv, vv):
        ok = ~np.isnan(vv)
        if ok.sum() < 2:
            return np.nan
        return float(np.interp(zt, zv[ok], vv[ok], left=np.nan, right=np.nan))

    def lapse(m):
        """Mean gradient (original form, kept for compatibility)."""
        if np.sum(m) < 5:
            return np.nan
        return float(np.nanmean(np.gradient(T[m], z[m])))

    def lapse_regression(m):
        """Linear regression of T on z. Positive = inversion."""
        if np.sum(m) < 3:
            return np.nan
        z_l = z[m]
        T_l = T[m]
        valid = ~np.isnan(T_l)
        if valid.sum() < 3:
            return np.nan
        return float(np.polyfit(z_l[valid], T_l[valid], 1)[0])

    def lapse_profile():
        """
        Lapse rate by linear regression in contiguous 200 m windows from 0 to 2000 m.
        Returns a dict {lapse_z0_200, lapse_z200_400, ..., lapse_z1800_2000}.
        Positive = inversion, negative = normal lapse.
        """
        result = {}
        for z_bot in range(0, 2000, 200):
            z_top = z_bot + 200
            m = (z_rel >= z_bot) & (z_rel < z_top) & ~np.isnan(T)
            key = f"lapse_z{z_bot}_{z_top}"
            if np.sum(m) < 3:
                result[key] = np.nan
            else:
                result[key] = float(np.polyfit(z[m], T[m], 1)[0])
        return result

    # ── TEMPERATURE ───────────────────────────────────────────────────────
    T_sfc = float(np.nanmedian(T[m0_200])) if np.any(m0_200) else float(T[0])

    T_m0_500 = smean(T, m0_500)
    T_m0_1km = smean(T, m0_1km)
    T_m1_3km = smean(T, m1_3km)

    p_sfc     = float(p[0]) if not np.isnan(p[0]) else 100000.0
    theta_sfc = T_sfc * (100000 / p_sfc) ** 0.286

    T_range_1km = (float(np.nanmax(T[m0_1km]) - np.nanmin(T[m0_1km]))
                   if np.sum(m0_1km) >= 3 else np.nan)

    if np.sum(m0_500) >= 5:
        theta_l = T[m0_500] * (
            100000 / np.where(np.isnan(p[m0_500]), p_sfc, p[m0_500])
        ) ** 0.286
        dth_dz = np.nanmean(np.gradient(theta_l, z[m0_500]))
        N2_sfc = float((9.81 / np.nanmean(theta_l)) * dth_dz)
    else:
        N2_sfc = np.nan

    # ── INVERSION - first local maximum within 0-1000 m ──────────────────
    m_inv   = m0_1km & ~np.isnan(T)
    inv_str = 0.0
    inv_h   = np.nan

    if np.sum(m_inv) >= 5:
        T_layer  = T[m_inv]
        z_layer  = z_rel[m_inv]
        T_smooth = uniform_filter1d(T_layer, size=3)
        local_max_idx = argrelmax(T_smooth, order=2)[0]
        if len(local_max_idx) > 0:
            first_max     = local_max_idx[0]
            candidate_str = float(T_smooth[first_max] - T_sfc)
            if candidate_str > 0:
                inv_str = candidate_str
                inv_h   = float(z_layer[first_max])

    kernel         = np.ones(5) / 5
    dTdz_smooth    = np.convolve(np.gradient(T, z), kernel, mode="same")
    max_grad_0_500 = (float(np.nanmax(dTdz_smooth[m0_500]))
                      if np.any(m0_500) else np.nan)

    # ── BROAD LAPSE RATES (regression, global stability) ─────────────────
    lapse_0_500_reg = lapse_regression(m0_500)
    lapse_0_2km_reg = lapse_regression(m0_2km)
    lapse_1_3km_reg = lapse_regression(m1_3km)
    lapse_3_6km_reg = lapse_regression(m3_6km)

    # ── LAPSE RATE PROFILE 0-2000 m (200 m windows) ──────────────────────
    lapse_dict = lapse_profile()

    # ── KATABATIC JET ─────────────────────────────────────────────────────
    ws_0     = float(ws[0]) if not np.isnan(ws[0]) else np.nan
    ws_500_v = interp1(500,  z_rel, ws)
    ws_1km_v = interp1(1000, z_rel, ws)
    ws_3km_v = interp1(3000, z_rel, ws)

    shear_0_500 = (float(ws_500_v - ws_0)
                   if not (np.isnan(ws_500_v) or np.isnan(ws_0)) else np.nan)

    if np.sum(m0_1km & ~np.isnan(ws)) >= 3:
        jet_spd = float(np.nanmax(ws[m0_1km]))
        jet_h   = float(z_rel[m0_1km][np.nanargmax(ws[m0_1km])])
        jet_sb  = float(jet_spd - ws_0)     if not np.isnan(ws_0)     else np.nan
        jet_sa  = float(ws_1km_v - jet_spd) if not np.isnan(ws_1km_v) else np.nan
    else:
        jet_spd = jet_h = jet_sb = jet_sa = np.nan

    # ── WIND BY LAYER ─────────────────────────────────────────────────────
    u_sfc_v = smean(u, m0_200)
    v_sfc_v = smean(v, m0_200)
    u_1km_v = interp1(1000, z_rel, u)
    v_1km_v = interp1(1000, z_rel, v)
    u_3km_v = interp1(3000, z_rel, u)
    v_3km_v = interp1(3000, z_rel, v)

    if not any(np.isnan([u_sfc_v, v_sfc_v, u_3km_v, v_3km_v])):
        turning = float(np.degrees(np.angle(
            np.exp(1j * (
                np.arctan2(-u_3km_v, -v_3km_v) -
                np.arctan2(-u_sfc_v, -v_sfc_v)
            ))
        )))
    else:
        turning = np.nan

    # ── HUMIDITY ──────────────────────────────────────────────────────────
    Td_v = np.where(Td >= 213.15, Td, np.nan)
    dd   = T - Td_v

    def es(t):
        return 6.112 * np.exp(17.67 * (t - 273.15) / (t - 273.15 + 243.5))

    RH_full  = np.clip(100 * es(Td_v) / es(T), 0, 105)
    valid_rh = ~np.isnan(RH_full)
    RH_max_v   = float(np.nanmax(RH_full))           if valid_rh.sum() >= 3 else np.nan
    z_RH_max_v = float(z_rel[np.nanargmax(RH_full)]) if valid_rh.sum() >= 3 else np.nan

    if np.sum(~np.isnan(Td_v) & ~np.isnan(p)) >= 5:
        q_sp   = 0.622 * es(Td_v) / (p / 100 - 0.378 * es(Td_v))
        ok_iwv = ~np.isnan(q_sp) & ~np.isnan(p)
        IWV    = (float(np.abs(np.trapezoid(q_sp[ok_iwv], -p[ok_iwv] / 9.81)))
                  if ok_iwv.sum() >= 5 else np.nan)
    else:
        IWV = np.nan

    # ── RETURN ────────────────────────────────────────────────────────────
    return {
        # TEMPERATURE
        "T_sfc_C":        T_sfc - 273.15,
        "T_mean_0_500_C": (T_m0_500 - 273.15) if not np.isnan(T_m0_500) else np.nan,
        "T_mean_0_1km_C": (T_m0_1km - 273.15) if not np.isnan(T_m0_1km) else np.nan,
        "T_mean_1_3km_C": (T_m1_3km - 273.15) if not np.isnan(T_m1_3km) else np.nan,
        "theta_sfc":      theta_sfc,
        "T_std_0_1km":    sstd(T, m0_1km),
        "T_range_0_1km":  T_range_1km,

        # INVERSION
        "inversion_strength": inv_str,
        "inversion_height":   inv_h,
        "max_grad_0_500":     max_grad_0_500,

        # BROAD LAPSE RATES
        "lapse_0_500":    lapse_0_500_reg,
        "lapse_0_2km":    lapse_0_2km_reg,
        "lapse_1_3km":    lapse_1_3km_reg,
        "lapse_3_6km":    lapse_3_6km_reg,

        "N2_surface":     N2_sfc,

        # KATABATIC JET
        "jet_nose_speed":  jet_spd,
        "jet_nose_height": jet_h,
        "jet_shear_below": jet_sb,
        "jet_shear_above": jet_sa,
        "shear_0_500":     shear_0_500,

        # WIND BY LAYER
        "ws_sfc":             ws_0,
        "ws_500":             ws_500_v,
        "ws_1km":             ws_1km_v,
        "ws_3km":             ws_3km_v,
        "ws_mean_0_1km":      smean(ws, m0_1km),
        "ws_mean_1_3km":      smean(ws, m1_3km),
        "u_sfc":              u_sfc_v,
        "v_sfc":              v_sfc_v,
        "u_1km":              u_1km_v,
        "v_1km":              v_1km_v,
        "u_3km":              u_3km_v,
        "v_3km":              v_3km_v,
        "wind_turning_0_3km": turning,

        # HUMIDITY
        "dd_0_500":      smean(dd, m0_500),
        "dd_1_3km":      smean(dd, m1_3km),
        "RH_mean_0_1km": smean(RH_full, m0_1km),
        "RH_mean_1_3km": smean(RH_full, m1_3km),
        "RH_max":        RH_max_v,
        "z_RH_max":      z_RH_max_v,
        "IWV":           IWV,

        # LAPSE RATE PROFILE 0-2000 m (200 m windows)
        **lapse_dict,
    }
