
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Diagnostic groups used to print cluster summaries
# ─────────────────────────────────────────────────────────────────────────────

_DIAG_GROUPS = {
    "TEMPERATURE":     ["T_sfc_C", "T_mean_0_500_C", "T_mean_0_1km_C", "T_mean_1_3km_C",
                        "theta_sfc", "T_std_0_1km", "T_range_0_1km"],
    "INVERSION":       ["inversion_strength", "inversion_height", "max_grad_0_500",
                        "lapse_0_500", "lapse_0_2km", "lapse_1_3km", "lapse_3_6km",
                        "N2_surface"],
    "KATABATIC JET":   ["jet_nose_speed", "jet_nose_height",
                        "jet_shear_below", "jet_shear_above", "shear_0_500"],
    "WIND BY LAYER":   ["ws_sfc", "ws_500", "ws_1km", "ws_3km",
                        "ws_mean_0_1km", "ws_mean_1_3km",
                        "u_sfc", "v_sfc", "u_1km", "v_1km", "u_3km", "v_3km",
                        "wind_turning_0_3km"],
    "HUMIDITY":        ["dd_0_500", "dd_1_3km", "RH_mean_0_1km", "RH_mean_1_3km",
                        "RH_max", "z_RH_max", "IWV"],
    "LAPSE PROFILE":   [f"lapse_z{z}_{z+200}" for z in range(0, 2000, 200)],
}

_DEFAULT_FEATURES = [
    'T_sfc_C', 'inversion_strength', 'inversion_height', 'lapse_0_500',
    'N2_surface', 'ws_sfc', 'ws_1km', 'ws_mean_0_1km', 'jet_nose_speed',
    'dd_0_500', 'RH_mean_0_1km', 'IWV',
]

_FEATURE_UNITS = {
    'T_sfc_C':            'degC',
    'T_mean_0_500_C':     'degC',
    'T_mean_0_1km_C':     'degC',
    'T_mean_1_3km_C':     'degC',
    'theta_sfc':          'K',
    'T_std_0_1km':        'K',
    'T_range_0_1km':      'K',
    'inversion_strength': 'K',
    'inversion_height':   'm',
    'max_grad_0_500':     'K m-1',
    'lapse_0_500':        'K m-1',
    'lapse_0_2km':        'K m-1',
    'lapse_1_3km':        'K m-1',
    'lapse_3_6km':        'K m-1',
    'N2_surface':         's-2',
    'jet_nose_speed':     'm s-1',
    'jet_nose_height':    'm',
    'jet_shear_below':    'm s-1',
    'jet_shear_above':    'm s-1',
    'shear_0_500':        'm s-1',
    'ws_sfc':             'm s-1',
    'ws_500':             'm s-1',
    'ws_1km':             'm s-1',
    'ws_3km':             'm s-1',
    'ws_mean_0_1km':      'm s-1',
    'ws_mean_1_3km':      'm s-1',
    'u_sfc':              'm s-1',
    'v_sfc':              'm s-1',
    'u_1km':              'm s-1',
    'v_1km':              'm s-1',
    'u_3km':              'm s-1',
    'v_3km':              'm s-1',
    'wind_turning_0_3km': 'deg',
    'dd_0_500':           'K',
    'dd_1_3km':           'K',
    'RH_mean_0_1km':      '%',
    'RH_mean_1_3km':      '%',
    'RH_max':             '%',
    'z_RH_max':           'm',
    'IWV':                'kg m-2',
}


# ─────────────────────────────────────────────────────────────────────────────
#  Basic feature functions (used by notebooks directly)
# ─────────────────────────────────────────────────────────────────────────────

def compute_gradient(var, z):
    dz = np.gradient(z)
    dvar = np.gradient(var)
    return dvar / dz

def temperature_features(df):
    z = df["altitude"].values
    T = df["t"].values - 273.15  # in deg C

    grad_T = compute_gradient(T, z)

    # mean lapse rate (0-3 km)
    mask_3km = z <= 3000
    lapse_rate = np.nanmean(grad_T[mask_3km])

    # inversion
    inversion_strength = np.max(grad_T)
    inversion_height = z[np.argmax(grad_T)]

    # mean temperature in lower layer (0-1 km)
    mask_1km = z <= 1000
    T_low = np.nanmean(T[mask_1km])

    return {
        "lapse_rate_0_3km": lapse_rate,
        "inversion_strength": inversion_strength,
        "inversion_height": inversion_height,
        "T_low_mean": T_low,
    }

def humidity_features(df):
    z = df["altitude"].values
    T = df["t"].values
    Td = df["td"].values

    RH = 100 * np.exp((17.625 * (Td - 273.15)) / (243.04 + (Td - 273.15))) / \
              np.exp((17.625 * (T - 273.15)) / (243.04 + (T - 273.15)))

    grad_RH = compute_gradient(RH, z)

    RH_mean = np.nanmean(RH)
    RH_grad = np.nanmean(grad_RH)

    saturated_fraction = np.mean(RH > 90)

    return {
        "RH_mean": RH_mean,
        "RH_gradient": RH_grad,
        "RH_saturated_frac": saturated_fraction,
    }

def wind_features(df):
    z = df["altitude"].values
    ff = df["ff"].values
    dd = df["dd"].values
    u = -ff * np.sin(np.radians(dd))
    v = -ff * np.cos(np.radians(dd))
    wind_speed = np.sqrt(u**2 + v**2)

    # mean
    ws_mean = np.nanmean(wind_speed)

    # shear 0-1 km
    mask_1km = z <= 1000
    if np.sum(mask_1km) > 2:
        shear_1km = wind_speed[mask_1km][-1] - wind_speed[mask_1km][0]
    else:
        shear_1km = np.nan

    # shear 0-3 km
    mask_3km = z <= 3000
    if np.sum(mask_3km) > 2:
        shear_3km = wind_speed[mask_3km][-1] - wind_speed[mask_3km][0]
    else:
        shear_3km = np.nan

    return {
        "wind_speed_mean": ws_mean,
        "shear_0_1km": shear_1km,
        "shear_0_3km": shear_3km,
    }

def structure_features(df):
    z = df["altitude"].values
    T = df["t"].values - 273.15

    grad_T = compute_gradient(T, z)

    return {
        "T_variance": np.nanvar(T),
        "max_temp_gradient": np.nanmax(np.abs(grad_T)),
    }


def extract_features(sounding):
    """Extract features from a single sounding dict."""
    df = sounding["data"]

    features = {}

    features.update(temperature_features(df))
    features.update(humidity_features(df))
    #features.update(wind_features(df))
    features.update(structure_features(df))

    return features

def build_feature_matrix(sounding_list):
    """Build a feature matrix from a list of soundings."""
    feature_list = []

    for s in sounding_list:
        feat = extract_features(s)
        feature_list.append(feat)

    return feature_list


# ─────────────────────────────────────────────────────────────────────────────
#  Full physics-based feature extraction (used by FeatureClusterProfiles)
# ─────────────────────────────────────────────────────────────────────────────

def build_features_full(soundings):
    """
    Compute a physics-based feature matrix from a list of sounding dicts.

    Each sounding is described by 25 scalar features covering surface temperature
    structure, inversion strength, katabatic jet, layer-by-layer wind components,
    and humidity/precipitable water.

    Parameters
    ----------
    soundings : list of sounding dicts (output of clean_and_interpolate)

    Returns
    -------
    df_features : pd.DataFrame  shape (n_valid_soundings, 25)
    """
    GAMMA_D = 9.8 / 1000  # K/m - dry adiabatic lapse rate

    features = []

    for s in soundings:
        df = s["data"].copy()

        required_cols = ["t", "altitude"]
        if not all(col in df.columns for col in required_cols):
            continue

        z = df["altitude"].values
        T = df["t"].values
        Td = df["td"].values if "td" in df.columns else np.full_like(T, np.nan)
        p = df["p_niv"].values if "p_niv" in df.columns else np.full_like(T, np.nan)

        if len(z) < 10:
            continue

        # Sort by altitude
        idx = np.argsort(z)
        z, T, Td, p = z[idx], T[idx], Td[idx], p[idx]
        z_rel = z - z[0]  # altitude relative to surface

        # Wind components (circular -> Cartesian)
        if "ff" in df.columns and "dd" in df.columns:
            ff_raw = df["ff"].values[idx]
            dd_raw = df["dd"].values[idx]
            theta = np.deg2rad(dd_raw)
            u = -ff_raw * np.sin(theta)   # zonal component (west = negative)
            v = -ff_raw * np.cos(theta)   # meridional component (south = negative)
            ws = np.sqrt(u**2 + v**2)
        else:
            u = v = ws = ff_raw = np.full_like(T, np.nan)

        # Layer masks
        m_0_200  = z_rel <= 200
        m_0_500  = z_rel <= 500
        m_0_1km  = z_rel <= 1000
        m_0_2km  = z_rel <= 2000
        m_1_3km  = (z_rel > 1000) & (z_rel <= 3000)
        m_2_6km  = (z_rel > 2000) & (z_rel <= 6000)
        m_1km_up = z_rel > 1000

        # Helper functions
        def safe_mean(x, mask=None):
            arr = x[mask] if mask is not None else x
            return np.nan if np.sum(~np.isnan(arr)) < 3 else np.nanmean(arr)

        def safe_std(x, mask=None):
            arr = x[mask] if mask is not None else x
            return np.nan if np.sum(~np.isnan(arr)) < 3 else np.nanstd(arr)

        def interp_at(z_target, z_arr, val_arr):
            valid = ~np.isnan(val_arr)
            if valid.sum() < 2:
                return np.nan
            return float(np.interp(z_target, z_arr[valid], val_arr[valid],
                                left=np.nan, right=np.nan))

        def circular_mean_dir(dd_arr):
            """Circular mean of wind direction in degrees."""
            valid = dd_arr[~np.isnan(dd_arr)]
            if len(valid) < 2:
                return np.nan
            rad = np.deg2rad(valid)
            return float(np.rad2deg(np.arctan2(np.nanmean(np.sin(rad)),
                                            np.nanmean(np.cos(rad)))) % 360)

        # BLOCK 1 - SURFACE THERMAL STRUCTURE AND INVERSION

        # Robust surface reference (median of first 3 levels)
        T_sfc_ref = np.nanmedian(T[z_rel < 200]) if np.any(z_rel < 200) else T[0]

        # lapse_0_500: raw thermal gradient in 0-500 m
        # Positive = inversion (T increases with z) = katabatic regime
        # Negative = instability or warm advection
        if np.sum(m_0_500) >= 5:
            lapse_0_500 = float(np.nanmean(np.gradient(T[m_0_500], z[m_0_500])))
        else:
            lapse_0_500 = np.nan

        # lapse_0_2km: raw thermal gradient in 0-2 km
        # Captures overall thermal structure of the lower troposphere
        if np.sum(m_0_2km) >= 5:
            lapse_0_2km = float(np.nanmean(np.gradient(T[m_0_2km], z[m_0_2km])))
        else:
            lapse_0_2km = np.nan

        # inversion_strength: surface inversion intensity (K)
        # T_max - T_sfc; high (>10K) = intense katabatic, low (<3K) = disturbed regime
        delta_T = T - T_sfc_ref
        inversion_strength = float(np.nanmax(delta_T)) if not np.all(np.isnan(delta_T)) else np.nan

        # inversion_height: altitude (m) of inversion top
        # Low inversion (<200 m) = pure radiative drainage
        # High inversion (500-1500 m) = synoptic-katabatic coupling
        inversion_height = float(z[np.nanargmax(delta_T)]) if not np.isnan(inversion_strength) else np.nan

        # max_grad_0_500: smoothed maximum thermal gradient in 0-500 m (K/m)
        # Captures the "nose" of the katabatic inversion in the lowest layer
        kernel = np.ones(5) / 5
        dTdz = np.gradient(T, z)
        dTdz_smooth = np.convolve(dTdz, kernel, mode="same")
        if np.any(m_0_500):
            max_grad_0_500 = float(np.nanmax(dTdz_smooth[m_0_500]))
        else:
            max_grad_0_500 = np.nan

        # T_std_0_1km: thermal variability in 0-1 km (K)
        # High std = complex structure (multiple inversions, strong gradient)
        T_std_0_1km = safe_std(T, m_0_1km)

        # theta_sfc: surface potential temperature (K)
        # Measures thermodynamic surface forcing independently of pressure
        # Low value = cold dense air favouring katabatic drainage
        p_sfc = p[0] if not np.isnan(p[0]) else 100000.0
        theta_sfc = T_sfc_ref * (100000 / p_sfc) ** 0.286 if not np.isnan(p_sfc) else np.nan

        # N2_surface: Brunt-Vaisala frequency squared in 0-500 m (s-2)
        # Measures static stability of the boundary layer
        # High N2 = extreme stratification (katabatic), N2 ~ 0 = neutral, N2 < 0 = unstable
        if np.sum(m_0_500) >= 5 and not np.isnan(theta_sfc):
            theta_layer = T[m_0_500] * (100000 / np.where(np.isnan(p[m_0_500]), p_sfc, p[m_0_500])) ** 0.286
            dtheta_dz = np.nanmean(np.gradient(theta_layer, z[m_0_500]))
            N2_surface = float((9.81 / np.nanmean(theta_layer)) * dtheta_dz)
        else:
            N2_surface = np.nan


        # BLOCK 2 - KATABATIC JET

        # jet_nose_speed: max wind speed in 0-1 km (m/s)
        # Katabatic jet intensity in the lowest layer
        if np.sum(m_0_1km & ~np.isnan(ws)) >= 3:
            jet_nose_speed = float(np.nanmax(ws[m_0_1km]))
            # jet_nose_height: altitude of jet nose (m)
            jet_nose_height = float(z_rel[m_0_1km][np.nanargmax(ws[m_0_1km])])
            # jet_shear_below: wind acceleration from surface to jet nose (m/s)
            jet_shear_below = float(jet_nose_speed - ws[0]) if not np.isnan(ws[0]) else np.nan
            # jet_shear_above: deceleration above jet nose to 1 km (m/s)
            ws_1km = interp_at(1000, z_rel, ws)
            jet_shear_above = float(ws_1km - jet_nose_speed) if not np.isnan(ws_1km) else np.nan
        else:
            jet_nose_speed = jet_nose_height = jet_shear_below = jet_shear_above = np.nan

        # shear_0_500: wind speed shear between surface and 500 m (m/s)
        ws_500 = interp_at(500, z_rel, ws)
        ws_sfc = ws[0] if not np.isnan(ws[0]) else np.nan
        shear_0_500 = float(ws_500 - ws_sfc) if not (np.isnan(ws_500) or np.isnan(ws_sfc)) else np.nan


        # BLOCK 3 - WIND DIRECTION AND TURNING

        # u_sfc, v_sfc: surface wind components (m/s)
        # Critical feature: distinguishes katabatic (SSE -> u<0, v<0)
        # from synoptic/marine (NW -> u>0, v<0 or variable)
        u_sfc = safe_mean(u, m_0_200)
        v_sfc = safe_mean(v, m_0_200)

        # u_1km, v_1km: wind components at 1 km altitude (m/s)
        # Synoptic flow in the free troposphere above the katabatic jet
        u_1km = interp_at(1000, z_rel, u)
        v_1km = interp_at(1000, z_rel, v)

        # u_3km, v_3km: wind components at 3 km (m/s)
        # Represents large-scale flow decoupled from the surface
        u_3km = interp_at(3000, z_rel, u)
        v_3km = interp_at(3000, z_rel, v)

        # wind_turning_0_3km: rotation of wind direction between surface and 3 km (degrees)
        # Veering (>0, clockwise) = warm synoptic advection
        # Backing (<0, anti-clockwise) = cold advection or persistent katabatic regime
        if not any(np.isnan([u_sfc, v_sfc, u_3km, v_3km])):
            dd_sfc_rad = np.arctan2(-u_sfc, -v_sfc)
            dd_3km_rad = np.arctan2(-u_3km, -v_3km)
            wind_turning_0_3km = float(np.degrees(
                np.angle(np.exp(1j * (dd_3km_rad - dd_sfc_rad)))
            ))
        else:
            wind_turning_0_3km = np.nan

        # BLOCK 4 - HUMIDITY

        # Mask unreliable Td values (Vaisala sensor floor ~ -60 C = 213.15 K)
        TD_SENSOR_FLOOR = 213.15
        Td_valid = np.where(Td >= TD_SENSOR_FLOOR, Td, np.nan)
        dd_dep = T - Td_valid  # dew-point depression (T - Td)

        # dd_0_500: surface dew-point depression (K)
        # Low (<5K) = moist marine air (intrusion), high (>15K) = dry katabatic air
        dd_0_500 = safe_mean(dd_dep, m_0_500)

        # dd_1_3km: dew-point depression in the lower free troposphere (K)
        dd_1_3km = safe_mean(dd_dep, m_1_3km)

        # Saturated vapour pressure (Magnus formula) — computed once and reused for IWV and RH
        e_s  = 6.112 * np.exp(17.67 * (T       - 273.15) / (T       - 273.15 + 243.5))
        e_td = 6.112 * np.exp(17.67 * (Td_valid - 273.15) / (Td_valid - 273.15 + 243.5))

        # IWV: integrated precipitable water over the profile (kg/m2)
        # Low (<1 kg/m2) = dry polar air, high (>3 kg/m2) = marine intrusion
        if np.sum(~np.isnan(Td_valid) & ~np.isnan(p)) >= 5:
            q_specific = 0.622 * e_td / (p / 100 - 0.378 * e_td)  # p in Pa -> hPa
            valid_iwv = ~np.isnan(q_specific) & ~np.isnan(p)
            if valid_iwv.sum() >= 5:
                IWV = float(np.abs(np.trapezoid(
                    q_specific[valid_iwv], -p[valid_iwv] / 9.81
                )))
            else:
                IWV = np.nan
        else:
            IWV = np.nan

        # RH_max: maximum relative humidity over the profile (%)
        RH_max = np.nan
        z_RH_max = np.nan
        if np.sum(~np.isnan(Td_valid) & ~np.isnan(T)) >= 5:
            RH_profile = np.clip(100 * e_td / e_s, 0, 105)
            valid_rh = ~np.isnan(RH_profile)
            if valid_rh.sum() >= 3:
                RH_max = float(np.nanmax(RH_profile))
                # z_RH_max: altitude of the most humid layer (m)
                z_RH_max = float(z_rel[np.nanargmax(RH_profile)])

        # BLOCK 5 - FINAL FEATURE VECTOR

        features.append([
            # SURFACE TEMPERATURE / INVERSION
            lapse_0_500,        # raw T gradient 0-500 m (K/m)
            lapse_0_2km,        # raw T gradient 0-2 km (K/m)
            inversion_strength, # surface inversion intensity (K)
            inversion_height,   # inversion top altitude (m)
            max_grad_0_500,     # smoothed max gradient 0-500 m (K/m)
            T_std_0_1km,        # thermal variability 0-1 km (K)
            theta_sfc,          # surface potential temperature (K)
            N2_surface,         # boundary-layer static stability (s-2)

            # KATABATIC JET
            jet_nose_speed,     # max wind speed 0-1 km (m/s)
            jet_nose_height,    # jet nose altitude (m)
            jet_shear_below,    # acceleration surface -> nose (m/s)
            jet_shear_above,    # deceleration nose -> 1 km (m/s)
            shear_0_500,        # wind speed shear 0-500 m (m/s)

            # WIND DIRECTION
            u_sfc,              # zonal surface wind component (m/s)
            v_sfc,              # meridional surface wind component (m/s)
            u_1km,              # zonal wind component at 1 km (m/s)
            v_1km,              # meridional wind component at 1 km (m/s)
            u_3km,              # zonal wind component at 3 km (m/s)
            v_3km,              # meridional wind component at 3 km (m/s)
            wind_turning_0_3km, # wind direction rotation 0->3 km (degrees)

            # HUMIDITY
            dd_0_500,           # dew-point depression surface 0-500 m (K)
            dd_1_3km,           # dew-point depression free troposphere 1-3 km (K)
            IWV,                # integrated precipitable water column (kg/m2)
            RH_max,             # maximum relative humidity over profile (%)
            z_RH_max,           # altitude of most humid layer (m)
        ])

    cols = [
        # temperature
        "lapse_0_500",
        "lapse_0_2km",
        "inversion_strength",
        "inversion_height",
        "max_grad_0_500",
        "T_std_0_1km",
        "theta_sfc",
        "N2_surface",
        # jet
        "jet_nose_speed",
        "jet_nose_height",
        "jet_shear_below",
        "jet_shear_above",
        "shear_0_500",
        # wind direction
        "u_sfc",
        "v_sfc",
        "u_1km",
        "v_1km",
        "u_3km",
        "v_3km",
        "wind_turning_0_3km",
        # humidity
        "dd_0_500",
        "dd_1_3km",
        "IWV",
        "RH_max",
        "z_RH_max",
    ]

    return pd.DataFrame(features, columns=cols)
