#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 14:19:14 2025

@author: corden
"""

def plot_radiosonde_skewt(sounding_dict, 
                           figsize = [7,8],
                           fontsize = 10, 
                           ylim_press = [1000, 100],
                           xlim = [-40, 40],
                           addalt = True,
                           heightaskm = False,
                           saveplot = False,
                           saveasplot = 'auto',
                           outpath = '',
                           dpi = 300,
                           site = 'DDU'):
    """
    Plot a SkewT diagram of a DDU HR radiosonde, with wind barbs.

    Parameters
    ----------
    sounding_dict : dict of three dataframes
        The output from the parse_rs function.
    figsize : list, optional
        The default is [6,3].
    fontsize : int, optional
        The default is 10.
    ylim_press : list, optional
        ylimits in pressure units (hPa). The default is [1000, 100].
    xlim : list, optional
        xlimits in degC. The default is [-30, 40].
    saveplot : boolean, optional
        Whether to save the plot. The default is False.
    saveasplot : string, optional
        filename. The default is 'auto'.
    outpath : string, optional
        path to folder where the plot should be saved. The default is ''.
    dpi : int, optional
        The default is 300.
    site : string, optional
        Only used for the title and plot name. The default is 'DDU'.

    Returns
    -------
    fig, ax.

    """
    
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.colors as mcolors
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator
    import os
    import numpy as np
    
    import metpy.calc as mpcalc

    from metpy.plots import SkewT
    from metpy.units import units
    
    plt.rcParams.update({'font.size': fontsize})
    
    df = sounding_dict['data']
    info = sounding_dict['header']
    
    #extract data and assign units for use with metpy
    p = df['p_niv'].values * units.Pa
    T = df['t'].values * units.kelvin
    Td = df['td'].values * units.kelvin
    
    #convert to the units we want to plot in
    p = p.to(units.hPa)
    T.ito('degC')
    Td.ito('degC')
    
    wind_speed = df['ff'].values * units.meter/units.second
    wind_dir = df['dd'].values * units.degrees

    wind_speed.ito('knots')
    u, v = mpcalc.wind_components(wind_speed, wind_dir)
    
    fig = plt.figure(figsize=figsize)


    skew = SkewT(fig, rotation=45)

    # Plot the data using normal plotting functions, in this case using
    # log scaling in Y, as dictated by the typical meteorological plot.
    skew.plot(p, T, 'r')
    skew.plot(p, Td, 'g')
    
    #plot wind barbs in knots, only plotting every 50th data point
    skew.plot_barbs(p[::50], u[::50], v[::50], xloc = 1.2, y_clip_radius = 0, length = 6)


    # Set some better labels than the default
    skew.ax.set_xlabel(f'Temperature [{T.units:~P}]')
    skew.ax.set_ylabel(f'Pressure [{p.units:~P}]')
    
    lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])
    skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black')

    # Calculate full parcel profile and add to plot as black line
    prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
    skew.plot(p, prof, 'k', linewidth=2)

    # Shade areas of CAPE and CIN
    skew.shade_cin(p, T, prof, Td)
    skew.shade_cape(p, T, prof)

    # An example of a slanted line at constant T -- in this case the 0
    # isotherm
    skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)

    # Add the relevant special lines
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()

    skew.ax.set_ylim(ylim_press)
    skew.ax.set_xlim(xlim)
    
    if addalt:
        # Create twin axis for altitude
        # Ensure increasing order for np.interp
        alt = df['altitude']
        p_inc = p.magnitude[::-1]  # reverse so pressure goes increasing (low → high)
        alt_dec = alt[::-1]  # reverse so altitude goes decreasing (high → low)

        def pressure_to_alt(pressures):
            return np.interp(pressures, p_inc, alt_dec)

        def alt_to_pressure(alts):
            return np.interp(alts, alt, p.magnitude)  # since z is already increasing

        secax = skew.ax.secondary_yaxis('right', functions=(pressure_to_alt, alt_to_pressure))
        secax.set_ylabel('Altitude [m ASL]')

        secax.set_yticks(np.arange(0, 30000, 1000))
        secax.set_yticklabels(np.arange(0, 30000, 1000))
        
        if heightaskm:
            y_vals = secax.get_yticks()
            secax.set_yticks(y_vals)
            secax.set_yticklabels(['{:,.0f}'.format(x /1000) for x in y_vals])
            secax.set_ylabel('Altitude [km ASL]')
            


    # Add title with launch time
    if sounding_dict['header']['date'].item() != "mean":
        startdate = info["date"].item()
        skew.ax.set_title(f'{site} {startdate.strftime("%d/%m/%Y %H:%M")} UTC', loc = 'right', color = 'grey')
    
    #machinery to save plot
    if saveplot:
        if saveasplot == 'auto':
            filename = f'{startdate.strftime("%Y%m%d%H%M%S")}_radiosonde_skewt_{site}'
        else:
            filename = saveasplot
            
        if not filename.endswith('.png'):
            filename +='.png'
            
        fig.savefig(os.path.join(outpath, filename), dpi = dpi, bbox_inches = 'tight')
        
        # check the file really saved where you thought
        if os.path.exists(os.path.join(outpath, filename)):
            print('The plot was saved at')
            print(os.path.abspath(os.path.join(outpath, filename)))
        else:
            print("The plot was not saved correctly, file not found")
    
    return fig, skew.ax


def plot_radiosonde_skewt_multi(
    sounding_list,
    figsize=(7,8),
    fontsize=10,
    ylim_press=(1000, 100),
    xlim=(-40, 40),
    quantiles=(0.05, 0.95),
    min_valid_frac=0.5,
    site="DDU"
):
    import numpy as np
    import matplotlib.pyplot as plt
    import metpy.calc as mpcalc
    from metpy.plots import SkewT
    from metpy.units import units

    plt.rcParams.update({'font.size': fontsize})

    # ----------------------------
    # Pressure grid (reference)
    # ----------------------------
    df0 = sounding_list[0]["data"]
    p = (df0["p_niv"].values * units.Pa).to("hPa")

    T_all, Td_all, alt_all = [], [], []

    # ----------------------------
    # stack profiles
    # ----------------------------
    for s in sounding_list:
        df = s["data"]

        T_all.append(df["t"].values - 273.15)
        Td_all.append(df["td"].values - 273.15)
        alt_all.append(df["altitude"].values)

    T_all = np.array(T_all)
    Td_all = np.array(Td_all)
    alt_all = np.array(alt_all)

    n_profiles, n_levels = T_all.shape
    min_valid = int(min_valid_frac * n_profiles)

    # ----------------------------
    # SAFE STATS
    # ----------------------------
    def safe(arr):
        valid = np.sum(~np.isnan(arr), axis=0)

        mean = np.full(n_levels, np.nan)
        q1 = np.full(n_levels, np.nan)
        q2 = np.full(n_levels, np.nan)

        ok = valid >= min_valid

        mean[ok] = np.nanmean(arr[:, ok], axis=0)
        q1[ok] = np.nanquantile(arr[:, ok], quantiles[0], axis=0)
        q2[ok] = np.nanquantile(arr[:, ok], quantiles[1], axis=0)

        return mean, q1, q2, ok

    T_mean, T_q1, T_q2, mask = safe(T_all)
    Td_mean, Td_q1, Td_q2, _ = safe(Td_all)

    alt_mean = np.nanmean(alt_all[:, mask], axis=0)

    # ----------------------------
    # CLEAN MASK (CRITICAL)
    # ----------------------------
    p = p[mask]
    T_mean = T_mean[mask]
    Td_mean = Td_mean[mask]
    T_q1 = T_q1[mask]
    T_q2 = T_q2[mask]
    Td_q1 = Td_q1[mask]
    Td_q2 = Td_q2[mask]

    # remove remaining NaN holes (IMPORTANT)
    valid_curve = ~np.isnan(T_mean) & ~np.isnan(Td_mean)

    p = p[valid_curve]
    T_mean = T_mean[valid_curve]
    Td_mean = Td_mean[valid_curve]
    T_q1 = T_q1[valid_curve]
    T_q2 = T_q2[valid_curve]
    Td_q1 = Td_q1[valid_curve]
    Td_q2 = Td_q2[valid_curve]
    alt_mean = alt_mean[valid_curve]

    # ----------------------------
    # PLOT SKewT
    # ----------------------------
    fig = plt.figure(figsize=figsize)
    skew = SkewT(fig, rotation=45)

    skew.plot(p, T_mean, 'r', lw=2, label="T mean")
    skew.plot(p, Td_mean, 'g', lw=2, label="Td mean")

    skew.ax.fill_betweenx(p, T_q1, T_q2, color='red', alpha=0.2)
    skew.ax.fill_betweenx(p, Td_q1, Td_q2, color='green', alpha=0.2)

    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()

    skew.ax.set_xlim(xlim)
    skew.ax.set_ylim(ylim_press)

  
    # ALTITUDE AXIS (RIGHT SIDE)
    def p_to_z(pp):
        return np.interp(pp, p.m[::-1], alt_mean[::-1])

    def z_to_p(zz):
        return np.interp(zz, alt_mean, p.m)

    secax = skew.ax.secondary_yaxis('right', functions=(p_to_z, z_to_p))
    secax.set_ylabel("Altitude [m ASL]")

    skew.ax.set_xlabel("Temperature [°C]")
    skew.ax.set_ylabel("Pressure [hPa]")

    skew.ax.set_title(f"{site} Multi SkewT ({n_profiles} profiles)")

    return fig, skew.ax