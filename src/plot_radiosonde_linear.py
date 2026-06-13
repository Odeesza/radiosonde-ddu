#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 14:58:28 2025

@author: corden
"""

# plot standard profiles of temp, humidity, wind from a metefrance radiosounding
# with linear axes (ie not skewT or similar)

def plot_radiosonde_linear(sounding_dict, 
                           figsize = [6,3],
                           fontsize = 10, 
                           ylim = [0, 15000], 
                           heightaskm = False, 
                           plotgrid = False,
                           saveplot = False,
                           saveasplot = 'auto',
                           outpath = '',
                           dpi = 300,
                           site = 'DDU'):
    """
    plot a basic profile of variables collected by a ddu radiosonde

    Parameters
    ----------
    sounding_dict : dict of three dataframes
        The output from the parse_rs function.
    figsize : list, optional
        The default is [6,3].
    fontsize : int, optional
        The default is 10.
    ylim : list, optional
        In m. The default is [0, 6000].
    heightaskm : boolean, optional
        Whether to plot the height axis in km. The default is False.
    plotgrid : boolean, optional
        The default is False.
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
    fig, (ax1, ax2, ax3, ax4)

    """
    
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.colors as mcolors
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator
    import os
    
    from metpy.calc import relative_humidity_from_dewpoint, saturation_vapor_pressure
    from metpy.units import units
    
    plt.rcParams.update({'font.size': fontsize})
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey = True, figsize = figsize, layout = 'constrained')
    
    info = sounding_dict['header']
    df = sounding_dict['data']
    df = df.where(df['altitude'] <= ylim[1]) #crop the data to have sensible automatic x limits
    
    # Temperature
    ax1.plot(df['t'] - 273.15, df['altitude'], label = r'$T$', c = 'tomato')
    ax1.plot(df['td'] - 273.15, df['altitude'], label = r'$T_d$', c = 'orange')
    ax1.set_xlabel('Temp. [°C]')
    ax1.legend()
    
    # Relative Humidity
    # calculate from the dew point temperature using metpy
    t = df['t'].values*units.kelvin
    rh = relative_humidity_from_dewpoint(df['t'].values*units.kelvin, df['td'].values*units.kelvin).to('percent')
    #rhi = relative_humidity_from_dewpoint(df['t'].values*units.kelvin, df['td'].values*units.kelvin, phase = 'solid').to('percent')
    rhi = rh*saturation_vapor_pressure(t, phase = 'liquid')/saturation_vapor_pressure(t, phase = 'solid')
    
    
    ax2.plot(rh, df['altitude'], label = r'$RH$', c = 'steelblue')
    ax2.plot(rhi, df['altitude'], label = r'$RH_i$', c = 'seagreen' )
    ax2.axvline(100, color = 'grey', linestyle = '-', linewidth = 0.5, zorder = 0)

    ax2.set_xlabel('Rel. Hum. [%]')
    ax2.legend()
    
    # Wind Speed
    ax3.plot(df['ff'], df['altitude'])
    ax3.set_xlabel('Speed [m/s]')
    
    # Wind Direction
    ax4.scatter(df['dd'], df['altitude'], s = 2)
    ax4.set_xlabel('Dir [°]')
    ax4.set_xlim([0, 360])
    
    ax1.set_ylabel('Altitude [m ASL]')
    ax1.set_ylim(ylim)
    
    if plotgrid == True:
        ax1.grid('major', 'both', color = 'grey', linestyle = '-', linewidth = 0.2 )
        ax2.grid('major', 'both', color = 'grey', linestyle = '-', linewidth = 0.2 )
        ax3.grid('major', 'both', color = 'grey', linestyle = '-', linewidth = 0.2 )
        
    if heightaskm:
        y_vals = ax1.get_yticks()
        ax1.set_yticks(y_vals)
        ax1.set_yticklabels(['{:,.0f}'.format(x /1000) for x in y_vals])
        ax1.set_ylabel('Altitude [km ASL]')
        
    ax1.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(4))    
    ax2.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax3.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax4.xaxis.set_minor_locator(MultipleLocator(45))
    ax4.set_xticks([0, 180])
    
    # Add title with launch time
    if sounding_dict['header']['date'].item() != "mean":
        startdate = info["date"].item()
        ax4.set_title(f'{site} {startdate.strftime("%d/%m/%Y %H:%M")} UTC', loc = 'right', color = 'grey')
    
    #machinery to save plot
    if saveplot:
        if saveasplot == 'auto':
            filename = f'{startdate.strftime("%Y%m%d%H%M%S")}_radiosonde_{site}'
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
    
    return fig, (ax1, ax2, ax3, ax4)


def plot_radiosonde_linear_multi(
    sounding_list,
    figsize=(8,4),
    fontsize=10,
    ylim=(0,15000),
    quantiles=(0.05, 0.95),
    plot_std=True,
    site="DDU"
):
    import numpy as np
    import matplotlib.pyplot as plt
    from metpy.calc import relative_humidity_from_dewpoint, saturation_vapor_pressure
    from metpy.units import units

    plt.rcParams.update({'font.size': fontsize})

    # ----------------------------
    # Extract aligned data
    # ----------------------------
    z = sounding_list[0]["data"]["altitude"].values
    mask = (z >= ylim[0]) & (z <= ylim[1])
    z = z[mask]

    T_all, Td_all, RH_all, RHi_all = [], [], [], []
    FF_all, U_all, V_all = [], [], []

    for s in sounding_list:
        df = s["data"]

        T = (df["t"].values - 273.15)[mask]
        Td = (df["td"].values - 273.15)[mask]
        FF = df["ff"].values[mask]
        DD = df["dd"].values[mask]

        #Wind components
        rad = np.deg2rad(DD)
        U = -FF * np.sin(rad)
        V = -FF * np.cos(rad)

        #RH
        t_k = (T + 273.15) * units.kelvin
        td_k = (Td + 273.15) * units.kelvin

        rh = relative_humidity_from_dewpoint(t_k, td_k).to('percent').magnitude
        rhi = rh * (
            saturation_vapor_pressure(t_k, phase='liquid') /
            saturation_vapor_pressure(t_k, phase='solid')
        ).magnitude

        T_all.append(T)
        Td_all.append(Td)
        RH_all.append(rh)
        RHi_all.append(rhi)
        FF_all.append(FF)
        U_all.append(U)
        V_all.append(V)

    # Convert to arrays
    def stack(x): return np.array(x)

    T_all, Td_all = stack(T_all), stack(Td_all)
    RH_all, RHi_all = stack(RH_all), stack(RHi_all)
    FF_all = stack(FF_all)
    U_all, V_all = stack(U_all), stack(V_all)

    # ----------------------------
    # Stats
    # ----------------------------
    def compute_stats(arr):
        mean = np.nanmean(arr, axis=0)
        q_low = np.nanquantile(arr, quantiles[0], axis=0)
        q_high = np.nanquantile(arr, quantiles[1], axis=0)
        std = np.nanstd(arr, axis=0)
        return mean, q_low, q_high, std

    T_mean, T_q1, T_q2, T_std = compute_stats(T_all)
    Td_mean, Td_q1, Td_q2, Td_std = compute_stats(Td_all)
    RH_mean, RH_q1, RH_q2, RH_std = compute_stats(RH_all)
    RHi_mean, RHi_q1, RHi_q2, RHi_std = compute_stats(RHi_all)
    FF_mean, FF_q1, FF_q2, FF_std = compute_stats(FF_all)

    # --- Wind mean via u/v
    U_mean = np.nanmean(U_all, axis=0)
    V_mean = np.nanmean(V_all, axis=0)

    DD_mean = (np.rad2deg(np.arctan2(-U_mean, -V_mean))) % 360
    FF_vec_mean = np.sqrt(U_mean**2 + V_mean**2)

    # ----------------------------
    # Plot
    # ----------------------------
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, sharey=True, figsize=figsize)

    def plot_with_band(ax, mean, q1, q2, std, color, label):
        ax.plot(mean, z, color=color, lw=2, label=label)
        ax.fill_betweenx(z, q1, q2, color=color, alpha=0.25)
        if plot_std:
            ax.fill_betweenx(z, mean-std, mean+std, color=color, alpha=0.1)

    # Temperature
    plot_with_band(ax1, T_mean, T_q1, T_q2, T_std, 'tomato', 'T')
    plot_with_band(ax1, Td_mean, Td_q1, Td_q2, Td_std, 'orange', 'Td')
    ax1.set_xlabel('Temp [°C]')
    ax1.legend()

    # RH
    plot_with_band(ax2, RH_mean, RH_q1, RH_q2, RH_std, 'steelblue', 'RH')
    plot_with_band(ax2, RHi_mean, RHi_q1, RHi_q2, RHi_std, 'seagreen', 'RHi')
    ax2.axvline(100, color='grey', lw=0.5)
    ax2.set_xlabel('RH [%]')

    # Wind speed
    plot_with_band(ax3, FF_mean, FF_q1, FF_q2, FF_std, 'black', 'Speed')
    ax3.plot(FF_vec_mean, z, '--', color='red', label='Vector mean')
    ax3.set_xlabel('m/s')
    ax3.legend()

    # Wind direction (now correct)
    ax4.plot(DD_mean, z, color='purple', lw=2)
    ax4.set_xlim(0, 360)
    ax4.set_xlabel('Dir [°]')

    ax1.set_ylabel('Altitude [m]')
    ax1.set_ylim(ylim)

    fig.suptitle(f"{site} Mean radiosonde profile ({len(sounding_list)} profiles)")

    return fig, (ax1, ax2, ax3, ax4)

    
    
    
    