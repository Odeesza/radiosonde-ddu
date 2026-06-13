#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 11:21:50 2025

@author: corden
"""

# function to plot the hoorizontal displacement of the ddu radiosonde over the map of ddu and d17

def plot_radiosonde_ddu_map(sounding_dict, 
                           figsize = [6,3],
                           fontsize = 10, 
                           extent = [139.6, 140.0949, -66.6420, -66.8],
                           scale = 11,
                           xspace = 0.05,
                           yspace = 0.05 ,
                           saveplot = False,
                           saveasplot = 'auto',
                           outpath = '',
                           dpi = 300,
                           sites_file = '/home/corden/Documents/campaigns/awaca/awaca_locations.csv'
                           ):
    """
    Plot a DDU radiosonde horizontal displacement over a openstreetmap background

    Parameters
    ----------
    sounding_dict : dict of three dataframes
        The output from the parse_rs function.
    figsize : list, optional
        The default is [6,3].
    fontsize : int, optional
        The default is 10.
    extent : list, optional
        Map extent in decimal deg. The default is [139.6, 140.0949, -66.6420, -66.8].
    scale : int, optional
        Used of rthe map scale. The default is 11, which is good for an area cvering ddu and d17.
    xspace : float, optional
        lon grid spacing. The default is 0.05.
    yspace : float, optional
        lat grid spacing. The default is 0.05.
    saveplot : boolean, optional
        Whether to save the plot. The default is False.
    saveasplot : string, optional
        filename. The default is 'auto'.
    outpath : string, optional
        path to folder where the plot should be saved. The default is ''.
    dpi : int, optional
        The default is 300.
    sites_file : string, optional
        path to csv containing site location. The default is '/home/corden/Documents/campaigns/awaca/awaca_locations.csv'.

    Returns
    -------
    fig : matplotlib fig
        
    ax : matplotlib ax
        

    """
    
    import matplotlib.pyplot as plt
    import pandas as pd
    
    import cartopy.crs as ccrs
    from cartopy.io.img_tiles import OSM
    import matplotlib.ticker as mticker
    
    import os
    
    
    plt.rcParams.update({'font.size': fontsize})
    
    sites = pd.read_csv(sites_file)
    launch_loc = [-66.663167, 140.001] #ddu radiosonde launch site, according to metefrance
    
    fig, ax= plt.subplots(1, 1,  figsize = figsize)
    
    info = sounding_dict['header']
    df = sounding_dict['data']
    
    imagery = OSM() # use openstreetmap background image

    # initialise figure
    fig = plt.figure(figsize = figsize)
    ax= plt.axes(projection = imagery.crs)
    ax.set_extent(extent, crs = ccrs.PlateCarree())

    # Add the imagery to the map.
    ax.add_image(imagery, int(scale), interpolation='spline36')

    #gridlines
    gl = ax.gridlines(draw_labels = True, x_inline = False, y_inline = False)
    gl.rotate_labels = False
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.MultipleLocator(base=xspace) 
    gl.ylocator = mticker.MultipleLocator(base=yspace) 
    
    
    lats = launch_loc[0] + df['dep_lat']
    lons = launch_loc[1] + df['dep_lon']
    ax.plot(lons, lats, transform=ccrs.PlateCarree())

    # add d17
    ax.scatter(sites.lon[1], sites.lat[1], transform = ccrs.PlateCarree(), s = 10, c = 'blue')
    ax.annotate(sites.location[1], xy = (sites.lon[1], sites.lat[1]), xytext=(-3,-4), textcoords='offset points', ha='right', va='top', transform = ccrs.PlateCarree())

    
    # Add title with launch time
    startdate = info["date"].item()
    ax.set_title(f'{startdate.strftime("%d/%m/%Y %H:%M")} UTC', loc = 'right', color = 'grey')
    
    #machinery to save plot
    if saveplot:
        if saveasplot == 'auto':
            filename = f'{startdate.strftime("%Y%m%d%H%M%S")}_radiosonde_ddu_map'
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
    
    return fig, ax
