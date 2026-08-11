import sys,os
import time
import datetime
import numpy as np
import scipy as sp
import pandas as pd
import xarray as xr
import netCDF4
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import seaborn as sns
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import wrf

from lib import wrf_diagnostics as wrfd
from lib.ops import find_nearest,trim,months,subset
from netCDF4 import Dataset
from wrf import getvar, interplevel, to_np, latlon_coords, get_cartopy, cartopy_xlim, cartopy_ylim, ALL_TIMES
#from cartopy.feature import NaturalEarthFeature, COLORS

def plot_temperature(wrf_output,month_name):
    # Calculates time-averaged temperature over a given month and plots
    # Inputs:
    #  wrf_output: The WRF file with variables
    # Returns:
    #  None
    # Extract and format variable of desired quantity; take average over specified time period

    air_temp = wrf_output['T2'].assign_coords({"lon":wrf_output['XLONG'],"lat":wrf_output['XLAT']})
    average_temp = air_temp.mean(dim='Time')

    # Plot time-averaged temperature

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    #upper_bound = np.round(average_temp.quantile(0.99),-2)
    lons = air_temp['lon'][0]
    lats = air_temp['lat'][0]
    data = average_temp
    levels = list(np.arange(250,320,7))

    avg_map = ax.contourf(lons, lats, data, levels,
                transform=ccrs.PlateCarree(),
                norm = colors.Normalize(vmin=250, vmax=320),
                cmap='nipy_spectral')
    ax.set_title(f'2-Meter Air Temperature over CONUS, Averaged for {month_name} 1981', fontsize = 20)
    ax.add_feature(cartopy.feature.STATES)
    ax.coastlines()
    ax.set_extent([-110,-70,25,50])

    gl = ax.gridlines(visible=False,draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    fig.colorbar(avg_map,label='2-Meter Air Temperature (K)')

    plt.show()

    # Change to output directory; save figure

    out_dir = ('../../images')
    os.chdir(out_dir)

    map_out = fig.get_figure()
    map_out.savefig(f'{month_name}_temp.jpg',bbox_inches='tight')

def plot_precip(wrf_output,month_name): 
    # Calculates accumulated precipitation over a given month and plots
    # Inputs:
    #  wrf_output: The WRF file with variables
    # Returns:
    #  None

    # Extract and format variable of desired quantity; take average over specified time period

    total_precip = (wrf_output['RAINC'] + wrf_output['RAINNC']).assign_coords({"lon":wrf_output['XLONG'],"lat":wrf_output['XLAT']})
    accum_precip = total_precip[-1] - total_precip[0] #<- Indexing needs to be in this order to descend into chunk of array, then time dimension
    
    #print('Final precipitation',np.mean(total_precip[-1][0].values))
    #print('Initial precipitation',np.mean(total_precip[0][0].values))
    # Plot accumulated monthly precipitation

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    #upper_bound = np.round(accum_precip.quantile(0.99),-2)
    lons = total_precip['lon'][0]
    lats = total_precip['lat'][0]
    data = accum_precip
    #print('Longitude array:',lons)
    print('Precip. data:',data)
    levels = list(np.arange(0,210,10))

    avg_map = ax.contourf(lons, lats, data, levels,
                transform=ccrs.PlateCarree(),
                norm = colors.Normalize(vmin=0, vmax=210),
                extend = 'max',
                cmap='Blues')
    ax.set_title(f'Accumulated Precipitation over CONUS for {month_name} 1981', fontsize = 20)
    ax.add_feature(cartopy.feature.STATES)
    ax.coastlines()
    ax.set_extent([-110,-70,25,50])

    gl = ax.gridlines(visible=False,draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    fig.colorbar(avg_map,label='Accumulated Precipitation (mm)')
    plt.show()

    # Change to output directory; save figure

    out_dir = ('../../images')
    os.chdir(out_dir)

    map_out = fig.get_figure()
    map_out.savefig(f'{month_name}_precip.jpg',bbox_inches='tight')

def plot_srh(num_month,month_name): 
    # Calculates average SRH over a given month and plots
    # Inputs:
    #  wrf_output: The WRF file with variables
    # Returns:
    #  None

    # Set up list of filenames
    file_list = [f for f in os.listdir() if f.startswith(f'wrfout_d01_1981-{num_month}')]
    print(len(file_list))

    # Set up an empty array to store results of getvar
    output_shape = (len(file_list),164,329)
    srh_final = np.empty(output_shape, np.float32)

    for index in range(output_shape[0]):
        f = Dataset(file_list[index])
        srh = getvar(f, 'helicity')

        srh_final[index,:] = srh[:]
        f.close()

    # Open reference file to extract spatial grid for plotting
    wrf_file = xr.open_dataset('wrfout_d01_1981-01-01_00:00:00')

    # Calculate time-averaged SRH
    average_srh = np.nanmean(srh_final, axis=0)

    # Plot time-averaged SRH

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    lons = wrf_file['XLONG'][0]
    lats = wrf_file['XLAT'][0]
    data = average_srh
    levels = list(np.arange(0,600,100))

    # Print a couple of sanity checks to the screen
    print('Dataset to plot is',data)
    #print('First element of this dataset is',average_cape[0])
    
    avg_map = ax.contourf(lons, lats, data, levels,
                transform = ccrs.PlateCarree(),
                norm = colors.Normalize(vmin=0, vmax=600),
                extend = 'max',
                cmap='Blues')
    ax.set_title(f'Average SRH over CONUS for {month_name} 1981', fontsize = 20)
    ax.add_feature(cartopy.feature.STATES)
    ax.coastlines()
    ax.set_extent([-110,-70,25,50])

    gl = ax.gridlines(visible=False,draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    fig.colorbar(avg_map,label='SRH (m^2/s^2)')

    plt.show()

    # Change to output directory; save figure

    out_dir = ('../../images')
    os.chdir(out_dir)

    map_out = fig.get_figure()
    map_out.savefig(f'{month_name}_SRH.jpg',bbox_inches='tight')

def plot_shear(num_month,month_name): 
    # Calculates average wind shear over a given month and plots
    # Inputs:
    #  wrf_output: The WRF file with variables
    # Returns:
    #  None

    # Set up list of filenames
    file_list = [f for f in os.listdir() if f.startswith(f'wrfout_d01_1981-{num_month}')]
    print(len(file_list))

    # Set up an empty array to store results of getvar
    output_shape = (len(file_list),2,164,329)
    shear_final = np.empty(output_shape, np.float32)

    for index in range(output_shape[0]):
        f = Dataset(file_list[index])
        wind = getvar(f, 'uvmet_wspd_wdir', timeidx=0)
        height = getvar(f, 'zstag')

        w_0 = find_nearest(height,0)
        w_6 = find_nearest(height,6000)
        wind_0 = wind[:,w_0,:,:]
        wind_6 = wind[:,w_6,:,:]

        #print('Heights:', height[:,0,0])
        #print('Index nearest to surface is', w_0)
        #print('Index nearest to 6 km is', w_6)
        #break

        shear = wind_6 - wind_0

        shear_final[index,:] = shear[:]
        f.close()

    
    # Open reference file to extract spatial grid for plotting
    wrf_file = xr.open_dataset('wrfout_d01_1981-01-01_00:00:00')

    # Calculate time-averaged SRH
    average_shear = np.nanmean(shear_final, axis=0)

    # Plot time-averaged SRH

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    lons = wrf_file['XLONG'][0]
    lats = wrf_file['XLAT'][0]
    data = average_shear[0]
    levels = list(np.arange(0,70,10))

    # Print a couple of sanity checks to the screen
    #print('Dataset to plot is',data)
    #print('First element of this dataset is',average_cape[0])
    
    avg_map = ax.contourf(lons, lats, data, levels,
                transform = ccrs.PlateCarree(),
                norm = colors.Normalize(vmin=0, vmax=60),
                extend = 'max',
                cmap='Blues')
    ax.set_title(f'Average 0-6 km Wind Shear over CONUS for {month_name} 1981', fontsize = 20)
    ax.add_feature(cartopy.feature.STATES)
    ax.coastlines()
    ax.set_extent([-110,-70,25,50])

    gl = ax.gridlines(visible=False,draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    fig.colorbar(avg_map,label='0-6 km Wind Shear (m/s)')

    plt.show()

    # Change to output directory; save figure

    out_dir = ('../../images')
    os.chdir(out_dir)

    map_out = fig.get_figure()
    map_out.savefig(f'{month_name}_shear.jpg',bbox_inches='tight')

def plot_cape(num_month,month_name): 
    # Calculates average CAPE over a given month and plots
    # Inputs:
    #  wrf_output: The WRF file with variables
    # Returns:
    #  None

    # Set up list of filenames
    file_list = [f for f in os.listdir() if f.startswith(f'wrfout_d01_1991-{num_month}')]
    print(len(file_list))

    # Set up an empty array to store results of getvar
    output_shape = (len(file_list),4,164,329)
    cape_final = np.empty(output_shape, np.float32)

    for index in range(output_shape[0]):
        f = Dataset(file_list[index])
        cape = getvar(f, 'cape_2d')

        cape_final[index,:] = cape[:]
        f.close()

    # Open reference file to extract spatial grid for plotting
    wrf_file = xr.open_dataset('wrfout_d01_1991-01-01_00_00_00')
    print(wrf_file)

    # Extract and format variable of desired quantity; take average over specified time period

    average_cape = np.nanmean(cape_final, axis=0)
    print('Computed average CAPE for the month!')

    # Plot time-averaged CAPE

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    #upper_bound = np.round(average_cape.quantile(0.99),-2)
    lons = wrf_file['XLONG'].squeeze('Time')
    lats = wrf_file['XLAT'].squeeze('Time')
    data = average_cape[0]
    levels = list(np.arange(0,600,100))

    # Print a couple of sanity checks to the screen
    #print('Dataset to plot is',data)
    #print('First element of this dataset is',average_cape[0])

    avg_map = ax.contourf(lons, lats, data, levels,
                transform=ccrs.PlateCarree(),
                norm = colors.Normalize(vmin=0, vmax=600),
                extend = 'max',
                cmap='Oranges')
    ax.set_title(f'Average CAPE over CONUS for {month_name} 1981', fontsize = 20)
    ax.add_feature(cartopy.feature.STATES)
    ax.coastlines()
    ax.set_extent([-110,-70,25,50])

    gl = ax.gridlines(visible=False,draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    fig.colorbar(avg_map,label='CAPE(J/kg)')

    plt.show()

    # Change to output directory; save figure

    out_dir = ('../../images/')
    if os.path.exists(out_dir):
        os.chdir(out_dir)
    else:
        os.mkdir(out_dir)
        os.chdir(out_dir)

    map_out = fig.get_figure()
    #map_out.savefig(f'{month_name}_CAPE.jpg',bbox_inches='tight')

def plot_cin(num_month,month_name): 
    # Calculates average CIN over a given month and plots
    # Inputs:
    #  wrf_output: The WRF file with variables
    # Returns:
    #  None

    # Set up list of filenames
    file_list = [f for f in os.listdir() if f.startswith(f'wrfout_d01_1981-{num_month}')]
    print(len(file_list))

    # Set up an empty array to store results of getvar
    output_shape = (len(file_list),4,164,329)
    cin_final = np.empty(output_shape, np.float32)

    for index in range(output_shape[0]):
        f = Dataset(file_list[index])
        cin = getvar(f, 'cape_2d')

        cin_final[index,:] = cin[:]
        f.close()

    # Open reference file to extract spatial grid for plotting
    wrf_file = xr.open_dataset('wrfout_d01_1981-01-01_00:00:00')
    print(wrf_file)

    # Extract and format variable of desired quantity; take average over specified time period

    average_cin = np.nanmean(cin_final, axis=0)
    print('Computed average CIN for the month!')

    # Plot time-averaged CIN

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    #upper_bound = np.round(average_cape.quantile(0.99),-2)
    lons = wrf_file['XLONG'].squeeze('Time')
    lats = wrf_file['XLAT'].squeeze('Time')
    data = average_cin[1]
    levels = list(np.arange(0,600,100))

    # Print a couple of sanity checks to the screen
    #print('Dataset to plot is',data)
    #print('First element of this dataset is',average_cin[0])

    avg_map = ax.contourf(lons, lats, data, levels,
                transform=ccrs.PlateCarree(),
                norm = colors.Normalize(vmin=0, vmax=600),
                extend = 'max',
                cmap='Oranges')
    ax.set_title(f'Average CIN over CONUS for {month_name} 1981', fontsize = 20)
    ax.add_feature(cartopy.feature.STATES)
    ax.coastlines()
    ax.set_extent([-110,-70,25,50])

    gl = ax.gridlines(visible=False,draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    fig.colorbar(avg_map,label='CIN(J/kg)')

    plt.show()

    # Change to output directory; save figure

    out_dir = ('../../images/')
    if os.path.exists(out_dir):
        os.chdir(out_dir)
    else:
        os.mkdir(out_dir)
        os.chdir(out_dir)

    map_out = fig.get_figure()
    map_out.savefig(f'{month_name}_CIN.jpg',bbox_inches='tight')

def plot_climo(climo_file,wrf_output,domain,epoch,month_name,variable):
    # Takes annually averaged data for a given month and plots climatology
    # Inputs:
    #  climo_file: The WRF file with variables
    #  month_name: The name of the month to analyze climatologies for
    #  variable: The month to analyze climatologies for
    # Returns:
    #  None
    # Extract and format variable of desired quantity; take average over specified time period
    
    # Things that need to be specified for each variable:
    # Variable name in file, long variable name, contours, colormap, units
    # Example function to handle
    # assign(variable) - Returns all above

    # Set name of data array to read
    if variable == 'T2':
        var = 'T2'
        name = '2-Meter Air Temperature'
        units = 'K'
    elif variable == 'PRECIP':
        var = 'PRECIP'
        name = 'Precipitation'
        units = 'mm'
    elif variable == 'CAPE':
        var = 'CAPE'
        name = 'Convective Available Potential Energy'
        units = 'J/kg'
        colorbar = 'Oranges'
        c_min = 0
        c_max = 1600
        step = 100
    elif variable == 'CIN':
        var = 'CIN'
        name = 'Convective Inhibition'
        units = 'J/kg'
        colorbar = 'Oranges'
        c_min = 0
        c_max = 350
        step = 50
    elif variable == 'SRH':
        var = 'SRH'
        name = 'Storm Relative Helicity'
        units = 'm^2/s^2'
        colorbar = 'Blues'
        c_min = 0
        c_max = 600
        step = 100
    elif variable == 'SHEAR':
        var = 'SHEAR'
        name = '0-6 km Wind Shear'
        units = 'm/s'
        colorbar = 'Blues'
        c_min = 0
        c_max = 60
        step = 10

    if epoch == 'hist':
        epoch_name = 'Historical'
    elif epoch == 'fut':
        epoch_name = 'Future'
    
    # Read in climatology file
    climo_geo = climo_file.assign_coords({"lon":wrf_output['XLONG'],"lat":wrf_output['XLAT']})
    climo_var = climo_geo[f'{var}']
    
    # Plot climatological values of variable

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    lons = climo_geo['lon'][0]
    lats = climo_geo['lat'][0]
    if variable == 'SHEAR':
        data = climo_var[0]
    else:
        data = climo_var.mean(dim='time').where(wrf_output['XLAND'] != 2.0).squeeze('Time') #<- Change this back and forth if using averages or not
    print(data)
    levels = list(np.arange(c_min,c_max,step))

    avg_map = ax.contourf(lons, lats, data, levels,
                transform=ccrs.PlateCarree(),
                norm = colors.Normalize(vmin=c_min, vmax=c_max),
                cmap=f'{colorbar}')
    if domain == 'd01':
        ax.set_title(f'Outer Domain {epoch_name} Climatology of\n {name}\n for {month_name} over CONUS', fontsize = 20)
    elif domain == 'd02':
        ax.set_title(f'4 km {epoch_name} Climatology of\n {name}\n for {month_name} over CONUS', fontsize = 20)
    elif domain == '27':
        ax.set_title(f'27 km {epoch_name} Climatology of\n {name}\n for {month_name} over CONUS', fontsize = 20)
    ax.add_feature(cartopy.feature.STATES)
    ax.coastlines()
    ax.set_extent([-110,-70,25,50])

    gl = ax.gridlines(visible=False,draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    fig.colorbar(avg_map,label=f'{name} ({units})')

    fig.tight_layout()

    plt.show()

    # Change to output directory; save figure

    out_dir = ('/pscratch/sd/n/nee2000/WRF-Prod/images_revisions/climatologies/')
    print(os.getcwd())
    if os.path.exists(out_dir):
        os.chdir(out_dir)
    else:
        os.mkdir(out_dir)
        os.chdir(out_dir)

    map_out = fig.get_figure()
    map_out.savefig(f'{month_name}_climo_{domain}_{epoch}_{var}.jpg',bbox_inches='tight')
    print('Saved climatology figure to', os.getcwd())

def plot_climo_diff(climo_file,wrf_output,domain,month_name,variable):
    # Takes annually averaged data for a given month and plots climatology
    # Inputs:
    #  climo_file: The WRF file with variables
    #  month_name: The name of the month to analyze climatologies for
    #  variable: The month to analyze climatologies for
    # Returns:
    #  None
    # Extract and format variable of desired quantity; take average over specified time period
    
    # Things that need to be specified for each variable:
    # Variable name in file, long variable name, contours, colormap, units
    # Example function to handle
    # assign(variable) - Returns all above

    # Set name of data array to read
    if variable == 'T2':
        var = 'T2'
        name = '2-Meter Air Temperature'
        units = 'K'
    elif variable == 'PRECIP':
        var = 'PRECIP'
        name = 'Precipitation'
        units = 'mm'
    elif variable == 'CAPE':
        var = 'CAPE'
        name = 'Convective Available Potential Energy'
        units = 'J/kg'
        colorbar = 'RdBu_r'
        c_min = -800
        c_max = 900
        step = 100
    elif variable == 'CIN':
        var = 'CIN'
        name = 'Convective Inhibition'
        units = 'J/kg'
        colorbar = 'RdGy_r'
        c_min = -300
        c_max = 350
        step = 50
    elif variable == 'SRH':
        var = 'SRH'
        name = 'Storm Relative Helicity'
        units = 'm^2/s^2'
        colorbar = 'BrBG'
        c_min = -300
        c_max = 300
        step = 50
    elif variable == 'SHEAR':
        var = 'SHEAR'
        name = '0-6 km Wind Shear'
        units = 'm/s'
        colorbar = 'PuOr'
        c_min = -10
        c_max = 10
        step = 2
    
    # Plot climatological values of variable

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    lons = wrf_output['XLONG'].squeeze('Time')
    lats = wrf_output['XLAT'].squeeze('Time')
    data = climo_file[f'{var}']
    print(data)
    if var == 'SHEAR':
        data = climo_file[f'{var}'][0]
    levels = list(np.arange(c_min,c_max,step))


    avg_map = ax.contourf(lons, lats, data, levels,
                transform=ccrs.PlateCarree(),
                norm = colors.Normalize(vmin=c_min, vmax=c_max),
                cmap=f'{colorbar}')
    #avg_map = ax.contourf(lons, lats, data, levels,
    #            transform=ccrs.PlateCarree(),
    #            norm = colors.TwoSlopeNorm(vmin=-200, vcenter=0, vmax = 800),
    #            cmap=f'{colorbar}')
    if domain == 'd01':
        ax.set_title(f'Climatological Difference of\n {name}\n for {month_name} over CONUS (Outer Domain)', fontsize = 20)
    else:
        ax.set_title(f'Climatological Difference of\n {name}\n for {month_name} over CONUS', fontsize = 20)
    ax.add_feature(cartopy.feature.STATES)
    ax.coastlines()
    ax.set_extent([-110,-70,25,50])

    gl = ax.gridlines(visible=False,draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    fig.colorbar(avg_map,label=f'{name} ({units})')

    fig.tight_layout()

    plt.show()

    # Change to output directory; save figure

    out_dir = ('images_final/')
    if os.path.exists(out_dir):
        os.chdir(out_dir)
    else:
        os.mkdir(out_dir)
        os.chdir(out_dir)
    print(os.getcwd())

    map_out = fig.get_figure()
    map_out.savefig(f'{month_name}_climodiff_{domain}_{var}.jpg',bbox_inches='tight',dpi=300)

def plot_env_diff(env_diff,wrf_output,domain,month_name,u_test):
    # Takes annually averaged data for a given month and plots climatology
    # Inputs:
    #  env_diff: The dataset for favorable environment differences
    #  month_name: The name of the month to analyze climatologies for
    #  variable: The month to analyze climatologies for
    # Returns:
    #  None
    # Extract and format variable of desired quantity; take average over specified time period
    
    # Things that need to be specified for each variable:
    # Variable name in file, long variable name, contours, colormap, units
    # Example function to handle
    # assign(variable) - Returns all above
 
    # Plot climatological probabilities of favorable environment

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    lons = wrf_output['XLONG'].squeeze('Time')
    lats = wrf_output['XLAT'].squeeze('Time')
    
    #data = env_diff.where((wrf_output['XLAND'] != 2.0)).fillna(0.3).squeeze('Time') # Try treating water as 0 and near-zero as NaN
    #data = env_diff.where(env_diff >= 0.2).fillna(-100) # Only using this line allows things to work properly
    
    # This set of lines together produces something close to correct
    init_data = env_diff#.where(wrf_output['XLAND'] != 2.0).fillna(1)
    data = init_data.where((init_data >= 1) | (init_data <= 0)).fillna(-100)
    data = data.where((wrf_output['XLAND'] != 2.0)).fillna(0.1).squeeze('Time') # Try treating water as 0 and near-zero as NaN
    
    #data = init_data.where((init_data <= 1) & (init_data >= 0.5)).fillna(-100).squeeze('Time')
    #print(data.values.min())
    #masked = np.ma.masked_equal(data,0.1)
    #print(np.count_nonzero(np.isnan(data)))
    #print(data)
    levels = list(np.arange(-10,11,1))
    
    # Colormap handling
    #cmap = mpl.cm.get_cmap('bwr').copy()
    cmap = mpl.cm.bwr
    #print(cmap)
    #cmap.set_under(color='lightgray')

    #plt.gca().set_facecolor('black')
    avg_map = ax.contourf(lons, lats, data, levels,
                transform=ccrs.PlateCarree(),
                norm = colors.Normalize(vmin=-10, vmax=11),
                cmap=cmap,extend='both')
    #data = data.transpose('y','x')
    #avg_map = data.plot.contourf(levels=levels,subplot_kws={'projection':ccrs.PlateCarree()})
   
    [m,n] = np.where((u_test[1] < 0.025) & (data != 0.1) & (data != -100))

    masked=np.zeros(u_test[1].shape)
    masked[m, n] = 1000

    #[m,n] = np.where(april_data > 0.5)

    #masked=np.zeros(april_data.shape)
    #masked[m, n] = 1000

    hatched = ax.contourf(lons, lats, masked, colors = 'none',
                      levels = 3, hatches = ["","."],
                    transform=ccrs.PlateCarree())
   
    ax.set_title(f'Average Future Change in Number of Days \nwith Favorable HCW Environment for {month_name}', fontsize = 20)
    ax.add_feature(cartopy.feature.STATES)
    #ax.add_feature(cartopy.feature.STATES,facecolor='gray',zorder=0)
    #ax.add_feature(cartopy.feature.STATES,facecolor='none',zorder=2)
    ax.coastlines()
    ax.set_extent([-105,-70,25,49])

    #avg_map.cmap.set_under('black')
    cmap = avg_map.get_cmap()
    cmap.set_under('lightgray')
    avg_map.set_cmap(cmap)

    gl = ax.gridlines(visible=False,draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size':14}
    gl.ylabel_style = {'size':14}

    cbar = fig.colorbar(avg_map)
    cbar.set_label('Average Number of Days \nwith Favorable Environment', size='x-large')
    cbar.ax.tick_params(labelsize=12)

    fig.tight_layout()

    plt.show()

    # Change to output directory; save figure

    out_dir = ('images_revisions/')
    if os.path.exists(out_dir):
        os.chdir(out_dir)
    else:
        os.mkdir(out_dir)
        os.chdir(out_dir)
    print('Image directory is: ',os.getcwd())

    map_out = fig.get_figure()
    map_out.savefig(f'fig5_{month_name}_favenvdiff_hatched.jpg',bbox_inches='tight',dpi=300)

def plot_rel_diff(env_diff,hist_data,wrf_output,domain,month_name,u_test):
    # Plot the relative change in environmental favorability between climate states
    # Inputs:
    #  env_diff: The dataset for favorable environment differences
    #  month_name: The name of the month to analyze climatologies for
    #  variable: The month to analyze climatologies for
    # Returns:
    #  None
    # Extract and format variable of desired quantity; take average over specified time period
    
    # Things that need to be specified for each variable:
    # Variable name in file, long variable name, contours, colormap, units
    # Example function to handle
    # assign(variable) - Returns all above

    # Plot climatological probabilities of favorable environment

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    cmin=-300
    cmax=300
    step=50

    lons = wrf_output['XLONG'].squeeze('Time')
    lats = wrf_output['XLAT'].squeeze('Time')
    data = env_diff['PROB_FAV'].fillna(0.1).where(wrf_output['XLAND'] != 2.0).squeeze('Time') * 100
    print('Mean change favorable environment probability is:', np.nanmean(data.values))
    print('95th percentile change of favorable environment probability is:', np.nanpercentile(data.values,95))
    
    # Generate levels
    levels = list(np.arange(cmin,cmax+50,50))
    #print(levels)
    
    # Colormap handling
    cmap = mpl.cm.get_cmap('coolwarm').copy()
    cmap.set_under(color='gray')

    avg_map = ax.contourf(lons, lats, data, levels,
                transform=ccrs.PlateCarree(),
                norm = colors.Normalize(vmin=cmin, vmax=cmax),
                cmap='coolwarm',
                extend='both')

    [m,n] = np.where((u_test[1] < 0.05) & ((data > 10) | (data < 0)))

    masked=np.zeros(u_test[1].shape)
    masked[m, n] = 1000

    #[m,n] = np.where(april_data > 0.5)

    #masked=np.zeros(april_data.shape)
    #masked[m, n] = 1000

    hatched = ax.contourf(lons, lats, masked, colors = 'none',
                      levels = 3, hatches = ["","."],
                    transform=ccrs.PlateCarree())
    ax.set_title(f'Future Relative Change in Probability \nof Favorable HCW Environment for {month_name}', fontsize = 20)
    ax.add_feature(cartopy.feature.STATES)
    ax.coastlines()
    ax.set_extent([-105,-70,25,49])
    #ax.set_extent([-110,-70,25,50])

    gl = ax.gridlines(visible=False,draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size':14}
    gl.ylabel_style = {'size':14}

    cbar = fig.colorbar(avg_map)
    cbar.set_label('Relative Change in Probability\n of Favorable Environment (%)', size='x-large')
    
    ticks=np.arange(cmin,cmax+step,step)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_yticklabels(list(map(str,list(ticks))))
    
    #ticks = np.logspace(np.log10(cmin), np.log10(cmax), num=11)
    #cbar = fig.colorbar(avg_map, ax=ax, label='Relative Change in Probability of Favorable Environment (%)', ticks=ticks)

    fig.tight_layout()

    plt.show()

    # Change to output directory; save figure

    out_dir = ('images_revisions/')
    if os.path.exists(out_dir):
        os.chdir(out_dir)
    else:
        os.mkdir(out_dir)
        os.chdir(out_dir)
    print('Image directory is: ',os.getcwd())

    map_out = fig.get_figure()
    map_out.savefig(f'fig7_{month_name}_{cmax}_relfavenvdiff_stippled.jpg',bbox_inches='tight',dpi=300)

def plot_envs(climo_probs,wrf_output,epoch,month_name,length):
    # Takes annually averaged data for a given month and plots climatology
    # Inputs:
    #  climo_file: The WRF file with variables
    #  month_name: The name of the month to analyze climatologies for
    #  variable: The month to analyze climatologies for
    # Returns:
    #  None
    # Extract and format variable of desired quantity; take average over specified time period
    
    # Things that need to be specified for each variable:
    # Variable name in file, long variable name, contours, colormap, units
    # Example function to handle
    # assign(variable) - Returns all above
 
    # Set name of epoch
    if epoch == 'hist':
        epoch_name = 'Historical'
    elif epoch == 'fut':
        epoch_name = 'Future'
    
    # Resample data
    #resample = climo_probs['PROB_FAV'].resample(time='1D').max()
    #print(resample)

    # Plot climatological probabilities of favorable environment

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    lons = wrf_output['XLONG'].squeeze('Time')
    lats = wrf_output['XLAT'].squeeze('Time')
    #print(climo_probs)
    #data = resample.where(wrf_output['XLAND'] != 2.0).squeeze('Time').mean(dim='time')# * length
    data = climo_probs
    print(data)
    levels = list(np.arange(0,31,2))
    avg_map = ax.contourf(lons, lats, data, levels,
                transform=ccrs.PlateCarree(),
                norm = colors.Normalize(vmin=0, vmax=31),
                cmap='Reds')
    ax.set_title(f'{epoch_name} Average Number of Days \nwith Favorable Environment for {month_name}', fontsize = 20)
    ax.add_feature(cartopy.feature.STATES)
    ax.coastlines()
    ax.set_extent([-105,-70,25,49])
    
    gl = ax.gridlines(visible=False,draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size':14}
    gl.ylabel_style = {'size':14}

    cbar = fig.colorbar(avg_map)
    cbar.set_label('Average Number of Days \nwith Favorable Environment', size='x-large')
    cbar.ax.tick_params(labelsize=12)

    fig.tight_layout()

    plt.show()

    # Change to output directory; save figure

    out_dir = ('images_final/')
    if os.path.exists(out_dir):
        os.chdir(out_dir)
    else:
        os.mkdir(out_dir)
        os.chdir(out_dir)
    print(os.getcwd())

    map_out = fig.get_figure()
    map_out.savefig(f'fig3_{month_name}_{epoch}_favenvs.jpg',bbox_inches='tight',dpi=300)
    # CHANGE THIS FOR EACH FIGURE SET

def plot_occ(prob_occ,wrf_output,epoch,month_name):
    # Takes annually averaged data for a given month and plots climatology
    # Inputs:
    #  climo_file: The WRF file with variables
    #  month_name: The name of the month to analyze climatologies for
    #  variable: The month to analyze climatologies for
    # Returns:
    #  None
    # Extract and format variable of desired quantity; take average over specified time period
    
    # Things that need to be specified for each variable:
    # Variable name in file, long variable name, contours, colormap, units
    # Example function to handle
    # assign(variable) - Returns all above
 
    # Set name of epoch
    if epoch == 'hist':
        epoch_name = 'Historical'
    elif epoch == 'fut':
        epoch_name = 'Future'

    # Plot climatological probabilities of HCW occurrence

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    lons = wrf_output['XLONG'].squeeze('Time')
    lats = wrf_output['XLAT'].squeeze('Time')
    #print(prob_occ)
    data = prob_occ['PROB_OCC'].squeeze('Time')*2480 #<- Multiplying by total number of time steps in epoch
    levels = list(np.arange(0,6,1))

    avg_map = ax.contourf(lons, lats, data, levels,
                transform=ccrs.PlateCarree(),
                norm = colors.Normalize(vmin=0, vmax=6),
                cmap='YlOrRd')
    #avg_map = ax.contourf(lons, lats, data, levels,
    #            transform=ccrs.PlateCarree(),
    #            norm = colors.TwoSlopeNorm(vmin=-200, vcenter=0, vmax = 800),
    #            cmap=f'{colorbar}')
    ax.set_title(f'Average {epoch_name} Number of Days with HCW Occurrence for {month_name}', fontsize = 20)
    ax.add_feature(cartopy.feature.STATES)
    ax.coastlines()
    ax.set_extent([-110,-70,25,50])

    gl = ax.gridlines(visible=False,draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    fig.colorbar(avg_map,label=f'# Time Steps of HCW Occurrence per Season')

    fig.tight_layout()

    plt.show()

    # Change to output directory; save figure

    out_dir = ('images/')
    if os.path.exists(out_dir):
        os.chdir(out_dir)
    else:
        os.mkdir(out_dir)
        os.chdir(out_dir)
    print(os.getcwd())

    map_out = fig.get_figure()
    #map_out.savefig(f'{month_name}_{epoch}_probocc.jpg',bbox_inches='tight')

def plot_kde(hist_4,fut_4,hist_27,fut_27,variable,region):
    var_name = variable
    if variable == 'CAPE':
        units = 'J/kg'
    elif variable == 'CIN':
        units = 'J/kg'
    elif variable == 'SRH':
        units = 'm^2/s^2'
    elif variable == 'SHEAR':
        units = 'm/s'

    if region == 'NGP':
        region_name = 'Northern Great Plains'
    elif region == 'SGP':
        region_name = 'Southern Great Plains'
    elif region == 'MID':
        region_name = 'Mid-Atlantic'
    elif region == 'SE':
        region_name = 'Southeast US'
    else:
        pass
    print(f'Plotting {variable} for {region}')

    hist_27 = hist_27.where((hist_27.lon.compute() > hist_4.lon.min().compute()) & (hist_27.lon.compute() < hist_4.lon.max().compute()) & (hist_27.lat.compute() > hist_4.lat.min().compute()) & (hist_27.lat.compute() < hist_4.lat.max().compute()),drop=True)
    fut_27 = fut_27.where((fut_27.lon.compute() > fut_4.lon.min().compute()) & (fut_27.lon.compute() < fut_4.lon.max().compute()) & (fut_27.lat.compute() > fut_4.lat.min().compute()) & (fut_27.lat.compute() < fut_4.lat.max().compute()),drop=True)
    print(hist_4.lat.min().values,hist_4.lat.max().values)
    print(hist_27)
    #hist_27 = hist_27.sel(lat=slice(hist_4.lat.min().values,hist_4.lat.max().values),lon=slice(hist_4.lon.min().values,hist_4.lon.max().values))
    #fut_27 = fut_27.sel(lat=slice(fut_4.lat.min().values,fut_4.lat.max().values),lon=slice(fut_4.lon.min().values,fut_4.lon.max().values))
    #print(hist_27)
    
    #fig,ax = plt.subplots()
    if variable == 'SRH':
        #sns.set_theme()
        start_time = time.time()
        hist_4,fut_4,hist_27,fut_27 = trim(hist_4,fut_4,hist_27,fut_27,variable,0.01,0.99)
        kde_plot = sns.kdeplot(data = np.random.choice(hist_4[f'{variable}'].values.flatten(),int(len(hist_4[f'{variable}'].values.flatten())/100)), label = 'Convection-Permitting Historical Climate')
        print('Completed plotting of historical 4km')
        print(f'Took {time.time() - start_time} seconds to plot') 
        kde_plot = sns.kdeplot(data = np.random.choice(fut_4[f'{variable}'].values.flatten(),int(len(fut_4[f'{variable}'].values.flatten())/100)), label = 'Convection-Permitting Future Climate')
        print('Completed plotting of future 4km')

        start_27 = time.time()
        kde_plot = sns.kdeplot(data = np.random.choice(fut_27[f'{variable}'].values.flatten(),int(len(fut_27[f'{variable}'].values.flatten())/100)), label = '27 km Future Climate')
        print('Completed plotting of future 27km')
        print(f'Took {time.time() - start_27} seconds to plot')
        kde_plot = sns.kdeplot(data = np.random.choice(hist_27[f'{variable}'].values.flatten(),int(len(hist_27[f'{variable}'].values.flatten())/100)), label = '27 km Historical Climate')
        print('Completed plotting of historical 27km')
        if region == 'ALL':
            kde_plot.set_title(f'Kernel Density Estimation of {variable}')
        else:
            kde_plot.set_title(f'Kernel Density Estimation of {variable}\n in {region_name}')
    
    elif variable == 'SHEAR':
        #sns.set_theme()
        start_time = time.time()
        #hist_4 = hist_4.where((hist_4[f'{variable}'] > np.quantile(hist_4[f'{variable}'].values,0.05)) & (hist_4[f'{variable}'] < np.quantile(hist_4[f'{variable}'].values,0.95)))
        kde_plot = sns.kdeplot(data = hist_4[f'{variable}'][0].values.flatten(), label = 'Convection-Permitting Historical Climate')
        print('Completed plotting of historical 4km')
        print(f'Took {time.time() - start_time} seconds to plot') 
        #fut_4 = fut_4.where((fut_4[f'{variable}'] > np.quantile(fut_4[f'{variable}'].values,0.05)) & (fut_4[f'{variable}'] < np.quantile(fut_4[f'{variable}'].values,0.95)))
        kde_plot = sns.kdeplot(data = fut_4[f'{variable}'][0].values.flatten(), label = 'Convection-Permitting Future Climate')
        print('Completed plotting of future 4km')

        start_27 = time.time()
        #fut_27 = fut_27.where((fut_27[f'{variable}'] > np.quantile(fut_27[f'{variable}'].values,0.05)) & (fut_27[f'{variable}'] < np.quantile(fut_27[f'{variable}'].values,0.95)))
        kde_plot = sns.kdeplot(data = fut_27[f'{variable}'][0].values.flatten(), label = '27 km Future Climate')
        print('Completed plotting of future 27km')
        print(f'Took {time.time() - start_27} seconds to plot')
        #hist_27 = hist_27.where((hist_27[f'{variable}'] > np.quantile(hist_27[f'{variable}'].values,0.05)) & (hist_27[f'{variable}'] < np.quantile(hist_27[f'{variable}'].values,0.95)))
        kde_plot = sns.kdeplot(data = hist_27[f'{variable}'][0].values.flatten(), label = '27 km Historical Climate')
        print('Completed plotting of historical 27km')
        if region == 'ALL':
            kde_plot.set_title(f'Kernel Density Estimation of {variable}')
        else:
            kde_plot.set_title(f'Kernel Density Estimation of {variable}\n in {region_name}')

    else:
        #sns.set_theme()
        start_time = time.time()
        kde_plot = sns.kdeplot(data = np.random.choice(hist_4[f'{variable}'].values.flatten(),int(len(hist_4[f'{variable}'].values.flatten())/10)), color='blue', log_scale = 10, label = 'Convection-Permitting Historical Climate')
        print('Completed plotting of historical 4km')
        print(f'Took {time.time() - start_time} seconds to plot') 
        kde_plot = sns.kdeplot(data = np.random.choice(fut_4[f'{variable}'].values.flatten(),int(len(fut_4[f'{variable}'].values.flatten())/10)), color='red', log_scale = 10, label = 'Convection-Permitting Future Climate')
        print('Completed plotting of future 4km')
        
        start_27 = time.time()
        kde_plot = sns.kdeplot(data = np.random.choice(hist_27[f'{variable}'].values.flatten(),int(len(hist_27[f'{variable}'].values.flatten())/10)), color='blue', linestyle='--', log_scale = 10, label = '27 km Historical Climate')
        print('Completed plotting of historical 27km')
        kde_plot = sns.kdeplot(data = np.random.choice(fut_27[f'{variable}'].values.flatten(),int(len(fut_27[f'{variable}'].values.flatten())/10)), color='red', linestyle='--', log_scale = 10, label = '27 km Future Climate')
        print('Completed plotting of future 27km')
        print(f'Took {time.time() - start_27} seconds to plot')
        if region == 'ALL':
            kde_plot.set_title(f'Kernel Density Estimation of {variable} (Logarithmic Base 10 Scale)')
        else:
            kde_plot.set_title(f'Kernel Density Estimation of {variable}\n in {region_name} (Logarithmic Base 10 Scale)')

    kde_plot.set_xlabel(f'{variable} ({units})')
    kde_plot.set_ylim(top=1.0)
    kde_plot.set_xlim(left=0.5)
    plt.legend(fontsize='x-small')

    #plt.show()
    out_dir = ('/pscratch/sd/n/nee2000/WRF-Prod/images_final/')
    if os.path.exists(out_dir):
        os.chdir(out_dir)
    else:
        os.mkdir(out_dir)
        os.chdir(out_dir)
    print(os.getcwd())

    kde_out = kde_plot.get_figure()
    if region == 'ALL':
        kde_out.savefig(f'fig8_{variable}_KDE_white.jpg',bbox_inches='tight',dpi=300)
    else:
        kde_out.savefig(f'fig8_{variable}_{region}_KDE_white.jpg',bbox_inches='tight',dpi=300)

def plot_ss_kde(hist_4,fut_4,variable,region):
    var_name = variable
    if variable == 'REF':
        units = 'dBZ'
    elif variable == 'W':
        units = 'm/s'
    elif variable == 'UH':
        units = 'm^2/s^2'

    if region == 'NGP':
        region_name = 'Northern Great Plains'
    elif region == 'SGP':
        region_name = 'Southern Great Plains'
    elif region == 'MID':
        region_name = 'Mid-Atlantic'
    elif region == 'SE':
        region_name = 'Southeast US'
    else:
        pass
    print(f'Plotting {variable} for {region}')

    # Set the percentile to calculate
    percentile = 0.95

    # Set up the figure
    fig,ax = plt.subplots()

    # Mask data less than 0; trim to given percentile
    hist_4 = hist_4.where(hist_4[f'{variable}'] > 0)
    fut_4 = fut_4.where(fut_4[f'{variable}'] > 0)

    hist_4,fut_4,hist_27,fut_27 = trim(hist_4,fut_4,None,None,variable,percentile,1)

    # Plot KDEs based on trimmed data
    kde_plot = sns.kdeplot(data = np.random.choice(hist_4[f'{variable}'].values.flatten(),int(len(hist_4[f'{variable}'].values.flatten())/10)), log_scale = 10, label = 'Convection-Permitting Historical Climate',ax=ax)
    #plt.show()
    kde_orig = plt.gcf()

    print('Completed plotting of historical 4km')
    kde_plot = sns.kdeplot(data = np.random.choice(fut_4[f'{variable}'].values.flatten(),int(len(fut_4[f'{variable}'].values.flatten())/10)), log_scale = 10, label = 'Convection-Permitting Future Climate',ax=ax)
    print('Completed plotting of future 4km')
    #print('--------------------------------')
    #print('Running some sanity checks')
    #print(kde_plot)
    #print(type(kde_plot))
    #print(type(hist_4),type(fut_4))
    #print('Completed sanity checks')
    #print('--------------------------------')

    plt.legend(f'{variable} ({units})')

    print('Region is:',region)
    
    #plt.show()

    if 'percentile' in locals(): # <- Change figure arguments as needed
        if region == 'ALL':
            kde_plot.set_title(f'Kernel Density Estimation of {str(percentile*100)[:2]}th Percentile of {variable}')
        else:
            kde_plot.set_title(f'Kernel Density Estimation of {str(percentile*100)[:2]}th Percentile of {variable}\n in {region_name}')
    else:
        if region == 'ALL':
            kde_plot.set_title(f'Kernel Density Estimation of {variable} (Log 10 Scale)')
        else:
            kde_plot.set_title(f'Kernel Density Estimation of {variable}\n in {region_name} (Log 10 Scale)')
    
    kde_plot.set_xlabel(f'{variable} ({units})')
    plt.legend(fontsize='small')

    #plt.show()
    out_dir = ('test_images')
    if os.path.exists(out_dir):
        os.chdir(out_dir)
    else:
        os.mkdir(out_dir)
        os.chdir(out_dir)
    print(os.getcwd())

    kde_out = kde_plot.get_figure()
    if region == 'ALL':
        kde_orig.savefig(f'{variable}_log_hist_KDE.jpg',bbox_inches='tight')
        kde_out.savefig(f'{variable}_log_fut_KDE.jpg',bbox_inches='tight')
    else:
        kde_orig.savefig(f'{variable}_log_{region}_hist_KDE.jpg',bbox_inches='tight')
        kde_out.savefig(f'{variable}_log_{region}_KDE.jpg',bbox_inches='tight')

def plot_bars(variable,option,domain_name,func,threshold):
    # Creates barplots of selected input variable
    # Inputs:
    #  data: Datasets to use for plotting
    #  variable: Variable to plot
    #  option: Type of stratification to use for barplot
    # Returns:
    #  None (Outputs image to file)
    # Set base directory
    base_dir = os.getcwd()

    # Set file paths
    domain = f'd0{domain_name}'
    if domain == 'd01':
        hist_path = 'hist/WRF-d01'
        fut_path = 'fut/WRF-d01'
    else:
        hist_path = 'hist/WRF-Monthly'
        fut_path = 'fut/WRF-Monthly'
    ref_path = 'WRF-Ref'

    # Run controls based on type of plotting to perform
    option = int(option)
    try:
        func = int(func)
    except ValueError:
        pass
    
    if variable == 'CAPE':
        units = 'J/kg'
    elif variable == 'CIN':
        units = 'J/kg'
    elif variable == 'SRH':
        units = 'm^2/s^2'
    elif variable == 'SHEAR':
        units = 'm/s'
    elif variable == 'W':
        units = 'm/s'
    else:
        pass

    #print(option)
    #print(option == 2)
    if option == 1: # Epochs only
        for i in range(20):
            try:
                # Read in climatological files
                print('Starting to open historical files')
                os.chdir(hist_path)
                hist_data = xr.open_mfdataset(f'{variable}_*.nc',concat_dim='time',combine='nested',parallel=True)
                print('Current directory is',os.getcwd())
                #hist_data = xr.open_mfdataset(f'{variable}_1991*.nc',concat_dim='time',combine='nested',parallel=True)

                print('Starting to open future files')
                os.chdir(os.path.join(base_dir + '/' + fut_path))
                fut_data = xr.open_mfdataset(f'{variable}_*.nc',concat_dim='time',combine='nested',parallel=True)
                #fut_data = xr.open_mfdataset(f'{variable}_2091*.nc',concat_dim='time',combine='nested',parallel=True)
                print('Opened all files')
                break
            except (RuntimeError,OSError):
                #print('Failed to open files, trying again')
                continue
        
        # Reset W to UVV for plotting
        #if variable == 'W':
        #    variable = 'UVV'

        # Trim SRH datasets
        if variable == 'SRH':
            hist_data,fut_data,hist_null,fut_null = trim(hist_data,fut_data,None,None,variable,0.01,0.99)
        
        # Convert to dataframes; combine data
        hist_ds_init = hist_data.to_dataframe()
        fut_ds_init = fut_data.to_dataframe()

        #if variable == 'CIN':
        #    operator = '>'
        #    hist_ds = hist_ds_init.where(hist_ds_init[f'{variable}'] > int(threshold)).dropna()
        #    fut_ds = fut_ds_init.where(fut_ds_init[f'{variable}'] > int(threshold)).dropna()
        #else:
        operator = '< -'
        hist_ds = hist_ds_init.where(hist_ds_init[f'{variable}'] > int(threshold)).dropna()
        fut_ds = fut_ds_init.where(fut_ds_init[f'{variable}'] > int(threshold)).dropna()
        
        #hist_avg = np.nanmean(hist_ds[f'{variable}'])
        #fut_avg = np.nanmean(fut_ds[f'{variable}'])

        #diff = fut_avg - hist_avg
        #print('Future percentage change is', np.round(diff,4))
        #raise RuntimeError('Stop here')
        
        print('Sent data to dataframe')

        hist_ds.insert(3, 'Epoch', pd.Series('Historical', index=hist_ds.index))
        fut_ds.insert(3, 'Epoch', pd.Series('Future', index=fut_ds.index))
        print('Inserted new columns into dataframes')

        if variable == 'SHEAR':
            ds = pd.concat([hist_ds.sample(frac=0.05),fut_ds.sample(frac=0.05)])
        else:
            ds = pd.concat([hist_ds.sample(frac=0.1),fut_ds.sample(frac=0.1)])
        print('Created full dataframe')
        
        # Reset W to UVV for plotting
        if variable == 'W':
            variable = 'UVV'
        #print('Historical frequency of CAPE > 500 J/kg is:', len(hist_ds)/len(hist_ds_init))
        #print('Future frequency of CAPE > 500 J/kg is:', len(fut_ds)/len(fut_ds_init))
        hist_freq = len(hist_ds)/len(hist_ds_init)
        fut_freq = len(fut_ds)/len(fut_ds_init)
        freqs = [hist_freq,fut_freq]
        epochs = ['Historical','Future']
        ds_freq = pd.DataFrame(data=[freqs,epochs],index=[f'{variable}','Epoch']).T
        print(ds_freq)
        print('Percentage change in UVV is:', ((fut_freq - hist_freq)/hist_freq) * 100)
        # Plot dataset; save to file
        #sns.set_theme()
        if int(threshold) > 0:
            #bar_plot = sns.countplot(data=ds,x='Epoch')
            bar_plot = sns.barplot(data=ds_freq,x='Epoch',y=f'{variable}')
            bar_plot.set(ylabel=f'Frequency of {variable} {operator} {int(threshold)} {units}', title=f'Distribution of {variable} across Climate Epochs')
            #bar_plot.title.set_size(16)
        else:
            bar_plot = sns.barplot(data=ds,x='Epoch',y=f'{variable}',estimator=func)
            bar_plot.set(ylabel=f'{variable} ({units})', title=f'Distribution of {variable} across Climate Epochs')

        img_dir = os.path.join(base_dir + '/images_final')
        if os.path.exists(img_dir):
            pass
        else:
            os.mkdir(img_dir)
        os.chdir(img_dir)

        bar_fig = bar_plot.get_figure()
        bar_fig.savefig(f'fig10_{variable}_barplot_white.jpg',bbox_inches='tight',dpi=300)
        print('Successfully plotted and saved to file')

    if option == 2: # Epochs and months
        #print(type(func))
        # Read in climatological files
        months = ['January','February','March','April']
        hist_freqs = []
        fut_freqs = []
        for num_month in range(4):
            print(f'Opening files for {months[num_month]}')
            for i in range(20):
                try:
                    os.chdir(os.path.join(base_dir + '/' + hist_path))
                    hist_data = xr.open_mfdataset(f'{variable}_*0{num_month+1}.nc',concat_dim='time',combine='nested',parallel=True)
                    #hist_data = xr.open_mfdataset(f'{variable}_1991*.nc',concat_dim='time',combine='nested',parallel=True)

                    os.chdir(os.path.join(base_dir + '/' + fut_path))
                    fut_data = xr.open_mfdataset(f'{variable}_*0{num_month+1}.nc',concat_dim='time',combine='nested',parallel=True)
                    #fut_data = xr.open_mfdataset(f'{variable}_2091*.nc',concat_dim='time',combine='nested',parallel=True)
                    print('Opened all files for month')
                    break
                except (RuntimeError,OSError):
                    print('Failed to open files; trying again')
                    continue
            
            if variable == 'SRH':
                hist_data,fut_data,hist_null,fut_null = trim(hist_data,fut_data,None,None,variable,0.01,0.99)
        
            # Convert to dataframes; combine data
            hist_ds_init = hist_data.to_dataframe()
            fut_ds_init = fut_data.to_dataframe()

            #if variable == 'CIN':
            #    operator = '>'
            #    hist_ds = hist_ds.where(hist_ds[f'{variable}'] > int(threshold)).dropna()
            #    fut_ds = fut_ds.where(fut_ds[f'{variable}'] > int(threshold)).dropna()
            #else:
            operator = '< -'
            hist_ds = hist_ds_init.where(hist_ds_init[f'{variable}'] > int(threshold)).dropna()
            fut_ds = fut_ds_init.where(fut_ds_init[f'{variable}'] > int(threshold)).dropna()
            
            hist_avg = np.nanmean(hist_ds[f'{variable}'])
            fut_avg = np.nanmean(fut_ds[f'{variable}'])

            #diff = fut_avg - hist_avg
            #print('Future percentage change is', np.round(diff,4))

            hist_ds.insert(3,'Month',pd.Series(months[num_month], index=hist_ds.index))
            fut_ds.insert(3,'Month',pd.Series(months[num_month], index=fut_ds.index))

            hist_ds.insert(3, 'Epoch', pd.Series('Historical', index=hist_ds.index))
            fut_ds.insert(3, 'Epoch', pd.Series('Future', index=fut_ds.index))
            
            #hist_ds = hist_ds_init
            #fut_ds = fut_ds_init

            if 'ds' in locals():
                if variable == 'SHEAR':
                    ds = pd.concat([ds,hist_ds.sample(frac=0.05),fut_ds.sample(frac=0.05)])
                else:
                    ds = pd.concat([ds,hist_ds.sample(frac=0.1),fut_ds.sample(frac=0.1)])
            else:
                if variable == 'SHEAR':
                    ds = pd.concat([hist_ds.sample(frac=0.05),fut_ds.sample(frac=0.05)])
                else:
                    ds = pd.concat([hist_ds.sample(frac=0.1),fut_ds.sample(frac=0.1)])
            hist_thresh = len(hist_ds)
            fut_thresh = len(fut_ds)

            hist_len = len(hist_ds_init)
            fut_len = len(fut_ds_init)
            
            hist_freq = hist_thresh/hist_len
            fut_freq = fut_thresh/fut_len
            
            hist_freqs.append(hist_freq)
            fut_freqs.append(fut_freq)
            print(hist_freqs,fut_freqs)
            print(f'Percentage change in UVV for {months[num_month]} is:', ((fut_freq - hist_freq)/hist_freq) * 100)
            #hist_jan = hist_ds[hist_ds['Month'] == 'January']

        #raise RuntimeError('Stop here')
        
        # Process all data as well to produce fifth bar
        os.chdir(os.path.join(base_dir + '/' + hist_path))
        #hist_data = xr.open_mfdataset(f'{variable}_*0{num_month+1}.nc',concat_dim='time',combine='nested',parallel=True)
        hist_data = xr.open_mfdataset(f'{variable}_*.nc',concat_dim='time',combine='nested',parallel=True)

        os.chdir(os.path.join(base_dir + '/' + fut_path))
        #fut_data = xr.open_mfdataset(f'{variable}_*0{num_month+1}.nc',concat_dim='time',combine='nested',parallel=True)
        fut_data = xr.open_mfdataset(f'{variable}_*.nc',concat_dim='time',combine='nested',parallel=True)
        
        hist_ds_init = hist_data.to_dataframe()
        fut_ds_init = fut_data.to_dataframe()
        
        hist_ds = hist_ds_init.where(hist_ds_init[f'{variable}'] > int(threshold)).dropna()
        fut_ds = fut_ds_init.where(fut_ds_init[f'{variable}'] > int(threshold)).dropna()
        
        hist_thresh = len(hist_ds)
        fut_thresh = len(fut_ds)

        hist_len = len(hist_ds_init)
        fut_len = len(fut_ds_init)
        
        hist_freq = hist_thresh/hist_len
        fut_freq = fut_thresh/fut_len
        
        hist_freqs.append(hist_freq)
        fut_freqs.append(fut_freq)
        print(hist_freqs,fut_freqs)

        hist_ds.insert(3,'Month',pd.Series('Historical (All Months)', index=hist_ds.index))
        fut_ds.insert(3,'Month',pd.Series('Future (All Months)', index=fut_ds.index))

        hist_ds.insert(3, 'Epoch', pd.Series('Historical', index=hist_ds.index))
        fut_ds.insert(3, 'Epoch', pd.Series('Future', index=fut_ds.index))
        
        #hist_ds = hist_ds_init
        #fut_ds = fut_ds_init

        if 'ds' in locals():
            if variable == 'SHEAR':
                ds = pd.concat([ds,hist_ds.sample(frac=0.05),fut_ds.sample(frac=0.05)])
            else:
                ds = pd.concat([ds,hist_ds.sample(frac=0.1),fut_ds.sample(frac=0.1)])
        # Reset W to UVV for plotting
        if variable == 'W':
            variable = 'UVV'
        
        freqs = hist_freqs+fut_freqs
        print(freqs)
        epochs = ['Historical']*5+['Future']*5
        months = ['January','February','March','April','All Months']*2
        print(epochs)
        ds_freq = pd.DataFrame(data=[freqs,epochs,months],index=[f'{variable}','Epoch','Month']).T
        print(ds_freq)
        # Plot dataset; save to file
        #sns.set_theme()
        if int(threshold) > 0:
            bar_plot = sns.barplot(data=ds_freq,x='Epoch',y=f'{variable}',hue='Month')
            bar_plot.set(ylabel=f'Frequency of {variable} {operator} {int(threshold)} {units}', title=f'Distribution of {variable}, Stratified across Months')
            #bar_plot.title.set_size(16)
        else:
            bar_plot = sns.barplot(data=ds,x='Epoch',y=f'{variable}',hue='Month',estimator=func)
            bar_plot.set(ylabel=f'{variable} ({units})', title=f'Distribution of {variable}, Stratified across Months') 
        
        img_dir = os.path.join(base_dir + '/images_final')
        if os.path.exists(img_dir):
            pass
        else:
            os.mkdir(img_dir)
        os.chdir(img_dir)

        bar_fig = bar_plot.get_figure()
        bar_fig.savefig(f'fig10_{variable}_months_barplot_white.jpg',bbox_inches='tight',dpi=300)
    
    if option == 3: # Epochs and regions
        # Read in climatological files
        regions = ['NGP','SGP','MID','SE']
        hist_freqs = []
        fut_freqs = []
        for region in regions:
            for i in range(20):
                try:
                    os.chdir(os.path.join(base_dir + '/' + hist_path))
                    hist_data = xr.open_mfdataset(f'{variable}_*.nc',concat_dim='time',combine='nested',parallel=True)
                    #hist_data = xr.open_mfdataset(f'{variable}_1991*.nc',concat_dim='time',combine='nested',parallel=True)

                    os.chdir(os.path.join(base_dir + '/' + fut_path))
                    fut_data = xr.open_mfdataset(f'{variable}_*.nc',concat_dim='time',combine='nested',parallel=True)
                    #fut_data = xr.open_mfdataset(f'{variable}_2091*.nc',concat_dim='time',combine='nested',parallel=True)
                    break
                except (RuntimeError,OSError):
                    continue

            if variable == 'SRH':
                hist_data,fut_data,hist_null,fut_null = trim(hist_data,fut_data,None,None,variable,0.01,0.99)
            
            # Subset data geographically
            hist_data,fut_data,hist_null,fut_null = subset(hist_data,fut_data,None,None,region)
            
            # Convert to dataframes; combine data
            hist_ds_init = hist_data.to_dataframe()
            fut_ds_init = fut_data.to_dataframe()

            #if variable == 'CIN':
            #    operator = '>'
            #    hist_ds = hist_ds.where(hist_ds[f'{variable}'] > int(threshold)).dropna()
            #    fut_ds = fut_ds.where(fut_ds[f'{variable}'] > int(threshold)).dropna()
            #else:
            operator = '>'
            hist_ds = hist_ds_init.where(hist_ds_init[f'{variable}'] > int(threshold)).dropna()
            fut_ds = fut_ds_init.where(fut_ds_init[f'{variable}'] > int(threshold)).dropna()
            
            hist_avg = np.nanmean(hist_ds[f'{variable}'])
            fut_avg = np.nanmean(fut_ds[f'{variable}'])

            #diff = fut_avg - hist_avg
            #print('Future percentage change is', np.round(diff,4))

            hist_ds.insert(3,'Region',pd.Series(region, index=hist_ds.index))
            fut_ds.insert(3,'Region',pd.Series(region, index=fut_ds.index))

            hist_ds.insert(3, 'Epoch', pd.Series('Historical', index=hist_ds.index))
            fut_ds.insert(3, 'Epoch', pd.Series('Future', index=fut_ds.index))

            if 'ds' in locals():
                if variable == 'SHEAR':
                    ds = pd.concat([ds,hist_ds.sample(frac=0.05),fut_ds.sample(frac=0.05)])
                else:
                    ds = pd.concat([ds,hist_ds.sample(frac=0.1),fut_ds.sample(frac=0.1)])
            else:
                if variable == 'SHEAR':
                    ds = pd.concat([hist_ds.sample(frac=0.05),fut_ds.sample(frac=0.05)])
                else:
                    ds = pd.concat([hist_ds.sample(frac=0.1),fut_ds.sample(frac=0.1)])
            
            hist_thresh = len(hist_ds)
            fut_thresh = len(fut_ds)

            hist_len = len(hist_ds_init)
            fut_len = len(fut_ds_init)
            
            hist_freq = hist_thresh/hist_len
            fut_freq = fut_thresh/fut_len
            
            hist_freqs.append(hist_freq)
            fut_freqs.append(fut_freq)
            #print(hist_freqs,fut_freqs)
            #print(f'Percentage change in UVV for {region} is:', ((fut_freq - hist_freq)/hist_freq) * 100)
            #hist_jan = hist_ds[hist_ds['Month'] == 'January']
        #raise RuntimeError('Stop here')
        
        # Process all data as well to produce fifth bar
        os.chdir(os.path.join(base_dir + '/' + hist_path))
        #hist_data = xr.open_mfdataset(f'{variable}_*0{num_month+1}.nc',concat_dim='time',combine='nested',parallel=True)
        hist_data = xr.open_mfdataset(f'{variable}_*.nc',concat_dim='time',combine='nested',parallel=True)

        os.chdir(os.path.join(base_dir + '/' + fut_path))
        #fut_data = xr.open_mfdataset(f'{variable}_*0{num_month+1}.nc',concat_dim='time',combine='nested',parallel=True)
        fut_data = xr.open_mfdataset(f'{variable}_*.nc',concat_dim='time',combine='nested',parallel=True)
        
        hist_ds_init = hist_data.to_dataframe()
        fut_ds_init = fut_data.to_dataframe()
        
        hist_ds = hist_ds_init.where(hist_ds_init[f'{variable}'] > int(threshold)).dropna()
        fut_ds = fut_ds_init.where(fut_ds_init[f'{variable}'] > int(threshold)).dropna()
        
        hist_thresh = len(hist_ds)
        fut_thresh = len(fut_ds)

        hist_len = len(hist_ds_init)
        fut_len = len(fut_ds_init)
        
        hist_freq = hist_thresh/hist_len
        fut_freq = fut_thresh/fut_len
        
        hist_freqs.append(hist_freq)
        fut_freqs.append(fut_freq)
        print(hist_freqs,fut_freqs)

        hist_ds.insert(3,'Month',pd.Series('Historical (All Months)', index=hist_ds.index))
        fut_ds.insert(3,'Month',pd.Series('Future (All Months)', index=fut_ds.index))

        hist_ds.insert(3, 'Epoch', pd.Series('Historical', index=hist_ds.index))
        fut_ds.insert(3, 'Epoch', pd.Series('Future', index=fut_ds.index))
        
        #hist_ds = hist_ds_init
        #fut_ds = fut_ds_init

        if 'ds' in locals():
            if variable == 'SHEAR':
                ds = pd.concat([ds,hist_ds.sample(frac=0.05),fut_ds.sample(frac=0.05)])
            else:
                ds = pd.concat([ds,hist_ds.sample(frac=0.1),fut_ds.sample(frac=0.1)])
        # Reset W to UVV for plotting
        if variable == 'W':
            variable = 'UVV'
         
        freqs = hist_freqs+fut_freqs
        #print(freqs)
        epochs = ['Historical']*5+['Future']*5
        regions.append('All Regions')
        region_list = regions*2
        #print(regions)
        ds_freq = pd.DataFrame(data=[freqs,epochs,region_list],index=[f'{variable}','Epoch','Region']).T
            
        # Plot dataset; save to file
        print(ds_freq)
        #sns.set_theme()
        if int(threshold) > 0:
            bar_plot = sns.barplot(data=ds_freq,x='Epoch',y=f'{variable}',hue='Region')
            bar_plot.set(ylabel=f'Frequency of {variable} {operator} {int(threshold)} {units}', title=f'Distribution of {variable}, Stratified across Regions')
            #bar_plot.title.set_size(16)
        else:
            bar_plot = sns.barplot(data=ds,x='Epoch',y=f'{variable}',hue='Region',estimator=func)
            bar_plot.set(ylabel=f'{variable} ({units})', title=f'Distribution of {variable}, Stratified across Regions')
    
        img_dir = os.path.join(base_dir + '/images_final')
        if os.path.exists(img_dir):
            pass
        else:
            os.mkdir(img_dir)
        os.chdir(img_dir)

        bar_fig = bar_plot.get_figure()
        bar_fig.savefig(f'fig10_{variable}_regions_barplot_white.jpg',bbox_inches='tight',dpi=300)
    
    if option == 4: # Epochs, months and regions
        print('Using option 4')
        # Read in climatological files
        months = ['January','February','March','April']
        regions = ['NGP','SGP','MID','SE']
        for num_month in range(4):
            for region in regions:
                for i in range(20):
                    try:
                        os.chdir(os.path.join(base_dir + '/' + hist_path))
                        hist_data = xr.open_mfdataset(f'{variable}_*0{num_month+1}*.nc',concat_dim='time',combine='nested',parallel=True)

                        os.chdir(os.path.join(base_dir + '/' + fut_path))
                        fut_data = xr.open_mfdataset(f'{variable}_*0{num_month+1}*.nc',concat_dim='time',combine='nested',parallel=True)
                        print(f'Opened all files for 0{num_month+1} to work with {region}')
                        break
                    except (RuntimeError,OSError):
                        print(os.getcwd())
                        print('Failed to open files; trying again')
                        continue

                if variable == 'SRH':
                    hist_data,fut_data,hist_null,fut_null = trim(hist_data,fut_data,None,None,variable,0.01,0.99)
                
                # Subset data geographically
                hist_data,fut_data,hist_null,fut_null = subset(hist_data,fut_data,None,None,region)
                
                # Convert to dataframes; combine data
                hist_ds = hist_data.to_dataframe()
                fut_ds = fut_data.to_dataframe()
                
                if variable == 'CIN':
                    operator = '>'
                    hist_ds = hist_ds.where(hist_ds[f'{variable}'] < int(threshold)).dropna()
                    fut_ds = fut_ds.where(fut_ds[f'{variable}'] < int(threshold)).dropna()
                else:
                    operator = '>'
                    hist_ds = hist_ds.where(hist_ds[f'{variable}'] > int(threshold)).dropna()
                    fut_ds = fut_ds.where(fut_ds[f'{variable}'] > int(threshold)).dropna()

                hist_ds.insert(3,'Region',pd.Series(region, index=hist_ds.index))
                fut_ds.insert(3,'Region',pd.Series(region, index=fut_ds.index))

                hist_ds.insert(3,'Month',pd.Series(months[num_month], index=hist_ds.index))
                fut_ds.insert(3,'Month',pd.Series(months[num_month], index=fut_ds.index))

                hist_ds.insert(3, 'Epoch', pd.Series('Historical', index=hist_ds.index))
                fut_ds.insert(3, 'Epoch', pd.Series('Future', index=fut_ds.index))
                
                if 'ds' in locals():
                    if variable == 'SHEAR':
                        ds = pd.concat([ds,hist_ds.sample(frac=0.05),fut_ds.sample(frac=0.05)])
                    else:
                        ds = pd.concat([ds,hist_ds.sample(frac=0.1),fut_ds.sample(frac=0.1)])
                else:
                    if variable == 'SHEAR':
                        ds = pd.concat([hist_ds.sample(frac=0.05),fut_ds.sample(frac=0.05)])
                    else:
                        ds = pd.concat([hist_ds.sample(frac=0.1),fut_ds.sample(frac=0.1)])
                print(f'Finished computing for {region} in month 0{num_month+1}!')
            print(f'Finished computing for month 0{num_month+1}!')
        #ds.to_csv(f'{variable}_months_regions.csv')
        #raise RuntimeError('Stop here (if we can even get here)')

        # Plot dataset; save to file
        #os.chdir('WRF-Stats/')
        #ds = pd.read_csv(f'{variable}_months_regions.csv')

        #raise RuntimeError('Stop here')
        sns.set_theme()
        if int(threshold) > 0:
            bar_plot = sns.countplot(data=ds,x='Epoch',hue='Month')
            bar_plot.set(ylabel=f'Frequency of {variable} {operator} {int(threshold)} {units}', title=f'Distribution of {variable}, Stratified across Months and Regions')
            #bar_plot.title.set_size(16)
        else: 
            bar_plot = sns.barplot(data=ds,x='Epoch',y=f'{variable}',hue='Month',estimator=func)
            bar_plot.set(ylabel=f'{variable} ({units})', title=f'Distribution of {variable}, Stratified across Months and Regions')

        img_dir = os.path.join(base_dir + '/images/barplots')
        if os.path.exists(img_dir):
            pass
        else:
            os.mkdir(img_dir)
        os.chdir(img_dir)

        bar_fig = bar_plot.get_figure()
        #bar_fig.savefig(f'{variable}_{func}_months_regions_barplot.jpg',bbox_inches='tight')

    return None

def plot_time_series(ts_file,month_name,variable):
    # Takes spatially averaged data and plots a time series
    # Inputs:
    #  ts_file: The file containing a time series of input data
    #  month_name: The month to plot time series for
    #  variable: The variable to plot

    # Set name of data array to read
    if variable == 'T2':
        var = 'T2'
        name = '2-Meter Air Temperature'
        units = 'K'
    elif variable == 'PRECIP':
        var = 'PRECIP'
        name = 'Precipitation'
        units = 'mm'
    elif variable == 'CAPE':
        var = 'CAPE'
        name = 'Convective Available Potential Energy'
        units = 'J/kg'
    elif variable == 'CIN':
        var = 'CIN'
        name = 'Convective Inhibition'
        units = 'J/kg'
    elif variable == 'SRH':
        var = 'SRH'
        name = 'Storm Relative Helicity'
        units = 'm^2/s^2'
    elif variable == 'SHEAR':
        var = 'SHEAR'
        name = '0-6 km Wind Shear'
        units = 'm/s'
    
    # Plot time series
    fig, ax = plt.subplots()
    ax.plot(ts_file.Time, ts_file[f'{variable}'])

    trend = np.polyfit(ts_file.Time, ts_file[f'{variable}'],3)
    fit = np.poly1d(trend)
    plt.plot(ts_file.Time,fit(ts_file.Time))

    ax.set_xlabel('Time')
    ax.set_ylabel(f'{name} {units}')
    ax.set_title('Climatological Time Series of {name}')
    plt.show()

    # Change to output directory; save figure
    out_dir = ('images/')
    if os.path.exists(out_dir):
        os.chdir(out_dir)
    else:
        os.mkdir(out_dir)
        os.chdir(out_dir)
    
    ts_out = fig.get_figure()
    ts_out.savefig(f'{month_name}-{variable}_ts.jpg',bbox_inches='tight') #<- Generalize this

def percentiles(wrf_output,epoch,num_month,month_name,variable,percentile):
    # Reads in convection-permitting data and calculates quantiles
    # Inputs:
    #  month_name: The month to plot time series for
    #  variable: The variable to plot
   
    # Set variable characteristics
    if variable == 'REF':
        var = 'REFD_MAX'
        name = '6-hour maximum reflectivity'
        units = 'dBZ'
        colorbar = 'gist_ncar'
        c_min = 0
        c_max = 90
        step = 10
    if variable == 'W':
        var = 'W_UP_MAX'
        name = '6-hour maximum updraft speed'
        units = 'm/s'
        colorbar = 'Greens'
        c_min = 0
        c_max = 35
        step = 5
    if variable == 'UH':
        var = 'UP_HELI_MAX'
        name = '6-hour maximum updraft helicity'
        units = 'm^2/s^2'
        colorbar = 'Oranges'
        c_min = 0
        c_max = 55
        step = 5
    
    if epoch == 'hist':
        epoch_name = 'Historical'
    elif epoch == 'fut':
        epoch_name = 'Future'
    
    # Set file path
    path = os.getcwd()
    print('File path is:', path)

    # Read in data; select variable to use
    data = xr.open_mfdataset(f'{epoch}/WRF-Monthly/{var}*-{num_month}*',concat_dim='Time',combine='nested',parallel=True) #<-Change for new datasets
    data_var = data[f'{variable}']

    # Calculate quantiles of data
    quant = float(f'0.{percentile}')
    data_percentile = data_var.chunk(dict(Time=-1)).quantile(quant, dim='Time')
    print(data_percentile)

    # Plot 95th quantile of variable

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    lons = wrf_output['XLONG'].squeeze('Time')
    lats = wrf_output['XLAT'].squeeze('Time')
    data = data_percentile
    levels = list(np.arange(c_min,c_max,step))

    avg_map = ax.contourf(lons, lats, data, levels,
                transform=ccrs.PlateCarree(),
                norm = colors.Normalize(vmin=c_min, vmax=c_max),
                cmap=colorbar)
    ax.set_title(f'{percentile}th Percentile of {name} in {epoch_name} Climate for {month_name} over CONUS', fontsize = 20)
    ax.add_feature(cartopy.feature.STATES)
    ax.coastlines()
    ax.set_extent([-110,-70,25,50])

    gl = ax.gridlines(visible=False,draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    fig.colorbar(avg_map,label=f'{var_name} {units})')

    fig.tight_layout()

    plt.show()

    # Change to output directory; save figure

    out_dir = ('images/')
    if os.path.exists(out_dir):
        os.chdir(out_dir)
    else:
        os.mkdir(out_dir)
        os.chdir(out_dir)
    print(os.getcwd())

    map_out = fig.get_figure()
    map_out.savefig(f'{var_name}-{percentile}_{month_name}_{epoch}.jpg',bbox_inches='tight')

def ss_dist(hist_thresh,fut_thresh):
    # Inputs:
    # Returns:
   
    # Set image directory for output
    os.chdir('../..')
    base_dir = os.getcwd()
    img_dir = os.path.join(base_dir + '/images')
    print(img_dir)

    # Create dataframes from datasets
    #hist_df = hist_data.to_dataframe()
    #fut_df = fut_data.to_dataframe()

    # Plot threshold datasets
    #sns.set_theme()
    #fig,ax = plt.subplots(1,2)
    #plt.subplots_adjust(top=0.8, hspace=0.2, wspace=0.5)
    #plt.suptitle('Updraft Vertical Velocity Density over \nJanuary-April for Each Climate Epoch', y=0.98)
    #for axis in ax[0],ax[1]:
    #    axis.set_ylim(15,35)
    #    axis.set_ylabel('Updraft Vertical Velocity (m/s)')

    #hist_kde = sns.kdeplot(data=hist_thresh, x='Time', y='W', ax=ax[0])
    #hist_kde.set_xticks([7670,7700,7730,7760,7790], labels=['01-01','02-01','03-01','04-01','04-30'], rotation=50)
    #hist_kde.set_title('Historical Updraft Vertical Velocities', pad=20)
    
    #fut_kde = sns.kdeplot(data=fut_thresh, x='Time', y='W', ax=ax[1])
    #fut_kde.set_xticks([44195,44225,44255,44285,44315], labels=['01-01','02-01','03-01','04-01','04-30'], rotation=50)
    #fut_kde.set_title('Future Updraft Vertical Velocities', pad=20)
    
    # Plot threshold datasets
    sns.set_theme()
    fig2,ax2 = plt.subplots(1,2,figsize=(12,12))
    plt.subplots_adjust(top=0.8, hspace=0.2, wspace=0.5)
    plt.suptitle('Updraft Vertical Velocity Density over \nJanuary-April for Each Climate Epoch', y=0.98)
    for axis in ax2[0],ax2[1]:
        axis.set_ylim(15,45)
        axis.set_ylabel('Updraft Vertical Velocity (m/s)')

    hist_reg = sns.regplot(data=hist_thresh, x='time', y='W', fit_reg=True, ax=ax2[0])
    #hist_reg.set_xticks([7670,7700,7730,7760,7790], labels=['01-01','02-01','03-01','04-01','04-30'], rotation=50)
    hist_reg.set_title('Historical Updraft Vertical Velocities', pad=20)
    r1,p1 = sp.stats.pearsonr(x=hist_thresh['time'], y=hist_thresh['W'])
    plt.text(.05,.8, 'r={:.4f}'.format(r1), transform=ax2[0].transAxes)

    fut_reg = sns.regplot(data=fut_thresh, x='time', y='W', fit_reg=True, ax=ax2[1])
    #fut_reg.set_xticks([44195,44225,44255,44285,44315], labels=['01-01','02-01','03-01','04-01','04-30'], rotation=50)
    fut_reg.set_title('Future Updraft Vertical Velocities', pad=20)
    r2,p2 = sp.stats.pearsonr(x=fut_thresh['time'], y=fut_thresh['W'])
    plt.text(.05,.8, 'r={:.4f}'.format(r2), transform=ax2[1].transAxes)

    #plt.show()

    # Save to output directory
    os.chdir(img_dir)
    #kde_fig = fig.get_figure()
    reg_fig = fig2.get_figure()
    #kde_fig.savefig('w_kde_full.jpg',bbox_inches='tight')
    reg_fig.savefig('w_reg.jpg',bbox_inches='tight')

def plot_occ_bars(variable,option,domain_name,func,threshold):
    # Creates barplots of selected input variable
    # Inputs:
    #  data: Datasets to use for plotting
    #  variable: Variable to plot
    #  option: Type of stratification to use for barplot
    # Returns:
    #  None (Outputs image to file)
    # Set base directory
    base_dir = os.getcwd()

    # Set file paths
    domain = f'd0{domain_name}'
    if domain == 'd01':
        hist_path = 'hist/WRF-d01'
        fut_path = 'fut/WRF-d01'
    else:
        hist_path = 'hist/WRF-Monthly'
        fut_path = 'fut/WRF-Monthly'
    ref_path = 'WRF-Ref'

    # Run controls based on type of plotting to perform
    option = int(option)
    try:
        func = int(func)
    except ValueError:
        pass
    
    if option == 2: # Epochs and months
        #print(type(func))
        # Read in climatological files
        months = ['January','February','March','April']
        hist_freqs = []
        fut_freqs = []
        for num_month in range(4):
            print(f'Opening files for {months[num_month]}')
            for i in range(20):
                try:
                    os.chdir(os.path.join(base_dir + '/' + hist_path))
                    hist_data = xr.open_mfdataset(f'{variable}_*0{num_month+1}.nc',concat_dim='time',combine='nested',parallel=True)
                    #hist_data = xr.open_mfdataset(f'{variable}_1991*.nc',concat_dim='time',combine='nested',parallel=True)

                    os.chdir(os.path.join(base_dir + '/' + fut_path))
                    fut_data = xr.open_mfdataset(f'{variable}_*0{num_month+1}.nc',concat_dim='time',combine='nested',parallel=True)
                    #fut_data = xr.open_mfdataset(f'{variable}_2091*.nc',concat_dim='time',combine='nested',parallel=True)
                    print('Opened all files for month')
                    break
                except (RuntimeError,OSError):
                    print('Failed to open files; trying again')
                    continue
           
            # Rename variable (because I screwed up)
            if variable == 'PROBOCC':
                variable = 'PROB_OCC'
            # Convert to dataframes; combine data
            hist_ds_init = hist_data.to_dataframe()
            fut_ds_init = fut_data.to_dataframe()

            operator = '>'
            hist_ds = hist_ds_init.where(hist_ds_init[f'{variable}'] > int(threshold)).dropna()
            fut_ds = fut_ds_init.where(fut_ds_init[f'{variable}'] > int(threshold)).dropna()
            
            hist_avg = np.nanmean(hist_ds[f'{variable}'])
            fut_avg = np.nanmean(fut_ds[f'{variable}'])

            hist_ds.insert(3,'Month',pd.Series(months[num_month], index=hist_ds.index))
            fut_ds.insert(3,'Month',pd.Series(months[num_month], index=fut_ds.index))

            hist_ds.insert(3, 'Epoch', pd.Series('Historical', index=hist_ds.index))
            fut_ds.insert(3, 'Epoch', pd.Series('Future', index=fut_ds.index))
            
            if 'ds' in locals():
                if variable == 'SHEAR':
                    ds = pd.concat([ds,hist_ds.sample(frac=0.05),fut_ds.sample(frac=0.05)])
                else:
                    ds = pd.concat([ds,hist_ds.sample(frac=0.1),fut_ds.sample(frac=0.1)])
            else:
                if variable == 'SHEAR':
                    ds = pd.concat([hist_ds.sample(frac=0.05),fut_ds.sample(frac=0.05)])
                else:
                    ds = pd.concat([hist_ds.sample(frac=0.1),fut_ds.sample(frac=0.1)])
            hist_thresh = len(hist_ds)
            fut_thresh = len(fut_ds)

            hist_len = len(hist_ds_init)
            fut_len = len(fut_ds_init)
            
            hist_freq = hist_thresh/hist_len
            fut_freq = fut_thresh/fut_len
            
            hist_freqs.append(hist_freq)
            fut_freqs.append(fut_freq)
            print(hist_freqs,fut_freqs)
            print(f'Percentage change in UVV for {months[num_month]} is:', ((fut_freq - hist_freq)/hist_freq) * 100)
            #hist_jan = hist_ds[hist_ds['Month'] == 'January']

        #raise RuntimeError('Stop here')
        
        # Process all data as well to produce fifth bar
        os.chdir(os.path.join(base_dir + '/' + hist_path))
        #hist_data = xr.open_mfdataset(f'{variable}_*0{num_month+1}.nc',concat_dim='time',combine='nested',parallel=True)
        hist_data = xr.open_mfdataset(f'{variable}_*.nc',concat_dim='time',combine='nested',parallel=True)

        os.chdir(os.path.join(base_dir + '/' + fut_path))
        #fut_data = xr.open_mfdataset(f'{variable}_*0{num_month+1}.nc',concat_dim='time',combine='nested',parallel=True)
        fut_data = xr.open_mfdataset(f'{variable}_*.nc',concat_dim='time',combine='nested',parallel=True)
        
        hist_ds_init = hist_data.to_dataframe()
        fut_ds_init = fut_data.to_dataframe()
        
        hist_ds = hist_ds_init.where(hist_ds_init[f'{variable}'] > int(threshold)).dropna()
        fut_ds = fut_ds_init.where(fut_ds_init[f'{variable}'] > int(threshold)).dropna()
        
        hist_thresh = len(hist_ds)
        fut_thresh = len(fut_ds)

        hist_len = len(hist_ds_init)
        fut_len = len(fut_ds_init)
        
        hist_freq = hist_thresh/hist_len
        fut_freq = fut_thresh/fut_len
        
        hist_freqs.append(hist_freq)
        fut_freqs.append(fut_freq)
        print(hist_freqs,fut_freqs)

        hist_ds.insert(3,'Month',pd.Series('Historical (All Months)', index=hist_ds.index))
        fut_ds.insert(3,'Month',pd.Series('Future (All Months)', index=fut_ds.index))

        hist_ds.insert(3, 'Epoch', pd.Series('Historical', index=hist_ds.index))
        fut_ds.insert(3, 'Epoch', pd.Series('Future', index=fut_ds.index))
        
        #hist_ds = hist_ds_init
        #fut_ds = fut_ds_init

        if 'ds' in locals():
            if variable == 'SHEAR':
                ds = pd.concat([ds,hist_ds.sample(frac=0.05),fut_ds.sample(frac=0.05)])
            else:
                ds = pd.concat([ds,hist_ds.sample(frac=0.1),fut_ds.sample(frac=0.1)])
        # Reset W to UVV for plotting
        if variable == 'W':
            variable = 'UVV'
        
        freqs = hist_freqs+fut_freqs
        print(freqs)
        epochs = ['Historical']*5+['Future']*5
        months = ['January','February','March','April','All Months']*2
        print(epochs)
        ds_freq = pd.DataFrame(data=[freqs,epochs,months],index=[f'{variable}','Epoch','Month']).T
        print(ds_freq)
        # Plot dataset; save to file
        #sns.set_theme()
        if int(threshold) > 0:
            bar_plot = sns.barplot(data=ds_freq,x='Epoch',y=f'{variable}',hue='Month')
            bar_plot.set(ylabel=f'Frequency of {variable} {operator} {int(threshold)} {units}', title=f'Distribution of {variable}, Stratified across Months')
            #bar_plot.title.set_size(16)
        else:
            bar_plot = sns.barplot(data=ds,x='Epoch',y=f'{variable}',hue='Month',estimator=func)
            bar_plot.set(ylabel=f'{variable} ({units})', title=f'Distribution of {variable}, Stratified across Months') 
        
        img_dir = os.path.join(base_dir + '/images_final')
        if os.path.exists(img_dir):
            pass
        else:
            os.mkdir(img_dir)
        os.chdir(img_dir)

        bar_fig = bar_plot.get_figure()
        bar_fig.savefig(f'fig10_{variable}_months_barplot_white.jpg',bbox_inches='tight',dpi=300)
    
    if option == 3: # Epochs and regions
        # Read in climatological files
        regions = ['NGP','SGP','MID','SE']
        hist_freqs = []
        fut_freqs = []
        for region in regions:
            for i in range(20):
                try:
                    os.chdir(os.path.join(base_dir + '/' + hist_path))
                    hist_data = xr.open_mfdataset(f'{variable}_*.nc',concat_dim='time',combine='nested',parallel=True)
                    #hist_data = xr.open_mfdataset(f'{variable}_1991*.nc',concat_dim='time',combine='nested',parallel=True)

                    os.chdir(os.path.join(base_dir + '/' + fut_path))
                    fut_data = xr.open_mfdataset(f'{variable}_*.nc',concat_dim='time',combine='nested',parallel=True)
                    #fut_data = xr.open_mfdataset(f'{variable}_2091*.nc',concat_dim='time',combine='nested',parallel=True)
                    break
                except (RuntimeError,OSError):
                    continue

            if variable == 'SRH':
                hist_data,fut_data,hist_null,fut_null = trim(hist_data,fut_data,None,None,variable,0.01,0.99)
            
            # Subset data geographically
            hist_data,fut_data,hist_null,fut_null = subset(hist_data,fut_data,None,None,region)
            
            # Convert to dataframes; combine data
            hist_ds_init = hist_data.to_dataframe()
            fut_ds_init = fut_data.to_dataframe()

            #if variable == 'CIN':
            #    operator = '>'
            #    hist_ds = hist_ds.where(hist_ds[f'{variable}'] > int(threshold)).dropna()
            #    fut_ds = fut_ds.where(fut_ds[f'{variable}'] > int(threshold)).dropna()
            #else:
            operator = '>'
            hist_ds = hist_ds_init.where(hist_ds_init[f'{variable}'] > int(threshold)).dropna()
            fut_ds = fut_ds_init.where(fut_ds_init[f'{variable}'] > int(threshold)).dropna()
            
            hist_avg = np.nanmean(hist_ds[f'{variable}'])
            fut_avg = np.nanmean(fut_ds[f'{variable}'])

            #diff = fut_avg - hist_avg
            #print('Future percentage change is', np.round(diff,4))

            hist_ds.insert(3,'Region',pd.Series(region, index=hist_ds.index))
            fut_ds.insert(3,'Region',pd.Series(region, index=fut_ds.index))

            hist_ds.insert(3, 'Epoch', pd.Series('Historical', index=hist_ds.index))
            fut_ds.insert(3, 'Epoch', pd.Series('Future', index=fut_ds.index))

            if 'ds' in locals():
                if variable == 'SHEAR':
                    ds = pd.concat([ds,hist_ds.sample(frac=0.05),fut_ds.sample(frac=0.05)])
                else:
                    ds = pd.concat([ds,hist_ds.sample(frac=0.1),fut_ds.sample(frac=0.1)])
            else:
                if variable == 'SHEAR':
                    ds = pd.concat([hist_ds.sample(frac=0.05),fut_ds.sample(frac=0.05)])
                else:
                    ds = pd.concat([hist_ds.sample(frac=0.1),fut_ds.sample(frac=0.1)])
            
            hist_thresh = len(hist_ds)
            fut_thresh = len(fut_ds)

            hist_len = len(hist_ds_init)
            fut_len = len(fut_ds_init)
            
            hist_freq = hist_thresh/hist_len
            fut_freq = fut_thresh/fut_len
            
            hist_freqs.append(hist_freq)
            fut_freqs.append(fut_freq)
            #print(hist_freqs,fut_freqs)
            #print(f'Percentage change in UVV for {region} is:', ((fut_freq - hist_freq)/hist_freq) * 100)
            #hist_jan = hist_ds[hist_ds['Month'] == 'January']
        #raise RuntimeError('Stop here')
        
        # Process all data as well to produce fifth bar
        os.chdir(os.path.join(base_dir + '/' + hist_path))
        #hist_data = xr.open_mfdataset(f'{variable}_*0{num_month+1}.nc',concat_dim='time',combine='nested',parallel=True)
        hist_data = xr.open_mfdataset(f'{variable}_*.nc',concat_dim='time',combine='nested',parallel=True)

        os.chdir(os.path.join(base_dir + '/' + fut_path))
        #fut_data = xr.open_mfdataset(f'{variable}_*0{num_month+1}.nc',concat_dim='time',combine='nested',parallel=True)
        fut_data = xr.open_mfdataset(f'{variable}_*.nc',concat_dim='time',combine='nested',parallel=True)
        
        hist_ds_init = hist_data.to_dataframe()
        fut_ds_init = fut_data.to_dataframe()
        
        hist_ds = hist_ds_init.where(hist_ds_init[f'{variable}'] > int(threshold)).dropna()
        fut_ds = fut_ds_init.where(fut_ds_init[f'{variable}'] > int(threshold)).dropna()
        
        hist_thresh = len(hist_ds)
        fut_thresh = len(fut_ds)

        hist_len = len(hist_ds_init)
        fut_len = len(fut_ds_init)
        
        hist_freq = hist_thresh/hist_len
        fut_freq = fut_thresh/fut_len
        
        hist_freqs.append(hist_freq)
        fut_freqs.append(fut_freq)
        print(hist_freqs,fut_freqs)

        hist_ds.insert(3,'Month',pd.Series('Historical (All Months)', index=hist_ds.index))
        fut_ds.insert(3,'Month',pd.Series('Future (All Months)', index=fut_ds.index))

        hist_ds.insert(3, 'Epoch', pd.Series('Historical', index=hist_ds.index))
        fut_ds.insert(3, 'Epoch', pd.Series('Future', index=fut_ds.index))
        
        #hist_ds = hist_ds_init
        #fut_ds = fut_ds_init

        if 'ds' in locals():
            if variable == 'SHEAR':
                ds = pd.concat([ds,hist_ds.sample(frac=0.05),fut_ds.sample(frac=0.05)])
            else:
                ds = pd.concat([ds,hist_ds.sample(frac=0.1),fut_ds.sample(frac=0.1)])
        # Reset W to UVV for plotting
        if variable == 'W':
            variable = 'UVV'
         
        freqs = hist_freqs+fut_freqs
        #print(freqs)
        epochs = ['Historical']*5+['Future']*5
        regions.append('All Regions')
        region_list = regions*2
        #print(regions)
        ds_freq = pd.DataFrame(data=[freqs,epochs,region_list],index=[f'{variable}','Epoch','Region']).T
            
        # Plot dataset; save to file
        print(ds_freq)
        #sns.set_theme()
        if int(threshold) > 0:
            bar_plot = sns.barplot(data=ds_freq,x='Epoch',y=f'{variable}',hue='Region')
            bar_plot.set(ylabel=f'Frequency of {variable} {operator} {int(threshold)} {units}', title=f'Distribution of {variable}, Stratified across Regions')
            #bar_plot.title.set_size(16)
        else:
            bar_plot = sns.barplot(data=ds,x='Epoch',y=f'{variable}',hue='Region',estimator=func)
            bar_plot.set(ylabel=f'{variable} ({units})', title=f'Distribution of {variable}, Stratified across Regions')
    
        img_dir = os.path.join(base_dir + '/images_final')
        if os.path.exists(img_dir):
            pass
        else:
            os.mkdir(img_dir)
        os.chdir(img_dir)

        bar_fig = bar_plot.get_figure()
        bar_fig.savefig(f'fig10_{variable}_regions_barplot_white.jpg',bbox_inches='tight',dpi=300)
    

# Various lines of code tested for plotting KDEs up to now
    #hist_4 = hist_4.where((hist_4[f'{variable}'] > np.quantile(hist_4[f'{variable}'].values,percentile)))
    #hist_4 = hist_4.where(hist_4[f'{variable}'] > 0)
    #fut_4 = fut_4.where(fut_4[f'{variable}'] > 0)
    #kde_plot = sns.displot(data = hist_4[f'{variable}'].values.flatten(), label = 'Convection-Permitting Historical Climate', log_scale = 10, kde = True)
    #sns.kdeplot(data = np.random.choice(fut_4[f'{variable}'].values.flatten(),int(len(fut_4[f'{variable}'].values.flatten())/10)), log_scale = 10, label = 'Convection-Permitting Future Climate',ax=ax)
    #print('Completed plotting of historical 4km')
    #print(f'Took {time.time() - start_time} seconds to plot') 
    #fut_4 = fut_4.where((hist_4[f'{variable}'] > np.quantile(fut_4[f'{variable}'].values,percentile)))
    #kde_plot = sns.displot(data = fut_4[f'{variable}'].values.flatten(), label = 'Convection-Permitting Future Climate', log_scale = 10, kde = True)
    #sns.kdeplot(data = np.random.choice(fut_4[f'{variable}'].values.flatten(),int(len(fut_4[f'{variable}'].values.flatten())/10)), log_scale = 10, label = 'Convection-Permitting Future Climate',ax=ax)
    #sns.kdeplot(data = {'Convection-Permitting Historical Climate': np.random.choice(hist_4[f'{variable}'].values.flatten(),int(len(hist_4[f'{variable}'].values.flatten())/10)),'Convection-Permitting Future Climate':np.random.choice(fut_4[f'{variable}'].values.flatten(),int(len(fut_4[f'{variable}'].values.flatten())/10))}, log_scale = 10,ax=ax)
