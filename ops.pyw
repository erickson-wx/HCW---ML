import sys,os
import glob
import xarray as xr
import numpy as np
import metpy as mp
import pandas as pd
import itertools

#from netCDF4 import Dataset
from wrf import getvar

def yearly_bounds(resolution,epoch):
    # Determines the yearly bounds to use depending on the simulation inputs
    # Inputs:
    #  resolution: The horizontal resolution of the simulations
    #  epoch: The time period (historical or future) in which the simulations take place
    # Returns:
    #  start_year,end_year: The starting and ending years to look for
    if (resolution == '27') & (epoch == 'hist'):
        start_year = '1981'
        end_year = '2001'
    elif (resolution == '27') & (epoch == 'fut'):
        start_year = '2081'
        end_year = '2101'
    elif (resolution == '4') & (epoch == 'hist'):
        start_year = '1991'
        end_year = '2001'
    elif (resolution == '4') & (epoch == 'fut'):
        start_year = '2091'
        end_year = '2101'
    return int(start_year),int(end_year)

def months():
    # Determines the numerical index and name of month for analysis based on user input
    # Inputs:
    #  Accepts user input from the command line
    # Returns:
    #  num_month: The numerical index of the month
    #  month_name: The name of the month
    months = {'01' : 'January',
              '02' : 'February',
              '03' : 'March',
              '04' : 'April'}

    inp = input('Please type the numerical index of the month for which you would like to calculate the diagnostic variable:')

    for step,month in enumerate(months.keys()):
        if inp == month:
            num_month = month
            month_name = list(months.values())[step]
            break

    if month_name == 'January':
        length = 31
    elif month_name == 'February':
        length = 28
    elif month_name == 'March':
        length = 31
    elif month_name == 'April':
        length = 30
    return num_month,month_name,length

def find_nearest(array, value):
    #height_array = array.values[:,0,0]
    height_array = array.values
    idx = (np.abs(height_array - value)).argmin()
    return idx

def climo_processor(resolution,epoch,start,end,num_month,variable,domain):
    # Sets up for averaging a particular variable across specified dimensions and calls function for that variable
    # Inputs:
    #  resolution: The horizontal resolution of the simulations
    #  epoch: The time period (horizontal or future) in which the simulations took place
    #  start: start_year, as returned by yearly_bounds
    #  end: end_year, as returned by yearly_bounds
    #  num_month: Numerical index of month, as returned by months
    #  variable: The diagnostic variable to process
    # Returns:
    #  None

    dir = os.getcwd()
    in_path = os.path.join(dir+f'/{epoch}')
    if domain == 'd01':
        out_path = os.path.join(dir+f'/{epoch}/WRF-d01/')
    else: 
        out_path = os.path.join(dir+f'/{epoch}/WRF-Averages/')
    #print('Input path is:',in_path)
    #print('Output path is:',out_path)
    #start = '1991' #<- Change me as needed
    #option = input('Please select the type of averaging to perform - 1 for climatology, 2 for time series: ')
    option = 1
    for year in range(int(start),int(end)+1): 
    #for year in year_list:
        print(f'Processing for {year}')
        path = os.path.join(in_path+f'/WRF-{year}/run/wrf/')
        #print('Current path is:',path)
        os.chdir(path)
        #wrf_file = xr.open_mfdataset(f'wrfout_d01_{year}-{num_month}*',concat_dim='Time',combine='nested',parallel=True)
       
        if variable == 'T2':
            wrf_file = xr.open_mfdataset(f'wrfout_d01_{year}-{num_month}*',concat_dim='Time',combine='nested',parallel=True)
            average_temp(wrf_file,num_month,year,option)
        if variable == 'PRECIP':
            wrf_file = xr.open_mfdataset(f'wrfout_d01_{year}-{num_month}*',concat_dim='Time',combine='nested',parallel=True)
            average_precip(wrf_file,num_month,year,option)
        if variable == 'CAPE':
            average_cape(num_month,year,option,out_path)
        if variable == 'CIN':
            average_cin(num_month,year,option,out_path)
        if variable == 'SRH':
            average_srh(num_month,year,option,out_path)
        if variable == 'SHEAR':
            average_shear(num_month,year,option,domain,out_path)
        print(f'Completed averaging for variable {variable} in {year}')
    #climo(variable,num_month,option)

def average_temp(wrf_file,month,year,option):
    # Calculates average temperature
    # Inputs:
    #  wrf_file: A WRF output file from which to extract temperature
    #  month: The month for which to perform the calculation
    #  year: The year in which to perform the calculation
    #  option: Command line input supplied by the user; 1 for climatological averaging, 2 for time-series averaging
    # Ouputs a netCDF file with average temperature

    print(f'Directory where file will be output is', os.getcwd())

    if int(option) == 1: # Climatological averaging    
        print('Running climatological averaging')
        # Extract and format variable; take average in time
        air_temp = wrf_file['T2'].assign_coords({"lon":wrf_file['XLONG'],"lat":wrf_file['XLAT']})
        annual_temp = air_temp.mean(dim='Time')
        average_temp = annual_temp.expand_dims('Time')
        
        #Output averages to file
        #print(average_temp)
        average_temp.to_netcdf(f'T2_{year}-{num_month}-avg.nc')

    elif int(option) == 2: # Time series
        print('Running time series averaging')
        # Extract and format variable; take averages in space
        air_temp = wrf_file['T2']
        average_temp = air_temp.mean(dim=('south_north','west_east')).expand_dims('Year')
        
        # Output averages to file
        average_temp.to_netcdf(f'T2_{year}-{num_month}-ts.nc')

def average_precip(wrf_file,num_month,year,option):
    # Calculates average precipitation
    # Inputs:
    #  wrf_file: A WRF output file from which to extract temperature
    #  month: The month for which to perform the calculation
    #  year: The year in which to perform the calculation
    #  option: Command line input supplied by the user; 1 for climatological averaging, 2 for time-series averaging
    # Ouputs a netCDF file with average precipitation

    print(f'Directory where file will be output is', os.getcwd())

    if int(option) == 1: # Climatological averaging    
        print('Running climatological averaging')
        # Extract and format variable; take average in time
        total_precip = (wrf_file['RAINC'] + wrf_file['RAINNC']).assign_coords({"lon":wrf_file['XLONG'],"lat":wrf_file['XLAT']}).rename('PRECIP')
        annual_precip = total_precip.mean(dim='Time')
        average_precip = annual_precip.expand_dims('Time')
        
        #Output averages to file
        #print(average_temp)
        average_precip.to_netcdf(f'PRECIP_{year}-{num_month}-avg.nc')

    elif int(option) == 2: # Time series
        print('Running time series averaging')
        # Extract and format variable; take averages in space
        total_precip = (wrf_file['RAINC'] + wrf_file['RAINNC']).rename('PRECIP')
        average_precip = total_precip.mean(dim=('south_north','west_east')).expand_dims('Year')
        
        # Output averages to file
        average_precip.to_netcdf(f'PRECIP_{year}-{num_month}-ts.nc')

def average_cape(num_month,year,option,out_path):
    # Calculates average CAPE/CIN
    # Inputs:
    #  month: The month for which to perform the calculation
    #  year: The year in which to perform the calculation
    #  option: Command line input supplied by the user; 1 for climatological averaging, 2 for time-series averaging
    # Ouputs netCDF files with average CAPE/CIN

    file_list = [f for f in os.listdir() if f.startswith(f'wrfout_d01_{year}-{num_month}')]

    # Set up an empty array to store results of getvar
    output_shape = (len(file_list),4,370,741) #<- Will need to generalize for 4 km runs
    cape_final = np.empty(output_shape, np.float32)

    for index in range(len(file_list)):
        f = Dataset(file_list[index])
        cape = getvar(f, 'cape_2d')

        cape_final[index,:] = cape[:]
        #print(f'Completed computation for {index}!')
        f.close()
    #raise RuntimeError('Stop here')
    
    # Open reference file to extract spatial grid
    wrf_file = xr.open_dataset(f'wrfout_d01_{year}-{num_month}-01_00_00_00')

    # Create dataset to prepare for output
    cape_ds = xr.Dataset(
            data_vars=dict(
                CAPE=(["time","cape_attrs","x","y"],cape_final)
                ),
            coords=dict(
                lon=(["x","y"],wrf_file['XLONG'].squeeze('Time').data),
                lat=(["x","y"],wrf_file['XLAT'].squeeze('Time').data),
                time=np.arange(len(file_list))
                ),
            attrs=dict(description="CAPE and related variables")
            )
    print(cape_ds)

    print('Output directory is', out_path)
    if os.path.exists(out_path):
        os.chdir(out_path)
    else:
        os.mkdir(out_path)
        os.chdir(out_path)
    cape_ds['CAPE'][:,0,:,:].to_netcdf(f'CAPE_d01_{year}-{num_month}.nc')
    cape_ds['CAPE'][:,1,:,:].to_netcdf(f'CIN_d01_{year}-{num_month}.nc')
    
    # Decision for climatology or time series
    if int(option) == 1: # Climatological averaging    
        print('Running climatological averaging')
        # Extract and format variable; take average in time
        average_cape = cape_ds['CAPE'][:,0,:,:].mean(dim='time')
        average_cin = cape_ds['CAPE'][:,1,:,:].mean(dim='time').rename('CIN') #<- Check here to make sure this renames as intended for climatology reading

        #Output averages to file
        average_cape.to_netcdf(f'CAPE_d01_{year}-{num_month}-avg.nc')
        average_cin.to_netcdf(f'CIN_d01_{year}-{num_month}-avg.nc')

    elif int(option) == 2: # Time series
        print('Running time series averaging')
        # Extract and format variable; take averages in space
        average_cape = cape_ds['CAPE'][:,0,:,:].mean(dim=('south_north','west_east')).expand_dims('Year')
        average_cin = cape_ds['CAPE'][:,1,:,:].mean(dim='time').rename('CIN')
        
        # Output averages to file
        average_cape.to_netcdf(f'CAPE_{year}-{num_month}-ts.nc')
        average_cin.to_netcdf(f'CIN_{year}-{num_month}-ts.nc')

def average_srh(num_month,year,option,out_path):
    # Calculates average SRH
    # Inputs:
    #  month: The month for which to perform the calculation
    #  year: The year in which to perform the calculation
    #  option: Command line input supplied by the user; 1 for climatological averaging, 2 for time-series averaging
    # Ouputs a netCDF file with average SRH

    print(f'Directory where file will be output is', os.getcwd())
    
    # Open reference file to extract spatial grid
    wrf_file = xr.open_dataset(f'wrfout_d01_{year}-{num_month}-01_00_00_00')

    file_list = [f for f in os.listdir() if f.startswith(f'wrfout_d01_{year}-{num_month}')]

    # Set up an empty array to store results of getvar
    output_shape = (len(file_list),370,741) #<- Will need to generalize for 4 km runs
    srh_final = np.empty(output_shape, np.float32)

    for index in range(output_shape[0]):
        f = Dataset(file_list[index])
        srh = getvar(f, 'helicity')

        srh_final[index,:] = srh[:]
        f.close()

    # Create dataset to prepare for output
    srh_ds = xr.Dataset(
            data_vars=dict(
                SRH=(["time","x","y"],srh_final)
                ),
            coords=dict(
                lon=(["x","y"],wrf_file['XLONG'].squeeze('Time').data),
                lat=(["x","y"],wrf_file['XLAT'].squeeze('Time').data),
                time=np.arange(len(file_list))
                ),
            attrs=dict(description="SRH")
            )
    print('Output directory is', out_path)
    if os.path.exists(out_path):
        os.chdir(out_path)
    else:
        os.mkdir(out_path)
        os.chdir(out_path)
        
    srh_ds.to_netcdf(f'SRH_d01_{year}-{num_month}.nc')

    # Decision for climatology or time series
    if int(option) == 1: # Climatological averaging    
        print('Running climatological averaging')
        # Extract and format variable; take average in time
        average_srh = srh_ds['SRH'].mean(dim='time')

        #Output averages to file
        average_srh.to_netcdf(f'SRH_d01_{year}-{num_month}-avg.nc')

    elif int(option) == 2: # Time series
        print('Running time series averaging')
        # Extract and format variable; take averages in space
        average_srh = srh_ds['SRH'][:,1,:,:].mean(dim=('south_north','west_east')).expand_dims('Year')
        
        # Output averages to file
        average_srh.to_netcdf(f'SRH_{year}-{num_month}-ts.nc')

def average_shear(num_month,year,option,domain,out_path):
    # Calculates average wind shear
    # Inputs:
    #  month: The month for which to perform the calculation
    #  year: The year in which to perform the calculation
    #  option: Command line input supplied by the user; 1 for climatological averaging, 2 for time-series averaging
    # Ouputs a netCDF file with average wind shear

    print(f'Directory where file will be output is', os.getcwd())

    file_list = [f for f in os.listdir() if f.startswith(f'wrfout_{domain}_{year}-{num_month}')]

    # Set up an empty array to store results of getvar
    if domain == 'd01':
        x = 370
        y = 741
    else:
        x = 609
        y = 621
    output_shape = (len(file_list),2,x,y) #<- Will need to generalize for 4 km runs
    shear_final = np.empty(output_shape, np.float32)

    # Need to hash out actual processing part down below
    for index in range(output_shape[0]):
        f = Dataset(file_list[index])
        wind = getvar(f, 'uvmet', timeidx=0)
        height = getvar(f, 'zstag')

        w_0 = find_nearest(height,0)
        w_6 = find_nearest(height,6000)
        wind_0 = wind[:,w_0,:,:]
        wind_6 = wind[:,w_6,:,:]

        u_shear = wind_6[0] - wind_0[0]
        v_shear = wind_6[1] - wind_0[1]
        shear = np.sqrt(u_shear**2 + v_shear**2)

        shear_final[index,:] = shear[:]
        f.close()

    # Open reference file to extract spatial grid
    wrf_file = xr.open_dataset(f'wrfout_{domain}_{year}-{num_month}-01_00_00_00')

    # Create dataset to prepare for output
    shear_ds = xr.Dataset(
            data_vars=dict(
                SHEAR=(["time","shear_comps","x","y"],shear_final)
                ),
            coords=dict(
                lon=(["x","y"],wrf_file['XLONG'].squeeze('Time').data),
                lat=(["x","y"],wrf_file['XLAT'].squeeze('Time').data),
                time=np.arange(len(file_list))
                ),
            attrs=dict(description="0-6 km wind shear")
            )
    print('Output directory is', out_path)
    if os.path.exists(out_path):
        os.chdir(out_path)
    else:
        os.mkdir(out_path)
        os.chdir(out_path)
    shear_ds.to_netcdf(f'SHEAR_{domain}_{year}-{num_month}.nc')

    # Decision for climatology or time series
    if int(option) == 1: # Climatological averaging    
        print('Running climatological averaging')
        # Extract and format variable; take average in time
        average_shear = shear_ds['SHEAR'].mean(dim='time')

        #Output averages to file
        average_shear.to_netcdf(f'SHEAR_d01_{year}-{num_month}-avg.nc')

    elif int(option) == 2: # Time series
        print('Running time series averaging')
        # Extract and format variable; take averages in space
        average_shear = shear_ds['SHEAR'][:,1,:,:].mean(dim=('south_north','west_east')).expand_dims('Year')
        
        # Output averages to file
        average_shear.to_netcdf(f'SHEAR_{year}-{num_month}-ts.nc')

def average_shear_ERA5(year,month,option,out_path):
    # Calculates average wind shear
    # Inputs:
    #  month: The month for which to perform the calculation
    #  year: The year in which to perform the calculation
    #  option: Command line input supplied by the user; 1 for climatological averaging, 2 for time-series averaging
    # Ouputs a netCDF file with average wind shear

    #print(f'Directory where file will be output is', os.getcwd())
    # Set file paths
    base_dir = os.getcwd()
    u_dir = os.path.join(base_dir + '/U/')
    v_dir = os.path.join(base_dir + '/V/')
    #print(base_dir, u_dir, v_dir)

    #file_list = [f for f in os.listdir() if f.startswith(f'wrfout_{domain}_{year}-{num_month}')]
    #'e5.oper.an.pl.128_131_u.ll025uv.1996'
    #u_list = [f for f in sorted(os.listdir(u_dir)) if (f.startswith(f'e5.oper.an.pl.128_131_u.ll025uv.{year}'))]
    #print(sorted(os.listdir(u_dir))[0][36:38])
    #print(sorted(os.listdir(u_dir))[0][36:38] == '01')
    u_list = [f for f in sorted(os.listdir(u_dir)) if (f.startswith(f'e5.oper.an.pl.128_131_u.ll025uv.{year}') & (f[36:38] == f'{month}'))]
    v_list = [f for f in sorted(os.listdir(v_dir)) if (f.startswith(f'e5.oper.an.pl.128_132_v.ll025uv.{year}') & (f[36:38] == f'{month}'))]
    #print(f'Number of u-wind files for {year} is:', len(u_list))
    #print(f'Number of v-wind files for {year} is:', len(v_list))

    # Set up an empty array to store results of getvar
    x = 1440
    y = 721
    days = len(u_list)
    hours = 24
    time = days * hours
    output_shape = (days,hours,y,x) #<- Will need to generalize for 4 km runs
    shear_final = np.empty(output_shape, np.float32)
    #print(output_shape)
    # Need to hash out actual processing part down below
    for index in range(output_shape[0]):
        os.chdir(u_dir)
        era5_u = xr.open_dataset(u_list[index])
        
        os.chdir(v_dir)
        era5_v = xr.open_dataset(v_list[index])
        #print(era5_u)

        pressure = era5_u['isobaricInhPa']

        height = mp.calc.pressure_to_height_std(pressure) * 1000

        height_0 = find_nearest(height,0)
        height_6 = find_nearest(height,6000)

        u_0 = era5_u['u'][:,height_0,:,:]
        v_0 = era5_v['v'][:,height_0,:,:]
        u_6 = era5_u['u'][:,height_6,:,:]
        v_6 = era5_v['v'][:,height_6,:,:]

        u_shear = u_6 - u_0
        v_shear = v_6 - v_0
        shear = np.sqrt(u_shear**2 + v_shear**2)
        
        shear_final[index,:] = shear[:]
        era5_u.close()
        era5_v.close()
        #print(shear)

    # Create dataset to prepare for output
    os.chdir(u_dir)
    era5_ref = xr.open_dataset(u_list[0])
    shear_da = np.reshape(shear_final,(time,y,x))
    #print(shear_da)
    shear_ds = xr.Dataset(
    # Just need to convert DataArray to Dataset
            data_vars=dict(
                SHEAR=(["Time","Latitude","Longitude"],shear_da)
                ),
            coords=dict(
                Longitude=(["Longitude"],era5_ref['longitude'].data),
                Latitude=(["Latitude"],era5_ref['latitude'].data),
                Time=(['Time'],pd.date_range(f'{year}-01-01',freq='1H',periods=time))
                ),
            attrs=dict(description="0-6 km wind shear")
            )
    shear_out = shear_ds#.stack(Time=('Date','time'),create_index=None)
    #print(shear_out)
    #print('Output directory is', out_path)
    if os.path.exists(out_path):
        os.chdir(out_path)
    else:
        os.mkdir(out_path)
        os.chdir(out_path)
    shear_out.to_netcdf(f'SHEAR_ERA5_{year}-{month}.nc')

    # Decision for climatology or time series
    if int(option) == 1: # Climatological averaging    
        #print('Running climatological averaging')
        # Extract and format variable; take average in time
        average_shear = shear_out['SHEAR'].mean(dim='Time')

        #Output averages to file
        average_shear.to_netcdf(f'SHEAR_ERA5_{year}-{month}-avg.nc')

    elif int(option) == 2: # Time series
        print('Running time series averaging')
        # Extract and format variable; take averages in space
        average_shear = shear_ds['SHEAR'][:,1,:,:].mean(dim=('south_north','west_east')).expand_dims('Year')
        
        # Output averages to file
        average_shear.to_netcdf(f'SHEAR_{year}-{num_month}-ts.nc')
    #raise RuntimeError('Stop here')

def climo(variable,domain,month,option):
    # Calculates climatology or time-series climatology from yearly averages
    # Inputs:
    #  variable: Variable for which to calculate climatologies
    #  month: Month over which to calculate the climatology
    #  option: Command line input supplied by the user; 1 for climatological averaging, 2 for time-series averaging
    # Outputs a netCDF file with climatological averages for the specified variable, to be used in plotting

    # Set file path
    path = os.getcwd()
    print('File path is:', path)

    if int(option) == 1:
        # Read in data; average data in time
        wrf_file = xr.open_mfdataset(f'{variable}*{month}.nc',concat_dim='Time',combine='nested',parallel=True)
        wrf_climo = wrf_file[f'{variable}'].mean(dim='Time') #<- Need to try to generalize this line
        print(f'Calculated climatology for {variable}')
        # Output data to climatology file
        filename = f'{path}/{variable}_{month}_climo.nc'
        if os.path.exists(filename):
         #   pass
         #   print('Climatology file already exists, moving on')
            wrf_climo.to_netcdf(f'{variable}_{month}_{domain}_climo.nc')
        else:
            wrf_climo.to_netcdf(f'{variable}_{month}_{domain}_climo.nc')
            print(f'Successfully output climatology for {variable}!')
    elif int(option) == 2:
        # Read in data; average data across years
        wrf_file = xr.open_mfdataset(f'wrf_{variable}*ts.nc',concat_dim='Year',combine='nested',parallel=True)
        wrf_climo = wrf_file[f'{variable}'].mean(dim='Year') #<- Need to try to generalize this line
        print(f'Calculated climatology for {variable}')
        # Output data to climatology file
        filename = f'{path}/wrf_{variable}_{month}_ts.nc'
        if os.path.exists(filename):
            pass
            print('Time series file already exists, moving on')
        else:
            wrf_climo.to_netcdf(f'wrf_{variable}_{month}_ts.nc')
            print(f'Successfully output time series for {variable}!')

def fav_env(resolution,epoch,start,end,num_month,year):
    # Calculates the favorability of an environment based upon thresholds of CAPE and SRH
    # Inputs:
    # Returns:

    # Set file paths
    dir = ''

    in_path = os.path.join(dir+f'/{epoch}/WRF-Monthly')
    out_path = os.path.join(dir+f'/{epoch}/WRF-Monthly/')
    ref_dir = os.path.join(dir+f'/WRF-Ref')
    os.chdir(in_path)
    #print('Current directory is:', in_path)

    # Set up list of files to read
    #file_list = [f for f in os.listdir() if f.startswith(f'wrfout_d02_{year}-{num_month}')]

    # Set up empty arrays to store results of getvar
    #output_shape = (len(file_list),609,621) #<- Will need to generalize for 4 km runs
    #cape_output_shape = (len(file_list),4,609,621)
    
    #srh_final = np.empty(output_shape, np.float32)
    #cape_final = np.empty(cape_output_shape, np.float32)

    # Calculate CAPE and SRH on grid
    #for index in range(len(file_list)):
    #    f = Dataset(file_list[index])
    #    cape = getvar(f, 'cape_2d')
    #    srh = getvar(f, 'srh')

    #    cape_final[index,:] = cape[:]
    #    srh_final[index,:] = srh[:]
        #print(f'Completed operation for file {index}!')
    #    f.close()
    #cape_out = cape_final[:,0,:,:]
    #cin_out = cape_final[:,1,:,:]
    #print(cape_final.shape)
    #print(srh_final.shape)

    # Read in files
    cape_data = xr.open_dataset(f'CAPE_{year}-{num_month}.nc')
    cin_data = xr.open_dataset(f'CIN_{year}-{num_month}.nc')
    srh_data = xr.open_dataset(f'SRH_{year}-{num_month}.nc')

    # Open reference file to extract spatial grid
    os.chdir(ref_dir)
    wrf_file = xr.open_dataset(f'wrfout_ref4.nc').rename({'south_north':'x','west_east':'y','Time':'Time_step'})
    
    # Set up arrays for computation
    cape_final = cape_data['CAPE']#.where(wrf_file['XLAND'] != 2.0)
    cin_final = cin_data['CIN']#.where(wrf_file['XLAND'] != 2.0)
    srh_final = srh_data['SRH']#.where(wrf_file['XLAND'] != 2.0)

    # Determine environmental favorability
    fav = ((cape_final > 500) & (cin_final < 100) & (srh_final > 100)).astype(int) # Change these lines for different CAPE/CIN thresholds(?)
    #print(fav.shape)

    #prob_fav = np.mean(fav, axis = 0)
    #prob_fav = prob_fav.expand_dims('time',axis=2)
    #print(prob_fav)
    print('Shape of favorable is:', fav.shape)
    print(len(cape_data.time))
    # Create dataset to prepare for output
    fav_ds = xr.Dataset(
            data_vars=dict(
                PROB_FAV=(["time","x","y"],fav.data)
                ),
            coords=dict(
                time=(["time"],pd.date_range(start=f'{year}-{num_month}-01',periods=len(cape_data.time),freq='3H')),
                lon=(["x","y"],wrf_file['XLONG'].squeeze('Time_step').data),
                lat=(["x","y"],wrf_file['XLAT'].squeeze('Time_step').data)
                ),
            attrs=dict(description="Probability of a favorable environment")
            )

    # Extract and format variable; take average in time
    #Output averages to file
    os.chdir(out_path)
    print('Directory for outputting files is:', out_path)
    fav_out = fav_ds#.expand_dims('time')
    print(fav_out)
    fav_out.to_netcdf(f'PROBFAV_{year}-{num_month}.nc')

def storm_scale(epoch,num_month,month_name,variable):
    # Control flow
    if epoch == 'hist':
        start = '1991'
        end = '2000'
    elif epoch == 'fut':
        start = '2091'
        end = '2100'

    if variable == 'REF':
        var = 'REFD_MAX'
    elif variable == 'W':
        var = 'W_UP_MAX'
    elif variable == 'UH':
        var = 'UP_HELI_MAX'

    base_path = os.getcwd()
    print('Base directory is:', base_path)
    out_path = os.path.join(base_path+f'/{epoch}/WRF-Averages/')

    for year in range(int(start),int(end)+1): 
        print(f'Processing for {year}')
        path = os.path.join(base_path+f'/{epoch}/WRF-{year}/run/wrf/')
        print('Current path is:',path)
        os.chdir(path)

        # Open reference file to extract spatial grid
        wrf_file = xr.open_dataset(f'wrfout_d02_{year}-{num_month}-01_00_00_00')
        print(wrf_file)

        file_list = [f for f in os.listdir() if f.startswith(f'wrfout_d02_{year}-{num_month}')]

        # Set up an empty array to store values
        output_shape = (len(file_list),609,621)
        ss_final = np.empty(output_shape, np.float32)

        for index in range(len(file_list)):
            f = xr.open_dataset(file_list[index])
            ss = f[f'{var}'].values
            ss_final[index,:] = ss[:]

        # Create dataset to prepare for output
        ss_ds = xr.Dataset(
                data_vars=dict(
                    W=(["time","x","y"],ss_final)
                    ),
                coords=dict(
                    lon=(["x","y"],wrf_file['XLONG'].squeeze('Time').data),
                    lat=(["x","y"],wrf_file['XLAT'].squeeze('Time').data),
                    time=np.arange(len(file_list))
                    ),
                attrs=dict(description=f"{var}")
                )
        print(ss_ds)

        print('Output directory is', out_path)
        if os.path.exists(out_path):
            os.chdir(out_path)
        else:
            os.mkdir(out_path) 
            os.chdir(out_path)
        print(f'File will be named {var}_{year}-{num_month}')
        ss_ds[f'{variable}'].to_netcdf(f'{var}_{year}-{num_month}.nc')
        # Calculate quantiles of data
        #quantile = var.quantile(0.95, dim='Time')
        #print(quantile)
        #os.chdir(base_path)

def occurrence(resolution,epoch,start,end,num_month,year):
    # Calculates the favorability of an environment based upon thresholds of CAPE and SRH
    # Inputs:
    # Returns:

    # Set file paths
    dir = ''

    wrf_path = os.path.join(dir+f'/WRF-Ref/')
    in_path = os.path.join(dir+f'/{epoch}/WRF-Monthly/')
    out_path = os.path.join(dir+f'/{epoch}/WRF-Averages/')
    os.chdir(in_path)
    print('Current directory is:', in_path)
    print(f'Computing for {year}')

    # Read in CAPE/SRH files
    cape_data = xr.open_mfdataset(f'CAPE_{year}-{num_month}.nc',concat_dim='Time',combine='nested',parallel=True)
    cin_data = xr.open_mfdataset(f'CIN_{year}-{num_month}.nc',concat_dim='Time',combine='nested',parallel=True)
    srh_data = xr.open_mfdataset(f'SRH_{year}-{num_month}.nc',concat_dim='Time',combine='nested',parallel=True)
    ss_data = xr.open_mfdataset(f'W_UP_MAX_{year}-{num_month}.nc',concat_dim='Time',combine='nested',parallel=True)
    
    cape = cape_data['CAPE']
    cin = cin_data['CIN']
    srh = srh_data['SRH']
    w = ss_data['W']

    w_mask = w >= 18
    #print(np.count_nonzero(~np.isnan(w)))
    #print(np.count_nonzero(w_mask))
    #print(w_mask.compute())
    cape_masked = cape.rolling(time=3,min_periods=1).max().where(w_mask) # Take the maximum CAPE (or CIN, or SRH) from current time step and two prior - min_periods = 1 means that only 1 real value is required for rolling calculation - mask where 6-hourly UVV max > 18
    # This rolling operation can be used for almost any neighborhood operation
    cin_masked = cin.rolling(time=3,min_periods=1).max().where(w_mask)
    srh_masked = srh.rolling(time=3,min_periods=1).max().where(w_mask)
   
    #print('Original CAPE array:', cape.compute()[0,0])
    #print('Rolling maximum CAPE array:', cape_masked.compute()[0,0])
    # Access and transform storm-scale variables
    #refl = np.array(wrf_out['REFD_MAX'].values)
    #uh = np.array(wrf_out['UP_HELI_MAX'].values)
    #w = np.array(wrf_out['W_UP_MAX'].values)
    #print('Reflectivity/updraft helicity shapes are:',refl.shape,uh.shape)
    #w_up = np.array(wrf_out['W_UP_MAX'].values)

    # Determine occurrence of HCW based on thresholds
    occ = ((cape_masked > 500) & (cin_masked < 100) & (srh_masked > 150)).astype(int).compute() # New occurrence computation
    #occ_orig = ((w > 18) & (cape > 500) & (cin < 100) & (srh > 150)).astype(int).compute() # Original version of the occurrence computation
    #print(occ)
    #print(occ.mean())
    #print(occ_orig.mean())

    #print('Number of occurrences with new calculation is:', np.count_nonzero(occ))
    #print('Number of occurrences with original calculation is:', np.count_nonzero(occ_orig))
    #raise RuntimeError('Stop here')
    #print(occ.shape)

    prob_occ = np.squeeze(occ)
    #print(squeezed)
    #prob_occ = np.mean(squeezed, axis = 0)
    #print(prob_occ)

    # Open reference file to extract spatial grid
    os.chdir(wrf_path)
    wrf_file = xr.open_dataset(f'wrfout_ref4.nc')

    # Create dataset to prepare for output
    occ_ds = xr.Dataset(
            data_vars=dict(
                PROB_OCC=(["time","x","y"],prob_occ.data)
                ),
            coords=dict(
                lon=(["x","y"],wrf_file['XLONG'].squeeze('Time').data),
                lat=(["x","y"],wrf_file['XLAT'].squeeze('Time').data),
                time=(["time"],pd.date_range(start=f'{year}-{num_month}-01',periods=len(prob_occ.time),freq='3H'))
                ),
            attrs=dict(description="Probability of HCW occurrence")
            )

    # Extract and format variable; take average in time
    #Output averages to file
    os.chdir(in_path)
    print('Directory for outputting files is:', out_path)
    occ_out = occ_ds#.expand_dims('time')
    print(occ_out)
    #raise RuntimeError('Stop here')
    occ_out.to_netcdf(f'PROBOCC_{year}-{num_month}-neighborhood.nc')
    
    avg_occ = occ_out.mean(dim='time')
    avg_occ.to_netcdf(f'PROBOCC_{year}-{num_month}-avg-neighborhood.nc')

def subset(hist_4,fut_4,hist_27,fut_27,region):
    # Subsets data into geographic regions based on user input
    # Inputs:
    # hist_4/fut_4/hist_27/fut_27: Datasets at various resolutions/epochs to be processed
    # region: Geographic region to subset for
    # Returns:
    # hist_4/fut_4/hist_27/fut_27: Subset datasets

    # Set regional bounds
    if region == 'ALL':
        return hist_4,fut_4,hist_27,fut_27
    elif region == 'NGP':
        lon_min = -102.5
        lon_max = -91
        lat_min = 41
        lat_max = 49
    elif region == 'SGP':
        lon_min = -102.5
        lon_max = -91
        lat_min = 26
        lat_max = 41
    elif region == 'MID':
        lon_min = -91
        lon_max = -72
        lat_min = 37
        lat_max = 49
    elif region == 'SE':
        lon_min = -91
        lon_max = -72
        lat_min = 26
        lat_max = 37
    else:
        raise ValueError('Please provide a valid region for analysis')

    # Subset data based on provided regions
    if hist_4 != None:
        #print('Initial dataset pre-subsetting is',hist_4)
        hist_4 = hist_4.where((hist_4.lon > lon_min) & (hist_4.lon < lon_max) & (hist_4.lat > lat_min) & (hist_4.lat < lat_max))
        #print('Dataset after subsetting is',hist_4)
        print('Subset historical 4 km data')
    if fut_4 != None:
        fut_4 = fut_4.where((fut_4.lon > lon_min) & (fut_4.lon < lon_max) & (fut_4.lat > lat_min) & (fut_4.lat < lat_max))
        print('Subset future 4 km data')
    if hist_27 != None:
        hist_27 = hist_27.where((hist_27.lon > lon_min) & (hist_27.lon < lon_max) & (hist_27.lat > lat_min) & (hist_27.lat < lat_max))
        print('Subset historical 27 km data')
    else:
        pass
    if fut_27 != None:
        fut_27 = fut_27.where((fut_27.lon > lon_min) & (fut_27.lon < lon_max) & (fut_27.lat > lat_min) & (fut_27.lat < lat_max))
        print('Subset future 27 km data')
    else:
        pass

    return hist_4,fut_4,hist_27,fut_27

def trim(hist_4,fut_4,hist_27,fut_27,variable,low,up):
    # Trims data to plot on KDEs based on defined percentiles
    # Inputs:
    # hist_4/fut_4/hist_27/fut_27: Datasets at various resolutions/epochs to be processed
    # Returns:
    # hist_4/fut_4/hist_27/fut_27: Trimmed datasets

    # Set percentiles to trim on
    lower_bound = low
    upper_bound = up

    # Subset data based on provided bounds
    hist_4 = hist_4.where((hist_4[f'{variable}'] > np.nanquantile(hist_4[f'{variable}'].values,lower_bound)) & (hist_4[f'{variable}'] < np.nanquantile(hist_4[f'{variable}'].values,upper_bound)))
    print('Trimmed historical 4 km data')
    print('Trimmed future 4 km data')
    fut_4 = fut_4.where((fut_4[f'{variable}'] > np.nanquantile(fut_4[f'{variable}'].values,lower_bound)) & (fut_4[f'{variable}'] < np.nanquantile(fut_4[f'{variable}'].values,upper_bound)))
    if hist_27 != None:
        hist_27 = hist_27.where((hist_27[f'{variable}'] > np.nanquantile(hist_27[f'{variable}'].values,lower_bound)) & (hist_27[f'{variable}'] < np.nanquantile(hist_27[f'{variable}'].values,upper_bound)))
        print('Trimmed historical 27 km data')
    else:
        pass
    if fut_27 != None:
        fut_27 = fut_27.where((fut_27[f'{variable}'] > np.nanquantile(fut_27[f'{variable}'].values,lower_bound)) & (fut_27[f'{variable}'] < np.nanquantile(fut_27[f'{variable}'].values,upper_bound)))
        print('Trimmed future 27 km data')
    else:
        pass

    return hist_4,fut_4,hist_27,fut_27

def cond_handler(epochs,months,regions,option):
    # Code to handle setup of computations for conditional HCW occurrence
    # Inputs: 
    #  epochs: Climate epochs (hist and fut) for iteration
    #  months: January-April, for iteration
    #  regions: Geographic regions (Norhtern Great Plains, Southern Great Plains, Southeast US, Mid-Atlantic) for iteration
    # Returns:
    #  None (Outputs conditional probabilities to file)
    # Set up directory structure
    base_dir = os.getcwd()
    os.chdir(os.path.join(base_dir + '/WRF-Ref'))
    wrf_file = xr.open_dataset('wrfout_ref4.nc').rename({'south_north':'x','west_east':'y','Time':'Time_step'})
    #print(wrf_file)
    os.chdir(base_dir)
    
    favs = []
    occs = []
    conds = []
    indices = []
    if option == 1: # Epochs only
        for epoch in epochs:
            io_dir = os.path.join(base_dir + f'/{epoch}/' + 'WRF-Monthly')
            out_dir = os.path.join(base_dir + '/WRF-Stats')
            os.chdir(io_dir)

            # Read in data
            for i in range(20):
                try:
                    cape_data = xr.open_mfdataset(f'CAPE_*.nc',concat_dim='Time',combine='nested',parallel=True)
                    cin_data = xr.open_mfdataset('CIN-*.nc',concat_dim='Time',combine='nested',parallel=True)
                    srh_data = xr.open_mfdataset('SRH_*.nc',concat_dim='Time',combine='nested',parallel=True)
                    w_data = xr.open_mfdataset('W_UP_MAX_*.nc',concat_dim='Time',combine='nested',parallel=True)
                    break
                except (RuntimeError,OSError):
                    print(f'Failed to read files for one variable; trying again')
                    continue

            # Compute conditional probabilities; append to lists
            prob_fav,prob_occ,cond_prob = cond_probs(cape_data,cin_data,srh_data,w_data,wrf_file)
            favs.append(prob_fav)
            occs.append(prob_occ)
            conds.append(cond_prob)
            indices.append(epoch)
            print(f'Computed probabilities for {epoch}!')

    elif option == 2: # Epochs and months
        for combo in itertools.product(epochs,months):
            io_dir = os.path.join(base_dir + f'/{combo[0]}/' + 'WRF-Monthly')
            out_dir = os.path.join(base_dir + '/WRF-Stats')
            os.chdir(io_dir)

            # Read in data
            for i in range(20):
                try:
                    cape_data = xr.open_mfdataset(f'CAPE_*{combo[1]}*.nc',concat_dim='Time',combine='nested',parallel=True)
                    cin_data = xr.open_mfdataset(f'CIN_*{combo[1]}*.nc',concat_dim='Time',combine='nested',parallel=True)
                    srh_data = xr.open_mfdataset(f'SRH_*{combo[1]}*.nc',concat_dim='Time',combine='nested',parallel=True)
                    w_data = xr.open_mfdataset(f'W_UP_MAX_*{combo[1]}*.nc',concat_dim='Time',combine='nested',parallel=True)
                    break
                except (RuntimeError,OSError):
                    #print(f'Failed to read files for one variable; trying again')
                    continue

            # Compute conditional probabilities; append to lists
            prob_fav,prob_occ,cond_prob = cond_probs(cape_data,cin_data,srh_data,w_data,wrf_file)
            favs.append(prob_fav)
            occs.append(prob_occ)
            conds.append(cond_prob)
            indices.append(f'{combo[0]} {combo[1]}')
            print(f'Computed probabilities for {combo[0]} {combo[1]}!')
 
    elif option == 3: # Epochs and regions
        for combo in itertools.product(epochs,regions):
            io_dir = os.path.join(base_dir + f'/{combo[0]}/' + 'WRF-Monthly')
            out_dir = os.path.join(base_dir + '/WRF-Stats')
            os.chdir(io_dir)

            # Read in data
            for i in range(20):
                try:
                    cape_data = xr.open_mfdataset(f'CAPE_*.nc',concat_dim='Time',combine='nested',parallel=True)
                    cin_data = xr.open_mfdataset('CIN_*.nc',concat_dim='Time',combine='nested',parallel=True)
                    srh_data = xr.open_mfdataset('SRH_*.nc',concat_dim='Time',combine='nested',parallel=True)
                    w_data = xr.open_mfdataset('W_UP_MAX_*.nc',concat_dim='Time',combine='nested',parallel=True)
                    break
                except (RuntimeError,OSError):
                    print(f'Failed to read files for one variable; trying again')
                    continue

            # Subset regionally
            region = combo[1]
            cape_data,cin_data,srh_data,w_data,wrf_subset = subset_vars(cape_data,cin_data,srh_data,w_data,wrf_file,region)

            # Compute conditional probabilities; append to lists
            prob_fav,prob_occ,cond_prob = cond_probs(cape_data,cin_data,srh_data,w_data,wrf_subset)
            favs.append(prob_fav)
            occs.append(prob_occ)
            conds.append(cond_prob)
            indices.append(f'{combo[0]} {combo[1]}')
            print(f'Computed probabilities for {combo[0]} {combo[1]}!')
 
    elif option == 4: # Epochs, months, and regions
        for combo in itertools.product(epochs,months,regions):
            io_dir = os.path.join(base_dir + f'/{combo[0]}/' + 'WRF-Monthly')
            out_dir = os.path.join(base_dir + '/WRF-Stats')
            os.chdir(io_dir)

            # Read in data
            for i in range(20):
                try:
                    cape_data = xr.open_mfdataset(f'CAPE_*{combo[1]}.nc',concat_dim='Time',combine='nested',parallel=True)
                    cin_data = xr.open_mfdataset(f'CIN-*{combo[1]}.nc',concat_dim='Time',combine='nested',parallel=True)
                    srh_data = xr.open_mfdataset(f'SRH_*{combo[1]}.nc',concat_dim='Time',combine='nested',parallel=True)
                    w_data = xr.open_mfdataset(f'W_UP_MAX_*{combo[1]}.nc',concat_dim='Time',combine='nested',parallel=True)
                    break
                except (RuntimeError,OSError):
                    print(f'Failed to read files for one variable; trying again')
                    continue
            
            # Subset regionally
            region = combo[2]
            cape_data,cin_data,srh_data,w_data,wrf_subset = subset_vars(cape_data,cin_data,srh_data,w_data,wrf_file,region)

            # Compute conditional probabilities; append to lists
            prob_fav,prob_occ,cond_prob = cond_probs(cape_data,cin_data,srh_data,w_data,wrf_subset)
            favs.append(prob_fav)
            occs.append(prob_occ)
            conds.append(cond_prob)
            indices.append(f'{combo[0]} {combo[1]} {combo[2]}')
            print(f'Computed probabilities for {combo[0]} {combo[1]} {combo[2]}!')

    else:
        raise ValueError('Please supply a valid option')

    # Create data structure with probabilities
    d = {'Favorable HCW Environment Probability':favs,'HCW Occurrence Probability':occs,'Conditional Probability of HCW Occurrence':conds}
    cond_df = pd.DataFrame(data=d,index=indices)
    cond_df.index = cond_df.index.rename('Stratification')

    # Output probabilities to file
    if os.path.exists(out_dir):
        pass
    else:
        os.mkdir(out_dir)
    os.chdir(out_dir)

    file_name = 'probs_18.csv'
    if os.path.exists(file_name):
        cond_file = pd.read_csv(file_name,index_col='Stratification')
        cond_new = pd.concat([cond_file,cond_df])
        cond_new#.to_csv(file_name)
    else:
        cond_df#.to_csv(file_name)

def cond_probs(cape_data,cin_data,srh_data,w_data,wrf_file):
    # Computes conditional probability of HCW based on environmental diagnostic inputs
    # Inputs:
    #  cape/cin/srh/w_data: Environmental diagnostics used as input to compute environmental favorability/occurrence
    # Returns:
    #  cond_prob: Conditional probability of HCW occurrence given environmental favorability
    
    cape = cape_data['CAPE']#.where(wrf_file['XLAND'] != 2.0)
    cin = cin_data['CIN']#.where(wrf_file['XLAND'] != 2.0)
    srh = srh_data['SRH']#.where(wrf_file['XLAND'] != 2.0)
    w = w_data['W']#.where(wrf_file['XLAND'] != 2.0)

    # Calculate conditional probability of HCW occurrence (Conditional on month/region)
    fav_env = xr.where((cape > 500) & (cin < 100) & (srh > 150),1,0)#.astype(int)
    occ = xr.where((cape > 500) & (cin < 100) & (srh > 150) & (w > 18),1,0)#.astype(int)

    print('-----------------------------------')
    #prob_fav = (np.count_nonzero(fav_env))/len(fav_env.values.flatten())
    prob_occ = (np.count_nonzero(occ))/len(occ.values.flatten())
    prob_fav = np.nanmean(fav_env.values)
    print('Probability of favorable HCW environment is', prob_fav)
    print('Probability of HCW occurrence is', prob_occ)
    print('------------------------------------')

    cond_prob = prob_occ/prob_fav
    print('Conditional proability of HCW occurrence given a favorable environment is', cond_prob)
    print('------------------------------------')

    return prob_fav,prob_occ,cond_prob

def subset_vars(cape_data,cin_data,srh_data,w_data,wrf_file,region):
    # Geographically subsets data for a particular region
    # Inputs:
    # cape_data/cin_data/srh_data/w_data: Datasets at various resolutions/epochs to be processed
    # region: Geographic region to subset for
    # Returns:
    # cape_data/cin_data/srh_data/w_data: Subset datasets

    # Set regional bounds
    if region == 'ALL':
        return cape_data,cin_data,srh_data,w_data
    elif region == 'NGP':
        lon_min = -102.5
        lon_max = -91
        lat_min = 41
        lat_max = 49
    elif region == 'SGP':
        lon_min = -102.5
        lon_max = -91
        lat_min = 26
        lat_max = 41
    elif region == 'MID':
        lon_min = -91
        lon_max = -72
        lat_min = 37
        lat_max = 49
    elif region == 'SE':
        lon_min = -91
        lon_max = -72
        lat_min = 26
        lat_max = 37
    else:
        raise ValueError('Please provide a valid region for analysis')

    # Subset data based on provided regions
    cape_data = cape_data.where((cape_data.lon > lon_min) & (cape_data.lon < lon_max) & (cape_data.lat > lat_min) & (cape_data.lat < lat_max), drop=True)
    print('Subset CAPE data')
    cin_data = cin_data.where((cin_data.lon > lon_min) & (cin_data.lon < lon_max) & (cin_data.lat > lat_min) & (cin_data.lat < lat_max), drop=True)
    print('Subset CIN data')
    srh_data = srh_data.where((srh_data.lon > lon_min) & (srh_data.lon < lon_max) & (srh_data.lat > lat_min) & (srh_data.lat < lat_max), drop=True)
    print('Subset SRH data')
    w_data = w_data.where((w_data.lon > lon_min) & (w_data.lon < lon_max) & (w_data.lat > lat_min) & (w_data.lat < lat_max), drop=True)
    print('Subset W data')
    wrf_file = wrf_file.where((wrf_file.XLONG > lon_min) & (wrf_file.XLONG < lon_max) & (wrf_file.XLAT > lat_min) & (wrf_file.XLAT < lat_max), drop=True)
    print('Subset reference file')
    return cape_data,cin_data,srh_data,w_data,wrf_file

def frequency_changes(variable,option,domain_name,threshold):
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
    
    if option == 1: # Epochs only
        for i in range(20):
            try:
                # Read in climatological files
                print('Starting to open historical files')
                os.chdir(hist_path)
                hist_data = xr.open_mfdataset(f'{variable}_*.nc',concat_dim='time',combine='nested',parallel=True)
                
                print('Starting to open future files')
                os.chdir(os.path.join(base_dir + '/' + fut_path))
                fut_data = xr.open_mfdataset(f'{variable}_*.nc',concat_dim='time',combine='nested',parallel=True)
                print('Opened all files')
                break
            except (RuntimeError,OSError):
                print('Failed to open files, trying again')
                continue
    
        # Trim SRH datasets
        if variable == 'SRH':
            hist_data,fut_data,hist_null,fut_null = trim(hist_data,fut_data,None,None,variable,0.01,0.99)

        # Convert to dataframes; combine data
        hist_ds = hist_data.to_dataframe()
        fut_ds = fut_data.to_dataframe()

        if variable == 'CIN':
            operator = '>'
            hist_ds = hist_ds.where(hist_ds[f'{variable}'] > int(threshold)).dropna()
            fut_ds = fut_ds.where(fut_ds[f'{variable}'] > int(threshold)).dropna()
        else:
            operator = '>'
            hist_ds = hist_ds.where(hist_ds[f'{variable}'] > int(threshold)).dropna()
            fut_ds = fut_ds.where(fut_ds[f'{variable}'] > int(threshold)).dropna()
        
        print('Sent data to dataframe')

        hist_ds.insert(3, 'Epoch', pd.Series('Historical', index=hist_ds.index))
        fut_ds.insert(3, 'Epoch', pd.Series('Future', index=fut_ds.index))
        print('Inserted new columns into dataframes')

        if variable == 'SHEAR':
            ds = pd.concat([hist_ds.sample(frac=0.05),fut_ds.sample(frac=0.05)])
        else:
            ds = pd.concat([hist_ds.sample(frac=0.1),fut_ds.sample(frac=0.1)])
        print('Created full dataframe')

        # Group by epochs; calculate percentage change between epochs
        cond_group = ds.groupby(['Epoch'])
        perc_change = (len(cond_group.get_group('Future')) - len(cond_group.get_group('Historical'))) / (len(cond_group.get_group('Historical'))) * 100
        print(f'Percentage frequency change of {variable} between climate states is', perc_change)
    
    if option == 2: # Epochs and months
        for month in range (1,5):
            for i in range(20):
                try:
                    # Read in climatological files
                    print('Starting to open historical files')
                    os.chdir(os.path.join(base_dir + '/' + hist_path))
                    hist_data = xr.open_mfdataset(f'{variable}_*-0{month}.nc',concat_dim='time',combine='nested',parallel=True)
                    
                    print('Starting to open future files')
                    os.chdir(os.path.join(base_dir + '/' + fut_path))
                    fut_data = xr.open_mfdataset(f'{variable}_*-0{month}.nc',concat_dim='time',combine='nested',parallel=True)
                    print('Opened all files')
                    break
                except (RuntimeError,OSError):
                    print('Failed to open files, trying again')
                    continue
        
            # Trim SRH datasets
            if variable == 'SRH':
                hist_data,fut_data,hist_null,fut_null = trim(hist_data,fut_data,None,None,variable,0.01,0.99)

            # Convert to dataframes; combine data
            hist_ds = hist_data.to_dataframe()
            fut_ds = fut_data.to_dataframe()

            if variable == 'CIN':
                operator = '>'
                hist_ds = hist_ds.where(hist_ds[f'{variable}'] > int(threshold)).dropna()
                fut_ds = fut_ds.where(fut_ds[f'{variable}'] > int(threshold)).dropna()
            else:
                operator = '>'
                hist_ds = hist_ds.where(hist_ds[f'{variable}'] > int(threshold)).dropna()
                fut_ds = fut_ds.where(fut_ds[f'{variable}'] > int(threshold)).dropna()
            
            print('Sent data to dataframe')

            hist_ds.insert(3, 'Epoch', pd.Series('Historical', index=hist_ds.index))
            fut_ds.insert(3, 'Epoch', pd.Series('Future', index=fut_ds.index))
            print('Inserted new columns into dataframes')

            if variable == 'SHEAR':
                ds = pd.concat([hist_ds.sample(frac=0.05),fut_ds.sample(frac=0.05)])
            else:
                ds = pd.concat([hist_ds.sample(frac=0.1),fut_ds.sample(frac=0.1)])
            print('Created full dataframe')

            # Group by epochs; calculate percentage change between epochs stratified across months
            cond_group = ds.groupby(['Epoch'])
            perc_change = (len(cond_group.get_group('Future')) - len(cond_group.get_group('Historical'))) / (len(cond_group.get_group('Historical'))) * 100
            print(f'Percentage frequency change of {variable} in 0{month} between climate states is', perc_change)
    
    if option == 3: # Epochs and regions
        # Read in climatological files
        regions = ['NGP','SGP','MID','SE']
        for region in regions:
            for i in range(20):
                try:
                    os.chdir(os.path.join(base_dir + '/' + hist_path))
                    hist_data = xr.open_mfdataset(f'{variable}_*.nc',concat_dim='time',combine='nested',parallel=True)

                    os.chdir(os.path.join(base_dir + '/' + fut_path))
                    fut_data = xr.open_mfdataset(f'{variable}_*.nc',concat_dim='time',combine='nested',parallel=True)
                    break
                except (RuntimeError,OSError):
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
                hist_ds = hist_ds.where(hist_ds[f'{variable}'] > int(threshold)).dropna()
                fut_ds = fut_ds.where(fut_ds[f'{variable}'] > int(threshold)).dropna()
            else:
                operator = '>'
                hist_ds = hist_ds.where(hist_ds[f'{variable}'] > int(threshold)).dropna()
                fut_ds = fut_ds.where(fut_ds[f'{variable}'] > int(threshold)).dropna()

            hist_ds.insert(3,'Region',pd.Series(region, index=hist_ds.index))
            fut_ds.insert(3,'Region',pd.Series(region, index=fut_ds.index))

            hist_ds.insert(3, 'Epoch', pd.Series('Historical', index=hist_ds.index))
            fut_ds.insert(3, 'Epoch', pd.Series('Future', index=fut_ds.index))

            if variable == 'SHEAR':
                ds = pd.concat([hist_ds.sample(frac=0.05),fut_ds.sample(frac=0.05)])
            else:
                ds = pd.concat([hist_ds.sample(frac=0.1),fut_ds.sample(frac=0.1)])
            
            # Group by epochs; calculate percentage change between epochs stratified across months
            cond_group = ds.groupby(['Epoch'])
            perc_change = (len(cond_group.get_group('Future')) - len(cond_group.get_group('Historical'))) / (len(cond_group.get_group('Historical'))) * 100
            print(f'Percentage frequency change of {variable} in {region} between climate states is', perc_change)

    if option == 4: # Epochs, months, and regions
        # Read in climatological files
        regions = ['NGP','SGP','MID','SE']
        for month in range(1,5):
            for region in regions:
                for i in range(20):
                    try:
                        os.chdir(os.path.join(base_dir + '/' + hist_path))
                        hist_data = xr.open_mfdataset(f'{variable}_*0{month}.nc',concat_dim='time',combine='nested',parallel=True)

                        os.chdir(os.path.join(base_dir + '/' + fut_path))
                        fut_data = xr.open_mfdataset(f'{variable}_*0{month}.nc',concat_dim='time',combine='nested',parallel=True)
                        break
                    except (RuntimeError,OSError):
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
                    hist_ds = hist_ds.where(hist_ds[f'{variable}'] > int(threshold)).dropna()
                    fut_ds = fut_ds.where(fut_ds[f'{variable}'] > int(threshold)).dropna()
                else:
                    operator = '>'
                    hist_ds = hist_ds.where(hist_ds[f'{variable}'] > int(threshold)).dropna()
                    fut_ds = fut_ds.where(fut_ds[f'{variable}'] > int(threshold)).dropna()

                hist_ds.insert(3,'Region',pd.Series(region, index=hist_ds.index))
                fut_ds.insert(3,'Region',pd.Series(region, index=fut_ds.index))

                hist_ds.insert(3, 'Epoch', pd.Series('Historical', index=hist_ds.index))
                fut_ds.insert(3, 'Epoch', pd.Series('Future', index=fut_ds.index))

                if variable == 'SHEAR':
                    ds = pd.concat([hist_ds.sample(frac=0.05),fut_ds.sample(frac=0.05)])
                else:
                    ds = pd.concat([hist_ds.sample(frac=0.1),fut_ds.sample(frac=0.1)])
                
                # Group by epochs; calculate percentage change between epochs stratified across months
                cond_group = ds.groupby(['Epoch'])
                perc_change = (len(cond_group.get_group('Future')) - len(cond_group.get_group('Historical'))) / (len(cond_group.get_group('Historical'))) * 100
                print(f'Percentage frequency change of {variable} in {region} and month 0{month} between climate states is', perc_change)

def ml_inputs(epoch,resolution,year,option):
    # Inputs:
    #  epoch: The epoch for which to prepare ML inputs
    #  resolution: The resolution of simulation being handled
    # Returns:
    #  None (Outputs input data for ML model to file)

    # Format resolution for later use
    resolution = int(resolution)
    
    # Set directory structure
    base_dir = ''
    if resolution == 4:
        input_dir = os.path.join(base_dir + '/' + f'{epoch}/WRF-Monthly')
    elif resolution == 27:
        input_dir = os.path.join(base_dir + '/' + f'27km/{epoch}/WRF-Monthly')
    #print(input_dir)
    ref_dir = os.path.join(base_dir + '/WRF-Ref')
    if option == 'occs':
        out_dir = os.path.join(base_dir + '/ML/output/occ_predict')
    else:
        out_dir = os.path.join(base_dir + f'/ML/temp')

    
    # Read in data to prep inputs

    # Read in environmental parameters
    os.chdir(input_dir)
    print(os.getcwd())
    for i in range(20):
        try:
            cape = xr.open_mfdataset(f'CAPE_{year}*.nc',concat_dim='time',combine='nested',parallel=True)
            cin = xr.open_mfdataset(f'CIN_{year}*.nc',concat_dim='time',combine='nested', parallel=True)
            srh = xr.open_mfdataset(f'SRH_{year}*.nc',concat_dim='time',combine='nested',parallel=True)
            break
        except (OSError):
            print('Trying again')
            #sys.exit(1)

    # Read in storm-scale variables of interest
    if resolution == 4:
        ss = xr.open_mfdataset(f'W_UP_MAX_{year}*.nc',concat_dim='time',combine='nested',parallel=True)
    else:
        ss = None

    # Read in WRF file for spatial dimensions
    os.chdir(ref_dir)
    wrf_file = xr.open_dataset(f'wrfout_ref{resolution}.nc')

    # Processing
    if resolution == 4:
        if option == 'occs':
            occs,null_final = occs_predict(cape,cin,srh,ss,wrf_file)
        else:
            occs,null_final = ml_occs(cape,cin,srh,ss,wrf_file)
        os.chdir(out_dir)
        occs.to_csv(f'occs_{resolution}_{year}.csv')
        null_final.to_csv(f'null_cases_{resolution}_{year}.csv')
        print(f'Successfully processed occurrences/null cases for {year}')
        print('----------------------------------------------------------')
        #raise RuntimeError('Stop here')
        print('Preparing to run environments code')
        envs,null_envs = ml_envs(cape,cin,srh,ss,occs,null_final,resolution)

        # Send dataframes to files
        envs.to_csv(f'occ_envs_{resolution}_{year}.csv')
        null_envs.to_csv(f'null_envs_{resolution}_{year}.csv')
    else:
        test = ml_27(cape,cin,srh,wrf_file)
        print(f'Successfully processed occurrences/null cases for {year}')
        os.chdir(out_dir)
        test.to_csv(f'testing_{resolution}_{year}.csv')
        
        print('Preparing to run environments code')
        test_envs,null = ml_envs(cape,cin,srh,ss,test,None,resolution)

        # Send dataframes to files
        test_envs.to_csv(f'test_envs_{resolution}_{year}.csv')
    print(f'Files output to', os.getcwd())
   
    # Processing environments
    print(f'Environment data ready to return to main program for {year}')
    print('------------------------------------------------------------')
    print('Successfully processed environments surrounding occurrences')
    del cape,cin,srh,ss,wrf_file

def ml_occs(cape,cin,srh,ss,wrf_file):
    # Inputs
    #  cape,cin,srh,ss: Datasets with information for CAPE/CIN/SRH + storm-scale variable
    # Returns:
    #  occs,null_final: Dataframes with occurrences + null cases

    # Find occurrences
    #print(len(cape.x),len(cape.y))
    output_shape = (len(cape.time),len(cape.x),len(cape.y))
    occ_final = np.zeros(output_shape, np.float32)

    # Modify land mask array
    land_mask = wrf_file['LANDMASK'].expand_dims({'time':len(cape.time)}).squeeze('Time').rename({'south_north':'x','west_east':'y'})
    print(land_mask)
    print(cape['CAPE'])
    occ_masked = np.ma.masked_where((cape['CAPE'] > 500) & (cin['CIN'] < 100) & (srh['SRH'] > 100) & (ss['W'] > 18) & (land_mask == 1.0), occ_final)
    null_masked = np.ma.masked_where((cape['CAPE'] > 500) & (cin['CIN'] < 100) & (srh['SRH'] > 100) & (ss['W'] < 18) & (land_mask == 1.0), occ_final) 
    #occ_masked = np.ma.masked_where((cape['CAPE'] > 500) & (cin['CIN'] < 100) & (srh['SRH'] > 100) & (ss['W'] > 18), occ_final)
    #null_masked = np.ma.masked_where((cape['CAPE'] > 500) & (cin['CIN'] < 100) & (srh['SRH'] > 100) & (ss['W'] < 18), occ_final) 

    occ = occ_masked.filled(fill_value=1)
    null = null_masked.filled(fill_value=2)
     
    # Assign to dataset with time/spatial coordinates
    #print('Made it here')
    #print(cape['CAPE'])
    #print(wrf_file['LANDMASK'])
    #print(wrf_file['LANDMASK'].values.min())
    occ_ds = xr.Dataset(
            data_vars=dict(
                OCC=(['time','x','y'],occ),
                CAPE=(['time','x','y'],cape['CAPE'].data),
                CIN=(['time','x','y'],cin['CIN'].data),
                SRH=(['time','x','y'],srh['SRH'].data),
                W=(['time','x','y'],ss['W'].data),
                ),
            coords=dict(
                lon=(['x','y'],wrf_file['XLONG'].squeeze('Time').data),
                lat=(['x','y'],wrf_file['XLAT'].squeeze('Time').data),
                time=cape.time
                ),
            attrs=dict(description='HCW Occurrence')
            )
    
    null_ds = xr.Dataset(
            data_vars=dict(
                OCC=(['time','x','y'],null),
                CAPE=(['time','x','y'],cape['CAPE'].data),
                CIN=(['time','x','y'],cin['CIN'].data),
                SRH=(['time','x','y'],srh['SRH'].data),
                W=(['time','x','y'],ss['W'].data),
                ),
            coords=dict(
                lon=(['x','y'],wrf_file['XLONG'].squeeze('Time').data),
                lat=(['x','y'],wrf_file['XLAT'].squeeze('Time').data),
                time=cape.time
                ),
            attrs=dict(description='HCW Occurrence')
            )
    #print('Made it HERE')

    # Spatial masking
    occ_ds_mask = occ_ds.where((occ_ds['lon'] > -102.5) & (occ_ds['lon'] < -72) & (occ_ds['lat'] > 26) & (occ_ds['lat'] < 49))
    null_ds_mask = null_ds.where((null_ds['lon'] > -102.5) & (null_ds['lon'] < -72) & (null_ds['lat'] > 26) & (null_ds['lat'] < 49))
    #raise RuntimeError('Stop here')

    # Separate occurrences/null cases into two datasets
    occurrences = occ_ds_mask.where(occ_ds['OCC'] == 1.0,drop=True)
    null_cases = null_ds_mask.where(null_ds['OCC'] == 2.0,drop=True)

    # Convert datasets to dataframes
    occ_df = occurrences.to_dataframe()
    null_df = null_cases.to_dataframe()

    # Remove NaNs from each dataframe
    nulls = null_df.dropna(subset='OCC')
    occs = occ_df.dropna(subset='OCC').sample(frac=0.01,replace=False)
    print('Number of occurrences is',len(occs))

    # Randomly sample null cases dataframe
    null_final = nulls.sample(n=len(occs)*1000,replace=False).fillna(value=0)
    print('Number of null cases is',len(null_final))

    return occs,null_final

def ml_27(cape,cin,srh,wrf_file):
    # Inputs
    #  cape,cin,srh: Datasets with information for CAPE/CIN/SRH
    # Returns:
    #  occs,null_final: Dataframes with favorable candidates + null cases

    # Find favorable candidates
    output_shape = (len(cape.time),len(cape.x),len(cape.y))
    occ_final = np.zeros(output_shape, np.float32)

    land_mask = wrf_file['LANDMASK'].expand_dims({'time':len(cape.time)}).squeeze('Time').rename({'south_north':'x','west_east':'y'})
    #print(land_mask)
    #print(cape['CAPE'])
    #occ_masked = np.ma.masked_where((cape['CAPE'] > 500) & (cin['CIN'] < 100) & (srh['SRH'] > 100) & (land_mask == 1.0), occ_final)
    occ_masked = np.ma.masked_where((cape['CAPE'] > 500) & (cin['CIN'] < 100) & (srh['SRH'] > 100), occ_final)
    #print(output_shape)
    #print(len(occ_final.flatten()))
    #print(np.count_nonzero(occ==0))
    print('Number of favorable points is', len(occ_final.flatten()) - np.count_nonzero(occ_masked==0))
    occ = occ_masked.filled(fill_value = 1)
    print(occ.sum())

    # Assign to dataset with time/spatial coordinates
    occ_ds = xr.Dataset(
            data_vars=dict(
                OCC=(['time','x','y'],occ),
                CAPE=(['time','x','y'],cape['CAPE'].data),
                CIN=(['time','x','y'],cin['CIN'].data),
                SRH=(['time','x','y'],srh['SRH'].data),
                ),
            coords=dict(
                lon=(['x','y'],wrf_file['XLONG'].squeeze('Time').data),
                lat=(['x','y'],wrf_file['XLAT'].squeeze('Time').data),
                time=cape.time
                ),
            attrs=dict(description='HCW Occurrence')
            )

    # Spatial masking
    occ_ds_mask = occ_ds.where((occ_ds['lon'] > -102.5) & (occ_ds['lon'] < -72) & (occ_ds['lat'] > 26) & (occ_ds['lat'] < 49))
    print(occ_ds_mask['OCC'])
    print(np.nansum(occ_ds_mask['OCC']))
    #print(occ_ds_mask)
    # Separate occurrences/null cases into two datasets
    occurrences = occ_ds_mask.where(occ_ds['OCC'] == 1.0,drop=True)
    #null_cases = occ_ds.where(occ_ds['OCC'] == 0.0,drop=True)
    
    # Convert datasets to dataframes
    occ_df = occurrences.to_dataframe()
    #null_df = null_cases.to_dataframe()
    #null_df = null_df.where(null_df['W'] > 0.1)
    #print(occ_df)
    #print(null_df)

    # Remove NaNs from each dataframe
    #nulls = null_df.dropna(subset='OCC')
    occs = occ_df.dropna(subset='OCC')

    occs = occs.sample(frac=0.01,replace=False)
    
    # Randomly sample null cases dataframe
    #null_final = occs.sample(n=len(occs)*500,replace=False).fillna(value=0)

    print(occs)
    #print(null_final)
    #return occs,null_final
    #raise RuntimeError('Stop here')
    return occs

def ml_envs(cape,cin,srh,ss,occs,null_final,resolution):
    # Inputs
    #  cape,cin,srh,ss: Datasets with information for CAPE/CIN/SRH + storm-scale variable
    #  occs: Dataframe with HCW occurrence instances
    #  null_final:
    #  resolution:
    # Returns:
    #  envs,null_envs: Dataframes with occurrence + null case environments

    # Concatenate occurrences + null cases into one dataframe
    #print('Number of occurrences is:', len(occs))
    #print('Number of null cases is:', len(null_final))
    print('Running environments code')
    if 'null_final' not in locals():
        instances = occs
        #raise RuntimeError('Null finals not in locals')
    else:
        instances = pd.concat([occs,null_final])
        #raise RuntimeError('Null finals in locals')
    print(instances.index)

    # Set up constants/lists for environmental variables
    x_len = len(cape.lon)
    y_len = len(cape.lat)

    cape_list = []
    cin_list = []
    srh_list = []

    # Iterate over occurrences/null cases to obtain mean environmental variables nearby
    if resolution == 4:
        boundary_dist = 6 #<- Set this to determine tolerance from to use relative to lateral boundary edge
        for occ in instances.itertuples():
            index = occ[0]
            t = index[0]
            i = index[1]
            j = index[2]

            if (i - boundary_dist) < 0:
                x_min = 0
                x_max = i + 6
            elif (i + boundary_dist) >= x_len:
                x_min = i - 6
                x_max = x_len
            else:
                x_min = i - 6
                x_max = i + 6

            if (j - boundary_dist) < 0:
                y_min = 0
                y_max = j + 6
            elif (j + boundary_dist) >= y_len:
                y_min = j - 6
                y_max = y_len
            else:
                y_min = j - 6
                y_max = j + 6

            loc_cape = cape.isel(x=slice(x_min,x_max+1),y=slice(y_min,y_max+1))
            loc_cin = cin.isel(x=slice(x_min,x_max+1),y=slice(y_min,y_max+1))
            loc_srh = srh.isel(x=slice(x_min,x_max+1),y=slice(y_min,y_max+1))

            prox_cape = loc_cape.mean()
            prox_cin = loc_cin.mean()
            prox_srh = loc_srh.mean()

            cape_list.append(float(prox_cape['CAPE'].values))
            cin_list.append(float(prox_cin['CIN'].values))
            srh_list.append(float(prox_srh['SRH'].values))
        print('Successfully completed processing of environments')
    elif resolution == 27:
        boundary_dist = 1 #<- Set this to determine tolerance from to use relative to lateral boundary edge
        for occ in instances.itertuples():
            index = occ[0]
            t = index[0]
            i = index[1]
            j = index[2]

            if (i - boundary_dist) < 0:
                x_min = 0
                x_max = i + 1
            elif (i + boundary_dist) >= x_len:
                x_min = i - 1
                x_max = x_len
            else:
                x_min = i - 1
                x_max = i + 1

            if (j - boundary_dist) < 0:
                y_min = 0
                y_max = j + 1
            elif (j + boundary_dist) >= y_len:
                y_min = j - 1
                y_max = y_len
            else:
                y_min = j - 1
                y_max = j + 1

            loc_cape = cape.isel(x=slice(x_min,x_max+1),y=slice(y_min,y_max+1))
            loc_cin = cin.isel(x=slice(x_min,x_max+1),y=slice(y_min,y_max+1))
            loc_srh = srh.isel(x=slice(x_min,x_max+1),y=slice(y_min,y_max+1))

            prox_cape = loc_cape.mean()
            prox_cin = loc_cin.mean()
            prox_srh = loc_srh.mean()

            cape_list.append(float(prox_cape['CAPE'].values))
            cin_list.append(float(prox_cin['CIN'].values))
            srh_list.append(float(prox_srh['SRH'].values))

    #raise RuntimeError('Stop it here')
    #lons = np.concatenate([occs.lon.values,null_final.lon.values])
    #lats = np.concatenate([occs.lat.values,null_final.lat.values])
    lons = instances.lon.values
    lats = instances.lat.values

    env_values = {'CAPE':cape_list,'CIN':cin_list,'SRH':srh_list,'lon':lons,'lat':lats}
    envs = pd.DataFrame(data=env_values,index=instances.index).fillna(value=0)
    if resolution == 4:
        occ_envs = envs.iloc[:len(occs),:]
        null_envs = envs.iloc[len(occs):,:]
        print(len(occ_envs))
        print(len(null_envs))
    elif resolution == 27:
        occ_envs = envs
        null_envs = None
        print(len(occ_envs))
    return occ_envs,null_envs

#def ml_trim(df):
#    df_trim = df.drop(df.where(([df['lat'] > 26) & (df['lat'] < 49) & (df['lon'] > -102.5) & (df['lon'] < -72))
#    return df_trim

def occs_predict(cape,cin,srh,ss,wrf_file):
    # Inputs
    #  cape,cin,srh,ss: Datasets with information for CAPE/CIN/SRH + storm-scale variable
    # Returns:
    #  occs,null_final: Dataframes with occurrences + null cases

    # Find occurrences
    print(len(cape.x),len(cape.y))
    output_shape = (len(cape.time),len(cape.x),len(cape.y))
    occ_final = null_final = np.zeros(output_shape, np.float32)

    for i in range(10):
        try:
            occ_masked = np.ma.masked_where((cape['CAPE'] > 500) & (cin['CIN'] < 100) & (srh['SRH'] > 100) & (ss['W'] > 18), occ_final)
            null_masked = np.ma.masked_where((cape['CAPE'] > 500) & (cin['CIN'] < 100) & (srh['SRH'] > 100) & (ss['W'] < 18), null_final)
            break
        except ValueError:
            continue

    occ = occ_masked.filled(fill_value=1)
    null = null_masked.filled(fill_value=2)
    print('Created masked arrays')

    # Assign to dataset with time/spatial coordinates
    occ_ds = xr.Dataset(
            data_vars=dict(
                OCC=(['time','x','y'],occ),
                CAPE=(['time','x','y'],cape['CAPE'].data),
                CIN=(['time','x','y'],cin['CIN'].data),
                SRH=(['time','x','y'],srh['SRH'].data),
                W=(['time','x','y'],ss['W'].data),
                ),
            coords=dict(
                lon=(['x','y'],wrf_file['XLONG'].squeeze('Time').data),
                lat=(['x','y'],wrf_file['XLAT'].squeeze('Time').data),
                time=cape.time
                ),
            attrs=dict(description='HCW Occurrence')
            )
    
    null_ds = xr.Dataset(
            data_vars=dict(
                OCC=(['time','x','y'],null),
                CAPE=(['time','x','y'],cape['CAPE'].data),
                CIN=(['time','x','y'],cin['CIN'].data),
                SRH=(['time','x','y'],srh['SRH'].data),
                W=(['time','x','y'],ss['W'].data),
                ),
            coords=dict(
                lon=(['x','y'],wrf_file['XLONG'].squeeze('Time').data),
                lat=(['x','y'],wrf_file['XLAT'].squeeze('Time').data),
                time=cape.time
                ),
            attrs=dict(description='HCW Null Cases from Favorable Environment')
            )
    print('Created datasets')

    # Separate occurrences/null cases into two datasets
    occurrences = occ_ds.where(occ_ds['OCC'] == 1.0,drop=True)
    null_cases = null_ds.where(null_ds['OCC'] == 2.0,drop=True)
    
    # Convert datasets to dataframes
    occ_df = occurrences.to_dataframe()
    null_df = null_cases.to_dataframe()
    print('Converted to dataframes')

    # Remove NaNs from each dataframe
    nulls = null_df.dropna(subset='OCC')
    occs = occ_df.dropna(subset='OCC')

    # Randomly sample null cases dataframe
    nulls_final = nulls.sample(n=len(occs),replace=False).fillna(value=0)
    nulls_final['OCC'] = nulls_final['OCC'].replace(2.0,0.0)
    print('Finished sampling')
    print(nulls_final)
    print(occs)
    #raise RuntimeError('Stop here')
    return occs,nulls_final

