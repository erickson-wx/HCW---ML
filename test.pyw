import sys,os
import glob
import time
import xarray as xr
import numpy as np
import pandas as pd
import itertools

#from netCDF4 import Dataset
from wrf import getvar

def ml_inputs(epoch,resolution,year,option):
    # Inputs:
    #  epoch: The epoch for which to prepare ML inputs
    #  resolution: The resolution of simulation being handled
    # Returns:
    #  None (Outputs input data for ML model to file)

    # Format resolution for later use
    resolution = int(resolution)
    
    # Set directory structure
    base_dir = '/pscratch/sd/n/nee2000/WRF-Prod'
    if resolution == 4:
        input_dir = os.path.join(base_dir + '/' + f'{epoch}/WRF-Monthly')
    elif resolution == 27:
        input_dir = os.path.join(base_dir + '/' + f'27km/{epoch}/WRF-Monthly')
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
        except (RuntimeError,OSError):
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
        #occs.to_csv(f'occs_{resolution}_{year}.csv')
        #null_final.to_csv(f'null_cases_{resolution}_{year}.csv')
    else:
        occs,null_final = ml_27(cape,cin,srh,wrf_file)
        os.chdir(out_dir)
        #occs.to_csv(f'occs_{resolution}_{year}.csv')
        #null_final.to_csv(f'null_cases_{resolution}_{year}.csv')
    print('Successfully processed occurrences/null cases')
   
   # Processing environments
    envs,null_envs = ml_envs(cape,cin,srh,ss,occs,null_final,resolution)
    print(f'Environment data ready to return to main program for {year}')
    print('------------------------------------------------------------')
    print('Successfully processed environments surrounding occurrences')

    # Send dataframes to files
    #envs.to_csv(f'occ_envs_{resolution}_{year}.csv')
    #null_envs.to_csv(f'null_envs_{resolution}_{year}.csv')
    print(f'Files output to', os.getcwd())

def ml_occs(cape,cin,srh,ss,wrf_file):
    # Inputs
    #  cape,cin,srh,ss: Datasets with information for CAPE/CIN/SRH + storm-scale variable
    # Returns:
    #  occs,null_final: Dataframes with occurrences + null cases
    start_time = time.time()
    # Find occurrences
    #print(len(cape.x),len(cape.y))
    output_shape = (len(cape.time),len(cape.x),len(cape.y))
    occ_final = np.zeros(output_shape, np.float32)

    occ_masked = np.ma.masked_where((cape['CAPE'] > 500) & (cin['CIN'] < 100) & (srh['SRH'] > 100) & (ss['W'] > 18), occ_final)
    #occ_masked = np.ma.masked_where(cin['CIN'] < 100, occ_final)
    #print(len(occ_final.flatten()))
    #print(np.count_nonzero(occ_masked==0))
    #print(occ_masked)

    occ = occ_masked.filled(fill_value=1)

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

    # Separate occurrences/null cases into two datasets
    occurrences = occ_ds.where(occ_ds['OCC'] == 1.0,drop=True)
    null_cases = occ_ds.where(occ_ds['OCC'] == 0.0,drop=True)

    # Convert datasets to dataframes
    occ_df = occurrences.to_dataframe()
    null_df = null_cases.to_dataframe()
    #null_df = null_df.where(null_df['W'] > 0.1)

    # Remove NaNs from each dataframe
    nulls = null_df.dropna(subset=['OCC','W'])
    occs = occ_df.dropna(subset='OCC').sample(frac=0.01,replace=False)
    #print('Proportion of occurrences to null cases w/ > 0.1 m/s UVV is', len(occs)/len(nulls))

    # Randomly sample null cases dataframe
    null_final = nulls.sample(n=len(occs)*1000,replace=False).fillna(value=0)
    print('Number of occurrences = ', len(occs))
    print('Number of null cases = ', len(null_final))
    print('Time to run occurrences code is', time.time() - start_time)
    #raise RuntimeError('Stop here')

    return occs,null_final

def ml_27(cape,cin,srh,wrf_file):
    # Inputs
    #  cape,cin,srh: Datasets with information for CAPE/CIN/SRH
    # Returns:
    #  occs,null_final: Dataframes with favorable candidates + null cases
    start_time = time.time()
    # Find favorable candidates
    output_shape = (len(cape.time),len(cape.x),len(cape.y))
    occ_final = np.zeros(output_shape, np.float32)

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

    # Separate occurrences/null cases into two datasets
    occurrences = occ_ds.where(occ_ds['OCC'] == 1.0,drop=True)
    null_cases = occ_ds.where(occ_ds['OCC'] == 0.0,drop=True)
    
    # Convert datasets to dataframes
    occ_df = occurrences.to_dataframe()
    null_df = null_cases.to_dataframe()
    #null_df = null_df.where(null_df['W'] > 0.1)
    #print(occ_df)
    #print(null_df)

    # Remove NaNs from each dataframe
    nulls = null_df.dropna(subset='OCC')
    occs = occ_df.dropna(subset='OCC')

    occs = occs.sample(frac=0.0001,replace=False)
    
    # Randomly sample null cases dataframe
    null_final = nulls.sample(n=len(occs)*1000,replace=False).fillna(value=0)
    print('Number of occurrences = ', len(occs))
    print('Number of null cases = ', len(null_final))
    print('Time to run occurrences code is', time.time() - start_time)
    #raise RuntimeError('Stop here')

    #print(occ_ds)
    #print(null_final)
    return occs,null_final

def ml_envs(cape,cin,srh,ss,occs,null_final,resolution):
    # Inputs
    #  cape,cin,srh,ss: Datasets with information for CAPE/CIN/SRH + storm-scale variable
    #  occs: Dataframe with HCW occurrence instances
    #  null_final:
    #  resolution:
    # Returns:
    #  envs,null_envs: Dataframes with occurrence + null case environments

    # Concatenate occurrences + null cases into one dataframe
    start_time = time.time()
    print('Number of occurrences is:', len(occs))
    print('Number of null cases is:', len(null_final))
    instances = pd.concat([occs,null_final])
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
    lons = np.concatenate([occs.lon.values,null_final.lon.values])
    lats = np.concatenate([occs.lat.values,null_final.lat.values])
    
    env_values = {'CAPE':cape_list,'CIN':cin_list,'SRH':srh_list,'lon':lons,'lat':lats}
    envs = pd.DataFrame(data=env_values,index=instances.index).fillna(value=0)
    occ_envs = envs.iloc[:len(occs),:]
    null_envs = envs.iloc[len(occs):,:]
    print(len(occ_envs))
    print(len(null_envs))
    print('Time to run environments code is', time.time() - start_time)
    raise RuntimeError('Stop here')
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

