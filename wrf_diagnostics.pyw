import wrf
import numpy as np

def cape_cin(wrf_file):
    # Calculates CAPE/CIN from WRF file input
    # Inputs:
    #  file: The WRF file with variables
    # Returns:
    #  CAPE: Data array containing CAPE/CIN
     
    # Assign constants & variables
    pressure = wrf_file['P']
    temperature = wrf_file['T']
    specific_humidity = wrf_file['QVAPOR']
    geopotential_morph = wrf_file['PH'].drop_sel(bottom_top_stag = 0)
    terrain = wrf_file['HGT']
    surface_pressure = wrf_file['PSFC']
    
    # Perform CAPE calculation
    cape = wrf.cape_3d(pressure,temperature,specific_humidity,geopotential_morph,terrain,surface_pressure,False)
    
    return cape

def dbz(wrf_file):
    # Calculates radar reflectivity from WRF file input
    # Inputs:
    #  file: The WRF file with variables
    # Returns:
    #  DBZ: Data array containing DBZ
    
    # Assign constants & variables
    pressure = wrf_file['P']
    temperature = wrf_file['T']
    specific_humidity = wrf_file['QVAPOR']
    rain_mixing_ratio = wrf_file['QRAIN']
    snow_mixing_ratio = wrf_file['QSNOW']
    graupel_mixing_ratio = wrf_file['QGRAUP']
    
    # Perform DBZ calculation
    dbz = wrf.dbz(pressure,temperature,specific_humidity,rain_mixing_ratio,snow_mixing_ratio,graupel_mixing_ratio,False) # <- Check the Boolean option here; has something to do with parameterizations
    
    return dbz

def srhel(wrf_file,top):
    # Calculates storm-relative helicity from WRF file input
    # Inputs:
    #  file: The WRF file with variables
    #  top: Upper bound of the layer in which to calculate SRH
    # Returns:
    #  SRH: Data array containing storm-relative helicity 
    
    # Assign constants & variables; interpolate u and v winds to WRF's mass grid
    map_fac = wrf_file['MAPFAC_M']
    u_wind = wrf_file['U'].interp(south_north = map_fac.south_north, west_east_stag = map_fac.west_east)
    v_wind = wrf_file['V'].interp(south_north_stag = map_fac.south_north, west_east = map_fac.west_east)
    geopotential_morph = wrf_file['PH'].drop_sel(bottom_top_stag = 0)
    terrain = wrf_file['HGT']
    
    # Perform storm-relative helicity calculation
    srhel = wrf.srhel(u_wind,v_wind,geopotential_morph,terrain,top) # <- How do I want to set the layer top?
    
    return srhel

def udhel(wrf_file):
    # Calculates updraft helicity from WRF file input
    # Inputs:
    #  file: The WRF file with variables
    # Returns:
    #  UH: Data array containing updraft helicity 
    
    # Assign constants & variables
    dx = dy = 27000
    geopotential = wrf_file['PH']
    map_fac = wrf_file['MAPFAC_M']
    u_wind = wrf_file['U'].interp(south_north = map_fac.south_north, west_east_stag = map_fac.west_east)
    v_wind = wrf_file['V'].interp(south_north = map_fac.south_north_stag, west_east = map_fac.west_east)
    
    # Perform updraft helicity calculation
    udhel = wrf.udhel(geopotential,map_fac,u_wind,v_wind,dx,dy)
    
    return udhel

def nearest_height(array, value):
    # Finds the height nearest to a specified value from a list of heights
    # Inputs:
    #  array: The array containing height values
    #  value: The height value to search nearest to
    # Returns:
    #  array[idx]: The height value closest to the specified value
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def wind_shear(wrf_file):
    # Calculates the wind shear between two prescribed height surfaces
    # Inputs:
    #  wrf_file: The WRF file with variables
    # Returns:
    #  wind_shear: Data array containing wind shear

    # Convert WRF pressure levels to heights; determine the upper and lower height bounds
    heights = [pressure_to_height_std(wrf_file['PB'][0]).values[index][0][0] for index in range(len(wrf_file['PB']['bottom_top']))]
    lower_bound = np.min(heights)
    upper_bound = np.min(nearest_height(heights,6))

    lower_idx = np.where(heights == lower_bound)
    upper_idx = np.where(heights == upper_bound)

    # Calculate wind shear between two prescribed levels
