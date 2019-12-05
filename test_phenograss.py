import pandas as pd
import numpy as np
from plotnine import *
import matplotlib.pyplot as plt

from GrasslandModels.models.phenograss import PhenoGrass
from GrasslandModels import utils

def read_example_data(filepath):
    df = pd.read_csv(filepath, skiprows=14, sep='\t',
                     names = ['year', 'doy', 'validation', 'gcc', 'tmax', 'tmin',
                              'prcp', 'day_length', 'incident_radiation',
                              'snow_water_equivalent', 'VPD'])
    
    #df = df[df.year.isin([2008,2009,2010,2011,2012,2013])]
    #df = df[df.year.isin([2013])]
    return df

def read_example_metdata(filepath):
    # read in the site metadata from the phenograss-example
    with open(filepath) as f:
        file_lines = [f.readlines(1) for i in range(7)]
    
    metadata = {}
    for attribute in file_lines:
        cleaned = attribute[0].strip('#').strip('\n').strip().split(':')
        try:
            cleaned[1] = float(cleaned[1].strip())
        except:
            pass
        
        metadata[cleaned[0]] = cleaned[1]

    return metadata

site_files = ['freemangrass_grass.csv','ibp_grass.csv','kansas_grass.csv',
              'lethbridge_grass.csv','marena_grass.csv','vaira_grass.csv']

#site_files = ['freemangrass_grass.csv','ibp_grass.csv','kansas_grass.csv']


test_data = read_example_data('data/'+site_files[0]) # read 1 file to get dimensions
n_sites = len(site_files)
timeseries_length = test_data.shape[0]

# initialize predictor arrays
precip = np.zeros((timeseries_length, n_sites))
evap   = np.zeros((timeseries_length, n_sites))
Ra     = np.zeros((timeseries_length, n_sites))
Tm     = np.zeros((timeseries_length, n_sites))
MAP    = np.zeros((n_sites))
Wcap   = np.zeros((n_sites))
Wp     = np.zeros((n_sites))

GCC    = np.zeros((n_sites, timeseries_length))

for site_i, site_file in enumerate(site_files):
    site_metadata = read_example_metdata('./data/'+site_file)
    site_data = read_example_data('./data/'+site_file)
    
    site_data['Site'] = site_metadata['Site']
    
    #site_lat = utils.deg2rad(np.array([32.589]))
    #site_Wcap = 351.419006347656 
    #site_WP   = 106.473999023438 

    site_data['gcc'] = site_data.gcc.replace(-9999, np.nan)
    site_metadata['Lat_r'] = utils.deg2rad(site_metadata['Lat'])

    # Estimate ET from tmin, tmax adn latitude
    solar_dec = utils.sol_dec(site_data.doy.values)
    sha = utils.sunset_hour_angle(site_metadata['Lat_r'], solar_dec)
    ird = utils.inv_rel_dist_earth_sun(site_data.doy.values)
    site_data['radiation'] = utils.et_rad(site_metadata['Lat_r'], solar_dec, sha, ird)
    site_data['et'] = utils.hargreaves(tmin = site_data.tmin.values, tmax = site_data.tmax.values, 
                                         et_rad = site_data.radiation.values)

    # A running 15 day avg using only the prior 15 days
    site_data['tmean'] = (site_data.tmin + site_data.tmax)/2
    site_data['tmean_15day'] = np.nan
    for i in range(16,len(site_data)):
        site_data['tmean_15day'][i] = np.mean(site_data.tmean[i-15:i].values)
    
    
    #site_data = site_data[site_data.year.isin([2008,2009,2010,2011,2012,2013])]

    site_MAP = site_data.groupby('year')['prcp'].agg('sum').mean()

    ###########################
    # convert year + day of year to actual date object
    site_data['date'] = pd.to_datetime(site_data.year, format='%Y') + pd.to_timedelta(site_data.doy-1, unit='d')
    site_data['month'] = site_data.date.dt.month

    precip[:,site_i] = site_data.prcp.values
    evap[:,site_i]   = site_data.et.values
    Tm[:,site_i]     = site_data.tmean_15day.values
    Ra[:,site_i]     = site_data.radiation.values
    MAP[site_i]    = site_MAP
    Wcap[site_i]   = site_metadata['WCAP']
    Wp[site_i]     = site_metadata['WP']
    


# Aggregate by month
#site_data = single_site.groupby(['year','month']).agg({'gcc':'mean','et':'sum','prcp':'sum'}).reset_index()
m = PhenoGrass()

koens_phenograss_params = {'b1':124.502121,
                           'b2':0.00227958267,
                           'b3':0.0755224228,
                           'b4':0.519348383,
                           'L':2.4991734,
                           'Phmin':8.14994431,
                           'h': 222.205673,
                           'Topt':33.3597641,
                           'Phmax':37.2918091}

V, W, Dt = m._apply_model(precip = precip,
                           evap   = evap,
                           Tm     = Tm,
                           Ra     = Ra,
                           MAP    = MAP,
                           Wcap = Wcap,
                           Wp   = Wp,
                           V_initial=0.01,
                           **koens_phenograss_params,
                           return_vars='all')

# put the modelled GCC back into site files
all_results = pd.DataFrame()

for site_i, site_file in enumerate(site_files):
    site_metadata = read_example_metdata('./data/'+site_file)
    site_data = read_example_data('./data/'+site_file)
    
    site_data['date'] = pd.to_datetime(site_data.year, format='%Y') + pd.to_timedelta(site_data.doy-1, unit='d')
    site_data['gcc'] = site_data.gcc.replace(-9999, np.nan)
    site_data['modelled_gcc'] = V[:,site_i]
    site_data['Site'] = site_metadata['Site']
    all_results = all_results.append(site_data)

all_results.to_csv('phenograss_test_runs_vectorized.csv', index=False)
