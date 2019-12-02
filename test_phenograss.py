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

all_results = pd.DataFrame()

for site_file in site_files:
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


    ###########################
    # convert year + day of year to actual date object
    site_data['date'] = pd.to_datetime(site_data.year, format='%Y') + pd.to_timedelta(site_data.doy-1, unit='d')
    site_data['month'] = site_data.date.dt.month

    # Aggregate by month
    #site_data = single_site.groupby(['year','month']).agg({'gcc':'mean','et':'sum','prcp':'sum'}).reset_index()
    m = PhenoGrass()
    
    #fitting_params = {'maxiter':200,
    # 'popsize':100,
    # 'mutation':(0.5,1),
    # 'recombination':0.25,
    # 'disp':True}
    #
    #def na_rmse_loss(obs, pred):
    #    return np.sqrt(np.nanmean((obs - pred)**2))
    
    #m = CholerLinear(parameters={'a1':(0,500),'a2':(0,500),
    #                             'a3':(0,500),'L':(0,3)})
    #m.fit(observations=single_site.gcc.values,
    #      predictors={'precip':single_site.prcp.values,
    #                  'evap':single_site.et.values,
    #                  'Wcap': np.array([422.4]),
    #                  'Wp':   np.array([158.5])},
    #    loss_function=na_rmse_loss,
    #     debug=True, optimizer_params=fitting_params)
    #
    #single_site['modelled_gcc'] = m.predict()
    
    koens_phenograss_params = {'b1':124.502121,
                               'b2':0.00227958267,
                               'b3':0.0755224228,
                               'b4':0.519348383,
                               'L':2.4991734,
                               'Phmin':8.14994431,
                               'h': 222.205673,
                               'Topt':33.3597641,
                               'Phmax':37.2918091}

    V, W, Dt = m._apply_model(precip = site_data.prcp.values,
                                       evap   = site_data.et.values,
                                       Tm     = site_data.tmean_15day.values,
                                       Ra     = site_data.radiation.values,
                                       Wcap = site_metadata['WCAP'],
                                       Wp   = site_metadata['WP'],
                                       V_initial=0.01,
                                       **koens_phenograss_params,
                                       return_vars='all')
    site_data['modelled_gcc'] = V
    
    # Scale the observed gcc using eq. 1. Need to calculated MAP
    
    MAP = site_data.groupby('year')['prcp'].agg('sum').mean()
    scaling_factor = MAP / (MAP + koens_phenograss_params['h'])

    # The model output is already scaled, so just need to scale the observed
    # gcc to the same range, which would have been used as model training data
    site_data['scaled_gcc'] = site_data.gcc * scaling_factor
    
    all_results = all_results.append(site_data)

all_results.to_csv('phenograss_test_runs.csv', index=False)
    
#plt.plot(np.arange(V.shape[0]), V, 'o',color='black')
#    
#(ggplot(single_site, aes('doy','gcc', color='factor(year)')) 
#  +geom_point())
#
#(ggplot(single_site, aes('doy','radiation', color='factor(year)')) 
#  +geom_point())