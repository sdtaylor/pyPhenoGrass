import pandas as pd
import numpy as np
from plotnine import *

from GrasslandModels.models.choler import CholerLinear
from GrasslandModels import utils

def read_example_data(filepath):
    df = pd.read_csv(filepath, skiprows=14, sep='\t',
                     names = ['year', 'doy', 'validation', 'gcc', 'tmax', 'tmin',
                              'prcp', 'day_length', 'incident_radiation',
                              'snow_water_equivalent', 'VPD'])
    
    return df

single_site = read_example_data('data/lethbridge_grass.csv')
single_site = single_site[single_site.year.isin([2012,2013])]

site_lat = utils.deg2rad(np.array([49.4]))
site_Wcap = 422.429
site_WP   = 158.47399

single_site['gcc'] = single_site.gcc.replace(-9999, np.nan)


# Estimate ET from tmin, tmax adn latitude
solar_dec = utils.sol_dec(single_site.doy.values)
sha = utils.sunset_hour_angle(site_lat, solar_dec)
ird = utils.inv_rel_dist_earth_sun(single_site.doy.values)
radiation = utils.et_rad(site_lat, solar_dec, sha, ird)
single_site['et'] = utils.hargreaves(tmin = single_site.tmin.values, tmax = single_site.tmax.values, et_rad = radiation)

###########################
# convert year + day of year to actual date object
single_site['date'] = pd.to_datetime(single_site.year, format='%Y') + pd.to_timedelta(single_site.doy-1, unit='d')
single_site['month'] = single_site.date.dt.month

# Aggregate by month
single_site = single_site.groupby(['year','month']).agg({'gcc':'mean','et':'mean','prcp':'sum'}).reset_index()

fitting_params = {'maxiter':200,
 'popsize':200,
 'mutation':(0.5,1),
 'recombination':0.25,
 'disp':True}

def na_rmse_loss(obs, pred):
    return np.sqrt(np.nanmean((obs - pred)**2))

m = CholerLinear()
m.fit(observations=single_site.gcc.values,
      predictors={'precip':single_site.prcp.values,
                  'evap':single_site.et.values,
                  'Wcap': np.array([422.4]),
                  'Wp':   np.array([158.5])},
    loss_function=na_rmse_loss,
     debug=True, optimizer_params=fitting_params)

single_site['modelled_gcc'] = m.predict()

#single_site['modelled_gcc']=m._apply_model(precip = single_site.prcp.values,
#                                           evap   = single_site.et.values,
#                                           Wcap = site_Wcap,
#                                           Wp   = site_WP,
#                                           a1 = 1.97,
#                                           a2 = 13.73,
#                                           a3 = 71.58,
#                                           L = 2)
        
    
(ggplot(single_site, aes('month','modelled_gcc', color='factor(year)')) 
  +geom_point())

