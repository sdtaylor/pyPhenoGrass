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
single_site = single_site[single_site.year == 2013]

site_lat = utils.deg2rad(np.array([49.4]))
site_Wcap = 422.429
site_WP   = 158.47399

#gcc_mean = np.nanmean(single_site.gcc.replace(-9999, np.nan))
single_site['gcc'] = single_site.gcc.replace(-9999, np.nan)


# Estimate ET from tmin, tmax adn latitude
solar_dec = utils.sol_dec(single_site.doy.values)
sha = utils.sunset_hour_angle(site_lat, solar_dec)
ird = utils.inv_rel_dist_earth_sun(single_site.doy.values)
radiation = utils.et_rad(site_lat, solar_dec, sha, ird)
et = utils.hargreaves(tmin = single_site.tmin.values, tmax = single_site.tmax.values, et_rad = radiation)

fitting_params = {'maxiter':200,
 'popsize':50,
 'mutation':(0.5,1),
 'recombination':0.25,
 'disp':True}

def na_rmse_loss(obs, pred):
    return np.sqrt(np.nanmean((obs - pred)**2))

m = CholerLinear(parameters={'L':2})
m.fit(observations=single_site.gcc.values,
      predictors={'precip':single_site.prcp.values,
                  'evap':et,
                  'Wcap': np.array([422.4]),
                  'Wp':   np.array([158.5])},
    loss_function=na_rmse_loss,
     debug=True, optimizer_params=fitting_params)


single_site['modelled_gcc']=m._apply_model(precip = single_site.prcp.values,
                                           evap   = et,
                                           Wcap = site_Wcap,
                                           Wp   = site_WP,
                                           a1 = 1.97,
                                           a2 = 13.73,
                                           a3 = 71.58,
                                           L = 2)
        
    
(ggplot(single_site, aes('doy','modelled_gcc')) 
  +geom_point())

