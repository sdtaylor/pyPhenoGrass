import pandas as pd
import numpy as np
from plotnine import *

from GrasslandModels.models.phenograss import PhenoGrass
from GrasslandModels import utils

def read_example_data(filepath):
    df = pd.read_csv(filepath, skiprows=14, sep='\t',
                     names = ['year', 'doy', 'validation', 'gcc', 'tmax', 'tmin',
                              'prcp', 'day_length', 'incident_radiation',
                              'snow_water_equivalent', 'VPD'])
    
    return df

single_site = read_example_data('data/lethbridge_grass.csv')

site_lat = utils.deg2rad(np.array([49.7092]))
site_Wcap = 422.429
site_WP   = 158.47399

single_site['gcc'] = single_site.gcc.replace(-9999, np.nan)


# Estimate ET from tmin, tmax adn latitude
solar_dec = utils.sol_dec(single_site.doy.values)
sha = utils.sunset_hour_angle(site_lat, solar_dec)
ird = utils.inv_rel_dist_earth_sun(single_site.doy.values)
single_site['radiation'] = utils.et_rad(site_lat, solar_dec, sha, ird)
single_site['et'] = utils.hargreaves(tmin = single_site.tmin.values, tmax = single_site.tmax.values, 
                                     et_rad = single_site.radiation.values)

single_site['tmean'] = (single_site.tmin + single_site.tmax)/2
single_site['tmean_15day'] = np.nan
for i in range(16,len(single_site)):
    single_site['tmean_15day'][i] = np.mean(single_site.tmean[i-15:i].values)


single_site = single_site[single_site.year.isin([2012,2013])]


###########################
# convert year + day of year to actual date object
single_site['date'] = pd.to_datetime(single_site.year, format='%Y') + pd.to_timedelta(single_site.doy-1, unit='d')
single_site['month'] = single_site.date.dt.month

# Aggregate by month
#single_site = single_site.groupby(['year','month']).agg({'gcc':'mean','et':'sum','prcp':'sum'}).reset_index()

fitting_params = {'maxiter':200,
 'popsize':100,
 'mutation':(0.5,1),
 'recombination':0.25,
 'disp':True}

def na_rmse_loss(obs, pred):
    return np.sqrt(np.nanmean((obs - pred)**2))

m = PhenoGrass()
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

single_site['modelled_gcc']=m._apply_model(precip = single_site.prcp.values,
                                           evap   = single_site.et.values,
                                           Tm     = single_site.tmean_15day.values,
                                           Ra     = single_site.radiation.values,
                                           Wcap = site_Wcap,
                                           Wp   = site_WP,
                                           V_initial=0.1,
                                           **koens_phenograss_params)
        
    
(ggplot(single_site, aes('month','modelled_gcc', color='factor(year)')) 
  +geom_point())

(ggplot(single_site, aes('doy','radiation', color='factor(year)')) 
  +geom_point())