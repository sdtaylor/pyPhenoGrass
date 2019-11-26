import pandas as pd
import numpy as np


def read_example_data(filepath):
    df = pd.read_csv(filepath, skiprows=14, sep='\t',
                     names = ['year', 'doy', 'validation', 'gcc', 'tmax', 'tmin',
                              'prcp', 'day_length', 'incident_radiation',
                              'snow_water_equivalent', 'VPD'])
    
    return df

single_site = read_example_data('data/ibp_grass.csv')
single_site = single_site[single_site.year == 2013]

site_lat = utils.deg2rad(np.array([33.1]))
site_Wcap = 351.429
site_WP   = 106.47399

gcc_mean = np.nanmean(single_site.gcc.replace(-9999, np.nan))
single_site['gcc'] = single_site.gcc.replace(-9999, gcc_mean)

from GrasslandModels.models.choler import CholerLinear
from GrasslandModels import utils


# Estimate ET from tmin, tmax adn latitude
solar_dec = utils.sol_dec(single_site.doy.values)
sha = utils.sunset_hour_angle(utils.deg2rad(33.1), solar_dec)
ird = utils.inv_rel_dist_earth_sun(single_site.doy.values)
radiation = utils.et_rad(site_lat, solar_dec, sha, ird)
et = utils.hargreaves(tmin = single_site.tmin.values, tmax = single_site.tmax.values, et_rad = radiation)


m = CholerLinear()
m._apply_model(precip = single_site.prcp.values,
               evap   = et,
               Wcap = site_Wcap,
               Wp   = site_WP,
               a1 = 1.97,
               a2 = 13.73,
               a3 = 71.58,
               L = 2)
        
    


