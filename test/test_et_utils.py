import pytest
import numpy as np
from GrasslandModels import et_utils
#                   lethbridge,  IBP,        vaira
latitude = np.array([49.7, 49.7, 32.6,  32.6, 38.4, 38.4])
#1980,so a leap year july31, jan15,jan13, aug8, feb22,may13
doy      = np.array([213,    15,   14,   221,   53,  134])
tmax     = np.array([29.5,    5,   20,    36,   15,  21.5])
tmin     = np.array([7.5,  -5.5,  0.5,    18,  7.5,   10])


def test_hargreaves():
    """
    This is just a check to confirm the model math is correct.
    True ET calculated using the above numbers but on https://asianturfgrass.shinyapps.io/ET_calculator/
    """
    true_et = np.array([6, 0.48, 2.2, 6.9, 1.7, 4.2])
    
    latitude_radians = et_utils.deg2rad(latitude)
    solar_dec = et_utils.sol_dec(doy)
    sha = et_utils.sunset_hour_angle(latitude_radians, solar_dec)
    ird = et_utils.inv_rel_dist_earth_sun(doy)
    radiation = et_utils.et_rad(latitude_radians, solar_dec, sha, ird)
    ET = et_utils.hargreaves(tmin = tmin, tmax = tmax, et_rad = radiation)
    
    # Just make sure the numbers are close
    assert np.all(np.abs(np.round(true_et - ET,2)) < 0.1)