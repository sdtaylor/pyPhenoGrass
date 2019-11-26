import numpy as np

"""
The following evapotranspiration helper functions are from https://github.com/woodcrafty/PyETo
and modified slightly to work with numpy arrays.
Copied/derived under the following license:
    
----------------------------------------
Copyright (c) 2015, Mark Richards

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""


def deg2rad(degrees):
    """
    Convert angular degrees to radians

    :param degrees: Value in degrees to be converted.
    :return: Value in radians
    :rtype: float
    """
    return degrees * (np.pi / 180.0)

# Internal constants
    
#: Solar constant [ MJ m-2 min-1]
SOLAR_CONSTANT = 0.0820

# Stefan Boltzmann constant [MJ K-4 m-2 day-1]
STEFAN_BOLTZMANN_CONSTANT = 0.000000004903  #
"""Stefan Boltzmann constant [MJ K-4 m-2 day-1]"""


# Latitude
_MINLAT_RADIANS = deg2rad(-90.0)
_MAXLAT_RADIANS = deg2rad(90.0)

# Solar declination
_MINSOLDEC_RADIANS = deg2rad(-23.5)
_MAXSOLDEC_RADIANS = deg2rad(23.5)

# Sunset hour angle
_MINSHA_RADIANS = 0.0
_MAXSHA_RADIANS = deg2rad(180)

def check_latitude_rad(latitude):
    if not np.logical_and(latitude >= _MINLAT_RADIANS, latitude <= _MAXLAT_RADIANS).all():
        raise ValueError('latitude outside valid range')

def check_sol_dec_rad(sd):
    """
    Solar declination can vary between -23.5 and +23.5 degrees.

    See http://mypages.iit.edu/~maslanka/SolarGeo.pdf
    """
    if not np.logical_and(sd >= _MINSOLDEC_RADIANS, sd <= _MAXSOLDEC_RADIANS).all():
        raise ValueError('solar declination outside valid range')

def check_doy(doy):
    """
    Check day of the year is valid.
    """
    if not np.logical_and(doy >= 1, doy <= 366).all():
        raise ValueError('day of year should be in the range 1-366')

def check_sunset_hour_angle_rad(sha):
    """
    Sunset hour angle has the range 0 to 180 degrees.

    See http://mypages.iit.edu/~maslanka/SolarGeo.pdf
    """
    if not np.logical_and(sha >= _MINSHA_RADIANS, sha <= _MAXSHA_RADIANS).all():
        raise ValueError('sunset hour angle outside valid range')

def sunset_hour_angle(latitude, sol_dec):
    """
    Calculate sunset hour angle (*Ws*) from latitude and solar
    declination.

    Based on FAO equation 25 in Allen et al (1998).

    :param latitude: Latitude [radians]. Note: *latitude* should be negative
        if it in the southern hemisphere, positive if in the northern
        hemisphere.
    :param sol_dec: Solar declination [radians]. Can be calculated using
        ``sol_dec()``.
    :return: Sunset hour angle [radians].
    :rtype: float
    """
    check_latitude_rad(latitude)
    check_sol_dec_rad(sol_dec)

    #cos_sha = -math.tan(latitude) * math.tan(sol_dec)
    cos_sha = -np.tan(latitude) * np.tan(sol_dec)
    # If tmp is >= 1 there is no sunset, i.e. 24 hours of daylight
    # If tmp is <= 1 there is no sunrise, i.e. 24 hours of darkness
    # See http://www.itacanet.org/the-sun-as-a-source-of-energy/
    # part-3-calculating-solar-angles/
    # Domain of acos is -1 <= x <= 1 radians (this is not mentioned in FAO-56!)
    #return math.acos(min(max(cos_sha, -1.0), 1.0))
    return np.arccos(np.minimum(np.maximum(cos_sha, -1.0), 1.0))

def sol_dec(day_of_year):
    """
    Calculate solar declination from day of the year.

    Based on FAO equation 24 in Allen et al (1998).

    :param day_of_year: Day of year integer between 1 and 365 or 366).
    :return: solar declination [radians]
    :rtype: float
    """
    return 0.409 * np.sin(((2.0 * np.pi / 365.0) * day_of_year - 1.39))

def inv_rel_dist_earth_sun(day_of_year):
    """
    Calculate the inverse relative distance between earth and sun from
    day of the year.

    Based on FAO equation 23 in Allen et al (1998).

    :param day_of_year: Day of the year [1 to 366]
    :return: Inverse relative distance between earth and the sun
    :rtype: float
    """
    check_doy(day_of_year)
    return 1 + (0.033 * np.cos((2.0 * np.pi / 365.0) * day_of_year))

def et_rad(latitude, sol_dec, sha, ird):
    """
    Estimate daily extraterrestrial radiation (*Ra*, 'top of the atmosphere
    radiation').

    Based on equation 21 in Allen et al (1998). If monthly mean radiation is
    required make sure *sol_dec*. *sha* and *irl* have been calculated using
    the day of the year that corresponds to the middle of the month.

    **Note**: From Allen et al (1998): "For the winter months in latitudes
    greater than 55 degrees (N or S), the equations have limited validity.
    Reference should be made to the Smithsonian Tables to assess possible
    deviations."

    :param latitude: Latitude [radians]
    :param sol_dec: Solar declination [radians]. Can be calculated using
        ``sol_dec()``.
    :param sha: Sunset hour angle [radians]. Can be calculated using
        ``sunset_hour_angle()``.
    :param ird: Inverse relative distance earth-sun [dimensionless]. Can be
        calculated using ``inv_rel_dist_earth_sun()``.
    :return: Daily extraterrestrial radiation [MJ m-2 day-1]
    :rtype: float
    """
    check_latitude_rad(latitude)
    check_sol_dec_rad(sol_dec)
    check_sunset_hour_angle_rad(sha)

    tmp1 = (24.0 * 60.0) / np.pi
    tmp2 = sha * np.sin(latitude) * np.sin(sol_dec)
    tmp3 = np.cos(latitude) * np.cos(sol_dec) * np.sin(sha)
    return tmp1 * SOLAR_CONSTANT * ird * (tmp2 + tmp3)

def hargreaves(tmin, tmax, et_rad, tmean = None):
    """
    Estimate reference evapotranspiration over grass (ETo) using the Hargreaves
    equation.

    Generally, when solar radiation data, relative humidity data
    and/or wind speed data are missing, it is better to estimate them using
    the functions available in this module, and then calculate ETo
    the FAO Penman-Monteith equation. However, as an alternative, ETo can be
    estimated using the Hargreaves ETo equation.

    Based on equation 52 in Allen et al (1998).

    :param tmin: Minimum daily temperature [deg C]
    :param tmax: Maximum daily temperature [deg C]
    :param tmean: Mean daily temperature [deg C]. If emasurements not
        available it can be estimated as (*tmin* + *tmax*) / 2.
    :param et_rad: Extraterrestrial radiation (Ra) [MJ m-2 day-1]. Can be
        estimated using ``et_rad()``.
    :return: Reference evapotranspiration over grass (ETo) [mm day-1]
    :rtype: float
    """
    # Note, multiplied by 0.408 to convert extraterrestrial radiation could
    # be given in MJ m-2 day-1 rather than as equivalent evaporation in
    # mm day-1
    if not tmean:
        tmean = (tmax + tmin)/2
        
    return 0.0023 * (tmean + 17.8) * (tmax - tmin) ** 0.5 * 0.408 * et_rad