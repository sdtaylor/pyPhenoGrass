import pandas as pd
import numpy as np
import pkg_resources
from . import models
from warnings import warn


def load_test_data(sites='all'):
    """Pre-loaded data for model testing.
    
    This is the data used in the paper Hufkens et al. 2016
    
    Available sites:
        ['freemangrass_grass',
         'ibp_grassland',
         'kansas_grassland',
         'lethbridge_grassland',
         'marena_canopy',
         'vaira_grass']
    
    
    Parameters:
        site : str, list
            Site or list of sites to load. 'all' (the default) returns
            data on all 6 sites
    
    Returns:
        A tuple with 2 values, the first is a numpy array of GCC values for
        the sites, the 2nd a dictionary of numpy arrays.
        
        {'precip': precip, # A timeseries of daily precipitation
         'evap'  : evap,   # A timeseries of daily evapotranspiration
         'Tm'    : Tm,     # A timeseries of daily mean temp of the prior 15 days
         'Ra'    : Ra,     # A timeseries of daily solar radiation (top of atmosphere)
         'MAP'   : MAP,    # site level Mean Average Precipitaiton
         'Wcap'  : Wcap,   # site level water holding capacity
         'Wp'    : Wp}     # site level Wilting poitn
        
        All arrays are shape (12410,n_sites)
    
    """
    
    if isinstance(sites, str):
        sites = [sites]
    
    available_sites = ['freemangrass_grass',
                       'ibp_grassland',
                       'kansas_grassland',
                       'lethbridge_grassland',
                       'marena_canopy',
                       'vaira_grass']
    
    if sites[0] == 'all':
        sites = available_sites[:]
    
    not_available =  [s for s in sites if s not in available_sites]
    if len(not_available) > 0:
        raise ValueError('Unknown sites: ' + ', '.join(not_available))
    
    
    site_data_filename = pkg_resources.resource_filename(__name__, 'data/site_data.csv.gz')
    site_metadata_filename = pkg_resources.resource_filename(__name__, 'data/site_metadata.csv')
    
    site_data = pd.read_csv(site_data_filename)
    site_metadata = pd.read_csv(site_metadata_filename)
    
    n_sites = len(sites)
    timeseries_length = 12410
    
    # initialize arrays
    precip = np.zeros((timeseries_length, n_sites)).astype('float32')
    evap   = np.zeros((timeseries_length, n_sites)).astype('float32')
    Ra     = np.zeros((timeseries_length, n_sites)).astype('float32')
    Tm     = np.zeros((timeseries_length, n_sites)).astype('float32')
    MAP    = np.zeros((n_sites)).astype('float32')
    Wcap   = np.zeros((n_sites)).astype('float32')
    Wp     = np.zeros((n_sites)).astype('float32')
    
    GCC    = np.zeros((timeseries_length, n_sites)).astype('float32')
    
    # Fill in everything
    for site_i, site_name in enumerate(sites):
        this_site_data = site_data[site_data.Site == site_name]
        this_site_metdata = site_metadata[site_metadata.Site == site_name]
        
        precip[:, site_i] = this_site_data.prcp.values
        evap[:, site_i]   = this_site_data.et.values
        Ra[:, site_i]     = this_site_data.radiation.values
        Tm[:, site_i]     = this_site_data.tmean_15day.values
        MAP[site_i]       = this_site_metdata.MAP.values[0]
        Wcap[site_i]      = this_site_metdata.WCAP.values[0]
        Wp[site_i]        = this_site_metdata.WP.values[0]
        
        GCC[:, site_i]    = this_site_data.gcc.values
        
        
    site_vars =  {'precip':precip,
                  'evap'  : evap,
                  'Tm'    : Tm,
                  'Ra'    : Ra,
                  'MAP'   : MAP,
                  'Wcap'  : Wcap,
                  'Wp'    : Wp}
    
    return GCC, site_vars