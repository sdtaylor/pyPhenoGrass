import pandas as pd
import numpy as np
import pkg_resources
from . import models
from warnings import warn


def load_test_data(sites='all', variables='all'):
    """Pre-loaded data for model testing.
    
    This is the data used in the paper Hufkens et al. 2016
    
    Available sites:
        ['freemangrass_grass',
         'ibp_grassland',
         'kansas_grassland',
         'lethbridge_grassland',
         'marena_canopy',
         'vaira_grass']
    
    Available variables
        ['precip',   # daily precipitation, mm
         'evap',     # daily evapotranspiration, estimated by GrasslandModels.et_utils.py
         'Tm',       # Daily mean temperature of the prior 15 days
         'Ra',       # daily top of atomosphere radiation, estimated by GrasslandModels.et_utils.py
         'MAP',      # Mean average precipitation
         'Wcap',     # soil water holding capacity
         'Wp']       # soil wilting point
    
    Parameters:
        site : str, list
            Site or list of sites to load. 'all' (the default) returns
            data on all 6 sites
        
        variables : str, list
            Variable or list of variables. 'all' (the default) returns
            all 7 variables. 
    
    Returns:
        A tuple with 2 values, the first is a numpy array of GCC values for
        the sites, the 2nd a dictionary of numpy arrays corresponding to
        selected variables. 
        
        {'precip': precip, # A timeseries of daily precipitation
         'evap'  : evap,   # A timeseries of daily evapotranspiration
         'Tm'    : Tm,     # A timeseries of daily mean temp of the prior 15 days
         'Ra'    : Ra,     # A timeseries of daily solar radiation (top of atmosphere)
         'MAP'   : MAP,    # site level Mean Average Precipitaiton
         'Wcap'  : Wcap,   # site level water holding capacity
         'Wp'    : Wp}     # site level Wilting poitn
        
        All arrays are shape (12410,n_sites)
    
    """
    def parse_options(selected, available):
        if isinstance(selected, str):
            selected = [selected]
        
        if selected[0] == 'all':
            selected = available[:]
    
        not_available =  [s for s in selected if s not in available]
        if len(not_available) > 0:
            raise ValueError('Unknown selection: ' + ', '.join(not_available))

        return selected
        
    available_sites = ['freemangrass_grass',
                       'ibp_grassland',
                       'kansas_grassland',
                       'lethbridge_grassland',
                       'marena_canopy',
                       'vaira_grass']
    
    available_vars = ['precip',
                      'evap',
                      'Tm',
                      'Ra',
                      'MAP',
                      'Wcap',
                      'Wp']
    
    sites = parse_options(sites, available_sites)
    variables = parse_options(variables, available_vars)

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
        
    all_site_vars =  {'precip':precip,
                  'evap'  : evap,
                  'Tm'    : Tm,
                  'Ra'    : Ra,
                  'MAP'   : MAP,
                  'Wcap'  : Wcap,
                  'Wp'    : Wp}
    
    site_vars = {v : all_site_vars[v] for v in variables}
    
    return GCC, site_vars




def load_model(name):
    """Load a model via a string

    Options are ``['PhenoGrass']``
    """
    if not isinstance(name, str):
        raise TypeError('name must be string, got' + type(name))
        
    if name == 'PhenoGrass':
        return models.PhenoGrass
    else:
        raise ValueError('Unknown model name: ' + name)


def load_model_parameters(model_info):
    # Load a model from a model_info dictionary

    # These ensemble methods have their own code for loading saved files
    if model_info['model_name'] == 'BootstrapModel':
        raise NotImplementedError('Ensembles not implemented yet')
        model = models.BootstrapModel(parameters=model_info)
    elif model_info['model_name'] == 'Ensemble':
        raise NotImplementedError('Ensembles not implemented yet')
        model = models.Ensemble(core_models=model_info)
    else:
        # For all other ones just need to pass the parameters
        Model = load_model(model_info['model_name'])
        model = Model(parameters=model_info['parameters'])

    return model

def load_saved_model(filename):
    """Load a previously saved model file

    Returns the model object with parameters preloaded.
    """
    if not isinstance(filename, str):
        raise TypeError('filename must be string, got' + type(filename))

    model_info = models.utils.misc.read_saved_model(filename)
    return(load_model_parameters(model_info))