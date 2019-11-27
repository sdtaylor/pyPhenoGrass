import pandas as pd
import numpy as np

def validate_predictors(predictors, required_predictors):
    """ Validate the required predictors

    Parameters
    ----------
    predictors : dict of model predictors in the  format {'name': array}
    
    required_predictors : dict of required model predictors, in the format
                          {'name':'type'} where type is either per_timestep or
                          per_site for timeseries and site level values,
                          respectively.

    Returns
    -------
    None. Will raise ValueError for invalid or missing predictors
                    
    """
    timeseries_shapes = []
    site_level_shapes = []
    for predictor_name, predictor_type in required_predictors.items():
        # Check all are present
        if predictor_name not in predictors:
            raise ValueError('Missing {p} in predictors'.format(p=predictor_name))
        
        # Must by numpy arrays
        if not isinstance(predictors[predictor_name], np.ndarray):
            raise ValueError('{p} Must be numpy array, even if a single value'.format(p=predictor_name))
        
        if predictor_type == 'per_timestep':
            timeseries_shapes.append(predictors[predictor_name].shape)
        elif predictor_type == 'per_site':
            site_level_shapes.append(predictors[predictor_name].shape)
        
        if np.unique(timeseries_shapes).shape != (1,):
            raise ValueError('Uneven shapes for timeseries variables')
        #if np.unique(site_level_shapes).shape != (0,):
        #    raise ValueError('Uneven shapes for site_level variables')

def validate_observations(observations, predictors):
    """ Validate the required observations. It should be a numpy array
        the same shape as the timeseries predictors

    Parameters
    ----------
    observations: np.array of the observations
    
    predictors : dict of model predictors in the  format {'name': array}
    
    Returns
    -------
    None. Will raise ValueError for invalid observations
                    
    """
    # Must by numpy arrays
    if not isinstance(observations, np.ndarray):
        raise ValueError('observations must be numpy array.')
            
    # Check the shape matches the predictor timeseries
    # precip should be used in most models, so just check against that
    if observations.shape != predictors['precip'].shape:
        raise ValueError('observations shape does not match predictors')

def validate_model(model_class):
    required_attributes = ['_apply_model', 'all_required_parameters', '_required_data',
                           '_organize_predictors', '_validate_formatted_predictors']
    for attribute in required_attributes:
        if not hasattr(model_class, attribute):
            raise RuntimeError('Missing model attribute: ' + str(attribute))
