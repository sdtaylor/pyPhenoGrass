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
        
        # Must be float32 numpy arrays
        if not isinstance(predictors[predictor_name], np.ndarray):
            raise ValueError('{p} Must be numpy array of type float32, even if a single value'.format(p=predictor_name))
        
        if predictors[predictor_name].dtype != np.float32:
            raise ValueError('{p} Must be numpy array of type float32, even if a single value'.format(p=predictor_name))
        
        if predictor_type == 'per_timestep':
            timeseries_shapes.append(predictors[predictor_name].shape)
        elif predictor_type == 'per_site':
            site_level_shapes.append(predictors[predictor_name].shape)
    
    # Must all be equal shapes
    for i in range(len(timeseries_shapes)):
        if timeseries_shapes[i] != timeseries_shapes[0]:
            raise ValueError('Uneven shapes for timeseries variables')
    
    for i in range(len(site_level_shapes)):
        if site_level_shapes[i] != site_level_shapes[0]:
            raise ValueError('Uneven shapes for site_level variables')
            
    # Site level variables that have a single value per site. 
    # They must match the shape, minus the time axis, of the timeseries vars
    if site_level_shapes[0][-1] != timeseries_shapes[0][-1]:
        raise ValueError('site level length does not match timeseries site number')
    

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
