import numpy as np
import pandas as pd
from . import utils, validation
import time
from collections import OrderedDict
from warnings import warn


class BaseModel():
    def __init__(self):
        self._fitted_params = {}
        self.obs_fitting = None
        self.temperature_fitting = None
        self.doy_series = None
        self.debug = False
        self.metadata = {}

    def fit(self, observations, predictors, loss_function='rmse',
            method='DE', optimizer_params='practical',
            verbose=False, debug=False, **kwargs):
        """Estimate the parameters of a model

        Parameters:
            observations: np.array 
                Timeseries of the observations. Axis 0 should be the time axis,
                with other axis corresponding to sites or locations. The shape
                of this array must match the shape of any timeseries predictors.
            
            predictors : dict
                dictionary of predictors specified by the model

            loss_function : str, or function
            
            A string for built in loss functions, or a customized function 
            which accpepts 2 arguments. obs and pred, both numpy arrays of 
            the same shape
            
            method : str
                Optimization method to use. Either 'DE' or 'BF' for differential
                evolution or brute force methods.

            optimizer_params : dict | str
                Arguments for the scipy optimizer, or one of 3 presets 'testing',
                'practical', or 'intensive'.

            verbose : bool
                display progress of the optimizer

            debug : bool
                display various internals

        """
        self.fit_load(observations = observations, predictors = predictors,
                      loss_function = loss_function)
    
        if debug:
            verbose = True
            self.debug = True
            self.model_timings = []
            print('estimating params:\n {x} \n '.format(x=self._parameters_to_estimate))
            print('array passed to optimizer:\n {x} \n'.format(x=self._scipy_bounds()))
            print('fixed params:\n {x} \n '.format(x=self._fixed_parameters))
        if verbose:
            fitting_start = time.time()

        fitted_params, fitting_info = utils.optimize.fit_parameters(function_to_minimize=self._scipy_error,
                                                                      bounds=self._scipy_bounds(),
                                                                      method=method,
                                                                      results_translator=self._translate_scipy_parameters,
                                                                      optimizer_params=optimizer_params,
                                                                      verbose=verbose)
        self._fitted_params = fitted_params
        self.update_metadata(fitting_info = fitting_info)
        
        if verbose:
            total_fit_time = round(time.time() - fitting_start, 5)
            print('Total model fitting time: {s} sec.\n'.format(s=total_fit_time))

        if debug:
            n_runs = len(self.model_timings)
            mean_time = np.mean(self.model_timings).round(5)
            print('Model iterations: {n}'.format(n=n_runs))
            print('Mean timing: {t} sec/iteration \n\n'.format(t=mean_time))
            self.debug = False
        self._fitted_params.update(self._fixed_parameters)

    def fit_load(self, observations, predictors, loss_function='rmse'):
        """
        Validate and load the data in preperation for model fitting.
        """
        
        validation.validate_predictors(predictors, self._required_predictors)
        validation.validate_observations(observations, predictors)
        self._set_loss_function(loss_function)
        if len(self._parameters_to_estimate) == 0:
            raise RuntimeError('No parameters to estimate')
        
        # Store these as they'll be used for fitting and subsequent predictions
        self.fitting_predictors = predictors
        self.obs_fitting = observations
    
    def set_internal_method(self, method):
        """ 
        Here for compatability. This is overidden by a model specific
        set_internal_method where appropriate.
        """
        pass
    
    def required_predictors(self):
        """
        Get a list of the required model predictors
        """
        return list(self._required_predictors.keys())
    
    def predict(self, predictors=None, return_variables='V', **kwargs):
        """Make predictions

        Make a prediction given predictor data..
        All model parameters must be set either in the initial model call
        or by running fit(). If no new data is passed then this will predict
        based off fitting data (if a fit was run)

        Parameters:
            predictors : dict
                dictionary of predictors specified by the model
                
            return_variables : str
                Which variables to return. V (the default) returns a numpy array
                of the modelled vegetation. 'all' returns all available 
                state variables as a dictionary of numpy arrays

        Returns:
            predictions : array or dict of arrays
                array the same shape of timeseries values in predictors. 
                Or if predictors=None, the same shape as observations used in fitting.

        """
        self._check_parameter_completeness()

        if isinstance(predictors, dict):
            # predictors is a dict containing data that can be
            # used directly in _apply_mode()
            validation.validate_predictors(predictors, self._required_predictors)

        elif predictors is None:
            # Making predictions on data used for fitting
            if self.obs_fitting is not None and self.fitting_predictors is not None:
                predictors = self.fitting_predictors
            else:
                raise TypeError('No new new predictors passed, and' +
                                'no fitting done. Nothing to predict')
        else:
            raise TypeError('Invalid arguments. predictors must be a dictionary ' +
                            'of new data to predict,' +
                            'or set to None to predict the data used for fitting')

        predictions = self._apply_model(**predictors,
                                        **self._fitted_params,
                                        return_vars = return_variables)

        return predictions

    def _set_loss_function(self, loss_function):
        """The loss function (ie. RMSE)

        Either a sting for a built in function, or a customized
        function which accpepts 2 arguments. obs, pred, both 
        numpy arrays of the same shape
        """
        if isinstance(loss_function, str):
            self.loss_function = utils.optimize.get_loss_function(method=loss_function)
        elif callable(loss_function):
            # validation.validate_loss_function(loss_function)
            self.loss_function = loss_function
        else:
            raise TypeError('Unknown loss_function. Must be string or custom function')

    def _organize_parameters(self, passed_parameters):
        """Interpret each passed parameter value to a model.
        They can either be a fixed value, a range to estimate with,
        or, if missing, implying using the default range described
        in the model.
        """
        parameters_to_estimate = {}
        fixed_parameters = {}

        if not isinstance(passed_parameters, dict):
            raise TypeError('passed_parameters must be either a dictionary or string')

        # This is all the required parameters updated with any
        # passed parameters. This includes any invalid ones,
        # which will be checked for in a moment.
        params = self.all_required_parameters.copy()
        params.update(passed_parameters)

        for parameter, value in params.items():
            if parameter not in self.all_required_parameters:
                raise RuntimeError('Unknown parameter: ' + str(parameter))

            if isinstance(value, tuple):
                if len(value) != 2:
                    raise RuntimeError('Parameter tuple should have 2 values')
                parameters_to_estimate[parameter] = value
            elif isinstance(value, slice):
                # Note: Slices valid for brute force method only.
                parameters_to_estimate[parameter] = value
            elif isinstance(value * 1.0, float):
                fixed_parameters[parameter] = value
            else:
                raise TypeError('unkown parameter value: ' + str(type(value)) + ' for ' + parameter)

        self._parameters_to_estimate = OrderedDict(parameters_to_estimate)
        self._fixed_parameters = OrderedDict(fixed_parameters)

        # If nothing to estimate then all parameters have been
        # passed as fixed values and no fitting is needed
        if len(parameters_to_estimate) == 0:
            self._fitted_params = fixed_parameters

    def get_params(self):
        """Get the fitted parameters

        Parameters:
            None

        Returns:
            Dictionary of parameters.
        """
        self._check_parameter_completeness()
        return self._fitted_params

    def _get_model_info(self):
        return {'model_name': type(self).__name__,
                'parameters': self._fitted_params,
                'metadata'  : self.metadata}

    def save_params(self, filename, overwrite=False):
        """Save the parameters for a model

        A model can be loaded again by passing the filename to the ``parameters``
        argument on initialization.

        Parameters:
            filename : str
                Filename to save parameter file

            overwrite : bool
                Overwrite the file if it exists
        """
        self._check_parameter_completeness()
        utils.misc.write_saved_model(model_info=self._get_model_info(),
                                     model_file=filename,
                                     overwrite=overwrite)

    def update_metadata(self, new_params=None, **kwargs):
        """Add 1 or more metdata entries.
        
        Metdata entires can be set to anything and do not affect model 
        functionality. They are saved in the model file and can be accessed
        by the model.metdata dictionary.
        
        Update via a single dictionary or several parameters as arguments.
        Any prior entries with the same name will be overwritten.
        
        model.update_metdatda(fit_date='2020-01-02', training_set='set1')
        """
        if new_params is not None:
            if isinstance(new_params, dict):
                self.metadata.update(new_params)
            else:
                raise TypeError('new_params must be a dictionary')
        self.metadata.update(kwargs)
    
    def clear_metadata(self):
        """Delete all metadata entries
        """
        self.metadata = {}
        
    def _get_initial_bounds(self):
        # TODO: Probably just return params to estimate + fixed ones
        raise NotImplementedError()

    def _translate_scipy_parameters(self, parameters_array):
        """Map parameters from a 1D array to a dictionary for
        use in phenology model functions. Ordering matters
        in unpacking the scipy_array since it isn't labeled. Thus
        it relies on self._parameters_to_estimate being an 
        OrdereddDict
        """
        # If only a single value is being fit, some scipy.
        # optimizer methods will use a single
        # value instead of list of length 1.
        try:
            _ = parameters_array[0]
        except IndexError:
            parameters_array = [parameters_array]
        labeled_parameters = {}
        for i, (param, value) in enumerate(self._parameters_to_estimate.items()):
            labeled_parameters[param] = parameters_array[i]
        return labeled_parameters

    def _scipy_error(self, x):
        """Error function for use within scipy.optimize functions.

        All scipy.optimize functions take require a function with a single
        parameter, x, which is the set of parameters to test. This takes
        x, labels it appropriately to be used as **parameters to the
        internal phenology model, and adds any fixed parameters.
        """
        parameters = self._translate_scipy_parameters(x)

        # add any fixed parameters
        parameters.update(self._fixed_parameters)

        if self.debug:
            start = time.time()

        doy_estimates = self._apply_model(**self.fitting_predictors,
                                          **parameters,
                                          return_vars = 'V')
        if self.debug:
            self.model_timings.append(time.time() - start)

        return self.loss_function(self.obs_fitting, doy_estimates)

    def _scipy_bounds(self):
        """Bounds structured for scipy.optimize input"""
        return [bounds for param, bounds in list(self._parameters_to_estimate.items())]

    def _parameters_are_set(self):
        """True if all parameters have been set from fitting or loading at initialization"""
        return len(self._fitted_params) == len(self.all_required_parameters)

    def _check_parameter_completeness(self):
        """Don't proceed unless all parameters are set"""
        if not self._parameters_are_set():
            raise RuntimeError('Not all parameters set')

    def score(self, metric='rmse', observations=None, predictors=None):
        """Evaluate a prediction given observed values

        Given no arguments this will return the RMSE on the dataset used for
        fitting (if fitting was done).
        To evaluate a new set of data set pass ``observations`` and ``predictors``,
        as used in ``model.predict()``. The predictions from these will be
        evluated against the true values in ``observations``.

        Metrics available are root mean square error (``rmse``) and AIC (``aic``).
        For AIC the number of parameters in the model is set to the number of
        parameters actually estimated in ``fit()``, not the total number of
        model parameters. 

        Parameters:
            metric : str, optional
                The metric used either 'rmse' for the root mean square error,
                or 'aic' for akaike information criteria.
                
            observations: np.array 
                Timeseries of the observations. Axis 0 should be the time axis,
                with other axis corresponding to sites or locations. The shape
                of this array must match the shape of any timeseries predictors.
            
            predictors : dict
                dictionary of predictors specified by the model
        
        Returns:
            The score as a float
        """
        # If both args are none use fitting data for scoring, if available.
        if all([observations is None, predictors is None]):
            if self.obs_fitting is not None and self.fitting_predictors is not None:
                observations = self.obs_fitting
                predictors   = self.fitting_predictors
        
            else:
                raise TypeError('No observations + predictors passed, and' +
                                'no fitting done. Nothing to score')
        
        # But if 1 is set then the other must be set. 
        elif any([observations is None, predictors is None]):
            raise TypeError('Observations + predictors must both be set for scoring')
            
        estimated = self.predict(predictors=predictors)

        error_function = utils.optimize.get_loss_function(method=metric)

        if metric == 'aic':
            error = error_function(observations, estimated,
                                   n_param=len(self._parameters_to_estimate))
        else:
            error = error_function(observations, estimated)

        return error
