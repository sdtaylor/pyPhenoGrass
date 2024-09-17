from GrasslandModels import models, utils
import pytest
from copy import deepcopy
import numpy as np

core_model_names = ['PhenoGrass',
                    'PhenoGrassNDVI',
                    'CholerPR1',
                    'CholerPR2',
                    'CholerPR3',
                    'CholerMPR2',
                    'CholerMPR3',
                    'CholerM1A',
                    'CholerM1B',
                    'CholerM2A',
                    'CholerM2B',
                    'Naive',
                    'Naive2',
                    'NaiveMAPCorrected',
                    'Naive2MAPCorrected']

GCC, predictor_vars = utils.load_test_data()

quick_testing_params = {'maxiter':3,
                         'popsize':2,
                         'mutation':(0.5,1),
                         'recombination':0.25,
                         'disp':False}





# Setup a list of test cases where each one = (model_name, fitted_model object)
fitted_models = []
for name in core_model_names:
    m = utils.load_model(name)()
    models.validation.validate_model(m)
    this_model_data = {k:predictor_vars[k] for k in m._required_predictors.keys()}
    m.fit(GCC, this_model_data.copy(), optimizer_params=quick_testing_params)
    fitted_models.append(m)

model_test_cases = list(zip(core_model_names, fitted_models))

#######################################################################

@pytest.mark.parametrize('model_name, fitted_model', model_test_cases)
def test_state_variables_match1(model_name, fitted_model):
    """All declared state variables should be in the returned dictionary"""
    listed_state_vars = fitted_model.state_variables
    returned_state_vars = fitted_model.predict(return_variables='all').keys()
    assert all([v in returned_state_vars for v in listed_state_vars])

@pytest.mark.parametrize('model_name, fitted_model', model_test_cases)
def test_state_variables_match2(model_name, fitted_model):
    """All returned state variables should be in the declared list. The inverse of above"""
    listed_state_vars = fitted_model.state_variables
    returned_state_vars = fitted_model.predict(return_variables='all').keys()
    assert all([v in listed_state_vars for v in returned_state_vars])

@pytest.mark.parametrize('model_name, fitted_model', model_test_cases)
def test_predict_output_length(model_name, fitted_model):
    """Predict output shape should equal input shape"""
    assert fitted_model.predict().shape == GCC.shape

@pytest.mark.parametrize('model_name, fitted_model', model_test_cases)
def test_score(model_name, fitted_model):
    """Score should return a single number"""
    assert isinstance(fitted_model.score(), np.floating)

def test_prefit_model_loading():
    """Ensure prefit models are loading correctly"""
    prefit_models = ['CholerPR1-original','CholerPR2-original','PhenoGrass-original']
    models = [utils.load_prefit_model(n) for n in prefit_models]
    assert all([isinstance(m.get_params(), dict) for m in models])

def test_metadata_save_load():
    new_entries = {'entry1':'123',
                   'entry2':'123',
                   'entry3':'123'}
    
    m = utils.load_prefit_model('CholerPR1-original')
    m.clear_metadata()
    m.update_metadata(new_entries)
    m.save_params('test_model_params.json', overwrite = True)
    
    m2 = utils.load_saved_model('test_model_params.json')
    assert new_entries == m2.metadata

def test_shape_validation1():
    m = utils.load_model('PhenoGrass')()
    uneven_data = predictor_vars.copy()
    # Drop a single site of mean avg precip data
    uneven_data['MAP'] = uneven_data['MAP'][:-1]
    with pytest.raises(ValueError):
        m.fit(GCC, uneven_data, optimizer_params = quick_testing_params)

def test_shape_validation2():
    m = utils.load_model('PhenoGrass')()
    uneven_data = predictor_vars.copy()
    # Drop a single timestep of evapotranspiration data
    uneven_data['evap'] = uneven_data['evap'][:-1]
    with pytest.raises(ValueError):
        m.fit(GCC, uneven_data, optimizer_params = quick_testing_params)
        
def test_shape_validation3():
    m = utils.load_model('PhenoGrass')()
    # Drop a single site of gcc data
    with pytest.raises(ValueError):
        m.fit(GCC[:,:-1], predictor_vars.copy(), optimizer_params = quick_testing_params)

def test_predictor_validation1():
    # Raise error when not all predictors are passed
    m = utils.load_model('PhenoGrass')()
    data_with_missing_var = predictor_vars.copy()
    _ = data_with_missing_var.pop('precip')
    with pytest.raises(ValueError):
        m.fit(GCC, data_with_missing_var, optimizer_params = quick_testing_params)
        
def test_predictor_validation2():
    # Raise error when unknown predictors are pass
    # CholerPR does not need everything
    m = utils.load_model('CholerPR1')()
    with pytest.raises(ValueError):
        m.fit(GCC, predictor_vars.copy(), optimizer_params = quick_testing_params)

@pytest.mark.parametrize('model_name, fitted_model', model_test_cases)
def test_fitting_data_manipultaiton(model_name, fitted_model):
    """
    Theres a chance models may change the internal fitting data. The individual models
    *shouldn't*, but its still possible. There used to be a deepcopy in the base method
    to guard against this, but that slows things down and seems uneccesary.
    This is a quick check to make sure that is not happening.
    Note than after predictor_vars is loaded at the top of this test script
    its always passed as a copy to the model fitting method just to ensure
    its fresh.
    """

    this_model_data = {k:predictor_vars[k] for k in fitted_model._required_predictors.keys()}
    # TODO: Issues here with comparing nans inside the numpy array
    # All np arrays in the predictor dict should be unchanged after model fitting
    c = [(fitted_model.fitting_predictors[k] == v).all() for k,v in this_model_data.items()]
    assert all(c)
        
@pytest.mark.parametrize('model_name, fitted_model', model_test_cases)
def test_internal_broadcasting(model_name, fitted_model):
    """
    In the apply_model code each model should be able to handle any number of
    sites, scenarios, etc as long as the 0 axis is the time axis.
    This checks that by ensuring predictors copied to different axis
    produce the same output.
    Note this only applies to the numpy, and not the cython, method. 
    broadcasting in cython is a bugger. 
    """
    def copy_to_last_axis(a):
        return np.append(np.expand_dims(a, -1), np.expand_dims(a, -1), axis=-1)
    
    this_model_data = {k:copy_to_last_axis(predictor_vars[k]) for k in fitted_model._required_predictors.keys()}
    fitted_model.set_internal_method('numpy')
    model_output = fitted_model.predict(predictors=this_model_data)

    assert np.allclose(model_output[:,:,0], model_output[:,:,1], equal_nan=True)
    
########################################################################
# Some PhenoGrass specific tests
def test_phenograss_fit():
    m = models.PhenoGrass(parameters= {'L':2})
    m.fit(GCC, predictor_vars.copy(), optimizer_params=quick_testing_params, debug = True)
    assert isinstance(m.get_params(), dict)


def test_phenograss_internal_methods():
    m1 = utils.load_prefit_model('PhenoGrass-original')
    m1.set_internal_method(method='cython')
    
    m2 = utils.load_prefit_model('PhenoGrass-original')
    m2.set_internal_method('numpy')
    
    m1_prediction = m1.predict(predictor_vars)
    m2_prediction = m2.predict(predictor_vars)

    diff = np.abs(m1_prediction - m2_prediction)
    assert np.nanmax(diff) < 1e-5
