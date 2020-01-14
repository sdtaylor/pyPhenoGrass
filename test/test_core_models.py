from GrasslandModels import models, utils
import pytest
import numpy as np

core_model_names = ['PhenoGrass',
                    'PhenoGrassNDVI',
                    'CholerPR1',
                    'CholerPR2',
                    'CholerPR3',
                    'Naive']

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
    this_model_data = {k:predictor_vars[k] for k in m._required_predictors.keys()}
    m.fit(GCC, this_model_data, optimizer_params=quick_testing_params)
    fitted_models.append(m)

model_test_cases = list(zip(core_model_names, fitted_models))

#######################################################################

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
    
    
########################################################################
# Some PhenoGrass specific tests
def test_phenograss_fit():
    m = models.PhenoGrass(parameters= {'L':2})
    m.fit(GCC, predictor_vars, optimizer_params=quick_testing_params, debug = True)
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
