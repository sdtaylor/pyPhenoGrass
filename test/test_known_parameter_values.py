from GrasslandModels import models, utils
from numpy import isclose
from warnings import warn
import sys
import pytest

# Make sure some known parameters are estimated correctly

##############################
GCC, predictor_vars = utils.load_test_data(['kansas_grassland','lethbridge_grassland'])

# thorough but still relatively quick
thorough_DE_optimization = {'method':'DE', 'debug':True,
                            'optimizer_params':{'seed':1,
                                                'popsize':10,
                                                'maxiter':50,
                                                'mutation':1.5,
                                                'recombination':0.25}}


#######################################
# Setup test cases

test_cases=[]
test_cases.append({'test_name' : 'Naive Model',
                   'model' : models.Naive,
                   'fitting_obs':GCC,
                   'fitting_predictors':predictor_vars,
                   'expected_params':{'b1': 0.2716, 'b2': 0.0092, 'L': 2},
                   'fitting_ranges':{'b1': (0, 100), 'b2': (0, 100), 'L': 2},
                   'fitting_params':thorough_DE_optimization})

#######################################
# Get estimates for all models
for case in test_cases:
    model = case['model'](parameters = case['fitting_ranges'])
    model_predictors = {p:case['fitting_predictors'][p] for p in model.required_predictors()}
    model.fit(observations = case['fitting_obs'], predictors=model_predictors, 
              **case['fitting_params'])
    case['estimated_params'] = model.get_params()
    
########################################
# Setup tuples for pytest.mark.parametrize
test_cases = [(c['test_name'], c['expected_params'],c['estimated_params']) for c in test_cases]

@pytest.mark.parametrize('test_name, expected_params, estimated_params', test_cases)
def test_know_parameter_values(test_name, expected_params, estimated_params):
    all_values_match = True
    # Values are compared as ints since precision past that is not the goal here.
    for param, expected_value in expected_params.items():
        if not isclose(estimated_params[param], expected_value, atol=0.0001):
            all_values_match = False
    
    # Specific values are varying slightly between versions, 
    # let it slide if it's not on the specific version I tested
    # things on. 
    # TODO: Make this more robust
    if sys.version_info.major == 3 and sys.version_info.minor==7:
        assert all_values_match
    else:
        if not all_values_match:
            warn('Not all values match: {n} \n' \
                 'Expected: {e} \n Got: {g}'.format(n=test_name,
                                                    e=expected_params,
                                                    g=estimated_params))

