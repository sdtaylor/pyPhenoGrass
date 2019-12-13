from GrasslandModels import models, utils
import pytest
import numpy as np



GCC, predictor_vars = utils.load_test_data()

quick_testing_params = {'maxiter':5,
                         'popsize':3,
                         'mutation':(0.5,1),
                         'recombination':0.25,
                         'disp':False}


original_phenograss_params = {'b1':124.502121,
                           'b2':0.00227958267,
                           'b3':0.0755224228,
                           'b4':0.519348383,
                           #'L':2.4991734,
                           'L':2,
                           'Phmin':8.14994431,
                           'h': 222.205673,
                           'Topt':33.3597641,
                           'Phmax':37.2918091}

def test_phenograss_fit():
    m = models.PhenoGrass(parameters= {'L':2})
    m.fit(GCC, predictor_vars, optimizer_params=quick_testing_params, debug = True)
    assert isinstance(m.get_params(), dict)


def test_phenograss_internal_methods():
    m1 = models.PhenoGrass(parameters = original_phenograss_params)
    m1.set_internal_method(method='cython')
    
    m2 = models.PhenoGrass(parameters = original_phenograss_params)
    m2.set_internal_method('numpy')
    
    m1_prediction = m1.predict(predictor_vars)
    m2_prediction = m2.predict(predictor_vars)

    diff = np.abs(m1_prediction - m2_prediction)
    assert np.nanmax(diff) < 1e-5
