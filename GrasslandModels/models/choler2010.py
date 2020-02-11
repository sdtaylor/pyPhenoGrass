from . import utils
from .base import BaseModel
import numpy as np


class CholerM1(BaseModel):
    """
    The "M1" four parameter model described in Choler et al. 2010
    """
    def __init__(self, parameters={}):
        BaseModel.__init__(self)
        self.all_required_parameters = {'a1': (0, 100), 'a2': (0, 100), 
                                        'a3': (0, 100), 
                                        'L': (1,30)}
        self._organize_parameters(parameters)
        #self._required_data = {'predictor_columns': ['site_id', 'year', 'doy', 'temperature'],
         #                      'predictors': ['pr','tasmin','tasmax']}
        self._required_predictors = {'precip': 'per_timestep',
                                     'evap'  : 'per_timestep',
                                     'Wcap'  : 'per_site'}

        self.state_variables = ['V','W','Dt']
        
        # Default to the faster cython version.
        self.set_internal_method(method='numpy')
    
    def set_internal_method(self, method = 'numpy'):
        if method == 'cython':
            raise NotImplementedError('cython method not implemented for this model')
        elif method == 'numpy':
            self._apply_model = self._apply_model_numpy
        else:
            raise ValueError('Unknown internal method: ' + method)

    def _apply_model_numpy(self,
                         # Site specific drivers
                         precip,  # precip, Daily vector
                         evap,    # potential ET, Daily vector
                         #T,       # ? mean temp ? not actually used in phenograss.f90
                         #Ra,      # TOA radiation, MJ m-2 s-1, daily vector
                         #Tm,      # Running mean T with 15 day lag
                         Wcap,    # field capacity, single value/site
                         #Wp,      # wilting point, single value/site
                         #MAP,     # Mean avg precip, used to scale model input(gcc) to output (fcover)
                                  # fCover = GCC * MAP/ (MAP+h), where h is an estimated  parameter
                         
                         # Model parameters
                         a1,  
                         a2,
                         a3,
                         L,
                         
                         # Contraints on vegetation. 
                         Vmin = 0.001, # Needs to be small non-zero value 
                         Vmax = 1.,    # 100% cause GCC is scaled 0-1
                         # Note in the original Choler 2011 paper, Vmax is a site
                         # specific value set to the maximum value observed at a site.
                         # This is not feasable for extrapolation though. 
                     
                         # Initial conditions
                         W_initial = 0,
                         Wstart    = 0,
                         V_initial = 0.001,
                         # Normally just the V (vegatation cover) should be returned,
                         # but for diagnostics use 'all' to get V, and Dt
                         return_vars = 'V'
                         ):
            """
            
            """
            L = int(L) # must be a whole number. and floats will be truncated.
            
            # Initialize everything
            # Primary state variables
            W = np.empty_like(precip).astype('float32')
            W[:] = W_initial
            
            V = np.empty_like(precip).astype('float32')
            V[:] = V_initial
    
            # Derived variables
            Dt = np.zeros_like(precip).astype('float32')
            
            # Site level vars such as lagged plant-water and
            # temp responses
            # In the Choler2010 paper this term is We. It's marked 
            # add here to match the other models
            Dtl  = np.empty_like(Wcap)
            Dtl1 = np.empty_like(Wcap)
            
            n_timesteps = precip.shape[0] - 1
            
            for i in range(1,n_timesteps):
                
                # if we are near the start of the timeseries then initialize
                # soil/plant water to something reasonable
                # Condition (iii)
                if i - L - 1 < 0:
                    Dt[i] = np.maximum(0, W[i] - a3)
                    Dtl[:] = Wstart
                    Dtl1[:] = Wstart
                else:
                    Dt[i] = np.maximum(0, W[i] - a3)
                    Dtl = Dt[i-L]
                    Dtl1 = Dt[i-L-1]
                               
                # Soil water
                W[i+1] = W[i] + precip[i] - a1 * (W[i]/Wcap) * evap[i]
                # Condition (ii)
                W[i+1] = np.maximum(0, np.minimum(Wcap, W[i+1]))
                
                # Primary veg growth equation
                V[i+1] = V[i] + a2 * (Dtl -Dtl1)
                
                # Condtiion (iv)
                # Constrain veg to 0-1
                V[i+1] = np.maximum(Vmin, np.minimum(Vmax, V[i+1]))
            
            if return_vars == 'V':
                return V
            elif return_vars == 'all':
                return {'V':V, 'W':W, 'Dt':Dt}



class CholerM1A(CholerM1):
    """
    The "M1A" model described in Choler et al. 
    
    This fixes the a3 parameter to 0
    """
    def __init__(self, parameters={}):
        CholerM1.__init__(self)
        self.all_required_parameters = {'a1': (0, 100), 'a2': (0, 100), 
                                        'a3': 0, 
                                        'L': (1,30)}
        self._organize_parameters(parameters)


class CholerM1B(CholerM1):
    """
    The "M1B" model described in Choler et al. 2010
    
    Parameterizing the full model essentially. Described here
    for completeness.
    """
    def __init__(self, parameters={}):
        CholerM1.__init__(self)

class CholerM2(BaseModel):
    """
    The "M2" model described in Choler et al. 2011
    """
    def __init__(self, parameters={}):
        BaseModel.__init__(self)
        self.all_required_parameters = {'b1': (0, 100), 'b2': (0, 100), 
                                        'b3': (0, 100), 'b4': (0, 100), 
                                        'b5': (0,100)}
        self._organize_parameters(parameters)
        #self._required_data = {'predictor_columns': ['site_id', 'year', 'doy', 'temperature'],
         #                      'predictors': ['pr','tasmin','tasmax']}
        self._required_predictors = {'precip': 'per_timestep',
                                     'evap'  : 'per_timestep',
                                     'Wcap'  : 'per_site'}

        self.state_variables = ['V','W','Dt']
        
        # Default to the faster cython version.
        self.set_internal_method(method='numpy')
    
    def set_internal_method(self, method = 'numpy'):
        if method == 'cython':
            raise NotImplementedError('cython method not implemented for this model')
        elif method == 'numpy':
            self._apply_model = self._apply_model_numpy
        else:
            raise ValueError('Unknown internal method: ' + method)

    def _apply_model_numpy(self,
                         # Site specific drivers
                         precip,  # precip, Daily vector
                         evap,    # potential ET, Daily vector
                         #T,       # ? mean temp ? not actually used in phenograss.f90
                         #Ra,      # TOA radiation, MJ m-2 s-1, daily vector
                         #Tm,      # Running mean T with 15 day lag
                         Wcap,    # field capacity, single value/site
                         #Wp,      # wilting point, single value/site
                         #MAP,     # Mean avg precip, used to scale model input(gcc) to output (fcover)
                                  # fCover = GCC * MAP/ (MAP+h), where h is an estimated  parameter
                         
                         # Model parameters
                         b1,  
                         b2,
                         b3,
                         b4,
                         b5,
                         #L, No lag component in this one
                         
                         # Contraints on vegetation. 
                         Vmin = 0.001, # Needs to be small non-zero value 
                         Vmax = 1.,    # 100% cause GCC is scaled 0-1
                         # Note in the original Choler 2011 paper, Vmax is a site
                         # specific value set to the maximum value observed at a site.
                         # This is not feasable for extrapolation though. 
                     
                         # Initial conditions
                         W_initial = 0,
                         Wstart    = 0,
                         V_initial = 0.001,
                         # Normally just the V (vegatation cover) should be returned,
                         # but for diagnostics use 'all' to get V, and Dt
                         return_vars = 'V'
                         ):
            """
            
            """
            #L = int(L) # must be a whole number. and floats will be truncated.
            
            # Initialize everything
            # Primary state variables
            W = np.empty_like(precip).astype('float32')
            W[:] = W_initial
            
            V = np.empty_like(precip).astype('float32')
            V[:] = V_initial
    
            # Derived variables
            Dt = np.zeros_like(precip).astype('float32')
            
            
            n_timesteps = precip.shape[0] - 1
            
            for i in range(1,n_timesteps):
                
                # plant available water
                # condition (iii)
                Dt[i] = np.maximum(0, W[i] - b5)
                                
                # Soil water
                W[i+1] = W[i] + precip[i] - b1 * (1 - V[i]) * (W[i]/Wcap) * evap[i] - b2 * V[i] * Dt[i]
                # condition (ii)
                W[i+1] = np.maximum(0, np.minimum(Wcap, W[i+1]))
                
                
                # Primary veg growth equation
                V[i+1] = b3 * (Dt[i]/(Wcap - b5)) * V[i] * (1-(V[i]/Vmax)) - (b4 * V[i])
                
                # Condtiion (iv)
                # Constrain veg to 0-1
                V[i+1] = np.maximum(Vmin, np.minimum(Vmax, V[i+1]))
            
            if return_vars == 'V':
                return V
            elif return_vars == 'all':
                return {'V':V, 'W':W, 'Dt':Dt}


class CholerM2A(CholerM2):
    """
    The "M2A" model described in Choler et al. 2011
    
    Here the b5 parameter is fixed at 0, essentially making plant available 
    water equal to total soil water.
    """
    def __init__(self, parameters={}):
        CholerM2.__init__(self)
        self.all_required_parameters = {'b1': (0, 100), 'b2': (0, 100), 
                                        'b3': (0, 100), 'b4': (0, 100), 
                                        'b5': 0}
        self._organize_parameters(parameters)

class CholerM2B(CholerM2):
    """
    The "M2B" model described in Choler et al. 2011
    
    Parameterizing the full model essentially. Described here
    for completeness.
    """
    def __init__(self, parameters={}):
        CholerM2.__init__(self)
