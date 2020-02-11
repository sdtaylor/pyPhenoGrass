from . import utils
from .base import BaseModel
import numpy as np

"""

"""
class CholerMPR2(BaseModel):
    """
    The "PR2" four parameter model described in Choler et al. 2011
    
    Modified to use wilting point as an input instead of estimating it
    via the b1 parameter. Now 3 parameters. 
    """
    def __init__(self, parameters={}):
        BaseModel.__init__(self)
        self.all_required_parameters = {'b2': (0, 100),'b3': (0, 100), 
                                        'b4': (0, 100), 'L': (1,30)}
        self._organize_parameters(parameters)
        #self._required_data = {'predictor_columns': ['site_id', 'year', 'doy', 'temperature'],
         #                      'predictors': ['pr','tasmin','tasmax']}
        self._required_predictors = {'precip': 'per_timestep',
                                     'evap'  : 'per_timestep',
                                     'Wcap'  : 'per_site',
                                     'Wp'    : 'per_site'}

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
                         Wp,      # wilting point, single value/site
                         #MAP,     # Mean avg precip, used to scale model input(gcc) to output (fcover)
                                  # fCover = GCC * MAP/ (MAP+h), where h is an estimated  parameter
                         
                         # Model parameters
                         #b1, Replaced by Wp in the modified model    
                         b2,
                         b3,
                         b4,
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
            Dtl  = np.empty_like(Wcap)
            Dtl1 = np.empty_like(Wcap)
            
            n_timesteps = precip.shape[0] - 1
            
            for i in range(1,n_timesteps):
                
                # if we are near the start of the timeseries then initialize
                # soil/plant water to something reasonable
                if i - L - 1 < 0:
                    Dt[i] = np.maximum(0, W[i] - Wp)
                    Dtl[:] = Wstart
                    Dtl1[:] = Wstart
                else:
                    Dt[i] = np.maximum(0, W[i] - Wp)
                    Dtl = Dt[i-L]
                    Dtl1 = Dt[i-L-1]
                
                # Condition (ii)
                # If plant available water is on the decline
                # then decay is 1 and senescensce sets in
                d = (Dtl <= Dtl1) * 1
                
                # Soil water
                W[i+1] = W[i] + precip[i] - b4 * ((Dt[i]/(Wcap - Wp))**2) * evap[i]
                W[i+1] = np.maximum(0, np.minimum(Wcap, W[i+1]))
                
                # Primary veg growth equation
                V[i+1] = V[i] + b2 * Dtl * (1 - (V[i]/Vmax)) - d * b3 * V[i]
                
                # Condtiion (i)
                # Constrain veg to 0-1
                V[i+1] = np.maximum(Vmin, np.minimum(Vmax, V[i+1]))
            
            if return_vars == 'V':
                return V
            elif return_vars == 'all':
                return {'V':V, 'W':W, 'Dt':Dt}


class CholerMPR3(BaseModel):
    """
    The "PR3" four parameter model described in Choler et al. 2011
    
    Modified to use wilting point as an input instead of estimating it
    via the b1 parameter. Now 3 parameters.
    """
    def __init__(self, parameters={}):
        BaseModel.__init__(self)
        self.all_required_parameters = {'b2': (0, 100), 'b3': (0, 100),
                                        'b4': (0, 100), 'L': (1,30)}
        self._organize_parameters(parameters)
        #self._required_data = {'predictor_columns': ['site_id', 'year', 'doy', 'temperature'],
         #                      'predictors': ['pr','tasmin','tasmax']}
        self._required_predictors = {'precip': 'per_timestep',
                                     'evap'  : 'per_timestep',
                                     'Wcap'  : 'per_site',
                                     'Wp'    : 'per_site'}

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
                         Wp,      # wilting point, single value/site
                         #MAP,     # Mean avg precip, used to scale model input(gcc) to output (fcover)
                                  # fCover = GCC * MAP/ (MAP+h), where h is an estimated  parameter
                         
                         # Model parameters
                         #b1, Replaced by Wp in the modified model  
                         b2,
                         b3,
                         b4,
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
            Dtl  = np.empty_like(Wcap)
            Dtl1 = np.empty_like(Wcap)
            
            n_timesteps = precip.shape[0] - 1
            
            for i in range(1,n_timesteps):
                
                # if we are near the start of the timeseries then initialize
                # soil/plant water to something reasonable
                if i - L - 1 < 0:
                    Dt[i] = np.maximum(0, W[i] - Wp)
                    Dtl[:] = Wstart
                    Dtl1[:] = Wstart
                else:
                    Dt[i] = np.maximum(0, W[i] - Wp)
                    Dtl = Dt[i-L]
                    Dtl1 = Dt[i-L-1]
                
                # Condition (ii)
                # If plant available water is on the decline
                # then decay is 1 and senescensce sets in
                d = (Dtl <= Dtl1) * 1
                
                # Soil water
                W[i+1] = W[i] + precip[i] - (1 - V[i]) * ((Dt[i]/(Wcap - Wp))**2) * evap[i] - V[i]* b4 *Dt[i]
                W[i+1] = np.maximum(0, np.minimum(Wcap, W[i+1]))
                
                
                # Primary veg growth equation
                V[i+1] = V[i] + b2 * Dtl * (1 - (V[i]/Vmax)) - d * b3 * V[i]
                
                # Condtiion (i)
                # Constrain veg to 0-1
                V[i+1] = np.maximum(Vmin, np.minimum(Vmax, V[i+1]))
            
            if return_vars == 'V':
                return V
            elif return_vars == 'all':
                return {'V':V, 'W':W, 'Dt':Dt}
