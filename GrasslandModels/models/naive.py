from . import utils
from .base import BaseModel
import numpy as np


class Naive(BaseModel):
    """
    A simple model where NDVI is a linear function of lagged precip.
    
    NDVI_t ~ b1 + b2 * (sum(precip_(t-L) - precip_(t)))
    
    where L is a lag in timesteps and is estimated along with b1 and b2. 
    """
    def __init__(self, parameters={}):
        BaseModel.__init__(self)
        self.all_required_parameters = {'b1': (0, 100), 'b2': (0, 100), 
                                        'L': (1,6)}
        self._organize_parameters(parameters)
        self._required_predictors = {'precip': 'per_timestep'}
        
        self.state_variables = ['V']
        
    def _apply_model(self,
                     # Site specific drivers
                     precip,  # precip, Daily vector
                    
                     # Model parameters
                     b1,  
                     b2,
                     L,
                     return_vars = 'V'
                     ):
            """
            
            """
            L = int(L) # must be a whole number. any floats will be truncated.
            
            summed_precip = np.zeros_like(precip)
            for t in range(L, precip.shape[0]):
                summed_precip[t] = precip[t-L:(t+1)].sum(0)
            
            V = b1 + b2*summed_precip
            
            if return_vars == 'V':
                return V
            elif return_vars == 'all':
                return {'V':V}
            
            
class Naive2(BaseModel):
    """
    A simple model where NDVI is a linear function of lagged precip and 
    current potential evapotranspiration.
    
    NDVI_t ~ b1 + b2 * (sum(precip_(t-L) - precip_(t))) + b3 * evap
    
    where L is a lag in timesteps and is estimated along with b1,b2, and b3. 
    """
    def __init__(self, parameters={}):
        BaseModel.__init__(self)
        self.all_required_parameters = {'b1': (0, 100), 'b2': (0, 100), 
                                        'b3': (0,100),'L': (1,6)}
        self._organize_parameters(parameters)
        self._required_predictors = {'precip': 'per_timestep',
                                     'evap'  : 'per_timestep'}
        
        self.state_variables = ['V']
        
    def _apply_model(self,
                     # Site specific drivers
                     precip,  # precip, Daily vector
                     evap,
                    
                     # Model parameters
                     b1,  
                     b2,
                     b3,
                     L,
                     return_vars = 'V'
                     ):
            """
            
            """
            L = int(L) # must be a whole number. any floats will be truncated.
            
            summed_precip = np.zeros_like(precip)
            for t in range(L, precip.shape[0]):
                summed_precip[t] = precip[t-L:(t+1)].sum(0)
            
            V = b1 + b2*summed_precip + b3*evap
            
            if return_vars == 'V':
                return V
            elif return_vars == 'all':
                return {'V':V}
            
class NaiveMAPCorrected(BaseModel):
    """
    A simple model where the VI is a linear function of lagged precip.
    
    VI_t ~ b1 + b2 * (sum(precip_(t-L) - precip_(t)))
    
    where L is a lag in timesteps and is estimated along with b1 and b2. 
    
    This is designed for Phenocam GCC data, where the site GCC is scaled 
    via mean annual precip (MAP) and  the h parameter.
    
    See Hufkins et al. 2016 or the PhenoGrass mdoel for a description of this. 
    """
    def __init__(self, parameters={}):
        BaseModel.__init__(self)
        self.all_required_parameters = {'b1': (-100, 100), 'b2': (-100, 100), 
                                        'L': (1,6),      'h': (0,1000)}
        self._organize_parameters(parameters)
        self._required_predictors = {'precip': 'per_timestep',
                                     'MAP'   : 'per_site'}
        
        self.state_variables = ['V', 'fCover']
        
    def _apply_model(self,
                     # Site specific drivers
                     precip,  # precip, Daily vector
                     MAP,     # mean average precip
                     # Model parameters
                     b1,  
                     b2,
                     L,
                     h,
                     return_vars = 'V'
                     ):
            """
            
            """
            L = int(L) # must be a whole number. any floats will be truncated.
            
            summed_precip = np.zeros_like(precip)
            for t in range(L, precip.shape[0]):
                summed_precip[t] = precip[t-L:(t+1)].sum(0)
            
            V = b1 + b2*summed_precip
            
            fCover = V[:]
            
            scaling_factor = MAP / (MAP + h)
            V = V / scaling_factor
            
            if return_vars == 'V':
                return V
            elif return_vars == 'all':
                return {'V':V, 'fCover':fCover}
            
            
class Naive2MAPCorrected(BaseModel):
    """
    A simple model where VI is a linear function of lagged precip and 
    current potential evapotranspiration.
    
    VI_t ~ b1 + b2 * (sum(precip_(t-L) - precip_(t))) + b3 * evap
    
    where L is a lag in timesteps and is estimated along with b1,b2, and b3. 
    
    This is designed for Phenocam GCC data, where the site GCC is scaled 
    via mean annual precip (MAP) and  the h parameter.
    
    See Hufkins et al. 2016 or the PhenoGrass mdoel for a description of this. 
    """
    def __init__(self, parameters={}):
        BaseModel.__init__(self)
        self.all_required_parameters = {'b1': (-100, 100), 'b2': (-100, 100), 
                                        'b3': (-100,100),'L': (1,6),
                                        'h':  (0,1000)}
        self._organize_parameters(parameters)
        self._required_predictors = {'precip': 'per_timestep',
                                     'evap'  : 'per_timestep',
                                     'MAP'   : 'per_site'}
        
        self.state_variables = ['V', 'fCover']
        
    def _apply_model(self,
                     # Site specific drivers
                     precip,  # precip, Daily vector
                     evap,
                     MAP,
                     
                     # Model parameters
                     b1,  
                     b2,
                     b3,
                     L,
                     h,
                     return_vars = 'V'
                     ):
            """
            
            """
            L = int(L) # must be a whole number. any floats will be truncated.
            
            summed_precip = np.zeros_like(precip)
            for t in range(L, precip.shape[0]):
                summed_precip[t] = precip[t-L:(t+1)].sum(0)
            
            V = b1 + b2*summed_precip + b3*evap
            
            fCover = V[:]
            
            scaling_factor = MAP / (MAP + h)
            V = V / scaling_factor
            
            if return_vars == 'V':
                return V
            elif return_vars == 'all':
                return {'V':V, 'fCover':fCover}