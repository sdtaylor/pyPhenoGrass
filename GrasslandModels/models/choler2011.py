from . import utils
from .base import BaseModel
import numpy as np


class CholerPR1(BaseModel):
    """
    The "PR1" model described in Choler et al. 2011
    """
    def __init__(self, parameters={}):
        BaseModel.__init__(self)
        self.all_required_parameters = {'b1': (0, 100), 'b2': (0, 100), 
                                        'b3': (0, 100), 'L': (1,30)}
        self._organize_parameters(parameters)
        self._required_predictors = {'precip': 'per_timestep'}

        self.state_variables = ['V','Dt']

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
                         
                         # Model parameters
                         b1,  
                         b2,
                         b3,
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
            L = int(L) # must be a whole number, any floats will be truncated.
                       
            V = np.empty_like(precip).astype('float32')
            V[:] = V_initial
    
            # Derived variables
            Dt = np.zeros_like(precip).astype('float32')
            
            # Site level vars such as lagged plant-water and
            # temp responses
            # Initialize empty to the shape (,n_sites)
            Dtl  = np.empty_like(precip[0])
            Dtl1 = np.empty_like(precip[0])
            
            n_timesteps = precip.shape[0] - 1
            
            for i in range(1,n_timesteps):
                
                # if we are near the start of the timeseries then initialize
                # soil/plant water to something reasonable
                if i - L - 1 < 0:
                    Dt[i] = np.maximum(0, precip[i] - b1)
                    Dtl[:] = Wstart
                    Dtl1[:] = Wstart
                else:
                    Dt[i] = np.maximum(0, precip[i] - b1)
                    Dtl = Dt[i-L]
                    Dtl1 = Dt[i-L-1]
                
                # Condition (ii)
                # If plant available water is on the decline
                # then decay is 1 and senescensce sets in
                d = (Dtl <= Dtl1) * 1
                
                # Primary veg growth equation
                V[i+1] = V[i] + b2 * Dtl * (1 - (V[i]/Vmax)) - d * b3 * V[i]
                
                # Condtiion (i)
                # Constrain veg to 0-1
                V[i+1] = np.maximum(Vmin, np.minimum(Vmax, V[i+1]))
            
            if return_vars == 'V':
                return V
            elif return_vars == 'all':
                return {'V':V,'Dt':Dt}

class CholerPR1Gcc(BaseModel):
    """
    The "PR1" model described in Choler et al. 2011.
    Made to work with Phenocam Gcc with the MAP transformation
    """
    def __init__(self, parameters={}):
        BaseModel.__init__(self)
        self.all_required_parameters = {'b1': (0, 100), 'b2': (0, 100), 
                                        'b3': (0, 100), 'L': (1,30),
                                        'h': (0,1000)}
        self._organize_parameters(parameters)
        self._required_predictors = {'precip': 'per_timestep',
                                     'MAP'   : 'per_site'}

        self.state_variables = ['V','Dt','fCover']

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
                         MAP,     # mean annual precip, per site
                         
                         # Model parameters
                         b1,  
                         b2,
                         b3,
                         L,
                         h,
                         
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
            L = int(L) # must be a whole number, any floats will be truncated.
                       
            V = np.empty_like(precip).astype('float32')
            V[:] = V_initial
    
            # Derived variables
            Dt = np.zeros_like(precip).astype('float32')
            
            # Site level vars such as lagged plant-water and
            # temp responses
            # Initialize empty to the shape (,n_sites)
            Dtl  = np.empty_like(precip[0])
            Dtl1 = np.empty_like(precip[0])
            
            n_timesteps = precip.shape[0] - 1
            
            for i in range(1,n_timesteps):
                
                # if we are near the start of the timeseries then initialize
                # soil/plant water to something reasonable
                if i - L - 1 < 0:
                    Dt[i] = np.maximum(0, precip[i] - b1)
                    Dtl[:] = Wstart
                    Dtl1[:] = Wstart
                else:
                    Dt[i] = np.maximum(0, precip[i] - b1)
                    Dtl = Dt[i-L]
                    Dtl1 = Dt[i-L-1]
                
                # Condition (ii)
                # If plant available water is on the decline
                # then decay is 1 and senescensce sets in
                d = (Dtl <= Dtl1) * 1
                
                # Primary veg growth equation
                V[i+1] = V[i] + b2 * Dtl * (1 - (V[i]/Vmax)) - d * b3 * V[i]
                
                # Condtiion (i)
                # Constrain veg to 0-1
                V[i+1] = np.maximum(Vmin, np.minimum(Vmax, V[i+1]))
            
            fCover = V[:]
        
            scaling_factor = MAP / (MAP + h)
            V = V / scaling_factor
            
            if return_vars == 'V':
                return V
            elif return_vars == 'all':
                return {'V':V,'Dt':Dt,'fCover':fCover}

class CholerPR2(BaseModel):
    """
    The "PR2" four parameter model described in Choler et al. 2011
    """
    def __init__(self, parameters={}):
        BaseModel.__init__(self)
        self.all_required_parameters = {'b1': (0, 100), 'b2': (0, 100), 
                                        'b3': (0, 100), 'b4': (0, 100), 
                                        'L': (1,30)}
        self._organize_parameters(parameters)
        self._required_predictors = {'precip': 'per_timestep',
                                     'evap'  : 'per_timestep',
                                     'Wcap'  : 'per_site'}

        self.state_variables = ['V','W','Dt']
        
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
                         Wcap,    # field capacity, single value/site
                         
                         # Model parameters
                         b1,  
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
                         # but for diagnostics use 'all' to get V, W, and Dt
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
                    Dt[i] = np.maximum(0, W[i] - b1)
                    Dtl[:] = Wstart
                    Dtl1[:] = Wstart
                else:
                    Dt[i] = np.maximum(0, W[i] - b1)
                    Dtl = Dt[i-L]
                    Dtl1 = Dt[i-L-1]
                
                # Condition (ii)
                # If plant available water is on the decline
                # then decay is 1 and senescensce sets in
                d = (Dtl <= Dtl1) * 1
                
                # Soil water
                W[i+1] = W[i] + precip[i] - b4 * ((Dt[i]/(Wcap - b1))**2) * evap[i]
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

class CholerPR2Gcc(BaseModel):
    """
    The "PR2" four parameter model described in Choler et al. 2011
    Made to work with Phenocam Gcc with the MAP transformation
    """
    def __init__(self, parameters={}):
        BaseModel.__init__(self)
        self.all_required_parameters = {'b1': (0, 100), 'b2': (0, 100), 
                                        'b3': (0, 100), 'b4': (0, 100), 
                                        'L': (1,30), 'h': (0,1000)}
        self._organize_parameters(parameters)
        self._required_predictors = {'precip': 'per_timestep',
                                     'evap'  : 'per_timestep',
                                     'Wcap'  : 'per_site',
                                     'MAP'   : 'per_site'}

        self.state_variables = ['V','W','Dt','fCover']
        
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
                         Wcap,    # field capacity, single value/site
                         MAP,     # Mean annual precip, single value/site
                         # Model parameters
                         b1,  
                         b2,
                         b3,
                         b4,
                         L,
                         h,
                         
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
                         # but for diagnostics use 'all' to get V, W, and Dt
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
                    Dt[i] = np.maximum(0, W[i] - b1)
                    Dtl[:] = Wstart
                    Dtl1[:] = Wstart
                else:
                    Dt[i] = np.maximum(0, W[i] - b1)
                    Dtl = Dt[i-L]
                    Dtl1 = Dt[i-L-1]
                
                # Condition (ii)
                # If plant available water is on the decline
                # then decay is 1 and senescensce sets in
                d = (Dtl <= Dtl1) * 1
                
                # Soil water
                W[i+1] = W[i] + precip[i] - b4 * ((Dt[i]/(Wcap - b1))**2) * evap[i]
                W[i+1] = np.maximum(0, np.minimum(Wcap, W[i+1]))
                
                # Primary veg growth equation
                V[i+1] = V[i] + b2 * Dtl * (1 - (V[i]/Vmax)) - d * b3 * V[i]
                
                # Condtiion (i)
                # Constrain veg to 0-1
                V[i+1] = np.maximum(Vmin, np.minimum(Vmax, V[i+1]))
            
            fCover = V[:]
        
            scaling_factor = MAP / (MAP + h)
            V = V / scaling_factor
            
            if return_vars == 'V':
                return V
            elif return_vars == 'all':
                return {'V':V,'W':W,'Dt':Dt,'fCover':fCover}

class CholerPR3(BaseModel):
    """
    The "PR3" four parameter model described in Choler et al. 2011
    """
    def __init__(self, parameters={}):
        BaseModel.__init__(self)
        self.all_required_parameters = {'b1': (0, 100), 'b2': (0, 100), 
                                        'b3': (0, 100), 'b4': (0, 100), 
                                        'L': (1,30)}
        self._organize_parameters(parameters)
        self._required_predictors = {'precip': 'per_timestep',
                                     'evap'  : 'per_timestep',
                                     'Wcap'  : 'per_site'}

        self.state_variables = ['V','W','Dt']
        
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
                         Wcap,    # field capacity, single value/site
                         
                         # Model parameters
                         b1,  
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
                         # but for diagnostics use 'all' to get V, W, and Dt
                         return_vars = 'V'
                         ):
            """
            
            """
            L = int(L) # must be a whole number, any floats will be truncated.
            
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
                    Dt[i] = np.maximum(0, W[i] - b1)
                    Dtl[:] = Wstart
                    Dtl1[:] = Wstart
                else:
                    Dt[i] = np.maximum(0, W[i] - b1)
                    Dtl = Dt[i-L]
                    Dtl1 = Dt[i-L-1]
                
                # Condition (ii)
                # If plant available water is on the decline
                # then decay is 1 and senescensce sets in
                d = (Dtl <= Dtl1) * 1
                
                # Soil water
                W[i+1] = W[i] + precip[i] - (1 - V[i]) * ((Dt[i]/(Wcap - b1))**2) * evap[i] - V[i]* b4 *Dt[i]
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


class CholerPR3Gcc(BaseModel):
    """
    The "PR3" four parameter model described in Choler et al. 2011
    Made to work with Phenocam Gcc with the MAP transformation
    """
    def __init__(self, parameters={}):
        BaseModel.__init__(self)
        self.all_required_parameters = {'b1': (0, 100), 'b2': (0, 100), 
                                        'b3': (0, 100), 'b4': (0, 100), 
                                        'L': (1,30), 'h':(0,1000)}
        self._organize_parameters(parameters)
        self._required_predictors = {'precip': 'per_timestep',
                                     'evap'  : 'per_timestep',
                                     'Wcap'  : 'per_site',
                                     'MAP'   : 'per_site'}

        self.state_variables = ['V','W','Dt','fCover']
        
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
                         Wcap,    # field capacity, single value/site
                         MAP,     # Mean ann. precip, single value/site
                         
                         # Model parameters
                         b1,  
                         b2,
                         b3,
                         b4,
                         h,
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
                         # but for diagnostics use 'all' to get V, W, and Dt
                         return_vars = 'V'
                         ):
            """
            
            """
            L = int(L) # must be a whole number, any floats will be truncated.
            
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
                    Dt[i] = np.maximum(0, W[i] - b1)
                    Dtl[:] = Wstart
                    Dtl1[:] = Wstart
                else:
                    Dt[i] = np.maximum(0, W[i] - b1)
                    Dtl = Dt[i-L]
                    Dtl1 = Dt[i-L-1]
                
                # Condition (ii)
                # If plant available water is on the decline
                # then decay is 1 and senescensce sets in
                d = (Dtl <= Dtl1) * 1
                
                # Soil water
                W[i+1] = W[i] + precip[i] - (1 - V[i]) * ((Dt[i]/(Wcap - b1))**2) * evap[i] - V[i]* b4 *Dt[i]
                W[i+1] = np.maximum(0, np.minimum(Wcap, W[i+1]))
                
                
                # Primary veg growth equation
                V[i+1] = V[i] + b2 * Dtl * (1 - (V[i]/Vmax)) - d * b3 * V[i]
                
                # Condtiion (i)
                # Constrain veg to 0-1
                V[i+1] = np.maximum(Vmin, np.minimum(Vmax, V[i+1]))
            
            fCover = V[:]
        
            scaling_factor = MAP / (MAP + h)
            V = V / scaling_factor
            
            if return_vars == 'V':
                return V
            elif return_vars == 'all':
                return {'V':V,'W':W,'Dt':Dt,'fCover':fCover}