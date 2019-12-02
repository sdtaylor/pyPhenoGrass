from . import utils
from .base import BaseModel
import numpy as np


class PhenoGrass(BaseModel):
    """PhenoGrass Model

    """

    def __init__(self, parameters={}):
        BaseModel.__init__(self)
        self.all_required_parameters = {'a1': (0, 100), 'a2': (0, 100), 
                                        'a3': (0, 100), 'L': (0,10)}
        self._organize_parameters(parameters)
        #self._required_data = {'predictor_columns': ['site_id', 'year', 'doy', 'temperature'],
         #                      'predictors': ['pr','tasmin','tasmax']}
        self._required_predictors = {'precip': 'per_timestep',
                                     'evap'  : 'per_timestep',
                                     'Ra' : 'per_timestep',
                                     'Tm'   : 'per_timestep',
                                     'Wcap'  : 'per_site',
                                     'Wp'    : 'per_site'}

    def _apply_model(self,
                     # Site specific drivers
                     precip,  # precip, Daily vector
                     evap,    # potential ET, Daily vector
                     #T,       # ? mean temp ? not actually used in phenograss.f90
                     Ra,      # TOA radiation, MJ m-2 s-1, daily vector
                     Tm,      # Running mean T with 15 day lag
                     Wcap,    # field capacity, single value/site
                     Wp,      # wilting point, single value/site
                     
                     # Model parameters
                     b1,  # Note b1 is set below to Wp as writtin the phenograss.f90. 
                           # TODO: sort that out
                     b2,
                     b3,
                     b4,
                     L,
                     Phmin,
                     Topt,
                     Phmax,
                     
                     h = None, # This is from Eq. 1 to help scale the fCover. It's denoted a
                               # "slope" in the phenograss parameter files. 
                     # Constants
                     Tmin = 0,  # Maximum temperature of the growth response curve
                     Tmax = 45,
                     
                     Vmin = 0.001, # Nees to be small non-zero value 
                     Vmax = 1.,    # 100% cause GCC is scaled 0-1
                     d    = 0,     # decay flag
                     
                     # Initial conditions
                     W_initial = 0,
                     Wstart    = 0,
                     V_initial = 0.001,
                     Sd        = 0,
                     m         = 3600, # Not actaully used anywhere but in phenograss.f90
                     ):
        
        L = int(L) # must be a whole number. and floats will be truncated.
        
        # b1 should be a parameter, but in the phenograss fortran code
        # it's set to Wp. 
        # All b params are +1, see  https://github.com/sdtaylor/GrasslandModels/issues/2
        b1 = Wp
        
        # Initialze state variables
        W = np.empty_like(precip)
        W[:] = W_initial
        
        V = np.empty_like(precip)
        V[:] = V_initial
        
        # Initialize empty vectors of derived variables
        Dt = np.zeros_like(precip)
        
        # TODO: checks on daily vector lengths, etc.
        n_timesteps = len(precip) - 1
        
        for i in range(1,n_timesteps):
            
            # Eq. 4
            # if they are near the start of the timeseries then initialize
            # to something reasonable
            if i - L - 1 < 0:
                Dt[i] = np.max([0, W[i] - b1])
                Dtl = Wstart
                Dtl1 = Wstart
            else:
                Dt[i] = np.max([0, W[i] - b1])
                Dtl = Dt[i-L]
                Dtl1 = Dt[i-L-1]
            
            # Eq. 7
            # If plant available water is on the decline
            # then decay is 1 and senescensce sets in via the last
            # part of Eq. 3
            if Dtl > Dtl1:
                d = 0
            else:
                d = 1
        
            # Eq. 10
            # Temperature response function
            g = ((Tmax - Tm[i]) / (Tmax - Topt)) * (((Tm[i] - Tmin) / (Topt - Tmin)) ** (Topt/(Tmax-Topt)))
            
            # Temperatures too hot or cold can result in NA values, so set
            # growth to 0 here.
            if np.isnan(g):
                g = 0
                #raise RuntimeWarning('Temperature response g resolved to nan in timestep ' + str(i))
        
            # Eq. 8
            # Enforce sensence based on radation
            # TODO: this doesn't exactly match eq 8
            # The if statement here seems to set the bounds at 0-1.
            dor = (Ra[i] - Phmin) / (Phmax - Phmin)
            if Ra[i] >= Phmax:
                dor = 1
            elif Ra[i] <= Phmin:
                dor = 0                
            
            # Eq. 2 Soil Water Content
            W[i+1] = W[i] + precip[i] - (1 - V[i]) * ((Dt[i]/(Wcap - b1))**2) * evap[i] - g * b4 * V[i] * Dt[i]
            
            # No negative SWC
            W[i + 1] = max(0, min(Wcap, W[i+1]))
            
            # Eq. 3 Vegetation growth
            # TODO: b2/b3 in the  fortran code, but b1/b2 in the paper math
            V[i+1] = V[i] + g * dor * b2 * Dtl * (1 - (V[i]/Vmax)) - d * b3 * V[i] * (1-V[i])
            
            # Constrain veg to 0-1
            print(V[i+1])
            V[i+1] = max(Vmin, min(Vmax, V[i+1]))
            print(V[i+1])
            print('##')
            
        return V