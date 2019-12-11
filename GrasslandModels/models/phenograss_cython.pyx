import numpy as np
cimport numpy



def apply_model(# Site specific drivers
                 precip,  # precip, Daily vector
                 evap,    # potential ET, Daily vector
                 #T,       # ? mean temp ? not actually used in phenograss.f90
                 Ra,      # TOA radiation, MJ m-2 s-1, daily vector
                 Tm,      # Running mean T with 15 day lag
                 Wcap,    # field capacity, single value/site
                 Wp,      # wilting point, single value/site
                 MAP,     # Mean avg precip, used to scale model input(gcc) to output (fcover)
                          # fCover = GCC * MAP/ (MAP+h), where h is an estimated  parameter
                 
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
                 
                 # Normally just the V (vegatation cover) should be returned,
                 # but for diagnostics use 'all' to get V, W, and Dtl
                 return_vars = 'V'
                 ):
    
    L = int(L) # must be a whole number. and floats will be truncated.
    
    # b1 should be a parameter, but in the phenograss fortran code
    # it's set to Wp. 
    # All b params are +1, see  https://github.com/sdtaylor/GrasslandModels/issues/2
    b1 = Wp
    
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
    Dtl  = np.empty_like(Wp)
    Dtl1 = np.empty_like(Wp)
    
    # Eq. 10
    # Temperature response function
    g = ((Tmax - Tm) / (Tmax - Topt)) * (((Tm - Tmin) / (Topt - Tmin)) ** (Topt/(Tmax-Topt)))
    
    # Temperatures too hot or cold can result in NA values, so set
    # growth to 0 here.
    g[np.isnan(g)] = 0
    #raise RuntimeWarning('Temperature response g resolved to nan in timestep ' + str(i))

    # Eq. 8
    # Enforce sensence based on radation
    # TODO: this doesn't exactly match eq 8
    # The if statement here seems to set the bounds at 0-1.
    dor = (Ra - Phmin) / (Phmax - Phmin)
    # If Ra >= Phmax. Must be done vector wise
    dor[Ra >= Phmax] = 1
    # if Ra <Phmin
    dor[Ra <= Phmin] = 0  
    
    # TODO: checks on daily vector lengths, etc.
    n_timesteps = precip.shape[0] - 1
    
    for i in range(1,n_timesteps):
        
        # Eq. 4
        # if they are near the start of the timeseries then initialize
        # to something reasonable
        if i - L - 1 < 0:
            Dt[i] = np.maximum(0, W[i] - b1)
            Dtl[:] = Wstart
            Dtl1[:] = Wstart
        else:
            Dt[i] = np.maximum(0, W[i] - b1)
            Dtl = Dt[i-L]
            Dtl1 = Dt[i-L-1]
        
        # Eq. 7
        # If plant available water is on the decline
        # then decay is 1 and senescensce sets in via the last
        # part of Eq. 3
        d = (Dtl <= Dtl1) * 1
        #if Dtl > Dtl1:
        #    d = 0
        #else:
        #    d = 1
    
             
        
        # Eq. 2 Soil Water Content
        W[i+1] = W[i] + precip[i] - (1 - V[i]) * ((Dt[i]/(Wcap - b1))**2) * evap[i] - g * b4 * V[i] * Dt[i]
        
        # No negative SWC
        W[i + 1] = np.maximum(0, np.minimum(Wcap, W[i+1]))
        
        # Eq. 3 Vegetation growth
        # TODO: b2/b3 in the  fortran code, but b1/b2 in the paper math
        V[i+1] = V[i] + g * dor * b2 * Dtl * (1 - (V[i]/Vmax)) - d * b3 * V[i] * (1-V[i])
        
        # Constrain veg to 0-1
        V[i+1] = np.maximum(Vmin, np.minimum(Vmax, V[i+1]))
    
    scaling_factor = (MAP) / (MAP + h)
    V = V / scaling_factor
    if return_vars == 'V':
        return V
    elif return_vars == 'all':
        return V, W, Dt


def apply_model_cython(
                 # Site specific drivers
                 float[:,:] precip,  # precip, Daily vector
                 float[:,:] evap,    # potential ET, Daily vector
                 #T,       # ? mean temp ? not actually used in phenograss.f90
                 float[:,:] Ra,      # TOA radiation, MJ m-2 s-1, daily vector
                 float[:,:] Tm,      # Running mean T with 15 day lag
                 float[:] Wcap,    # field capacity, single value/site
                 float[:] Wp,      # wilting point, single value/site
                 float[:] MAP,     # Mean avg precip, used to scale model input(gcc) to output (fcover)
                          # fCover = GCC * MAP/ (MAP+h), where h is an estimated  parameter
                 
                 # Model parameters
                 #b1,  # Note b1 is set below to Wp as writtin the phenograss.f90. 
                       # TODO: sort that out
                 float b2,
                 float b3,
                 float b4,
                 int   L,
                 float Phmin,
                 float Topt,
                 float Phmax,
                 
                 float h, # This is from Eq. 1 to help scale the fCover. It's denoted a
                           # "slope" in the phenograss parameter files. 
                 # Constants
                 float Tmin = 0.,  # Maximum temperature of the growth response curve
                 float Tmax = 45.,
                 
                 float Vmin = 0.001, # Nees to be small non-zero value 
                 float Vmax = 1.,    # 100% cause GCC is scaled 0-1
                 
                 # Initial conditions
                 float W_initial = 0,
                 float Wstart    = 0.,
                 float V_initial = 0.001,
                 float Sd        = 0.,
                 float m         = 3600., # Not actaully used anywhere but in phenograss.f90
                 
                 # Normally just the V (vegatation cover) should be returned,
                 # but for diagnostics use 'all' to get V, W, and Dtl
                 #return_vars = 'V'
                 ):
    
    #L = int(L) # must be a whole number. and floats will be truncated.
    
    # b1 should be a parameter, but in the phenograss fortran code
    # it's set to Wp. 
    # All b params are +1, see  https://github.com/sdtaylor/GrasslandModels/issues/2
    cdef float[:] b1 = Wp
    
    # Initialize everything
    # Primary state variables
    cdef float[:,:] W = precip
    W[:] = W_initial
    
    cdef float[:,:] V = precip
    V[:] = V_initial

    # Plant available water
    cdef float[:,:] Dt = precip
    
    # Eq. 7, 8, 10 indicator vars
    cdef float g, d, dor
    
    # Lagged water availability, and  lagged minues 1 day
    cdef float Dtl, Dtl1
        
    #cdef float[:] scaling_factor = MAP / (MAP + h)
    
    # TODO: checks on daily vector lengths, etc.
    cdef int n_timesteps = precip.shape[0] - 1
    cdef int n_sites     = precip.shape[1]
    
    # This type is optimized for indexing
    cdef Py_ssize_t t,site
    for t in range(1,n_timesteps):
        for site in range(n_sites):
            
            # Eq. 4
            # if they are near the start of the timeseries then initialize
            # to something reasonable
            if t - L - 1 < 0:
                Dt[t,site] = max(0, W[t,site] - b1[site])
                Dtl = Wstart
                Dtl1 = Wstart
            else:
                Dt[t,site] = max(0, W[t,site] - b1[site])
                Dtl = Dt[t-L, site]
                Dtl1 = Dt[t-L-1, site]
            
            # Eq. 7
            # If plant available water is on the decline
            # then decay is 1 and senescensce sets in via the last
            # part of Eq. 3
            if Dtl > Dtl1:
                d = 0.
            else:
                d = 1.       
            
            # Eq. 10
            # Temperature response function
            g = ((Tmax - Tm[t, site]) / (Tmax - Topt)) * (((Tm[t, site] - Tmin) / (Topt - Tmin)) ** (Topt/(Tmax-Topt)))
            # Temperatures too hot or cold can result in NA values, so set
            # growth to 0 here. Nan filling in numba works on only 1D arrays, so a
            # quick reshape is needed here. 
            if np.isnan(g):
                g = 0
            
            # Eq. 8
            # Enforce sensence based on radation
            # TODO: this doesn't exactly match eq 8
            # The if statement here seems to set the bounds at 0-1.
            # See above for reshaping reasons.
            dor = (Ra[t,site] - Phmin) / (Phmax - Phmin)
            if Ra[t,site] >= Phmax:
                dor = 1
            elif Ra[t,site] <= Phmin:
                dor = 0   
            
            # Eq. 2 Soil Water Content
            W[t+1, site] = W[t, site] + precip[t, site] - (1 - V[t, site]) * ((Dt[t, site]/(Wcap[site] - b1[site]))**2) * evap[t, site] - g * b4 * V[t, site] * Dt[t, site]
            
            # No negative SWC
            W[t + 1, site] = max(0, min(Wcap[site], W[t+1,site]))
            
            # Eq. 3 Vegetation growth
            # TODO: b2/b3 in the  fortran code, but b1/b2 in the paper math
            V[t+1, site] = V[t,site] + g * dor * b2 * Dtl * (1 - (V[t,site]/Vmax)) - d * b3 * V[t,site] * (1-V[t,site])
            
            # Constrain veg to 0-1
            V[t+1,site] = max(Vmin, min(Vmax, V[t+1,site]))
    
    #V = V / scaling_factor
    return V