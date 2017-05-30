import matplotlib.pyplot as plt
import numpy.matlib
import numpy as np
from scipy import special, optimize
import warnings
import sys

def makeSimStruct_GLM(nkt,dtStim,dtSp):
#==============================================================================
#      Creates a list with default parameters for a GLM model
#      Input: nkt = number of time bins for stimulus filter
#             dtStim = bin size for sampling of stimulus kernel in sec
#             dtSp = bin size for sampling post-spike kernel in sec
#     
#      Output: (list)
#              filt - stimulus filter
#              nlfun - nonlinearity (exponential by default)
#              dc - dc input to cell
#              ih - post-spike current
#              ihbas - basis for post-spike current
#              iht - time lattice for post-spike current
#              dtsim - default time bin size for simulation
#              ihbasprs - basis for post-spike current
#==============================================================================
    
    # Check that stimulus bins bigger than spike bins
    if dtStim < dtSp:
        print 'dtStim must be greater than dtSp'
        sys.exit()
    
    # Check that spike bin size evenly divides stimulus bin size
    if (np.round(dtStim % dtSp) != 0):
        print 'makeSimStruct_GLM: dtSp doesn''t evenly divide dtStim: rounding dtSp to nearest even divisor'
        dtSp = dtStim/np.round(dtStim/dtSp)
        print 'dtSp reset to', dtSp
        
    # Create a default temporal stimulus filter
    tk = np.arange(nkt)
    b1 = nkt/32.0
    b2 = nkt/16.0
    k1 = (1/(special.gamma(6.0)*b1)*(tk/b1)**5)*np.exp(-tk/b1)
    k2 = (1/(special.gamma(6.0)*b2)*(tk/b2)**5)*np.exp(-tk/b2)
    k = np.flipud(k1-k2/1.5)
    k = k/np.linalg.norm(k)/2;
        
    # Represent this filter (approximately) in temporal basis
    ktbasprs = {}
    ktbasprs['neye'] = np.minimum(5,nkt) # Number of "identity" basis vectors near time of spike;
    ktbasprs['ncos'] = np.minimum(5,nkt) # Number of raised-cosine vectors to use
    ktbasprs['kpeaks'] = np.array([0,((nkt-ktbasprs['neye'])/2.0)]) # Position of 1st and last bump
    ktbasprs['b'] = 1.0 # Offset of nonlinear scailing (larger -> more linear)
    ktbas, _ = makeBasis_StimKernel(ktbasprs,nkt)
    k = ktbas.dot(np.linalg.lstsq(ktbas,k)[0]); # np.linalg.lstsq: solving the linear system using least square algorithm. MATLAB equivalent is a backslash operator
    
    # Nonlinearity
    nlinF = lambda x: expfun(x)
    
    # Make basis fro post-spike (h) filter
    ihbasprs = {}
    ihbasprs['ncols'] = 5 # Number of basis vectors for post-spike kernel
    ihbasprs['hpeaks'] = np.array([dtSp,dtSp*50.0]) # Peak location for first and last vectors
    ihbasprs['b'] = dtSp*5.0 # Determines how nonlinear to make spacings
    ihbasprs['absref'] = np.array([]) # absolute refractory period (optional)
    iht,_,ihbasis = makeBasis_PostSpike(ihbasprs,dtSp)
    ih = np.dot(ihbasis,np.array([-10,.5,1,.25,-1])[:,None])
    
    # Place parameters in structure
    S = {}
    S['k'] = 2*k # stimulus filter
    S['nlfun'] = nlinF # nonlinearity
    S['dc'] = 1.5 # dc input
    S['ih'] = ih # post-spike current
    S['iht'] = iht # time indices of aftercurrent
    S['dtStim'] = dtStim # time bin size for stimulus (s)
    S['dtSp'] = dtSp # time bin size for spikes (s)
    S['ihbasprs'] = ihbasprs # params for ih basis
    S['ktbasprs'] = ktbasprs # params for k time-basis

    return S

def makeBasis_PostSpike(ihprs,dt,iht0=None):
# Make nonlinearly stretched basis consisting of raised cosines
# -------
# Inputs: 
#     prs = param structure with fields:
#            ncols = # of basis vectors
#            hpeaks = 2-vector containg [1st_peak  last_peak], the peak 
#                     location of first and last raised cosine basis vectors
#            b = offset for nonlinear stretching of x axis:  y = log(x+b) 
#                     (larger b -> more nearly linear stretching)
#            absref = absolute refractory period (optional).  If specified,
#                     this param creates an additional "square" basis
#                     vector with support n [0,absref] (i.e., with a hard
#                     cutoff at absref)
#
#     dt = grid of time points for representing basis
#     iht (optional) = cut off time (or extend) basis so it matches this
#  --------
#  Outputs:  iht = time lattice on which basis is defined
#            ihbas = orthogonalized basis
#            ihbasis = original (non-orthogonal) basis 
#
#  -------------
#  Example call:
#
#  ihbasprs.ncols = 5;  
#  ihbasprs.hpeaks = [.1 2];  
#  ihbasprs.b = .5;  
#  ihbasprs.absref = .1;  (optional)
#  [iht,ihbas,ihbasis] = makeBasis_PostSpike(ihprs,dt);
    ncols = ihprs['ncols']
    b = ihprs['b']
    hpeaks = ihprs['hpeaks']
    if 'absref' in ihprs:
        absref = ihprs['absref']
    else:
        absref = 0
    
    # Check input values
    if (hpeaks[0]+b) < 0:
        sys.exit('b + first peak location: must be greater than 0')
    if absref >= dt: # use one fewer "cosine-shaped" basis vector    
        ncols = ncols - 1
    elif absref > 0:
        warnings.warn('Refractory period too small for time-bin sizes')
    
    # nonlinearity for stretching x axis (and its inverse)
    def nlin(x): return np.log(x+1e-20)
    def invnl(x): return np.exp(x)-1e-20
             
    #  Generate basis of raised cosines
    yrnge = nlin(hpeaks+b)
    db = np.diff(yrnge)[0]/(ncols-1)
    ctrs = np.arange(yrnge[0],yrnge[1]+db,db) # in matlab, it's yrnge(1):db:yrnge(2). length is always 1 less than matlab, so I have to manually add one more step.
    mxt = invnl(yrnge[1]+2*db)-b
    iht = np.arange(dt,mxt,dt)[:,None]
    nt = np.size(iht)
    def ff(x,c,dc): return (np.cos(np.maximum(-np.pi,np.minimum(np.pi,(x-c)*np.pi/dc/2)))+1)/2
    ihbasis = ff(np.squeeze(np.tile(nlin(iht+b)[:,None],ncols)),np.tile(ctrs,(nt,1)),db)
    
    # create first basis vector as step-function for absolute refractory period
    if absref >= dt:
        ii = np.nonzero(iht<absref)
        ih0 = np.zeros((ihbasis.shape[0],1))
        ih0[ii] = 1
        ihbasis[ii,:] = 0;
        ihbasis = np.concatenate((ih0,ihbasis),axis=1)
        
    # compute orthogonalized basis
    ihbas = orth(ihbasis)
    
    # Add more time bins (or remove bins) if this basis doesn't match iht0
    try:
        iht0
    except NameError:
        if iht0[1]-iht0[0] != dt:
            sys.exit('iht passed in has different time-bin size')
        niht = iht0.size
        if iht[-1] > iht0[-1]:
            iht = iht0
            ihbasis = ihbasis[0:niht-1,:]
            ihbas = ihbas[0:niht-1,:]
        elif iht[-1] < iht0[-1]:
            nextra = niht - iht.size
            iht = iht0
            ihbasis = np.concatenate((ihbasis,np.zeros((nextra,ncols))),axis=0)
            ihbas = np.concatenate((ihbas,np.zeros((nextra,ncols))),axis=0)
    else:
        pass
    
    return iht,ihbas,ihbasis

def expfun(x):
    # exponential nonlinearity with 3 outputs (first and second derivatives)
    return np.exp(x),np.exp(x),np.exp(x)

def makeBasis_StimKernel(ktbasprs,nkt):
#==============================================================================
#     Generates a basis consisting of raised cosines and several columns of
#     identity matrix vectors for temporal structure of stimulus kernel
#
#     Args: kbasprs = dictionary with fields:
#               neye = number of identity basis vectors at front
#               ncos = # of vectors that are raised cosines
#               kpeaks = 2-vector, with peak position of 1st and last vector,
#                   relative to start of cosine basis vectors (e.g. [0 10])
#               b = offset for nonlinear scaling. larger values -> more linear
#                   scaling of vectors. bk must be >= 0
#           nkt = number of time samples in basis (optional)
#
#     Output:
#           kbasorth = orthogonal basis
#           kbasis = standard raised cosine (non-orthogonal) basis
#==============================================================================
    neye = ktbasprs['neye']
    ncos = ktbasprs['ncos']
    kpeaks = ktbasprs['kpeaks']
    b = ktbasprs['b']
    
    kdt = 1.0; # spacing of x axis must be in units of 1
    
    # nonlinearity for stretching x axis (and its inverse)
    def nlin(x): return np.log(x+1e-20)
    def invnl(x): return np.exp(x)-1e-20
             
    # Generate basis or raised cosines
    yrnge = nlin(kpeaks+b)
    db = np.diff(yrnge)/(ncos-1)
    ctrs = np.arange(yrnge[0],yrnge[1]+db,db) # in matlab, it's yrnge(1):db:yrnge(2). length is always 1 less than matlab, so I have to manually add one more step.
    mxt = invnl(yrnge[1]+2*db)-b
    kt0 = np.arange(0,mxt,kdt)[:,None]
    nt = np.size(kt0)
    def ff(x,c,dc): return (np.cos(np.maximum(-np.pi,np.minimum(np.pi,(x-c)*np.pi/dc/2)))+1)/2
    kbasis0 = ff(np.squeeze(np.tile(nlin(kt0+b)[:,None],ncos)),np.tile(ctrs,(nt,1)),db[0])
    
    # Concatenate identity-vectors
    nkt0,nkttemp = kt0.shape
    kbasis = np.concatenate((np.concatenate((np.identity(neye),np.zeros((nkt0,neye))),axis=0),np.concatenate((np.zeros((neye,ncos)),kbasis0),axis=0)),axis=1)
    kbasis = np.flipud(kbasis) # flip so fine timescales are at the end
    nkt0,nkttemp = kbasis.shape
    
    try:
        nkt
    except:
        pass
    else:
       if nkt0 < nkt:
           kbasis = np.concatenate((np.zeros((nkt-nkt0,ncos+neye)),kbasis),axis=0)
       elif nkt0 > nkt:
           kbasisM,kbasisN = kbasis.shape
           kbasis = kbasis[kbasisM-(nkt+1)+1:kbasisM][:]
    
    # normalize columns to be unit vectors
    for i in range(0,kbasisN):
        kbasis[:,i] = kbasis[:,i]/np.sqrt(np.sum(kbasis[:,i]**2))
    
    kbasorth = orth(kbasis)
    
    return kbasorth,kbasis

def orth(A):
#==============================================================================
#     ORTH     Orthogonalization.
#           Q = ORTH(A) is an orthonormal basis for the range of A. That is, Q'
#           Q = I, the columns of Q span the same space as the columns of A, and
#           the number of columns of Q is the rank of A.
#           
#==============================================================================
    Q,s,v = np.linalg.svd(A)
    vM,vN = v.shape
    Q = Q[:,0:vM]
    AM,AN = A.shape
    tol = np.maximum(AM,AN)*np.finfo(float).eps
    r = (s>tol).sum()
    QM,QN = Q.shape
    Q[:,r:QN] = np.empty([QM,np.arange(r,QN).shape[0]])
    
    return Q

def normpdf(x,mu = 0,sigma = 1):
#NORMPDF Normal probability density function (pdf).
#   Y = NORMPDF(X,MU,SIGMA) returns the pdf of the normal distribution with
#   mean MU and standard deviation SIGMA, evaluated at the values in X.
#   The size of Y is the common size of the input arguments.  A scalar
#   input functions as a constant matrix of the same size as the other
#   inputs.
#
#   Default values for MU and SIGMA are 0 and 1 respectively.
#
#   See also NORMCDF, NORMFIT, NORMINV, NORMLIKE, NORMRND, NORMSTAT.
#
#   References:
#      [1] Evans, M., Hastings, N., and Peacock, B. (1993) Statistical
#          Distributions, 2nd ed., Wiley, 170pp.

    try:
        y = np.exp(-0.5*(((x-mu)/(sigma))**2.0))/(np.sqrt(2.0*np.pi)*sigma)
    except:
        print 'error: Input size mismatch'
        sys.exit()
    
    return y[0]

def simGLM(glmprs,Stim):
# Compute response of glm to stimulus Stim.
#
# Uses time rescaling instead of Bernouli approximation to conditionally
# Poisson process
#
# Dynamics:  Filters the Stimulus with glmprs.k, passes this through a
# nonlinearity to obtain the point-process conditional intensity.  Add a
# post-spike current to the linear input after every spike.
#
# Input: 
#   glmprs - struct with GLM params, has fields 'k', 'h','dc' for params
#              and 'dtStim', 'dtSp' for discrete time bin size for stimulus
#              and spike train (in s).
#     Stim - stimulus matrix, with time running vertically and each
#              column corresponding to a different pixel / regressor.
# Output:
#   tsp - list of spike times (in s)
#   sps - binary matrix with spike times (at resolution dtSp).
#  Itot - summed filter outputs 
#  Istm - just the spike-history filter output

    def simGLMsingle(glmprs,Stim,upSampFactor):
        # Sub-function within simGLM
        #
        # Simulates the GLM point process model for a single neuron using time-rescaling
        nbinsPerEval = 100 # Default number of bins to update for each spike
        dt = glmprs['dtSp'] # bin size for simulations
        
        slen = Stim.shape[0] # length of stimulus
        rlen = slen*upSampFactor # length of binned spike response
        hlen = glmprs['ih'].shape[0] # length of post-spike filter
        
        print slen,rlen,hlen
        tsp = 0
        sps = 0
        Itot = 0
        Istm = 0
        return tsp,sps,Itot,Istm
    
    def simGLMcpl(glmprs,Stim,upSampFactor):
        tsp = 0
        sps = 0
        Itot = 0
        Istm = 0
        return tsp,sps,Itot,Istm
    
    
    # Check nonlinearity (default is exponential)
    if 'nlfun' in glmprs:
        pass
    else:
        glmprs['nlfun'] = lambda x: np.expfun(x)
        
    upSampFactor = glmprs['dtStim']/glmprs['dtSp'] # number of spike bins per Stim bin
    assert (np.mod(upSampFactor,1) == 0)
    
    # Determine which version to run
    if len(glmprs['k'].shape) > 2:
        tsp,sps,Itot,Istm = simGLMcpl(glmprs,Stim,upSampFactor)
    else:
        tsp,sps,Itot,Istm = simGLMsingle(glmprs,Stim,upSampFactor)
        
    return tsp,sps,Itot,Istm

