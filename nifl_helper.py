## Helper functions:
## Load in flowlines from CSV
## Identify evenly-spaced points along them
## Pull given dataset at these points
## Export a point time series to CSV
from math import sqrt
from scipy import interpolate
import collections
import datetime
import numpy as np
import iceutils as ice



def Flowline_CSV(filename, has_width=False, flip_order=False):
    """Read in glacier flowlines in CSV format.
    Based on automatically-selected flowline format produced by EHU.

    Parameters
    ----------
    filename : string
        Name/path of CSV.
    has_width : Boolean, optional
        Whether the flowline has glacier width stored. The default is False.
    flip_order : Boolean, optional
        Whether lines need flipping to run from terminus upstream. The default is False.

    Returns
    -------
    centrelines_list : list
        list of flowlines
    """
    
    f = open(filename,'r')
    
    header = f.readline() #header line
    hdr = header.strip('\r\n')
    keys = hdr.split(',') #get names of variables
    data = {k : [] for k in keys} 
    data['Length_ID'] = collections.OrderedDict() #new dictionary that counts how many points (i.e. lines of file) are in each flowline.  Must be ordered for later iteration!
    data['Lineslist'] = []
    
    lines = f.readlines()
    f.close()
    
    temp = []
    j = 0
    for i,l in enumerate(lines):
        linstrip = l.strip('\r\n')
        parts = linstrip.split(',')
        
        x_coord = float(parts[1])
        y_coord = float(parts[2])
        
        if parts[0] not in data['Length_ID'].keys(): #finding out where lines separate 
            temp = []
            data['Lineslist'].append(temp) #initialize new empty array that can be modified in-place later
            data['Length_ID'][parts[0]] = 1
            j+=1 
        else:
            data['Length_ID'][parts[0]] += 1
        
        if has_width:
            width = float(parts[3])
            temp.append((x_coord, y_coord, width))
        else:
            temp.append((x_coord, y_coord))
            
        data['Lineslist'][j-1] = np.array(temp) #need to modify an existing array rather than append to keep correct indexing
    
    nlines = len(data['Length_ID'].keys())
    
    if flip_order:          
        centrelines_list = [np.array(data['Lineslist'][j])[::-1] for j in range(nlines)] #making arrays, reversed to start at terminus rather than peak
    else:
        centrelines_list = [np.array(data['Lineslist'][j]) for j in range(nlines)] # arrays already start at terminus
    return centrelines_list

def ds(num, flarr):
    """Find path length between adjacent entries of a 2D array.
    

    Parameters
    ----------
    num : int
        Index of first array element.
    flarr : array
        The array containing 2D coordinate positions to examine.

    Returns
    -------
    ds : float
        Cartesian distance between flarr[num] and flarr[num-1].
    """
    if num==0:
        return 0
    else:
        return sqrt((flarr[num,0]-flarr[num-1,0])**2 + (flarr[num,1]-flarr[num-1,1])**2)

def ArcArray(flowline):
    """Compute arc length along a flowline of coordinates

    Parameters
    ----------
    flowline : array
        2D array of coordinates (xi, yi).

    Returns
    -------
    arcarr: array
        Corresponding arc length values at each point
    """
    arcarr = []
    xs = range(len(flowline))
    for n in xs:
        if n==0:
            arcarr.append(0)
        else:
            arcarr.append(arcarr[-1]+ds(n, flowline))
    return arcarr

## Set up design matrix and perform lasso regression, according to Bryan's documentation
def build_collection(dates):
    """
    Function that creates a list of basis functions for a given datetime vector dates.
    """
    # Get date bounds
    tstart, tend = dates[0], dates[-1]

    # Initalize a collection and relevant basis functions
    collection = ice.tseries.timefn.TimefnCollection()
    bspl = ice.tseries.timefn.fnmap['bspline']
    ispl = ice.tseries.timefn.fnmap['isplineset']
    poly = ice.tseries.timefn.fnmap['poly']

    # Add polynomial first for secular components
    collection.append(poly(tref=tstart, order=1, units='years'))

    # Use B-splines for seasonal (short-term) signals
    Δtdec = 1.0 / 3.0 # years
    Δt = datetime.timedelta(days=int(Δtdec*365))
    t_current = tstart
    while t_current <= tend:
        collection.append(bspl(order=3, scale=Δtdec, units='years', tref=t_current))
        t_current += Δt
        
    # Integrated B-slines for transient signals
    # In general, we don't know the timescales of transients a prior
    # Therefore, we add integrated B-splines of various timescales where the
    # timescale is controlled by the 'nspl' value (this means to divide the time
    # vector into 'nspl' equally spaced spline center locations)
    for nspl in [128, 64, 32, 16, 8, 4]:
        collection.append(ispl(order=3, num=nspl, units='years', tmin=tstart, tmax=tend))
    
    # Done
    return collection

# Build a priori covariance matrix, mainly for repeating B-splines
def computeCm(collection, b_spl_sigma=0.5):
    from scipy.linalg import block_diag

    # Define some prior sigmas (large prior sigmas for secular and transient)
    sigma_secular = 100.0
    sigma_bspl = b_spl_sigma
    sigma_ispl = 100.0

    # Get lengths of different model partitions
    fnParts = ice.tseries.timefn.getFunctionTypes(collection)
    nBspl = len(fnParts['seasonal'])
    nIspl = len(fnParts['transient'])
    nSecular = len(fnParts['secular'])

    # Diagonal prior for secular components
    C_sec = sigma_secular**2 * np.eye(nSecular)

    # Get decimal times of B-spline centers   
    tdec = []
    for basis in collection.data:
        if basis.fnname == 'BSpline':
            tdec.append(ice.datestr2tdec(pydtime=basis.tref))
    tdec = np.array(tdec)

    # Correlation with splines w/ similar knot times
    # Decaying covariance w/ time
    tau = 1.0
    C_bspl = np.zeros((nBspl, nBspl))
    for i in range(nBspl):
        weight_decay = np.exp(-1.0 * np.abs(tdec - tdec[i]) / tau)
        ind = np.zeros((nBspl,), dtype=bool)
        ind[i::5] = True
        ind[i::-5] = True
        C_bspl[i,ind] = weight_decay[ind]
    C_bspl *= sigma_bspl**2

    # Diagonal prior for integrated B-splines
    C_ispl = sigma_ispl**2 * np.eye(nIspl)

    return block_diag(C_sec, C_bspl, C_ispl)


def VSeriesAtPoint(pt, vel_stack, collection, model, model_pred, solver, t_grid,
                   sigma=1.5, data_key='igram'):
    """
    Invert for a continuous 1D velocity series at a given point

    Parameters
    ----------
    pt : tuple
        Position (x,y) at which to pull series.
    vel_stack : ice.MagStack
        Stack of 2D velocity fields including this point.
    collection : ice.tseries TimefnCollection
        Basis functions for the tseries dates in vel_stack
    model : ice.tseries.model.Model
        Model instance for inversion
    model_pred : ice.tseries.model.Model
        Model instance for prediction - evenly spaced dates
    solver : ice.tseries solver
        Instance of solver that will do the inversion.  LassoRegression has worked well.
    t_grid : ndarray
        Evenly spaced decimal times at which to sample spline-fit velocity
    sigma : float, optional
        B-spline sigma for a priori covariance at this point. The default is 1.5.
    data_key : str, optional
        Convention for accessing HDF5 data. The default is 'igram' (per B. Riel).

    Returns
    -------
    pred : dict
        Model prediction for this point. Keys are 'full', 'seasonal', 'transient', 'secular', 'step'
    short_term : ndarray
        Series with long-term signals removed
    long_term : ndarray
        Series with short-term signals removed

    """
    series = vel_stack.timeseries(xy=pt, key=data_key)
    
    # Construct a priori covariance
    Cm = computeCm(collection, b_spl_sigma=sigma)
    solver.regMat = np.linalg.inv(Cm)
    
    # Perform inversion to get coefficient vector and coefficient covariance matrix
    status, m, Cm = solver.invert(model.G, series) # fit near-terminus (series[0]) first
    assert status == ice.SUCCESS, 'Failed inversion'
    
    # Model will perform predictions
    pred = model_pred.predict(m)

    # Separate out seasonal (short-term) and secular + transient (long-term) signals
    short_term = pred['seasonal']
    long_term = pred['secular'] + pred['transient']

    # Remove long-term signals from data
    series_short_term = series - np.interp(vel_stack.tdec, t_grid, long_term)

    # Remove short-term signals from data
    series_long_term = series - np.interp(vel_stack.tdec, t_grid, short_term)
    
    return pred, series_short_term, series_long_term
    

def SmbXcorr(pt, smb_dictionary, smb_dates, velocity_pred, t_grid, diff=1, normalize=True):
    """
    Compute cross-correlation on coincident series of SMB, velocity.

    Parameters
    ----------
    pt : tuple
        Position (x,y) at which to pull series.
    smb_dictionary : dict
        SMB field by date. Keys should be same format as smb_dates
    smb_dates : pandas DatetimeIndex
        Dates of SMB time slices
    velocity_series : dict
        Output of iceutils prediction.
    t_grid : ndarray
        Evenly spaced decimal times at which spline-fit velocity is sampled
    diff : int, optional
        Number of discrete differences to apply to data. Default is 1.
        Setting diff=0 will process the input data as-is.
    normalize : bool, optional
        Whether to normalize for a cross-correlation in [-1,1]. Default is True.
        This makes the output inter-comparable with normalized output for other
        variables.  If set to False, the signal amplitude will be larger but
        the correlation values may exceed 1.

    Returns
    -------
    corr : array
        Cross-correlation coefficients between SMB, velocity
    lags : array
        Time lag for each correlation value
    ci : array
        Confidence intervals for evaluation

    """
    smb_series = [float(smb_dictionary[d](pt[0],pt[1])) for d in smb_dates]
    
    ## Now interpolate time series and pull values coincident with satellite shots
    smb_d = [d.utctimetuple() for d in smb_dates]
    dates_interp = [ice.timeutils.datestr2tdec(d[0], d[1], d[2]) for d in smb_d]
    smb_series_func = interpolate.interp1d(dates_interp, smb_series, bounds_error=False)
    coincident_dates = t_grid[t_grid<=max(dates_interp)]
    coincident_smb = smb_series_func(coincident_dates) # sample at same dates as helheim-tseries_decomp
    vel_series = velocity_pred['full'][t_grid<=max(dates_interp)]

    smb_diff = np.diff(coincident_smb, n=diff)
    vel_diff = np.diff(vel_series, n=diff)
    if normalize:
        smb_diff = (smb_diff - np.mean(smb_diff)) / (np.std(smb_diff)*len(smb_diff))
        vel_diff = (vel_diff - np.mean(vel_diff)) / (np.std(vel_diff))
    corr = np.correlate(smb_diff, vel_diff, mode='full')
    lags = range(int(-0.5*len(corr)), int(0.5*len(corr)+1))
    ci = [2/np.sqrt(len(coincident_smb)-abs(k)) for k in lags]

    ## convert lags to physical units
    lags = np.mean(np.diff(t_grid))*365.26*np.asarray(lags)
    
    return corr, lags, ci

def RunoffXcorr(pt, runoff_func, runoff_dates, velocity_pred, t_grid, diff=1, normalize=True):
    """
    Compute cross-correlation on coincident series of catchment-integrated runoff
    and velocity at a point.

    Parameters
    ----------
    pt : tuple
        Position (x,y) at which to pull velocity series.
    runoff_func : interpolate.interp1d
        1D-interpolated function with values of catchment-integrated runoff over time.
    runoff_dates : list
        Decimal dates of runoff data points
    velocity_series : dict
        Output of iceutils prediction.
    t_grid : ndarray
        Evenly spaced decimal times at which spline-fit velocity is sampled
    diff : int, optional
        Number of discrete differences to apply to data. Default is 1.
        Setting diff=0 will process the input data as-is.
    normalize : bool, optional
        Whether to normalize for a cross-correlation in [-1,1]. Default is True.
        This makes the output inter-comparable with normalized output for other
        variables.  If set to False, the signal amplitude will be larger but
        the correlation values may exceed 1.

    Returns
    -------
    corr : array
        Cross-correlation coefficients between SMB, velocity
    lags : array
        Time lag for each correlation value
    ci : array
        Confidence intervals for evaluation

    """
    coincident_dates = t_grid[t_grid<=max(runoff_dates)]
    coincident_runoff = runoff_func(coincident_dates) # sample at same dates as velocity series
    
    runoff_diff = np.diff(coincident_runoff, n=diff)
    vel_series = velocity_pred['full'][t_grid<=max(runoff_dates)]
    vel_diff = np.diff(vel_series, n=diff)
    if normalize:
        runoff_diff = (runoff_diff-np.mean(runoff_diff)) / (np.std(runoff_diff)*len(runoff_diff))
        vel_diff = (vel_diff-np.mean(vel_diff)) / (np.std(vel_diff))
    corr = np.correlate(runoff_diff, vel_diff, mode='full')
    lags = range(int(-0.5*len(corr)), int(0.5*len(corr)+1))
    ci = [2/np.sqrt(len(coincident_runoff)-abs(k)) for k in lags]

    ## convert lags to physical units
    lags = np.mean(np.diff(t_grid))*365.26*np.asarray(lags)
    
    return corr, lags, ci

def Xcorr1D(pt, series_func, series_dates, velocity_pred, t_grid, t_limits, diff=1, normalize=True):
    """
    Compute cross-correlation on coincident series of a 1D time series
    (e.g. catchment-integrated runoff or SMB) versus velocity at a point.

    Parameters
    ----------
    pt : tuple
        Position (x,y) at which to pull velocity series.
    series_func : interpolate.interp1d
        1D-interpolated function with values of data series over time.
    series_dates : list
        Decimal dates of data points
    velocity_series : dict
        Output of iceutils prediction.
    t_grid : ndarray
        Evenly spaced decimal times at which spline-fit velocity is sampled
    t_limits : tuple
        Start and end dates (decimal) of the time period to study
    diff : int, optional
        Number of discrete differences to apply to data. Default is 1.
        Setting diff=0 will process the input data as-is.
    normalize : bool, optional
        Whether to normalize for a cross-correlation in [-1,1]. Default is True.
        This makes the output inter-comparable with normalized output for other
        variables.  If set to False, the signal amplitude will be larger but
        the correlation values may exceed 1.

    Returns
    -------
    corr : array
        Cross-correlation coefficients between SMB, velocity
    lags : array
        Time lag for each correlation value
    ci : array
        Confidence intervals for evaluation

    """
    t_min = max(min(series_dates), t_limits[0])
    t_max = min(max(series_dates), t_limits[1])
    coincident_dates = np.asarray([t for t in t_grid if (t>=t_min and t<t_max)])
    coincident_series = series_func(coincident_dates) # sample at same dates as velocity series
    
    series_diff = np.diff(coincident_series, n=diff)
    vel_series_0 = velocity_pred['full'][np.where(t_grid>=t_min)]
    vel_series = vel_series_0[np.where(t_grid[np.where(t_grid>=t_min)]<t_max)] # trim dates to match t_limits
    vel_diff = np.diff(vel_series, n=diff)
    if normalize:
        series_diff = (series_diff-np.mean(series_diff)) / (np.std(series_diff)*len(series_diff))
        vel_diff = (vel_diff-np.mean(vel_diff)) / (np.std(vel_diff))
    corr = np.correlate(series_diff, vel_diff, mode='full')
    lags = range(int(-0.5*len(corr)), int(0.5*len(corr)+1))
    ci = [2/np.sqrt(len(coincident_series)-abs(k)) for k in lags]

    ## convert lags to physical units
    lags = np.mean(np.diff(t_grid))*365.26*np.asarray(lags)
    
    return corr, lags, ci
    