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
import statsmodels.api as sm
import pandas as pd
import iceutils as ice
from iceutils import tseries, timeutils



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
    

def Xcorr1D(pt, series_func, series_dates, velocity_pred, t_grid, t_limits, diff=1, normalize=True, pos_only=False):
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
    pos_only : bool, optional
    	Whether to analyse only xcorrs with positive lag values.  Default is False.
    	This allows a bidirectional causal relationship.  For a causal relationship 
    	hypothesised to be single-directional, choose True to display only positive
    	lag values.

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
    
    if pos_only:
    	corr = corr[np.argwhere(lags>=0)].squeeze()
    	ci = np.asarray(ci)[np.argwhere(lags>=0)].squeeze()
    	lags = lags[np.argwhere(lags>=0)].squeeze()
    	
    return corr, lags, ci
    
def Arima_ResidFunc(series, normalize=True, diff=1):
    """
    Fit an ARIMA model of appropriate order to the input series, compute the 
    residuals we will use for the cross-correlation with velocity.

    Parameters
    ----------
    series : pd.Series
        The "input variable" (SMB, runoff, terminus position...) expressed as a
        one-dimensional time series.  The timestamp of each data point 
        should be the "index" of the pandas Series.
    normalize : bool, optional
        Whether to normalize series before fitting ARIMA. Default is True.
        Normalized residuals will make the output of XCorr1D inter-comparable 
        with normalized output for other variables.  If set to False, the signal 
        amplitude will be larger but the correlation values may exceed 1.
    diff : int, optional
        Number of discrete differences to apply to data. Default is 1.
        We do this to hone in on the effect of our variables on each other,
        rather than a mutual response (background trend) to a shared 
        un-modelled factor.

    Returns
    -------
    mod_fit: statsmodels.tsa.arima.model.ARIMAResultsWrapper
        The ARIMA model fit to the series, which we will use in filtering
        the velocity data for cross-correlation.
        (Ref: https://online.stat.psu.edu/stat510/lesson/9/9.1)
    resid_func : interpolate.interp1d
        1D-interpolated function with values of residuals (series-ARIMA) over time.
        We will use this to sample the residuals at the same time intervals
        as the velocity, which is necessary for the cross-correlation.
    series_resid : pd.Series , optional
        A pd.Series of the residuals

    """
    if normalize:
        series_normed = (series-np.mean(series)) / (np.std(series)*len(series))
    else:
        series_normed = series
    
    ## Prewhitening data: fit ARIMA model to series data
    d = diff # order of differencing to apply in ARIMA model 
    ## partial autocorrelation function helps us choose the AR term (p) in ARIMA
    lag_pacf = sm.tsa.stattools.pacf(series_normed, nlags=20, method='ols')
    upper_confidence = 1.96/np.sqrt(len(series_normed))
    p_candidates = np.argwhere(lag_pacf > upper_confidence).squeeze()
    p = p_candidates[p_candidates >0][0] # choose first nonzero value that exceeds upper CI
    ## autocorrelation function helps us choose the MA term (q) in ARIMA
    lag_acf = sm.tsa.stattools.acf(series_normed, nlags=20) 
    q_candidates = np.argwhere(lag_acf > upper_confidence).squeeze()
    q = q_candidates[q_candidates >0][0] # choose first nonzero value that exceeds upper CI
    print('Fitting ARIMA of order ({},{},{})'.format(p,d,q))
    
    mod = sm.tsa.arima.ARIMA(series_normed, order=(p,d,q))
    mod_fit = mod.fit()

    series_resid = pd.DataFrame(mod_fit.resid).squeeze()
    resid_d = [d.utctimetuple() for d in series_resid.index]
    resid_d_interp = [ice.timeutils.datestr2tdec(d[0], d[1], d[2]) for d in resid_d]
    resid_func = interpolate.interp1d(resid_d_interp, series_resid)
    
    return mod_fit, resid_func, series_resid

def Xcorr1D_prewhitened(resid_func, series_dates, velocity_pred, t_grid, t_limits, 
            mod_fit, normalize=True, pos_only=False):
    """
    Compute cross-correlation on coincident series of a 1D time series
    (e.g. catchment-integrated runoff or SMB) versus velocity at a point.
    
    Note that unlike XCorr1D, we do not include the option to difference the 
    velocity time series here.  Rather, we apply a convolution filter defined
    by the parameters of the ARIMA model mod_fit.

    Parameters
    ----------
    resid_func : interpolate.interp1d
        1D-interpolated function with values of data series over time.
        We assume that this is already normalized.
    series_dates : list
        Decimal dates of data points
    velocity_series : dict
        Output of iceutils prediction.
    t_grid : ndarray
        Evenly spaced decimal times at which spline-fit velocity is sampled
    t_limits : tuple
        Start and end dates (decimal) of the time period to study
    mod_fit : statsmodels.tsa.arima.model.ARIMAResultsWrapper
        An ARIMA model fit to the input variable, which we will use in filtering
        the velocity data for cross-correlation.
        (Ref: https://online.stat.psu.edu/stat510/lesson/9/9.1)
    normalize : bool, optional
        Whether to normalize for a cross-correlation in [-1,1]. Default is True.
        This makes the output inter-comparable with normalized output for other
        variables.  If set to False, the signal amplitude will be larger but
        the correlation values may exceed 1.
    pos_only : bool, optional
    	Whether to analyse only xcorrs with positive lag values.  Default is False.
    	This allows a bidirectional causal relationship.  For a causal relationship 
    	hypothesised to be single-directional, choose True to display only positive
    	lag values.

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
    coincident_resid = resid_func(coincident_dates) # sample at same dates as velocity series
    
    vel_series_0 = velocity_pred['full'][np.where(t_grid>=t_min)]
    vel_series = vel_series_0[np.where(t_grid[np.where(t_grid>=t_min)]<t_max)] # trim dates to match t_limits
    if normalize:
        vel_series = (vel_series-np.mean(vel_series)) / (np.std(vel_series))
    v_filt = sm.tsa.filters.convolution_filter(vel_series, filt=mod_fit.params)[1:-1]  #trim nans from ends
    # TODO: confirm whether this whitens correctly

        
    corr = np.correlate(coincident_resid[1:-1], v_filt, mode='full')
    lags = range(int(-0.5*len(corr)), int(0.5*len(corr)+1))
    ci = [2/np.sqrt(len(coincident_resid)-abs(k)) for k in lags]

    ## convert lags to physical units
    lags = np.mean(np.diff(t_grid))*365.26*np.asarray(lags)
    
    if pos_only:
    	corr = corr[np.argwhere(lags>=0)].squeeze()
    	ci = np.asarray(ci)[np.argwhere(lags>=0)].squeeze()
    	lags = lags[np.argwhere(lags>=0)].squeeze()
    
    return corr, lags, ci