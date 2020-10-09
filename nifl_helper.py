## Helper functions:
## Load in flowlines from CSV
## Identify evenly-spaced points along them
## Pull given dataset at these points
## Export a point time series to CSV
from math import sqrt
import collections
import numpy as np


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

    