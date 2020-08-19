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
    