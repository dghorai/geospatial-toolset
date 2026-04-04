# Set the maximum extend length first 
# and then run the script 
# and then clip the output from the reference layer.

# Import modules
import arcpy
import collections

from operator import add
from math import hypot


# Input arguments
layer = r"C:\Work\TempWork.gdb\terminals"
distance = float(25.0)

# Computes new coordinates x3,y3 at a specified distance
# along the prolongation of the line from x1,y1 to x2,y2
def newcoord(coords, dist):
    (x1,y1),(x2,y2) = coords
    dx = x2 - x1
    dy = y2 - y1
    linelen = hypot(dx, dy)
    x3 = x2 + dx/linelen * dist
    y3 = y2 + dy/linelen * dist    
    return x3, y3

# accumulate([1,2,3,4,5]) --> 1 3 6 10 15
# Equivalent to itertools.accumulate() which isn't present in Python 2.7
def accumulate(iterable):    
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = add(total, element)
        yield total

# OID is needed to determine how to break up flat list of data by feature.
coordinates = [[row[0], row[1]] for row in arcpy.da.SearchCursor(layer, ["OID@", "SHAPE@XY"], explode_to_points=True)]
oid, vert = zip(*coordinates)

# Construct list of numbers that mark the start of a new feature class.
# This is created by counting OIDS and then accumulating the values.
vertcounts = list(accumulate(collections.Counter(oid).values()))

# Grab the last two vertices of each feature
lastpoint = [point for x,point in enumerate(vert) if x+1 in vertcounts or x+2 in vertcounts]

# Convert flat list of tuples to list of lists of tuples.
# Obtain list of tuples of new end coordinates.
newvert = [newcoord(y, distance) for y in zip(*[iter(lastpoint)]*2)]    

j = 0
with arcpy.da.UpdateCursor(layer, "SHAPE@XY", explode_to_points=True) as rows:
    for i,row in enumerate(rows):
        if i+1 in vertcounts:            
            row[0] = newvert[j]
            j+=1
            rows.updateRow(row)
