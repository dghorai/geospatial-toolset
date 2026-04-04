# Global Cyclone Path Downloading

# Data: Files by year
# Import Module
import urllib

# Input arguments
outpath = r"E:\PhD_Working\Cyclone"
main_url = 'ftp://eclipse.ncdc.noaa.gov/pub/ibtracs/v03r06/all/shp/year/'

# Local variables
year = 'Year.'
typ = ['.ibtracs_all_lines.', '.ibtracs_all_points.']
ver = 'v03r06'
ext = ['.dbf', '.prj', '.shp', '.shx']

# Data downloading
for i in range(2012, 2015):
    for t in typ:
        for e in ext:
            fname = year+str(i)+t+ver+e
            print(fname)
            file_url = main_url+year+str(i)+t+ver+e
            urlopener = urllib.URLopener()
            urlopener.retrieve(file_url, outpath+"\\"+fname)
