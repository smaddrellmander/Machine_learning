import urllib3
import shutil

# TODO:
# 1. Read in from file the URL list
# 2. Rescale the images to correct size
#
#
#


url = 'http://www.solarspace.co.uk/PlanetPics/Neptune/NeptuneAlt1.jpg'
c = urllib3.PoolManager()
filename = 'save.png'

with c.request('GET',url, preload_content=False) as resp, open(filename, 'wb') as out_file:
    shutil.copyfileobj(resp, out_file)

resp.release_conn()


import os, sys
from PIL import Image

infile = 'save.png'
size = 32, 32


outfile = os.path.splitext(infile)[0] + "small.png"
if infile != outfile:
    try:
        im = Image.open(infile)
        im.thumbnail(size, Image.ANTIALIAS)
        im.save(outfile)
    except IOError:
        print ("cannot create thumbnail for '%s'", infile)
