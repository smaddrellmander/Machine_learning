import urllib3
import shutil

# TODO:
# 1. Read in from file the URL list
# 2. Rescale the images to correct size
#
#
#


url = 'https://is3-ssl.mzstatic.com/image/thumb/Purple30/v4/2d/86/b3/2d86b3bc-f6c2-8de7-3fa0-f64820e3926a/source/256x256bb.jpg'
c = urllib3.PoolManager()
filename = 'save.png'

with c.request('GET',url, preload_content=False) as resp, open(filename, 'wb') as out_file:
    shutil.copyfileobj(resp, out_file)

resp.release_conn()


import os, sys
from PIL import Image

infile = 'save.png'
size = 32, 32


outfile = os.path.splitext(infile)[0] + "dog.png"
if infile != outfile:
    try:
        im = Image.open(infile)
        im.thumbnail(size, Image.ANTIALIAS)
        im.save(outfile)
    except IOError:
        print ("cannot create thumbnail for '%s'", infile)
