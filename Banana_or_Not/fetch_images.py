import urllib3
import shutil
import os, sys
from PIL import Image
import requests
from io import StringIO

from urllib import request

i = 0
f = open('imagenet.synset.geturls.txt')
for line in f:
    i += 1
    print('{0}/{1}'.format(i, 'len(f)'))
    try:
        img = request.urlretrieve(line, "banana.png")
    except:
        continue
    infile = 'banana.png'
    size = 32, 32


    outfile = 'bananas/'+os.path.splitext(infile)[0] + "thumb"+str(i)+".png"
    if infile != outfile:
        try:
            im = Image.open(infile)
            im = im.resize(size, Image.ANTIALIAS)
            im.save(outfile)
        except IOError:
            print ("cannot create thumbnail for '{}'".format(len(infile)))
