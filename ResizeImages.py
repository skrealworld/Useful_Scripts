#Resize images present in a directory. 

import os
import sys
import glob
from PIL import Image,ImageOps 

IMAGES_DIR = "/Users/SK_Mac/Documents/Github/startbootstrap-creative-gh-pages/img/portfolio/thumbnails/"
RESIZE_RESOLUTION=(400,400) 
images_name=glob.glob(IMAGES_DIR + "*.jpg")

for file_name in images_name:
	"""
	"""
	_image=Image.open(file_name)
	_image=ImageOps.fit(_image,RESIZE_RESOLUTION,Image.ANTIALIAS)
	_image.save(file_name,"JPEG")



