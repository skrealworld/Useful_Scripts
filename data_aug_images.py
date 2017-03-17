#Python script for data augmentation for images. 

import glob 
import sys
import os
import numpy as np 
from skimage import io 
from scipy import ndimage,misc
from skimage import filters,img_as_uint,exposure
from skimage.transform import resize,swirl,AffineTransform
from skimage import transform
io.use_plugin('freeimage')
#first argument would be to provide the directory path. 
fileNames=glob.glob(sys.argv[1]+"*.jpg")

def horizontalFlip(image_object,_file):
	flipped_image=np.fliplr(image_object)
	io.imsave(_file[:-4]+'_flipped'+'.jpg',flipped_image)

def rotateImage(image_object,_file):
	rotated_image=ndimage.rotate(image_object,15)
	io.imsave(_file[:-4]+'_rotated'+'.jpg',rotated_image)

def blurImage(image_object,_file):
	blurredImage = filters.gaussian(image_object,sigma=5,multichannel=True)
	misc.imsave(_file[:-4]+'_blurred'+'.jpg',blurredImage)

def zoomIn(image_object,_file):
	row=image_object.shape[0]
	col=image_object.shape[1] 
	image_object=image_object[int(row*.15):int(row-row*.15),int(col*.15):int(col-col*.15)]
	image_object=resize(image_object,(row, col), mode='reflect')
	misc.imsave(_file[:-4]+'_zoomed'+'.jpg',image_object)

def randomNoise(image_object,_file):
    row,col,ch = image_object.shape
    gauss = np.random.randn(row,col,ch)
    gauss = gauss.reshape(row,col,ch) 
    noisy = image_object +5*gauss
    misc.imsave(_file[:-4]+'_random_noised'+'.jpg',noisy)

def spNoise(image_object,_file):
	row,col,ch = image_object.shape
	s_vs_p = 0.8
	amount = 0.01
	out = np.copy(image_object)
	num_salt = np.ceil(amount * image_object.size * s_vs_p)
	coords = [np.random.randint(0, i - 1, int(num_salt))for i in image_object.shape]
	out[coords] = 1
	num_pepper = np.ceil(amount* image_object.size * (1. - s_vs_p))
	coords = [np.random.randint(0, i - 1, int(num_pepper))for i in image_object.shape]
	out[coords] = 0
	misc.imsave(_file[:-4]+'_sp_noised'+'.jpg',out)

def swirlImage(image_object,_file):
	row,col,_=image_object.shape
	swirled_image=swirl(image_object,center=(row/2,col/2),strength=2,radius=1000,rotation=0)
	misc.imsave(_file[:-4]+'_swirled'+'.jpg',swirled_image)

def affine(image_object,_file):
	afine_tf = AffineTransform(shear=0.1)
	modified = transform.warp(image_object,afine_tf)
	misc.imsave(_file[:-4]+'_affined'+'.jpg',modified)

def contrast(image_object,_file):
	image_object[image_object > 230] = 255
	image_object[image_object < 30] = 0
	misc.imsave(_file[:-4]+'contrast'+'.jpg',image_object)


def increaseIntensity(image_object,_file):
	gamma_corrected = exposure.adjust_gamma(image_object,.5)
	misc.imsave(_file[:-4]+'intensity_inc'+'.jpg',gamma_corrected)

def decreaseIntensity(image_object,_file):
	gamma_corrected = exposure.adjust_gamma(image_object,1.5)
	misc.imsave(_file[:-4]+'intensity_dec'+'.jpg',gamma_corrected)

def histEqui(image_object,_file):
	gamma_corrected = exposure.equalize_hist(image_object,nbins=1000)
	misc.imsave(_file[:-4]+'hist_equi'+'.jpg',gamma_corrected)


for _file in fileNames:
	
	"""
	import image and perform data augmentation
	and save the images in the same directory. 
	"""
	imageObj=io.imread(_file) 

	#Perform the operations
	horizontalFlip(imageObj,_file)
	rotateImage(imageObj,_file)
	blurImage(imageObj,_file)
	zoomIn(imageObj,_file)
	randomNoise(imageObj,_file)
	spNoise(imageObj,_file)
	swirlImage(imageObj,_file)
	affine(imageObj,_file)
	contrast(imageObj,_file)
	increaseIntensity(imageObj,_file)
	decreaseIntensity(imageObj,_file)
	#histEqui(image_object,_file)


	