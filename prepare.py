#!/usr/bin/env python
import os
import sys
import datetime
import pprint
import time
from pathlib import Path
import PIL
from PIL import Image
from shutil import copyfile
import random

# get urls of each pitcure
def getListUrlsPictures():
	f = open('input/input_url_file.txt','r')
	urlsPictures = f.readlines()
	f.close()
	return urlsPictures

# get description of each pitcure
def getDescriptionPictures():
	f = open('input/input_description_file.txt','r')
	list_name = f.readlines()
	f.close()
	return list_name

def copy_image( source , text , c):
	classes=[
		"cartier",
		"Audemars Piguet",
		"Richard Mille",
		"IWC",
		"Jaeger-Lecoultre",
		"Hublot",
		"Rolex",
		"Patek Philippe"
	]
	for name in classes:
		for train in ["train","val"]:
			directory="watches_classed/"+train+"/"+name
			if not os.path.exists(directory):
				os.makedirs(directory)
	for name in classes:
		if name in text:
			if random.randint(0, 4)==4:
				cible = "watches_classed/val/"+name+"/"+str(c)+".jpg"
			else:
				cible = "watches_classed/train/"+name+"/"+str(c)+".jpg"
	copyfile( source , cible )

def isFile(my_file):
	return my_file.is_file()

# for each image in list yet downloaded, we check further points.
def preparePicturesInClasses(urlsPictures,descriptionsPictures,output_path,output_text):
	c=0
	for descr in descriptionsPictures: 
		sourcePicture = Path( "watches/" + str(c) + ".jpg" )
		if isFile(sourcePicture):
			list_res = []
			for l in urlsPictures[max(0,c-10):c]:
				list_res = list_res + [ l.split('/')[-1].split('px')[-1] ]
			if descr.split('/')[-1].split('px')[-1] not in list_res:
				copy_image( "watches_prepared/"+str(c)+".jpg" , descr.replace('\n','') ,c)
				output_path.write( str(c) +"\n")
				output_text.write( str(descr.replace('\n','')) +"\n")	
		if c%100==0:
			print( str(c)+" / "+str(len(descriptionsPictures))+" done..." )
		c=c+1

print("prepare...")
urlsPictures = getListUrlsPictures()
descriptionsPictures = getDescriptionPictures()

output_path = open('output/output_path.txt','w+')
output_text = open('output/output_text.txt','w+')

preparePicturesInClasses(urlsPictures,descriptionsPictures,output_path,output_text)

output_path.close()
output_text.close()