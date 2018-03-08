#!/usr/bin/env python
# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import PIL
from PIL import Image

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to image")
args = vars(ap.parse_args())
source_image = args["images"]

def detect_watches(image,name,minneighbors):
	car_cascade = cv2.CascadeClassifier( name )
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	n_wacthes=0
	# Detect cars
	#scaleFactor	Parameter specifying how much the image size is reduced at each image scale.
	#minNeighbors	Parameter specifying how many neighbors each candidate rectangle should have to retain it.
	#flags	Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. It is not used for a new cascade.
	#minSize	Minimum possible object size. Objects smaller than that are ignored.
	#maxSize	Maximum possible object size. Objects larger than that are ignored. If maxSize == minSize model is evaluated on single scale.
	rects = car_cascade.detectMultiScale(gray, scaleFactor=1.1 , minNeighbors=minneighbors, flags=0, minSize=(50,50), maxSize=(800,800))

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	# Draw border
	for (x, y, w, h) in rects:
		cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 1)
		print("(?) found...")
		n_wacthes=+1
	return n_wacthes

def display_image(image):
	cv2.imwrite( 'res.png' , image )
	img = Image.open('res.png')
	img.show()

try:
	image = cv2.imread( source_image )
	image = imutils.resize(image, width=800 )
except:
	print("(?) image reading error...\n(*) ./opencv_cars.py --i test/1.jpg")

minneighbors=2
n_wacthes = detect_watches(image,"watches.xml",minneighbors)
if n_wacthes==0:
	n_wacthes = detect_watches(image,"watches.xml",minneighbors-1)

# show the output images
display_image(image)
