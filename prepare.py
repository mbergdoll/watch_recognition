#!/usr/bin/env python
import os
import sys
import datetime
import pprint
import time
from pathlib import Path

# mpiexec -n 10 python download

# extract url from a list of url to download
def extract_name_from_name():
	f = open('Base10000/input_url_file.txt','r')
	list_name = f.readlines()
	f.close()
	return list_name

# extract text from a list of text associated to url downloaded
def extract_text_from_input_file():
	f = open('Base10000/input_description_file.txt','r')
	list_name = f.readlines()
	f.close()
	return list_name

# for each image in list yet downloaded, we check further points.
def Analyse_and_filter(list_name):
	c=0
	list_text = extract_text_from_input_file()
	tot=len(list_name)
	output_path = open('output_path.txt','w+')
	output_text = open('output_text.txt','w+')
	for name in list_name:
		path = "Base10000/image/" + str(c) + ".jpg" 
		my_file = Path( path )
		if my_file.is_file():
			# image exists
			path = c
			text = list_text[c]
			list_res = []
			for l in list_name[max(0,c-10):c]:
				list_res = list_res + [ l.split('/')[-1].split('px')[-1] ]
			if name.split('/')[-1].split('px')[-1] not in list_res:
				text = text.replace('\n','')
				output_path.write( str(c) +"\n")
				output_text.write( str(text) +"\n")	
		if c%1000==0:
			print( str(c)+" / "+str(tot)+" made..." )
		c=c+1
	output_path.close()
	output_text.close()


print("prepare...")
list_name = extract_name_from_name()
Analyse_and_filter(list_name)