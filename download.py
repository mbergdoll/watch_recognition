#!/usr/bin/env python
import os
import sys
import datetime
import pprint
import time
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup, Comment
import requests
from mpi4py import MPI
from pathlib import Path

# mpiexec -n 10 python download.py

# extract url from a list of url to download
def extract_name_from_name():
	f = open('input/input_url_file.txt','r')
	list_name = f.readlines()
	f.close()
	return list_name

# download and save image from url (name) with a rank c.
def save_image_from_url( name , c ):
	img_data = requests.get(name).content
	with open( "watches/"+str(c)+".jpg" , 'wb') as handler:
		handler.write(img_data)

# for each list of url between begin and end
def boucle( begin , end , tot , list_name , c):
	for name in list_name:
		try:
			if c>begin and c<=end :
				path = "watches/" + str(c) + ".jpg" 
				my_file = Path( path )
				#print( my_file )
				pr = name.split('/')[-1][0:25].replace('\n','')
				if my_file.is_file():
				    # exists
					print( str(tot)+" / "+str(c)+" --"+pr+"-- (yet analysed)" )
				else:
				    # doesn't exist
					save_image_from_url( name.replace('\n','') , c )
					remaining = end-c
					print( str(remaining)+" remaining..."+str(name.split('/')[-1][0:30].replace('\n','') ) )
		except:
			print("error...")
		c=c+1

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#print( "--"+ str(size) )

list_name = extract_name_from_name()
tot = len( list_name )
c=1
begin = 0
pas = int( (tot-begin)/size )+1


# case download image
for i in range(size):
	if( rank==0 ):
		print( "begin:" + str(begin) )
		print( "tot:" + str(tot) )
		print( "pas:" + str(pas) )
	if( rank==i ):
		boucle( begin+(i*pas) , min( tot , begin+((i+1)*pas) ) , tot , list_name , c)

print("copying images in watch_prepared/ with the same size 32x52 in jpg format")
os.system("")
