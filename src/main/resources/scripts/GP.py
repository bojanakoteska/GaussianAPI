#!/usr/bin/python
import mainSc
import sys
import requests
import os

file_url = sys.argv[1]
sigma = float(sys.argv[2])
beta = float(sys.argv[3])
theta = float (sys.argv[4])
delta = float(sys.argv[5])
d = float (sys.argv[6])
c = float(sys.argv[7])
N = int(sys.argv[8])
b2dropusername=sys.argv[9]
b2droppassword=sys.argv[10]
#email = sys.argv[9]
#r = requests.get(file_url)
f=open(file_url)
open('ref.xyz', 'wb').write(f.read())
z = mainSc.GP(b2dropusername,b2droppassword,'ref.xyz',sigma,beta,theta,delta,d,c,N)

