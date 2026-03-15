#!/usr/bin/python
import mainSc
import sys
import requests
import os
import tldextract

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
ext = tldextract.extract(file_url)
domain='.'.join(ext[:2])
if domain=='b2drop.eudat':
    file_url=file_url+'/download'
elif domain=='www.dropbox':
    file_url=file_url[:-1]+'1'
r = requests.get(file_url)
open('ref.xyz', 'wb').write(r.content)
z = mainSc.GP(b2dropusername,b2droppassword,'ref.xyz',sigma,beta,theta,delta,d,c,N)

