#!/usr/bin/python
from __future__ import print_function
import numpy as np
#import matplotlib.pyplot as plt
import sys
#sys.path.append('/home/debian/gprep-release_v1.0/GP_regression_multi') # path to code
from gp_sparse_linear import GP_sparse_linear # class for Gaussian process regression
from spline_func import Spline # class for reading in repulsion spline from .skf file
#from prediction_statistics2 import Prediction_Stats # class for comparing the prediction performance 
#from plot_functions import visualize_training_data, plot_repulsion, plot_RMSE_vs_hyper # some plotting tools
import os
import uuid
import cutoff_analysis as ca
import easywebdav



def GP(b2dropusername,b2droppassword,file,sigma,beta_global,theta_global,delta,d,c,N):

    #First, the reference data is processed to extract the relevant forces and pair distances:
    #os.system('python /home/debian/gprep-release_v1.0/GP_regression_multi/get_distances.py -c 5. -o tmp.rep /home/debian/gprep-release_v1.0/reference_data.xyz')
    #os.system('python /home/debian/gprep-release_v1.0/GP_regression_multi/get_distances.py -c 5. -o tmp.rep %s' %file)
    filename=str(uuid.uuid4())
    os.system('python /usr/local/tomcat9/gaussian/ROOT/WEB-INF/classes/scripts/get_distances.py -c %.2f -o tmp%s.rep ref.xyz' %(c,filename))
    #os.system('python /users/bojana.koteska/Downloads/GaussianAPI/src/main/resources/scripts/get_distances.py -c %.2f -o tmp%s.rep ref.xyz' %(c,filename))
     
    # training data
    training_file = 'tmp%s.rep' %filename
 #   training_file= os.popen('python /usr/local/tomcat9/webapps/GaussianAPI/WEB-INF/classes/scripts/get_distances.py -c 5. ref.xyz').read()
    bond_file = 'bond_dists_tmp%s.rep' %filename # list of bond distances in the training file
                                     # (additional output of get_distances.py)
    
    # the covariance function (kernel) k expresses our prior assumptions about the fit function.
    # the parameters of the kernel are called hyperparameters:

    # cutoff: range of the repulsive potential
    # sigma : data noise (if delta = 1, then data noise to prior function variance ratio) --> prevents overfitting
    # delta: prior function variance
    # theta: length scale of covariance (over which length are function values similar to each other?)
    # beta: exponential damping (for smoother function in the long range)
    # X_pseudo: representative bond lengths

    # all hyperparameters, except sigma, and the pseudo inputs are bond type specific
    # and are passed in a 'bond dictionary'

    sigma = sigma
    beta_global = beta_global
    theta_global = theta_global


    with open(bond_file) as f:
    	content = f.readlines()

    indexOC = [x for x in range(len(content)) if 'O-C' in content[x] or 'C-O' in content[x]]
    start_CO=round(float(content[indexOC[0]+1]),1)

    indexCC = [x for x in range(len(content)) if 'C-C' in content[x]]
    start_CC=round(float(content[indexCC[0]+1]),1)

    indexHH = [x for x in range(len(content)) if 'H-H' in content[x]]
    start_HH=round(float(content[indexHH[0]+1]),1)

    indexOO = [x for x in range(len(content)) if 'O-O' in content[x]]
    start_OO=round(float(content[indexOO[0]+1]),1)

    indexCH = [x for x in range(len(content)) if 'C-H' in content[x] or 'H-C' in content[x]] 
    start_CH=round(float(content[indexCH[0]+1]),1)

    indexOH = [x for x in range(len(content)) if 'O-H' in content[x] or 'H-O' in content[x]]
    start_OH=round(float(content[indexOH[0]+1]),1)

    cutoff_CC = ca.cuttof(bond_file,'C-C','C-C')
    start_CC = start_CC # smallest bond distance in training file 
    CC_dict = {'X_pseudo':np.linspace(start_CC, cutoff_CC, 20),
               'beta':beta_global,
               'theta':theta_global,
               'delta':delta,
               'd':d,
               'cutoff':cutoff_CC}

    cutoff_CH = ca.cuttof(bond_file,'C-H','H-C')
    start_CH = start_CH
    CH_dict = {'X_pseudo':np.linspace(start_CH, cutoff_CH, 20),
               'beta':beta_global,
               'theta':theta_global,
               'delta':delta,
               'd':d,
               'cutoff':cutoff_CH}

    cutoff_CO = ca.cuttof(bond_file,'C-O','O-C')
    start_CO = start_CO
    CO_dict = {'X_pseudo':np.linspace(start_CO, cutoff_CO, 20),
               'beta':beta_global,
               'theta':theta_global,
               'delta':delta,
               'd':d,
               'cutoff':cutoff_CO}

    cutoff_OO = ca.cuttof(bond_file,'O-O','O-O')
    start_OO = start_OO
    OO_dict = {'X_pseudo':np.linspace(start_OO, cutoff_OO, 20),
               'beta':beta_global,
               'theta':theta_global,
               'delta':delta,
               'd':d,
               'cutoff':cutoff_OO}

    cutoff_OH = ca.cuttof(bond_file,'O-H','H-O')
    start_OH = start_OH
    OH_dict = {'X_pseudo':np.linspace(start_OH, cutoff_OH, 20),
               'beta':beta_global,
               'theta':theta_global,
               'delta':delta,
               'd':d,
               'cutoff':cutoff_OH}

    cutoff_HH = ca.cuttof(bond_file,'H-H','H-H')
    start_HH = start_HH
    HH_dict = {'X_pseudo':np.linspace(start_HH, cutoff_HH, 20),
               'beta':beta_global,
               'theta':theta_global,
               'delta':delta,
               'd':d,
               'cutoff':cutoff_HH}

    bond_dict = {'C-C':CC_dict, 'C-H':CH_dict, 'C-O':CO_dict, 'O-O':OO_dict, 'O-H':OH_dict, 'H-H':HH_dict}

    # kernel: function that expresses covariance between function values
    kernel = 'SE_damped_cut' # Squared Exponential covariance function with damping and cutoff

    # training
    gp = GP_sparse_linear(bond_dict=bond_dict,
                          training_file=training_file,
                          kernel=kernel,
                          sigma=sigma,
                          rel_sigma=False)

    
    # spline and write to .skf
    gp.make_skf_spline(s0=start_CC, s1=start_CC+0.1, N=N, outputfile='/usr/local/tomcat9/gaussian/ROOT/C-C_GPrep_SI%s.skf' %filename, bond_type='C-C')
    gp.make_skf_spline(s0=start_CH, s1=start_CH+0.1, N=N, outputfile='/usr/local/tomcat9/gaussian/ROOT/C-H_GPrep_SI%s.skf' %filename, bond_type='C-H')
    gp.make_skf_spline(s0=start_CO, s1=start_CO+0.1, N=N, outputfile='/usr/local/tomcat9/gaussian/ROOT/C-O_GPrep_SI%s.skf' %filename, bond_type='C-O')
    gp.make_skf_spline(s0=start_OO, s1=start_OO+0.1, N=N, outputfile='/usr/local/tomcat9/gaussian/ROOT/O-O_GPrep_SI%s.skf' %filename, bond_type='O-O')
    gp.make_skf_spline(s0=start_OH, s1=start_OH+0.1, N=N, outputfile='/usr/local/tomcat9/gaussian/ROOT/O-H_GPrep_SI%s.skf' %filename, bond_type='O-H')
    gp.make_skf_spline(s0=start_HH, s1=start_HH+0.1, N=N, outputfile='/usr/local/tomcat9/gaussian/ROOT/H-H_GPrep_SI%s.skf' %filename, bond_type='H-H')

    
    
    try:
        webdav = easywebdav.connect('b2drop.eudat.eu', username=b2dropusername, password=b2droppassword, protocol='https', port=443, path='/remote.php/webdav/')
        webdav.upload('/usr/local/tomcat9/gaussian/ROOT/C-C_GPrep_SI%s.skf' %filename, 'C-C_GPrep_SI%s.skf' %filename)
        webdav.upload('/usr/local/tomcat9/gaussian/ROOT/C-H_GPrep_SI%s.skf' %filename, 'C-H_GPrep_SI%s.skf' %filename)
        webdav.upload('/usr/local/tomcat9/gaussian/ROOT/C-O_GPrep_SI%s.skf' %filename, 'C-O_GPrep_SI%s.skf' %filename)
        webdav.upload('/usr/local/tomcat9/gaussian/ROOT/O-O_GPrep_SI%s.skf' %filename, 'O-O_GPrep_SI%s.skf' %filename)
        webdav.upload('/usr/local/tomcat9/gaussian/ROOT/O-H_GPrep_SI%s.skf' %filename, 'O-H_GPrep_SI%s.skf' %filename)
        webdav.upload('/usr/local/tomcat9/gaussian/ROOT/H-H_GPrep_SI%s.skf' %filename, 'H-H_GPrep_SI%s.skf' %filename)
        print("Files were successfully uploaded to b2drop.")
    except:
        print("Files were not uploaded to b2drop. Please check your username and password.")
    
    
    os.system('rm /usr/local/tomcat9/gaussian/ROOT/C-C_GPrep_SI%s.skf' %filename)
    os.system('rm /usr/local/tomcat9/gaussian/ROOT/C-H_GPrep_SI%s.skf' %filename)
    os.system('rm /usr/local/tomcat9/gaussian/ROOT/C-O_GPrep_SI%s.skf' %filename)
    os.system('rm /usr/local/tomcat9/gaussian/ROOT/O-O_GPrep_SI%s.skf' %filename)
    os.system('rm /usr/local/tomcat9/gaussian/ROOT/O-H_GPrep_SI%s.skf' %filename)
    os.system('rm /usr/local/tomcat9/gaussian/ROOT/H-H_GPrep_SI%s.skf' %filename)

    
    #os.system('cp C-C_GPrep_SI%s.skf /Users/bojana.koteska/Downloads/' %filename)
    #os.system('cp C-H_GPrep_SI%s.skf /Users/bojana.koteska/Downloads/' %filename)
    #os.system('cp C-O_GPrep_SI%s.skf /Users/bojana.koteska/Downloads/' %filename)
    #os.system('cp O-O_GPrep_SI%s.skf /Users/bojana.koteska/Downloads/' %filename)
    #os.system('cp O-H_GPrep_SI%s.skf /Users/bojana.koteska/Downloads/' %filename)
    #os.system('cp H-H_GPrep_SI%s.skf /Users/bojana.koteska/Downloads/' %filename)


    #print ('https://gaussian.chem-api.finki.ukim.mk/C-C_GPrep_SI%s.skf' %filename)
    #print ('https://gaussian.chem-api.finki.ukim.mk/C-H_GPrep_SI%s.skf' %filename)
    #print ('https://gaussian.chem-api.finki.ukim.mk/C-O_GPrep_SI%s.skf' %filename)
    #print ('https://gaussian.chem-api.finki.ukim.mk/O-O_GPrep_SI%s.skf' %filename)
    #print ('https://gaussian.chem-api.finki.ukim.mk/O-H_GPrep_SI%s.skf' %filename)
    #print ('https://gaussian.chem-api.finki.ukim.mk/H-H_GPrep_SI%s.skf' %filename)
           

#     #send email
#     import smtplib
#     fromaddr = 'gaussianrestapi@gmail.com'
#     toaddrs  = '%s' %(email)
#     msg = "\r\n".join([
#       "From: gaussianrestapi@gmail",
#       "To: %s" %(email),
#       "Subject: GP potentials",
#       "",
#           """
#         Dear user, \r\n
# 
#         Please find below the links to download the GP potentials .skf files
# 
#         
#           https://gaussian.chem-api.finki.ukim.mk/C-C_GPrep_SI%s.skf \r\n
#           https://gaussian.chem-api.finki.ukim.mk/C-H_GPrep_SI%s.skf \r\n
#           https://gaussian.chem-api.finki.ukim.mk/C-O_GPrep_SI%s.skf \r\n
#           https://gaussian.chem-api.finki.ukim.mk/O-O_GPrep_SI%s.skf \r\n
#           https://gaussian.chem-api.finki.ukim.mk/O-H_GPrep_SI%s.skf \r\n
#           https://gaussian.chem-api.finki.ukim.mk/H-H_GPrep_SI%s.skf \r\n """ %(filename,filename,filename,filename,filename,filename)
# 
#       ])
#     username = 'gaussianrestapi'
#     password = 'apiapi777'
#     server = smtplib.SMTP('smtp.gmail.com:587')
#     server.ehlo()
#     server.starttls()
#     server.login(username,password)
#     server.sendmail(fromaddr, toaddrs, msg)
#     server.quit()

    #print to console
    #print("C-C")
    #print()
    #file = open("C-C_GPrep_SI.skf", "r") 
    #print (file.read())
    #print()
    #print()
    #print()
    #print("C-H")
    #print()
    #file = open("C-H_GPrep_SI.skf", "r") 
    #print (file.read())
    #print()
    #print()
    #print()
    #print("C-O")
    #print()
    #file = open("C-O_GPrep_SI.skf", "r") 
    #print (file.read())
    #print()
    #print()
    #print()
    #print("C-O")
    #print()
    #file = open("O-O_GPrep_SI.skf", "r") 
    #print (file.read())
    #print()
    #print()
    #print()
    #print("O-H")
    #print()
    #file = open("O-H_GPrep_SI.skf", "r") 
    #print (file.read())
    #print()
    #print()
    #print()
    #print("H-H")
    #print()
    #file = open("H-H_GPrep_SI.skf", "r") 
    #print (file.read())
    
