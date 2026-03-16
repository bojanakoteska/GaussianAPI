# Gaussian API  - RESTful web service for fitting repulsive potentials in density-functional tight-binding with Gaussian process regression

Developers: [Bojana Koteska](mailto:bojana.koteska@finki.ukim.mk)

Project Supervisors: [Anastas Mishev](mailto:anastas.mishev@finki.ukim.mk)

Scientific Advisors: [Ljupco Pejov](mailto:ljupcop@pmf.ukim.mk)

Contributors: Thanks to Vojdan Kjorverziroski for his help in deploying the code.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19039928.svg)](https://doi.org/10.5281/zenodo.19039928)


## Overview 

The RESTful web service for fitting repulsive potentials in density-functional tight-binding with Gaussian process regression (Gaussian API) fits repulsive potentials in density-functional tight-binding (DFTB) with Gaussian process regression. The service documentation can be found on the following link: [Gaussian API User Manual](https://gaussian.chem-api.finki.ukim.mk/static/GaussianAPI_user_manual.html).

## Software metadata

### Domain

Computational Physics

### Funder

European Commission under the Horizon 2020 - NI4OS-Europe, grant agreement no. 857645

### Progamming languages

Python 3, Java

### Date created

2026-03-15

### Keywords

REST API, RESTful web service, Gaussian regression

## Service descripiton

It provides two methods: **GPrep** (POST method) and **GPrepRemote** (GET method).

The only difference between these two REST API methods is that in case of **GPrep**, the user should provide an input file by browsing the file system on the local device, while in case of **GPrepRemote** the user should provide a public URL where the input file can be accessed. 

The provided URL should be a direct link to a public file (e.g.  https://gaussian.chem-api.finki.ukim.mk/static/reference_data.xyz) or public **Dropbox** link (e.g. https://www.dropbox.com/s/qnk7r3ey6pkfzb9/reference_dataB.xyz?dl=0) or public **B2DROP** link (e.g. https://b2drop.eudat.eu/s/QWPRFGwYHEno99P). **B2DROP** is a secure and trusted data exchange service for researchers and scientists to keep their research data synchronized and up-to-date and to exchange with other researchers. More information about **B2DROP** can be found at https://eudat.eu/services/b2drop.

The output Slater-Koster files (.skf) with potentials will be uploaded to the user **B2DROP** account https://b2drop.eudat.eu/apps/files/. The user should log in to **B2DROP** (preferably by using her/his institutional account) and to generate username and password. All steps needed to generate username and password are described in the section "**Mounting B2DROP using the WebDAV client**" on the following link https://eudat.eu/services/userdoc/b2drop. 


Both methods have the same additional parameters described below. For all parameters, except for **b2dropUsername** and **b2dropPassword**, users can use the default provided values or they can enter their own values. For **b2dropUsername** and **b2dropPassword** users must provide their own values.

Parameters:

- **file** - reference data file from which the relevant forces and pair distances are extracted 
    - **GPrep** method: user should upload the file (*an example file can be found [here](https://gaussian.chem-api.finki.ukim.mk/static/reference_data.xyz)*) 
    - **GPrepRemote** method: user should provide public URL of the file (*default value: https://gaussian.chem-api.finki.ukim.mk/static/reference_data.xyz*)
- **sigma** - data noise standard deviation (*default value 0.05*) 
- **beta** - exponential damping factor (*default value 3.0*)
- **theta** - latent function length scale (*default value 1.0*)
- **delta** - latent function standard deviation (*default value 1.0*)
- **d** - cutoff transition width (*default value 1.0*)
- **c** - cutoff (*default value = 5.0*) 
- **N** - number of data points (*default value = 100*)
- **b2dropUsername** - **B2DROP** generated username
- **b2dropPassword** - **B2DROP** generated password

## Service usage examples

### GPrep method

Since **GPrep** method is a POST method, one way to be tested is to enter the parameters and upload the input file directly on the Gaussian API home page (supported by Swagger) at https://gaussian.chem-api.finki.ukim.mk/. Other possibilites to test this service include using specific API testing tools such as [Postman](https://www.postman.com/).

First step is to click the **GPrep** method from the list:
<img src="https://b2drop.eudat.eu/s/LLaaZwyogFq3QHB/download" width="500"/>

Then, user should click the button **Try it out**:
<img src="https://b2drop.eudat.eu/s/qgqb8i2G5WdpPX6/download" width="1000"/>

Next, user should enter the input parameters. The **B2DROP** username and password must be entered in the b2dropUsername and b2dropPassword textboxs in order to receive the output files (to be uploaded on the user's **B2DROP** account).

<img src="https://b2drop.eudat.eu/s/k97ekonNMq8Faqw/download" width=400/>

User should select an input file from the local device by clicking the **Choose File** button and click the **Execute** button. Please be aware that the caluclation can take some time since it depends on the size of the input file.

<img src="https://b2drop.eudat.eu/s/fCZsjRzbo6j5TEZ/download" width="1000"/>

If output files are successfully uploaded to the user's **B2DROP** account the following text will be shown in the **Response Body** form below. 

<img src="https://b2drop.eudat.eu/s/ppi3jo5YqQoqr2w/download" width="400"/>

### GPrepRemote method

**GPrepRemote** method is a GET method and it can be tested again by entering the parameters directly on the Gaussian API home page (supported by Swagger) at https://gaussian.chem-api.finki.ukim.mk/. Other possibilites to test this service include using browser URL bar, consuming the service in your source code or again by using specific API testing tools such as Postman.

First step is to click **GPrepRemote** method from the list and then click the **Try it out** button.

**GPrepRemote** does not have a file attachment button, but there is a field for entering the URL of the input file (e.g. https://gaussian.chem-api.finki.ukim.mk/static/reference_data.xyz).

<img src="https://b2drop.eudat.eu/s/zWFKtN5zCnznRww/download" width="400"/>

Same as in the previous method, the B2DROP username and password must be entered in the b2dropUsername and b2dropPassword textboxs in order to receive the output files (to be uploaded on the user's B2DROP account). All other steps are same as in the **GPrep** method.

Other way to use this REST API method is to access it directly from the browser address bar. 

https://gaussian.chem-api.finki.ukim.mk/GPrepRemote?b2dropUsername=YOUR_B2DROPUSERNAME&b2dropPassword=YOUR_B2DROPPASSWORD&file=YOUR_FILE_LOCATION

If user preffers to change other paramerets, they can be added as **&PARAMETER=VALUE** 

Another option is to consume this method in a program source code. An example in Python is provided below.

```python
import requests
response = requests.get('https://gaussian.chem-api.finki.ukim.mk/GPrepRemote?b2dropUsername=YOUR_B2DROPUSERNAME\
&b2dropPassword=YOUR_B2DROPPASSWORD&file=https://gaussian.chem-api.finki.ukim.mk/static/reference_data.xyz\
&sigma=0.05&beta=3.0&theta=1.0&delta=1.0&d=1.0&c=5.0&N=100')
if response.status_code == 200:
        print(response.content.decode('utf-8')) 
else:
        print("None")
```

## License
This project is licensed under the MIT License; for more details, see the [LICENSE](https://github.com/bojanakoteska/GaussianAPI/blob/main/LICENSE) file.

