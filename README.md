# Do You Need to Live a Good Life to be Happy?
# Exploring the Relationship Between Life Evaluation Factors and World Happiness Rankings
Group 10 <br />
**Project 1 Leader:** Zoe Averill <br />
**Members:** Sujith Panchumarthy, Carson Crenshaw <br />
DS 4002 <br />
Last Updated: 11/20/2023

## Description
MI3 deliverable for the third project cycle of DS 4002. <br />

The objective of this project is to determine which life evaluation factors are associated with high national happiness scores and to construct a time-series regression model utilizing the predetermined factors to successfully predict future happiness scores. This project group will attempt to fulfill this objective by answering the following questions: Are individual evaluations of life satisfaction indicative of a nation’s happiness in the aggregate? If a correlation can be established between happiness and life evaluations, can these regressors accurately predict a country’s happiness in the near future? How can a predictive model for happiness be leveraged by policymakers in the implementation of new national programs? This project hopes to highlight which socioeconomic components of a nation are most important for the well-being of its citizens in order to achieve a stronger social fabric and better public welfare. 

## Contents of the Repository

The contents of this repository include the source code, data, figures, license, and references for MI3.

## [SRC](https://github.com/C-Crenshaw/Project3_DS4002/tree/388225ef8d5d4637a992764c78967781d607fdce/SRC)

This section contains all the source code for the project. All code has been consolidated in one python/jupyter notebook file. The original source code can be found [online](https://colab.research.google.com/drive/1v_eI8NASrvFKHwKa-J2pDHTHxRSty-Ns?usp=sharing). 

### Installing/Building Code

* The code can be downloaded from the SRC folder in either python or Jupyter Notebook format. The finalized and most updated version of the code file can be found online on Google Colaboratory. The link to this file is shared above.

### Usage
The usage section focuses on the most complex aspects of the final code, highlighting the most important aspects of the final code. A supplemental documentation of the code usage can be found in the SRC folder on the main repository page (linked above). 

* This section describes key steps for setting up the development environment and running the time-series regression model. All necessary imports are loaded into the 'P3_Source_Code.py' and 'P3_Source_Code.ipynb' file. These necessary packages are repeated below. These imports include those standard to most Python projects and the Sklearn package utilized for the construction and testing of a regression model.
```
#Standard imports
import pandas as pd
import numpy as np

#Install and import package for continents
!pip install pycountry_convert
import pycountry_convert as pc
```
```
#Read in data
WHR = pd.read_excel(r"/content/WHR Data.xls")
```
```
#Drop rows with nan values
WHR = WHR.dropna()
#Drop irrelevant columns
WHR = WHR.drop(columns=["Positive affect","Negative affect"])
#Rename columns to more user friendly names
WHR = WHR.rename(columns={"Country name":"Country","year":"Year","Life Ladder":"Happiness Score","Log GDP per capita":"Log GDP per Cap","Social support":"Social Support","Healthy life expectancy at birth":"Life Expectancy",
                    "Freedom to make life choices":"Freedom","Perceptions of corruption":"Corruption"})

#Select only countries with complete data from 2013 to 2019
WHR = WHR[WHR["Year"].isin(range(2013,2020))]
countries = np.unique(WHR["Country"], return_counts=True)[0]
occurances = np.unique(WHR["Country"], return_counts=True)[1] == 7
keepCountries = countries[np.where(occurances)]
WHR = WHR[WHR["Country"].isin(keepCountries)]

#Rename countries with different names to alpha-2 country code list
WHR["Country"] = WHR["Country"].replace(["Congo (Brazzaville)","Turkiye"],["Congo","Turkey"])

#Set dataframe index to Country so it's easier to drop countries
WHR = WHR.set_index("Country")

#Assign continents
WHR["Country Code"] = WHR.index.map(lambda x: pc.country_name_to_country_alpha2(x, cn_name_format="default"))
WHR["Continent Code"] = WHR["Country Code"].map(lambda x: pc.country_alpha2_to_continent_code(x))
WHR["Continent"] = WHR["Continent Code"].map(lambda x: pc.convert_continent_code_to_continent_name(x))
WHR = WHR.drop(columns=["Country Code", "Continent Code"]).reset_index()

#Cleaned dataset
WHR
```











