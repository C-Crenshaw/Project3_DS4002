# Source Code Folder for Project 3

Updated verisons of the source code are included in this file. Python file and Jupyter Notebook file types of "P3_Source_Code" are included. 


Within the files, the structure of the code is largely oriented around four sections: Data Cleaning, Data Visualization, Machine Learning, and Future Analysis. Users should follow each section carefully to reproduce the desired outputs. This ReadMe includes a supplemental documentation to the main repository and gives a detailed account of each section's code, specifically with respect to the imports necessary for each section. 


### Data Cleaning
The analysis documented within this Github repository is dependent upon the completion of a data cleaning process. In order to faciliate the creation of a competent time-series regression model, edits such as removing all NaN values and restricting the years of interest were undertaken. See the codeblocks below. 

The original dataset, as well as the cleaned version and its respective code, is referenced and acknowledged in the main repository ReadMe. The raw datasets can also be found in the Data folder.  

The standard imports are included within this section of the code, as well as the pycountry_convert package which provides conversion functions between ISO country names, country-codes, and continent names.

```
#Standard imports
import pandas as pd
import numpy as np

#Install and import package for continents
!pip install pycountry_convert
import pycountry_convert as pc
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
```

### Data Visualization
A unique facet of this project and analysis is the necessity of conducting exploratory data analyses before building the regression model. Graphical representations of the relationship between social indexes and happiness scores are utilized to determine which variables should be included in the final model. 

The plots incorporate the time-series data of the cleaned dataset by clearly demonstrating the change over time for each variable. Plotly express is imported and must be run for the interactive graphics to work. Plotly express is a high-level data visualization package that allows users to create interactive plots. It is built on top of the Plotly Graph Objects package which provides a lower-level interface for developing custom visualizations.

```
#Data viz import
import plotly.express as px
```

### Machine Learning and Future Analysis
For the purposes of building, training, testing, and evaluating a time-series regression model, the sklearn package is used. Scikit-learn is an open source machine learning library that supports supervised and unsupervised learning. This package provides various tools for model fitting, data preprocessing, model selection, and model evaluation, all of which are utilized within this project.

```
#Machine learning imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
```
