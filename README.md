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

* As a consequence of limited international information sharing, some values were missing and not all countries were present across the time measured. To account for this unfortunate constraint, the project group decided to carry out this analysis specifically for the years 2013-2019. Despite the presence of much data from 2020 and 2021, these years were excluded because they were unprecedented years with clear and strong confounding factors. The final range (2013-2019) was selected due to the retention of countries as detailed in the data cleaning process. In other words, this data cleaning process (which only includes countries that had a complete reference of data for the time period aforementioned) left the largest complete dataset possible. However, data from 2022 will be used to assess the model’s accuracy beyond the train test split for 2013-2019.

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

#Exporting dataset
WHR.to_csv('WHR.csv', index=False)
```

* After cleaning the dataset using the code above, the resulting table should look similar to the one provided below:
  
![CleanedData](https://github.com/C-Crenshaw/Project3_DS4002/blob/519cd6610566cef58de11e2b2f7d6661f64622d0/FIGURES/CleanedDatasetTable.png)https://github.com/C-Crenshaw/Project3_DS4002/blob/519cd6610566cef58de11e2b2f7d6661f64622d0/FIGURES/CleanedDatasetTable.png)

* An extensive exploratory data analysis was conducted with respect to countries’ happiness scores and the six life evaluation variables. The project group constructed interactive interfaces that allow users to investigate the cleaned dataset, ultimately highlighting which of the variables demonstrate a clear relationship with happiness scores. The subsequent regression analysis was be built from the conclusions derived from the graphical representations as they will indicate which of the six variables may have an illustrated relationship with happiness scores.

```
#Data viz import
import plotly.express as px
```
```
#Happiness Score by Continent over Time
px.line(WHR.groupby(by=["Continent","Year"]).agg("mean").reset_index(),"Year","Happiness Score",color="Continent")
```
```
#Happiness Score vs GDP
px.scatter(WHR, "Log GDP per Cap", "Happiness Score", color = "Continent", hover_name = "Country", animation_frame="Year", animation_group="Country", title="Happiness Score vs. GDP")
```
```
#Happiness Score vs Social Support
px.scatter(WHR, "Social Support", "Happiness Score", color = "Continent", hover_name = "Country", animation_frame="Year", animation_group="Country", title="Happiness Score vs. Social Support")
```
```
#Happiness Score vs Life Expectancy
px.scatter(WHR, "Life Expectancy", "Happiness Score", color = "Continent", hover_name = "Country", animation_frame="Year", animation_group="Country", title="Happiness Score vs. Life Expectancy")
```
```
#Happiness Score vs Freedom
px.scatter(WHR, "Freedom", "Happiness Score", color = "Continent", hover_name = "Country", animation_frame="Year", animation_group="Country", title="Happiness Score vs. Freedom")
```
```
#Happiness Score vs Generosity
px.scatter(WHR, "Generosity", "Happiness Score", color = "Continent", hover_name = "Country", animation_frame="Year", animation_group="Country", title="Happiness Score vs. Generosity")
```
```
#Happiness Score vs Corruption
px.scatter(WHR, "Corruption", "Happiness Score", color = "Continent", hover_name = "Country", animation_frame="Year", animation_group="Country", title="Happiness Score vs. Corruption")
```

* A time-series regression analysis was used to develop a model which was trained and tested on the cleaned dataset. The accuracy of the model was then evaluated both on the cleaned dataset and on the reserved data from 2022. The final and most significant life evaluation factors (as determined by the previous graphs) were therefore used to predict future happiness scores.

```
#Read in cleaned dataset
WHR = pd.read_csv(r"/content/WHR.csv")

#Machine learning imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

#Keep columns relavent for ML
ML_WHR = WHR.drop(columns=["Country", "Year", "Continent", "Generosity", "Corruption"])

#Divide into X & y
X = StandardScaler().fit_transform(ML_WHR.drop(columns=["Happiness Score"]))
y = ML_WHR["Happiness Score"]
#Split data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
#Create and train model
mult_reg = LinearRegression()
mult_reg.fit(X_train,y_train)
```
```
#Generate Predictions
predicted = mult_reg.predict(X_test)
actual = np.array(y_test)

#MSE & RMSE
mse = mean_squared_error(predicted,actual)
rmse = mean_squared_error(predicted,actual,squared=False)
print("MSE is ", round(mse,5), " and RMSE is ", round(rmse,5))
```
```
#Beta Coefficients
coef = np.around(mult_reg.coef_,3)
intercept = round(mult_reg.intercept_,3)
factor_impact = (coef - coef[1]) / coef[1]

print(factor_impact)
```
```
#Multiple Linear Regression Equation
print("Score =", coef[0],"*GDPperCap +", coef[1],"*SocialSup +",coef[2],"*LifeExpect +",coef[3],"*Freedom +", intercept)
```
```
#Future Analysis
#Read in original dataset
WHR22 = pd.read_excel(r"/content/WHR Data.xls")

#Drop rows with nan values
WHR22 = WHR22.dropna()
#Drop irrelevant columns
WHR22 = WHR22.drop(columns=["Positive affect","Negative affect"])
#Rename columns to more user friendly names
WHR22 = WHR22.rename(columns={"Country name":"Country","year":"Year","Life Ladder":"Happiness Score","Log GDP per capita":"Log GDP per Cap","Social support":"Social Support","Healthy life expectancy at birth":"Life Expectancy",
                    "Freedom to make life choices":"Freedom","Perceptions of corruption":"Corruption"})

#Select only 2022 data
WHR22 = WHR22[WHR22["Year"] == 2022].reset_index(drop=True)

#Scale factors just as they were done for ML
WHR22[["Log GDP per Cap","Social Support","Life Expectancy","Freedom"]] = StandardScaler().fit_transform(WHR22[["Log GDP per Cap","Social Support","Life Expectancy","Freedom"]])

#Create new column with predictions
predictions = []
for i in WHR22.index:
   predictions = np.append(predictions, coef[0]*WHR22["Log GDP per Cap"][i] + coef[1]*WHR22["Social Support"][i] + coef[2]*WHR22["Life Expectancy"][i] +
                           coef[3]*WHR22["Freedom"][i] + intercept)
WHR22["Predicted Happiness Score"] = predictions

#Finalized WHR22 dataset
WHR22

#MSE & RMSE
mse = mean_squared_error(predictions,WHR22["Happiness Score"])
rmse = mean_squared_error(predictions,WHR22["Happiness Score"],squared=False)
print("MSE is ", round(mse,5), " and RMSE is ", round(rmse,5))
```

* After running the regression model in order to generate happiness score predictions for 2022, the resulting table with the appended scores should look similar to the one provided below:
  
![PredictedData](https://github.com/C-Crenshaw/Project3_DS4002/blob/25f2f6d8c43f9a39bce51e6054142064107889d5/FIGURES/PredictedDatasetTable.png)





