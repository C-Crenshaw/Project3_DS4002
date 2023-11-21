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
  
![CleanedData](https://github.com/C-Crenshaw/Project3_DS4002/blob/519cd6610566cef58de11e2b2f7d6661f64622d0/FIGURES/CleanedDatasetTable.png)

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

## [Data](https://github.com/C-Crenshaw/Project3_DS4002/tree/775436d2e125fa8099843f56c243f77f771fd2b5/DATA)

This section contains all of the data for this project. Data dictionaries are provided below and are organized by dataset. Relevant notes about use of data are also included. 

The original dataset is pulled directly from the United Nation’s annual [World Happiness Report](https://worldhappiness.report/data/). This dataset contains happiness scores for all internationally recognized countries beginning in the early 2000s [1]. The happiness scores are based on surveyed individuals’ own assessments of their lives [2]. With the happiness scores, the dataset also includes six additional variables that contextualize the state of the referenced countries. The dataset reviews the current state of happiness in the world and illustrates national variations in happiness. 

As a consequence of limited international information sharing, some values were missing and not all countries were present across the time measured. To account for this unfortunate constraint, the project group decided to carry out this analysis specifically for the years 2013-2019. Despite the presence of much data from 2020 and 2021, these years were excluded because they were unprecedented years with clear and strong confounding factors. The final range (2013-2019) was selected due to the retention of countries as detailed in the data cleaning process. In other words, this data cleaning process (which only includes countries that had a complete reference of data for the time period aforementioned) left the largest complete dataset possible. However, data from 2022 will be used to assess the model’s accuracy beyond the train test split for 2013-2019.

The updated dataset ('WHR.csv') is a cleaned version of the original and is largely used throughout the analysis. Within the python file it is used/uploaded in excel format, but can also be downloaded as a .csv when cleaned. 

**WHR.csv**
| 	Column Name	 | 	Description	 | 	Data Type	 |  
| 	:-----:	 | 	:-----:	 | 	:-----:	 |
| Country	| 	Name of the Country	| 	String	 | 
| Year	| 	Year for the data provided	| 	Integer	 | 
| Happiness Score	| 	Average reported happiness score from 0 to 10. Lower score indicates less happiness.	| 	Float	 | 
| Log GDP per Cap		| 	Logarithmic GDP per capita determined by WHR	| 	Float	 | 
| Social Support	| 	Social support score from 0 to 1. Lower score indicates less social support as determined by WHR	| 	Float	 | 
| Life Expectancy	| 	Life expectancy at birth in years	| 	Float	 | 
| Freedom	| 	Freedom to make life choices score from 0 to 1. Lower score indicates less freedom as determined by WHR	| 	Float	 | 
| Generosity	| 	Generosity score where a lower score indicates less generosity as determined by WHR	| 	Float	 | 
| Corruption	| 	Citizens’ perception of government corruption score from 0 to 1. Lower score indicates lesser belief of corruption as determined by WHR	| 	Float	 | 
| Continent	| 	Corresponding continent for the country	| 	String	 | 

Note: Log GDP per Cap is used rather than simply GDP per capita due to the exponential growth rate of GDP per capita. Since a relatively large time frame is being looked at, this will allow for there to be less exaggeration of this exponential growth. 

At the conclusion of the regression analysis, a subsequent table is published that contains the predicted happiness scores for countries in the year 2022. This table is not exported as a new dataset, but could be if determined necessary. This dataset is named "WHR22" within the SRC code files. 

## [Figures](https://github.com/C-Crenshaw/Project3_DS4002/tree/aadce70ed70bc3618120e175f115e9f0564facfe/FIGURES)

This section contains all of the figures generated by the project. A complete list and summary description of each figure is organized in the markdown table below.

| 	Figure Title	 | 	Description	 | 
| 	:-----:	 | 	:-----:	 |
| 	[Line Graph of Happiness Score by Continent over Time](https://github.com/C-Crenshaw/Project3_DS4002/blob/aadce70ed70bc3618120e175f115e9f0564facfe/FIGURES/HappinessScorebyContinentOverTime.png)	| 	An initial exploration into the dataset; demonstrates a first overview of the relationship between countries and happiness scores over time. Grouped by continent, this graphical representation demonstrates a clear difference between countries of various continents and their average happiness score. 	| 
| 	[Scatterplot of Happiness Score vs GDP](https://github.com/C-Crenshaw/Project3_DS4002/blob/aadce70ed70bc3618120e175f115e9f0564facfe/FIGURES/HappinessScorevsGDP.png)	| 	Raw comparison between happiness scores and log GDP per capita.	| 
| 	[Scatterplot of Happiness Score vs Social Support](https://github.com/C-Crenshaw/Project3_DS4002/blob/aadce70ed70bc3618120e175f115e9f0564facfe/FIGURES/HappinessScorevsSocialSupport.png)	| 	Raw comparison between happiness scores and social support.	| 
| 	[Scatterplot of Happiness Score vs Life Expectancy](https://github.com/C-Crenshaw/Project3_DS4002/blob/aadce70ed70bc3618120e175f115e9f0564facfe/FIGURES/HappinessScorevsLifeExpectancy.png)	| 	Raw comparison between happiness scores and life expectancy.	| 
| 	[Scatterplot of Happiness Score vs Freedom](https://github.com/C-Crenshaw/Project3_DS4002/blob/aadce70ed70bc3618120e175f115e9f0564facfe/FIGURES/HappinessScorevsFreedom.png)	| 	Raw comparison between happiness scores and freedom.	| 
| 	[Scatterplot of Happiness Score vs Generosity](https://github.com/C-Crenshaw/Project3_DS4002/blob/aadce70ed70bc3618120e175f115e9f0564facfe/FIGURES/HappinessScorevsGenerosity.png)	| 	Raw comparison between happiness scores and generosity.	| 
| 	[Scatterplot of Happiness Score vs Corruption](https://github.com/C-Crenshaw/Project3_DS4002/blob/aadce70ed70bc3618120e175f115e9f0564facfe/FIGURES/HappinessScorevsCorruption.png)	| 	Raw comparison between happiness scores and corruption.	| 
| 	Final Presentation	| 	Link to the final Project 3 powerpoint presentation.	|

As an addition to the table that organizes all of the figures contained within this repository, a video demonstration is submitted which illustrates the interactive nature of the scatterplots created. 
[![Scatterplots](https://github.com/C-Crenshaw/Project3_DS4002/blob/aadce70ed70bc3618120e175f115e9f0564facfe/FIGURES/HappinessScorevsCorruption.png)](https://www.youtube.com/watch?v=QudqajEl0hg&t=21s)

## [License](https://github.com/C-Crenshaw/Project1_DS4002/blob/19a904af31fb17d7c6a334f728b05b2a784e7304/LICENSE)

This project is licensed under the MIT License. See the LICENSE.md file for details. 

## References
[Link-to-MI1-Doc](https://docs.google.com/document/d/18SwcHu3ZOJcJB3VtFBrnd3wXZO9DUk7lSsIfbL_4J0w/edit?usp=sharing)

[Link-to-MI2-Doc](https://docs.google.com/document/d/1Lxu5n4c0crAGLUrOInCG0xXO40D9k8R2-GWcNupuO1A/edit?usp=sharing)

[1] “Foreword | The World Happiness Report,” Mar. 18, 2022. Available:
https://worldhappiness.report/ed/2022/foreword. [Accessed Nov. 5, 2023].

[2] “About | The World Happiness Report.”Available: https://worldhappiness.report/about/.
[Accessed Nov. 5, 2023]. 

## Acknowledgments
This README structure is adopted from [@DomPizzie](https://gist.github.com/DomPizzie) on Github. 
