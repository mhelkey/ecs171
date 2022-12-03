# ECS 171 Final Project

**Group 21**

- Nandhini Sridhar
- Trevor Lopez
- Dylan Ang
- Matthew Helkey



## About the Dataset


The [data set](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease) comes from the CDC’s Behavioral Risk Factor Surveillance System, in which roughly 400,000 phone interviews are conducted every year. The original dataset has approximately 300 features, but this one has been previously reduced to 17. The dataset includes factors like BMI and age, lifestyle factors like smoking and alcohol use, and previous medical conditions like asthma and cancer.
 
> https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

## Abstract

According to the CDC, heart disease is the leading cause of death in the US. To help classify causes for heart disease, the CDC conducts over 400,00 phone interviews annually, and compiles the results into a public dataset. We aim to use a subset of this data to determine important factors associated with heart disease. This analysis can be used to both inform the general public for practical lifestyle decisions and develop a predictive model for medical practices in considering the risk factors of patients. Using supervised learning algorithms, we plan to train a neural network to classify whether individuals have heart disease based on 17 features related to lifestyle and health conditions.

## Methods

We used the scikit-learn Python library in a [jupyter notebook](main.ipynb). 

### Data Exploration
We first found that none of the observations had missing data or null values, so we did not have to do anything with those. 

#### Categorical Data
We then figured out which variables in our data were categorical, and what unique values each variable had. We then plotted the frequency of the unique values of each categorical variable in bar charts. We found that a majority of survey participants did not have preexisting conditions such as skin cancer, heart disease, a previous stroke, difficulty walking, diabetes, asthma, or kidney disease. There were more non-smokers than smokers. The participants were almnost equally split between female and male, with more females. Additionally, there were more older participants from the ages 65-69, and older age groups tended to have more survey partcipants than younger age groups. More participants tended to rate their health as better rather than worse, with the most rating their general health as very good and the least rating their health as poor. We saw a huge disparity in our race distribution, with a clear majority of the survey participants being white.

We recognized this as a potential source of bias if race emds up being an important factor in our prediction, as our model would likely be more effective at predicting heart disease in white people than other racial groups due to sample size.

#### Numerical Data
We created a heatmap and a pairplot to explore our numeric data. We chose not to include our categorical data even after enocding since most were binary valued and not ordered scales.

[DESCRIBE CORRELATION MATRIX HERE]

Our pairplot showed us the relationships between the various numeric variables as well as the distributions of each. BMI seemed to have a sort of skewed normal distribution, but all the others seemed to have distributions that were distinctly separated by the various data values. The scatter plots did not show much of a linear relationship between any of the variables. This is in accordance with the low correlations shown in the correlation matrix.

### Preprocessing
After performing exploratory data anlysis, we label encoded our categorical variables. We opted for one-hot encoding for encoding the race of the person, as race is not a scale, nor is it binary. We then explored the distributions of race in our data set and discovered that over 75% of the data is listed as white. This could lead to our final model being better at predicting heart disease in white people than in other races. Finally, we normalized the numerical variables in our data to be between 0 and 1.

### Model 1
Our initial model was built with three layers: an input layer with a tanh activation function, a hidden layer with a relu activation function, and an output layer with a sigmoid activation function, since our aim is to do binary classification. Predictions were then thresholded with a threshold of 0.5. 

### Model 2

## Results
### Model 1
After predictions were thresholded with a threshold of .5, the initial model had a training data accuracy of about 92% and a testing data accuracy of about 91%. The model was not at the absolute minima of the fitting graph, but instead one step up with a higher loss than the minima. 

### Model 2

## Discussion

## Conclusion

## Collaboration




