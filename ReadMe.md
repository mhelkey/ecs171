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

## Analysis

We used the scikit-learn Python library in a [jupyter notebook](main.ipynb). After performing exploratory data anlysis with a pairplot and a correlation matrix, we label encoded our categorical variables. We opted for one-hot encoding for encoding the race of the person, as race is not a scale, nor is it binary. We then explored the distributions of race in our data set and discovered that over 75% of the data is listed as white. This could lead to our final model being better at predicting heart disease in white people than in other races. 