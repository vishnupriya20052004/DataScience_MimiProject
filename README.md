
# EX NO : 10 DataScience_MiniProject  
## Rainfall Analysis of India

## DATE : 12/11/2023

## Description :
As is widely recognized, rainfall data is necessary for the mathematical modelling of extreme hydrological events, such as droughts or floods, as well as for evaluating surface and subsurface water resources and their quality. The phase, quantity, and elevation of generic hydrometeors in the atmosphere can be estimated by ground-based radars. Satellites can provide images with visible and infrared radiation, and they can also serve as platforms for radiometers to derive the quantity and phase of hydrometeors. Radars and satellites provide spatial information on precipitation at wide scales, avoiding many problems connected to local ground measurements, including those for the areal inhomogeneity of a network. However, direct rainfall observations at point scale can be obtained only by rain gauges installed at the soil surface.

## KEY FEATURES :
Three main characteristics of rainfall are its amount, frequency and intensity, the values of which vary from place to place, day to day, month to month and also year to year. Precise knowledge of these three main characteristics is essential for planning its full utilization.

## CODE :
DEVELOPED BY : VISHNU PRIYA.S

REGISTER NO : 212221040181
```
import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import pickle
from sklearn.model_selection import train_test_split
#read the whole file 
df.iloc[:]
df.info()
print(f'Number of Rows: {df.shape[0]}')
print(f'Number of Columns: {df.shape[1]}')
df.columns
row3 = df.iloc[3]
print(row3)
print("\n")
row115 = df.iloc[115]
print (row115)
description = df.describe()
print(description)
#plot the actual rain 
dg = df[['jun','jul','aug','sep']]
dg.plot(figsize=(20, 5));
#plot shows the total rain 
dh = df[['total']]
dh.plot(figsize=(10, 10));
#dy represent dataframe for 10 last rows which is rainfall from year 2007 to 2016 
dy = df.tail(10)
print(dy)
#plot diagram shows only "TOTAL" rain from dy dataframe (rainfall from year 2007 to 2016)
de = dy[['total']]
de.plot(figsize=(10, 10));
#rainfall from 2007 to 2016
#plot actual rain and departure percentage 
a_d = dy[['jun', 'jul', 'aug', 'sep', 'jun_p', 'jul_p', 'aug_p', 'sep_p']]
a_d.plot(figsize=(20, 10));
#rainfall from 2007 to 2016
#only plot actual rain 
actual = dy[['jun', 'jul', 'aug', 'sep']]
actual
actual = dy[['jun', 'jul', 'aug', 'sep']]
# Calculate the rainfall ranges for each month
rainfall_ranges = actual.max() - actual.min()
# Display the rainfall ranges, minimum, and maximum values
for month, rainfall_range in rainfall_ranges.items():
    min_rainfall = actual[month].min()
    max_rainfall = actual[month].max()
    
    print(f"Rainfall Range for {month}: Range = {rainfall_range} mm,", end=' ')
    print(f"Min = {min_rainfall} mm, Max = {max_rainfall} mm")
actual.plot(figsize=(20, 10))
# Add a title and labels for the axes
plt.title('Actual Rainfall (mm) vs Year')
plt.xlabel('Year')
plt.ylabel('Actual Rainfall (mm)')
plt.show()
#rainfall from 2007 to 2016
#plot departure percentage 
de = dy[['jun_p', 'jul_p', 'aug_p', 'sep_p']]
de
de = dy[['jun_p', 'jul_p', 'aug_p', 'sep_p']]
# Calculate the rainfall ranges for each month
rainfall_ranges = de.max() - de.min()
# Display the rainfall ranges, minimum, and maximum values
for month, rainfall_range in rainfall_ranges.items():
    min_rainfall = de[month].min()
    max_rainfall = de[month].max()
    
    print(f"Rainfall Range for {month}: Range = {rainfall_range} %,", end=' ')
    print(f"Min = {min_rainfall} %, Max = {max_rainfall} %")
print("\n")
de.plot(figsize=(20, 10))
# Add a title and labels for the axes
plt.title('Departure Percentage vs Year')
plt.xlabel('Month')
plt.ylabel('Departure Percentage (%)')
plt.show()
#From 1901 to 2016
actual_df = df[['jun', 'jul', 'aug', 'sep']]
actual_df
# Calculate the average rainfall for each month
av_df = actual_df.mean()
av_df
#Stacked area chart
#Calculate the cumulative rainfall for each month or season
cumulative_rainfall = df[['jun', 'jul', 'aug', 'sep']].cumsum(axis=1)
# Plot the stacked area chart
plt.figure(figsize=(10, 6))
plt.stackplot(df.index, cumulative_rainfall.values.T, labels=['Jun', 'Jul', 'Aug', 'Sep'])
plt.title('Cumulative Rainfall Over Time')
plt.xlabel('Year')
plt.ylabel('Cumulative Rainfall (mm)')
plt.legend(loc='upper left')
plt.show()
#Scatter plot
# Create a scatter plot with trendline
plt.figure(figsize=(10, 6))
sns.regplot(x=actual_df.index, y='jun', data=actual_df, scatter=True, label='June')
sns.regplot(x=actual_df.index, y='jul', data=actual_df, scatter=True, label='July')
sns.regplot(x=actual_df.index, y='aug', data=actual_df, scatter=True, label='August')
sns.regplot(x=actual_df.index, y='sep', data=actual_df, scatter=True, label='September')
plt.title('Rainfall Variation over Time')
plt.xlabel('Year')
plt.ylabel('Rainfall (mm)')
plt.legend()
plt.show()
# Reset display options to default
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
actual_df
# Plotting the predicted values versus the actual values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.show()
```
## OUPUT:
![image](https://github.com/Anuayshh/Mini-Project/assets/127651217/3951899a-1c76-47ea-b082-d5f390a4616e)
![image](https://github.com/Anuayshh/Mini-Project/assets/127651217/7b72f3e0-6726-4a55-aad6-7c8e605cfc17)
![image](https://github.com/Anuayshh/Mini-Project/assets/127651217/03a9c408-6a16-4ee5-a7a6-df7a7e5a18cc)

print(f'Number of Rows: {df.shape[0]}') : to print the number of rows
print(f'Number of Columns: {df.shape[1]}') : to print number of columns
![image](https://github.com/Anuayshh/Mini-Project/assets/127651217/65b0bb3a-e18e-4b55-9e8c-8713789b8296)

df.columns is to retrieve the column labels or column names of a DataFrame
to shows data at row 3 (year 1904) and row 115 (year 2016)
![image](https://github.com/Anuayshh/Mini-Project/assets/127651217/81406b91-12d4-4d0f-8336-237c5ba90bca)
![image](https://github.com/Anuayshh/Mini-Project/assets/127651217/8648c3a4-742f-4878-80e8-6bdc571679d2)
![image](https://github.com/Anuayshh/Mini-Project/assets/127651217/a8d4a293-8118-4100-90e2-97a37a63b7ac)
![image](https://github.com/Anuayshh/Mini-Project/assets/127651217/d40f794e-2069-44ea-a532-d11111d4e069)
![image](https://github.com/Anuayshh/Mini-Project/assets/127651217/993dc66a-5650-4b6a-a012-526d725f5047)
![image](https://github.com/Anuayshh/Mini-Project/assets/127651217/293e6529-3814-4e92-a354-19f08e56ea10)
![image](https://github.com/Anuayshh/Mini-Project/assets/127651217/7016fc23-3624-412a-9b83-57b087c4604e)
![image](https://github.com/Anuayshh/Mini-Project/assets/127651217/baa1763e-91d7-4c2d-aafe-7071467a9101)
![image](https://github.com/Anuayshh/Mini-Project/assets/127651217/6b2f5ab5-17e0-4a45-a0ec-7206dc904ba3)
![image](https://github.com/Anuayshh/Mini-Project/assets/127651217/e38b2f30-5cd7-4bd2-8fcf-a7d0481cdbb5)
![image](https://github.com/Anuayshh/Mini-Project/assets/127651217/c96bdd63-9d40-42a8-8723-2cf02487172b)
![image](https://github.com/Anuayshh/Mini-Project/assets/127651217/7f9eb9e9-6c7e-4904-b3f3-5217abc1d369)
![image](https://github.com/Anuayshh/Mini-Project/assets/127651217/e5da59c0-250f-4ed0-85f3-06c67177b4e9)
![image](https://github.com/Anuayshh/Mini-Project/assets/127651217/595a4ce4-d1d5-4cd9-a6dd-82d646371eab)
![image](https://github.com/Anuayshh/Mini-Project/assets/127651217/a1ae278f-56ff-4f96-88c4-2137659cbe32)
![image](https://github.com/Anuayshh/Mini-Project/assets/127651217/7c623e35-9a23-4cd2-b54d-9befe42ea958)

Based on the cumulative rainfall data for the months of June, July, August, and September from 1901 to 2016,
The cumulative rainfall values represent the total amount of rainfall received from June to September for each year. 
The data shows variations in cumulative rainfall over time, indicating fluctuations in precipitation patterns.
![image](https://github.com/Anuayshh/Mini-Project/assets/127651217/c41e54c5-9769-48bb-8b50-0642634f3104)
![image](https://github.com/Anuayshh/Mini-Project/assets/127651217/11b8f249-2db7-484d-93a8-0cc60e3d8601)



## RESULT:
Thus the rainfall analysis has been executed successfully.
