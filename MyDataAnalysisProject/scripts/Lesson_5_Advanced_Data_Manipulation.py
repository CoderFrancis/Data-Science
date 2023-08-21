import pandas as pd
import numpy as np

# Sample data
data = {
    'Department': ['HR', 'IT', 'IT', 'Sales', 'Sales', 'HR'],
    'Employee': ['Mike', 'Tom', 'Sue', 'Jane', 'John', 'Mary'],
    'Salary': [50000, 80000, 90000, 75000, 85000, 52000]
}

# Creating DataFrame
df = pd.DataFrame(data)

# Grouping data by department
grouped = df.groupby('Department')

# Calculating the average salary by department
# print(grouped['Salary'].mean())

# Creating two DataFrames
data1 = {'key': [1, 2, 3], 'value': ['A', 'B', 'C']}
data2 = {'key': [2, 3, 4], 'value': ['X', 'Y', 'Z']}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Merging DataFrames on key column
merged = pd.merge(df1, df2, on='key')
# print(merged)

# Creating a pivot table of average salary by department
pivot_table = pd.pivot_table(df, values='Salary', index='Department', aggfunc=np.mean)
# print(pivot_table)

# Creating time series data
time_data = pd.date_range('2023-01-01', periods=10, freq='D')
time_series = pd.Series(range(10), index=time_data)

# Resampling time series data (downsampling to 3-day intervals)
resampled = time_series.resample('3D').mean()
print(resampled)

# Lesson 6: Introduction to Machine Learning

