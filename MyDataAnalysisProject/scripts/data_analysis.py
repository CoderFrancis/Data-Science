import pandas as pd
import os
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np


os.chdir('C:/Users/Mikha/Documents/GitHub/Data-Science/MyDataAnalysisProject')

data = {'name': ['John', 'Jane', 'Sue'], 'age': [28, 34, 45], 'city': ['New York', 'Los Angeles', 'Chicago']}
df = pd.read_excel('data/data.xlsx')
# print(df.head())
names = df['name']
first_row = df.loc[0]
# print(first_row)

older_than_30 = df[df['age'] > 30]
# print(older_than_30)
cleaned_df = df.dropna()
filled_df = df.fillna(0)
# print(df.describe())

# Matplotlib

# Sample data
# x = [1, 2, 3, 4, 5]
# y = [2, 4, 6, 8, 10]

# Create a figure and axis
# plt.figure()

# Plot the data
# plt.plot(x, y, label='y = 2x')

# Add labels and title
# plt.xlabel('X Axis')
# plt.ylabel('Y Axis')
# plt.title('Simple Line Plot')
# plt.legend()

# Display the plot
# plt.show()

# Seaborn

# Load a built-in dataset
# tips = sns.load_dataset('tips')

# Create a boxplot
# sns.boxplot(x='day', y='total_bill', data=tips)

# Add labels and title
# plt.xlabel('Day of the Week')
# plt.ylabel('Total Bill')
# plt.title('Total Bill Distribution by Day')

# Display the plot
# plt.show()

# numpy

a = np.array([1, 2, 3])
b = np.array([(1.5, 2, 3), (4, 5, 6)], dtype=float)

c = a + 2
d = a * 2
e = a + b[0]

first_element = a[0]
sub_array = b[0, 1:3]

mean = np.mean(a)
max_value = np.max(b)

reshaped_array = b.reshape(3, 2)

above_one = a[a > 1]

