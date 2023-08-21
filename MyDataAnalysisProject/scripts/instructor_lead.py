import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# data = {
#    'Name': ['John', 'Alice', 'Bob', 'Eve'],
#    'Age': [24, 28, 22, 30],
#    'City': ['New York', 'Los Angeles', 'Chicago', 'Miami']
# }

# df = pd.DataFrame(data)
# print(df)

# ages = df['Age']
# print(ages)

# print(df.describe())

# names = ['John', 'Alice', 'Bob', 'Eve']
# ages = [24, 28, 22, 30]

# plt.plot(names, ages)
# plt.xlabel('Names')
# plt.ylabel('Ages')
# plt.show()

# Example DataFrame with missing values
# data = {'A': [1, 2, np.nan], 'B': [4, np.nan, 6], 'C': [7, 8, 9]}
# df = pd.DataFrame(data)

# Filling missing values with a placeholder
# df.fillna(value=0, inplace=True)

# Grouping data by a column and calculating mean on the groups
# grouped = df.groupby('A').mean()
# print(grouped)

# Split data into 80% train and 20% test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a linear regression model
# model = LinearRegression()

# Train the model
# model.fit(X_train, y_train)

# Make predictions on the test data
# predictions = model.predict(X_test)

# Calculate the mean squared error of our predictions
# mse = mean_squared_error(y_test, predictions)
# print(f"Mean Squared Error: {mse}")

# from sklearn.datasets import load_iris
# data = load_iris()
# X = data['data']
# y = data['target']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# classifier = KNeighborsClassifier(n_neighbors=5)

# classifier.fit(X_train, y_train)

# y_pred = classifier.predict(X_test)

# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

ages = [22, 24, 26, 28, 30, 32, 34]
salaries = [40000, 45000, 50000, 55000, 60000, 65000, 70000]

plt.plot(ages, salaries)
plt.xlabel('Ages')
plt.ylabel('Salaries')
plt.title('Salary by Age')
# plt.show()

# Creating a sample dataset
data = {'Ages': ages, 'Salaries': salaries}
df = pd.DataFrame(data)

# Scatter plot with regression line
sns.regplot(x='Ages', y='Salaries', data=df)
# plt.show()

plt.hist(salaries, bins=5, edgecolor='k')
plt.xlabel('Salaries')
plt.ylabel('Frequency')
plt.title('Histogram of Salaries')
plt.show()