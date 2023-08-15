import pandas as pd

data = {'name': ['John', 'Jane', 'Sue'], 'age': [28, 34, 45], 'city': ['New York', 'Los Angeles', 'Chicago']}
df = pd.read_csv('data/data.csv')
print(df.head())
