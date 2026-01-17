import pandas as pd

dataset = pd.read_excel("default of credit card clients.xls", header=1)

print(dataset.shape)
print(dataset.head)
print(dataset.info())
print(dataset['default payment next month'].value_counts())
