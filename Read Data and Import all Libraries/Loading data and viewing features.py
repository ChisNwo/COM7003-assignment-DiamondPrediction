# Dataset load from CSV file
import pandas as pd
import sns
from matplotlib import pyplot as plt
import seaborn as sns
from pandas import DataFrame

# Loading the data correctly
data: DataFrame = pd.read_csv(r'C:\Users\nwoko\OneDrive\Desktop\Data Science and AI Program\Artificial Intelligence Course\AI Assignment\diamonds.csv', encoding='ascii')

# Checking for null values
print(f'There are {data.isna().sum().sum()} null values in the dataset.')

#Checking for duplicate entries
print(f'There are {data.duplicated().sum()} duplicate entries in the dataset.')

# Displaying the first few rows of the dataset
print('Dataset head:')
print(data.head())

# Displaying the last few rows of the dataset
print('Dataset tail:')
print(data.tail())

# Print dataset info, including null values and data types
print('\nDataset info:')
print(data.info())

# Summary statistics for numerical features
print('\nSummary statistics:')
print(data.describe())

# Showing the features of the price column
print(data['price'].describe())

