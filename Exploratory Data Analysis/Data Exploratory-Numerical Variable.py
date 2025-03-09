# Exploratory data analysis (Numerical Variables)
# Plotting the distribution of numeric variables (both histogram and barplots).
# This is important to understand the shape of the distribution and to see if there are outliers,
# and values of skewness and kurtosis of the distribution will provide more details on the distribution.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Loading data
data = pd.read_csv(r'C:\Users\nwoko\OneDrive\Desktop\Data Science and AI Program\Artificial Intelligence Course\AI Assignment\diamonds.csv', encoding='ascii')

# Selecting only numeric columns
num_cols = [col for col in data.columns if data[col].dtypes != 'O']

# Looping through numeric columns and plotting histogram & box-plots
for col in num_cols:
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))

    # Histogram to show value distribution for each numeric column
    sns.histplot(data[col], bins=30, kde=True, ax=ax1)
    ax1.set_title(f'Histogram of {col}', size=19)
    ax1.set_xlabel(col)

    # Boxplot to highlight outliers and spread the data
    sns.boxplot(x=data[col], ax=ax2)
    ax2.set_title(f'Boxplot of {col}', size=19)

    # Overall Title
    plt.suptitle(f'Distribution of {col}', size=25)

    # Adjusting layout and showing plots
    plt.tight_layout()
    plt.show()


# Ensure the correlation is calculated on numeric columns
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns

# Compute the correlation matrix for only numeric columns
correlation_matrix = data[numeric_cols].corr()

# Plotting the heatmap
plt.figure(figsize=(20, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Diamond Features', fontsize=20)
plt.show()
# From the heatmap, carat is the biggest factor to determine the price of a diamond.
# It obviously that X,Y and Z have a direct impact on the carat weight. The other features have a small value compare to it


# Scatterplots for numeric features
sns.scatterplot(x=data['price'], y=data['x'], color='red')
plt.title("Price vs 'x'")
plt.show()

sns.scatterplot(x=data['price'], y=data['y'], color='blue')
plt.title("Price vs 'y'")
plt.show()

sns.scatterplot(x=data['price'], y=data['z'], color='green')
plt.title("Price vs 'z'")
plt.show()

sns.scatterplot(x=data['price'], y=data['table'], color='purple')
plt.title("Price vs 'table'")
plt.show()

# Boxplots for categorical features
sns.boxplot(x=data['cut'], y=data['price'], color='orange')
plt.title("Price vs 'cut'")
plt.show()

sns.boxplot(x=data['color'], y=data['price'], color='cyan')
plt.title("Price vs 'color'")
plt.show()

sns.boxplot(x=data['clarity'], y=data['price'], color='brown')
plt.title("Price vs 'clarity'")
plt.show()

sns.boxplot(x=data['depth'], y=data['price'], color='pink')
plt.title("Price vs 'depth'")
plt.show()


# Calculating Skewness and Kurtosis of numeric columns
from scipy.stats import skew, kurtosis

# Loop through numeric columns and compute skewness & kurtosis

for col in num_cols:
    skewness = skew(data[col], nan_policy='omit')  # Handles NaN values if any
    kurt = kurtosis(data[col], nan_policy='omit')  # Handles NaN values if any

    print(f"Skewness of {col}: {skewness:.4f}") # To display result up to 4 decimal places for readability
    print(f"Kurtosis of {col}: {kurt:.4f}")
    print("-" * 40)  # Add separator for clarity


# List to store outlier percentages

outliers_perc = []

print("Percentage of outliers in the columns\n")

# Defining the list inside function to avoid global variable issues
def outliers_perc_search(data, num_features):
    outliers_perc = []

    for col in num_features:
        # Ensure the column is numeric
        if data[col].dtype != 'O':
            q1 = data[col].quantile(0.25)  # First quartile (25%)
            q3 = data[col].quantile(0.75)  # Third quartile (75%)
            IQR = q3 - q1  # Interquartile Range calculation

            # Find values outside 1.5*IQR range (outliers)
            outliers = data[(data[col] < (q1 - 1.5 * IQR)) | (data[col] > (q3 + 1.5 * IQR))]

            # Calculating percentages of outliers
            perc = len(outliers) * 100.0 / len(data)
            outliers_perc.append((col, round(perc, 1)))

            print(f"Column {col} outliers = {perc:.1f}%")

    return outliers_perc  # Return the list of outliers

# Run function and store results
outlier_results = outliers_perc_search(data, num_cols)

# NOTE
# All distributions (except 'x') either have a relatively high value of kurtosis and/or skewness.
# All variables have outliers, that may have to be dealt with before carrying out the regression analysis.
# The percentage of outliers in the columns is less than 5% (except in the 'price' column). It is worth investigating whether I can just drop them or not.