# Data Exploration - Categorical Columns
# Plotting the histograms for the categorical variables, including a 5% threshold that highligths
# those classes with a frequency smaller than 0.05. A typical value chosen as threshold to distinguish
# between more common classes and less frequent ones. Dealing with rare classes in the categorical
# variables can result in problems during the regression stage, such as overfitting.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Loading data
data = pd.read_csv(r'C:\Users\nwoko\OneDrive\Desktop\Data Science and AI Program\Artificial Intelligence Course\AI Assignment\diamonds.csv', encoding='ascii')

# Defining a 5% threshold of the total amount of data
five_perc_threshold = len(data) * 0.05

cat_cols = [col for col in data.columns if data[col].dtypes == 'O']

fig, (ax1,ax2,ax3) = plt.subplots(ncols=3, figsize=(12,6))

ax1 = sns.histplot(x=data[cat_cols[0]], ax=ax1)
ax1.set_xticklabels(ax1.get_xticklabels(), size=8)
ax1.axhline(five_perc_threshold, ls='--', color='red')
ax1.text(3, five_perc_threshold+500, "5% threshold", color='red')
ax1.set_title(f'Histogram of {cat_cols[0]}', size=20)

ax2 = sns.histplot(x=data[cat_cols[1]], ax=ax2)
ax2.axhline(five_perc_threshold, ls='--', color='red')
ax2.text(4, five_perc_threshold+250, "5% threshold", color='red')
ax2.set_title(f'Histogram of {cat_cols[1]}', size=20)

ax3 = sns.histplot(x=data[cat_cols[2]], ax=ax3)
ax3.axhline(five_perc_threshold, ls='--', color='red')
ax3.text(5, five_perc_threshold+250, "5% threshold", color='red')
ax3.set_xticklabels(ax3.get_xticklabels(), size=8)
ax3.set_title(f'Histogram of {cat_cols[2]}', size=20)

plt.suptitle('Distribution of the Categorical Variables', size=30)

plt.tight_layout()

# Price vs Cut, Price vs Color and Price vs Clarity
# Creating a new column for price per carat
data['price_per_carat'] = data['price'] / data['carat']

# Computing mean prices per categorical feature
price_cut = data.groupby('cut')[['price', 'price_per_carat']].mean().reset_index().sort_values('price_per_carat', ascending=False)
price_color = data.groupby('color')[['price', 'price_per_carat']].mean().reset_index().sort_values('price_per_carat', ascending=False)
price_clarity = data.groupby('clarity')[['price', 'price_per_carat']].mean().reset_index().sort_values('price_per_carat', ascending=False)

# Storing grouped data for easy iteration
grouped_data = [[price_cut, 'cut'], [price_color, 'color'], [price_clarity, 'clarity']]

# Plotting histograms for each categorical feature
for df, category in grouped_data:
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))

    # Barplot for average price
    sns.barplot(data=df, x=category, y='price', ax=ax1)
    ax1.set_title(f'Avg Price vs {category}', fontsize=19)
    ax1.set_xlabel(category)
    ax1.set_ylabel('Avg Price')

    # Barplot for average price per carat
    sns.barplot(data=df, x=category, y='price_per_carat', ax=ax2)
    ax2.set_title(f'Avg Price per Carat vs {category}', fontsize=19)
    ax2.set_xlabel(category)
    ax2.set_ylabel('Avg Price per Carat')

    # Overall title
    plt.suptitle(f'Average Price vs {category}', fontsize=25)

    # Improve layout & display plot
    plt.tight_layout()
    plt.show()

# It can be deduced that:

# A more relevant comparison between classes is obtained by considering 'price_per_carat' instead of 'price' as the y variable.
# There are some anomalies in the histograms. One of them concerns the classification of the average prices of diamonds in
# terms of their cut. One can notice that the diamond cuts with the highest average price per carat are those with a 'Premium'
# cut, while one would expect the 'Ideal' cut to be, on average, the most expensive.
# Even in the case of 'clarity' one can notice the presence of some anomalies, given that 'VVS1' diamonds do not hold the second position in the price per carat chart.
# Finally, the color chart seems completely off.
# These problems or anomalies could be due to the presence of outliers in the data.


# PRICE VS CARAT
# plotting 'price' vs 'carat'  and making sure that the categorical variables is hue.
# Expecting that lines with different steepness will be displayed, given the diamonds'
# specific cuts, colors and clarity. The more the steepness of the line, the more the expensive cost of the diamond.

# Assuming categorical_columns (cat_cols) are defined, for example:
# cat_cols = ['cut', 'color', 'clarity']

for col in cat_cols:
    # Create lmplot with hue = categorical column
    g = sns.lmplot(data=data, x='carat', y='price', hue=col)

    # Set the title for the plot
    plt.title(f'Price vs Carat (hue = {col})', size=22)

    # Adjust the figure size for better visibility
    g.fig.set_figwidth(9)
    g.fig.set_figheight(6)

    # Show the plot (necessary in PyCharm)
    plt.show()

#These results are not so clear and, moreover, they apparently conflict with those of the histograms. In particular, one can notice that:

#The 'cut' lines have the steepness order (in terms of price): 'Ideal' > 'Premium' and 'Very Good', that what one expects.
#The 'clarity' lines have the steepness order (in terms of price) 'IF' > 'VVS1' > 'VVS2', which is fine.
#The 'color' lines steepness do not follow a clear pattern.