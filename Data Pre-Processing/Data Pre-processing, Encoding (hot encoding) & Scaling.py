# Defining Lib
from time import time

import pandas as pd
import tf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from tensorboard.notebook import display
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from xgboost import XGBRegressor

# Load dataset
data = pd.read_csv(r'C:\Users\nwoko\OneDrive\Desktop\Data Science and AI Program\Artificial Intelligence Course\AI Assignment\diamonds.csv')

# Defining numerical columns (excluding categorical features)
num_cols = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]


# Function to remove outliers using IQR
def remove_outliers_iqr(data, num_cols):
    """Removes outliers from numerical columns using the IQR method."""

    filtered_data = data.copy()  # Make a copy of the dataset

    for col in num_cols:
        Q1 = filtered_data[col].quantile(0.25)
        Q3 = filtered_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Applying filter on the specific column
        filtered_data = filtered_data[(filtered_data[col] >= lower_bound) & (filtered_data[col] <= upper_bound)]

    return filtered_data  # Return the cleaned dataset


# ✅ Call the function **AFTER** defining `num_cols`
data_filtered = remove_outliers_iqr(data, num_cols)

# Check how much data was removed
print(f"Original data shape: {data.shape}")
print(f"Filtered data shape: {data_filtered.shape}")
print(f"Number of rows removed: {data.shape[0] - data_filtered.shape[0]}")


# Subplots axes array
import matplotlib.pyplot as plt
import seaborn as sns

for col in num_cols:
    fig, axs = plt.subplots(ncols=4, figsize=(19, 9))

    # Histogram before filtering
    sns.histplot(x=data[col], bins=15, ax=axs[0])
    axs[0].set_title(f'{col} (before filtering)', size=19)

    # Boxplot before filtering
    sns.boxplot(x=data[col], ax=axs[1])
    axs[1].set_title(f'{col} (before filtering)', size=19)

    # Histogram after filtering
    sns.histplot(x=data_filtered[col], bins=15, ax=axs[2])
    axs[2].set_title(f'{col} (after filtering)', size=19)

    # Boxplot after filtering
    sns.boxplot(x=data_filtered[col], ax=axs[3])
    axs[3].set_title(f'{col} (after filtering)', size=19)

    # Super Title for better visualization
    plt.suptitle(f'Distribution of {col}', size=25)

    plt.tight_layout()
    plt.show()

# Number of rows before and after
    import sys

    # Ensure `data_filtered` exists
    if 'data_filtered' in locals():
        # Print summary **only once**
        print(f'Number of rows before filtering: {len(data)}')
        print(f'Number of rows after filtering: {len(data_filtered)}')
        print(f'Percentage of rows lost: {(len(data) - len(data_filtered)) / len(data) * 100:.0f}%.')

        # Ensure PyCharm prints the output immediately
        sys.stdout.flush()
    else:
        print("⚠️ Warning: 'data_filtered' is not defined. Ensure the filtering function ran correctly.")


# ENCODING

# Identify categorical columns *after* filtering
cat_cols = data_filtered.select_dtypes(include=['object', 'category']).columns

# Print value distribution for each categorical column
for col in cat_cols:
    print(f"Category Distribution for: {col}")
    print(data_filtered[col].value_counts(normalize=True).reset_index())
    print("=" * 50)  # Separator for better readability

# CLARITY: Measurement of how clear the diamond is [I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best)].
# 'I1' will be moved into the same class as 'SI2' and I will also move 'IF' into the 'VVS1' class.

# Replace clarity categories in one step
data_filtered['clarity'] = data_filtered['clarity'].replace({
    'I1': 'I1+SI2',
    'SI2': 'I1+SI2',
    'IF': 'IF+VVS1',
    'VVS1': 'IF+VVS1'
})

# Print the updated clarity distribution
print(data_filtered['clarity'].value_counts(normalize=True).reset_index())

# CUT: moving 'Fair' diamonds into the 'Good' class.
# Replace cut categories in one step
data_filtered['cut'] = data_filtered['cut'].replace({
    'Fair': 'Fair+Good',
    'Good': 'Fair+Good'
})

# Print the updated cut distribution
print(data_filtered['cut'].value_counts(normalize=True).reset_index())

# TRAIN-TEST SPLIT BEFORE ENCODING
import numpy as np
from sklearn.model_selection import train_test_split, validation_curve, GridSearchCV

# Check if 'price_per_carat' exists before dropping
cols_to_drop = ['price', 'price_per_carat']
cols_to_drop = [col for col in cols_to_drop if col in data_filtered.columns]

X = data_filtered.drop(cols_to_drop, axis=1)
y = data_filtered['price']

# Handle missing or infinite values
X = X.dropna().replace([np.inf, -np.inf], np.nan).dropna()
y = y[X.index]  # Ensure `y` matches `X` indices

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print final shapes
print("X_train:", X_train.shape, "| X_test:", X_test.shape)
print("y_train:", y_train.shape, "| y_test:", y_test.shape)


# Identify categorical columns
object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

# Debugging: Print categorical columns
print("Categorical columns found:", object_cols)

# Handle missing values in categorical columns
X_train[object_cols] = X_train[object_cols].fillna("Missing")
X_test[object_cols] = X_test[object_cols].fillna("Missing")

# Columns that can be safely ordinal encoded
good_label_cols = [col for col in object_cols if set(X_test[col]).issubset(set(X_train[col]))]

# Problematic columns that will be dropped
bad_label_cols = list(set(object_cols) - set(good_label_cols))

# Debugging: Check mismatches
for col in bad_label_cols:
    print(f"🚨 `{col}` has unseen categories in test set:", set(X_test[col]) - set(X_train[col]))

print("\n✅ Categorical columns that will be encoded:", good_label_cols)
print("\n⚠️ Categorical columns that will be dropped:", bad_label_cols)

# The categorical variables are of ordinal type. Therefore, ordinal encoding can be performed on all of them.
# However, there are no clear patterns in the 'cut', 'color' and 'clarity' features. Hence, why one-hot encoding will be done instead of ordinal encoding.

### One-hot encoding ###
oh_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').set_output(transform="pandas")

# Fit and transform the categorical columns
OHE_train = pd.DataFrame(oh_encoder.fit_transform(X_train[cat_cols]))
OHE_test  = pd.DataFrame(oh_encoder.transform(X_test[cat_cols]))

# One-hot encoding removed index; put it back
OHE_train.index = X_train.index
OHE_test.index  = X_test.index

# Remove categorical columns (will replace with one-hot encoding)
num_train = X_train.drop(cat_cols,axis=1)
num_test  = X_test.drop(cat_cols,axis=1)

# Add one-hot encoded columns to numerical features
OHE_X_train = pd.concat([num_train,OHE_train],axis=1)
OHE_X_test  = pd.concat([num_test,OHE_test],axis=1)


# SCALING #
from sklearn.preprocessing import StandardScaler

num_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

std_scaler = StandardScaler()

OHE_X_train[num_cols] = std_scaler.fit_transform(OHE_X_train[num_cols])
OHE_X_test[num_cols]  = std_scaler.transform(OHE_X_test[num_cols])

y_train = std_scaler.fit_transform(y_train.values.reshape(-1,1))
y_test  = std_scaler.transform(y_test.values.reshape(-1,1))

OHE_X_train.head()


