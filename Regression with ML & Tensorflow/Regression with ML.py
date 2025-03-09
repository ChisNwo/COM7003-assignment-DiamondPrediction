# Importing Libs

from time import time

import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from tensorboard.notebook import display
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv(r'C:\Users\nwoko\OneDrive\Desktop\Data Science and AI Program\Artificial Intelligence Course\AI Assignment\diamonds.csv')

# Defining numerical columns (excluding categorical features)
num_cols = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]

# Encode categorical features using One-Hot Encoding
categorical_features = ['cut', 'color', 'clarity']  # Adjust based on your dataset
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Avoid multicollinearity

# Apply One-Hot Encoding to categorical columns
encoded_features = encoder.fit_transform(data[categorical_features])

# Convert encoded features to DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

# Drop original categorical columns and merge encoded features
data = data.drop(columns=categorical_features).reset_index(drop=True)
OHE_X_train = pd.concat([data, encoded_df], axis=1)  # Ensure `OHE_X_train` is a DataFrame

# Define y_train (target variable)
y_train = data['price']  # Assuming 'price' is the target variable
OHE_X_train = OHE_X_train.drop(columns=['price'])  # Remove target from features



    ### Regression with Machine Learning Models ###


 # Comparing the performance of Four models
def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: target training set
       - X_test: features testing set
       - y_test: target testing set
    '''

    results = {}

    # Fit the learner to the training data using slicing with 'sample_size'
    start = time()  # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time()  # Get end time

    # Calculate the training time
    results['train_time'] = end - start

    # Get the predictions on the test set,
    # then get predictions on the first 300 training samples
    start = time()  # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time()  # Get end time

    # Calculate the total prediction time
    results['pred_time'] = end - start

    # Compute R^2 score on the first 300 training samples
    results['r2_train'] = r2_score(y_train[:300], predictions_train)

    # Compute R^2 score on test set
    results['r2_test'] = r2_score(y_test, predictions_test)

    # Compute MAE on the first 300 training samples
    results['mae_train'] = mean_absolute_error(y_train[:300], predictions_train)

    # Compute MAE on the test set
    results['mae_test'] = mean_absolute_error(y_test, predictions_test)

    # Success
    print(f"{learner.__class__.__name__} trained on {sample_size} samples.")

    # Return the results
    return results

# Initialize the regressors
reg_A = RandomForestRegressor(random_state=42)
reg_B = GradientBoostingRegressor(random_state=42)
reg_C = XGBRegressor(random_state=42)
reg_D = LinearRegression()

# Calculate the number of samples for 1%, 10%, and 100% of the training data
# Ensure OHE_X_train is properly defined as a DataFrame
samples_1   = int(round(len(OHE_X_train) / 100))
samples_10  = int(round(len(OHE_X_train) / 10))
samples_100 = len(OHE_X_train)

# Collect results on the learners
results = {}

from sklearn.model_selection import train_test_split

# Splitting data into train and test sets
OHE_X_train, OHE_X_test, y_train, y_test = train_test_split(OHE_X_train, y_train, test_size=0.2, random_state=42)

for reg in [reg_A, reg_B, reg_C, reg_D]:
    reg_name = reg.__class__.__name__
    results[reg_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[reg_name][i] = \
        train_predict(reg, samples, OHE_X_train, y_train, OHE_X_test, y_test)



# Printing out the values
for model_name, model_results in results.items():
    print(f"\n{model_name}")  # Print model name with a newline for better readability

    # Convert dictionary to DataFrame and rename columns
    df_results = pd.DataFrame(model_results).rename(columns={0: '1%', 1: '10%', 2: '100%'})

    # Print DataFrame
    print(df_results.to_string())  # to_string() ensures it prints in a readable format


## Fine-Tuning the Random Forest Regressor ##
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import validation_curve, GridSearchCV


def plot_validation_curve(regressor, X, y, param_name, param_range):
    train_scores, validation_scores = validation_curve(
        estimator=regressor,
        X=X,
        y=y.to_numpy(),  # Convert to NumPy array if needed
        param_name=param_name,
        param_range=param_range,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1  # Use all available CPU cores
    )

    # Convert negative MAE scores back to positive values
    train_scores_mean = -np.mean(train_scores, axis=1)
    validation_scores_mean = -np.mean(validation_scores, axis=1)

    # Plot validation curve
    plt.figure(figsize=(8, 5))
    plt.plot(param_range, train_scores_mean, marker='o', linestyle='-', label='Training Error')
    plt.plot(param_range, validation_scores_mean, marker='s', linestyle='--', label='Validation Error')

    plt.ylabel('Mean Absolute Error (MAE)', fontsize=14)
    plt.xlabel(f'{param_name}', fontsize=14)
    plt.title(f'Validation Curve: {param_name}', fontsize=16)
    plt.legend()
    plt.grid()
    plt.show()


from sklearn.metrics import mean_squared_error, r2_score

def get_test_scores(model_name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MSE: {mse:.4f}, R2 Score: {r2:.4f}")
    return pd.DataFrame({'Model': [model_name], 'MSE': [mse], 'R2': [r2]})


      # Initialize model
random_forest = RandomForestRegressor(random_state=42)

# Define parameter range for n_estimators
param_range = [10, 25, 50, 100]

# Call function with fixed y_train (ensure it's 1D if needed)
plot_validation_curve(random_forest, OHE_X_train, y_train, 'n_estimators', param_range)


      # Fine-tuning Gradient Boosting Regressor ##
gbr = GradientBoostingRegressor(random_state=42)

plot_validation_curve(gbr, OHE_X_train, y_train, 'n_estimators', [25,50,100,200,300,400])

plot_validation_curve(gbr, OHE_X_train, y_train, 'max_depth', [1,2,3,4,5,6,7])
plot_validation_curve(gbr, OHE_X_train, y_train, 'min_samples_split', [0.1,0.2,0.4,0.6,0.8,1.])

# Grid Search for Gradient Boosting Regressor
gbr_grid = GridSearchCV(estimator=gbr, param_grid={
    'n_estimators': [450, 500, 550, 600],
    'min_samples_split': [2, 5, 10]  # Use integer values or a larger fraction (e.g., 0.1, 0.2)
}, cv=5)


gbr_grid.fit(OHE_X_train, y_train)

# Predictions
train_preds_gbr = gbr_grid.predict(OHE_X_train)
test_preds_gbr  = gbr_grid.predict(OHE_X_test)

# Generate test scores
def get_test_scores(param, train_preds_gbr, y_train):
    passgbr_test_GSCV_results = get_test_scores('GradientBoosting + GridSearchCV (test)', test_preds_gbr, y_test)


gbr_train_GSCV_results = get_test_scores('GradientBoosting + GridSearchCV (train)', train_preds_gbr, y_train)


gbr_test_GSCV_results  = get_test_scores('GradientBoosting + GridSearchCV (test)', test_preds_gbr, y_test)

gbr_results = pd.concat([gbr_train_GSCV_results, gbr_test_GSCV_results], axis=0)

# Print Results
print(gbr_grid.best_params_)
print(gbr_grid.best_estimator_)
print(gbr_results)

         # Feature Importance
# Random Forest Model
rf = RandomForestRegressor(n_estimators=100)
rf.fit(OHE_X_train, y_train)

sorted_idx = (-rf.feature_importances_).argsort()

list_of_tuples = list(zip(OHE_X_train.columns[sorted_idx], rf.feature_importances_[sorted_idx]))

feat_importance = pd.DataFrame(list_of_tuples, columns=['feature', 'feature importance'])

fig = plt.figure(figsize=(12,6))

fig = sns.barplot(data=feat_importance[feat_importance['feature importance'] > 0.01], x='feature', y='feature importance')
plt.title('Feature Importance > 0.01', fontsize=25)
plt.xticks(fontsize=8, rotation=45)

plt.tight_layout()

