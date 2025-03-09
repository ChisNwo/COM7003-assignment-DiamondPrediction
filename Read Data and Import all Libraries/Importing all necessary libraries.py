# Importing all necessary libraries and configurations
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Scikit-learn libraries for modeling
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics #(vital for notebook)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import validation_curve

from xgboost import XGBRegressor
from scipy.stats import kurtosis, skew

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


from time import time

# Suppress warnings for a clean output
from warnings import simplefilter
simplefilter("ignore")

# To supress the informational message from TensorFlow regarding oneDNN optimisations
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Print the current versions of the important packages
print('Pandas version:', pd.__version__)
print('NumPy version:', np.__version__)
print('Matplotlib version:', matplotlib.__version__)
print('Seaborn version:', sns.__version__)