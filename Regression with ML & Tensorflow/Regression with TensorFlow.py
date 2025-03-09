# Importing Libraries
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Load dataset
data = pd.read_csv(r'C:\Users\nwoko\OneDrive\Desktop\Data Science and AI Program\Artificial Intelligence Course\AI Assignment\diamonds.csv')

# Perform One-Hot Encoding and ensure only 23 features are used
OHE_X_train = pd.get_dummies(data.drop(columns=['price']), drop_first=True)

# Select only the first 23 columns
OHE_X_train = OHE_X_train.iloc[:, :23]

# Define target variable
y_train = data['price']

# Split into training and validation sets
X_train2, X_val, y_train2, y_val = train_test_split(OHE_X_train, y_train, test_size=0.2, random_state=42)

# Print shape to confirm
print(X_train2.shape, X_val.shape, y_train2.shape, y_val.shape)  # Should be (*, 23)

# Define an early stopping callback
early_stopping = EarlyStopping(
    min_delta=0.001,
    patience=20,
    restore_best_weights=True
)

# Set the random seed
tf.random.set_seed(42)

# Create a TF model with input shape 23
model1 = tf.keras.Sequential([
    tf.keras.layers.Dense(units=16, input_shape=[23]),  # Fixed input shape
    tf.keras.layers.Dense(units=1)
])

# Compile the model
model1.compile(optimizer='adam',
               loss='mse',
               metrics=['mse', tf.keras.metrics.RootMeanSquaredError()])

# Train the model
r1 = model1.fit(
    X_train2, y_train2,
    validation_data=(X_val, y_val),
    batch_size=16,
    epochs=500,
    callbacks=[early_stopping],
    verbose=0
)

# Evaluate the model
print("Train MSE:", model1.evaluate(X_train2, y_train2))
print("Test MSE:", model1.evaluate(X_val, y_val))  # Fixed test dataset reference

# Model summary
model1.summary()

# Plot the train and validation loss curves
plt.plot(r1.history['loss'], label='train loss')
plt.plot(r1.history['val_loss'], label='validation loss')
plt.xlabel('epochs', fontsize=12)
plt.ylabel('loss', fontsize=12)
plt.legend()
plt.title('Train and Validation Losses of Model #1', fontsize=20)
plt.tight_layout()

# Second Model
# Create a TF model
model2 = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1)
])

# Compile the model
model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
               loss='mse',
               metrics=['mse', 'RootMeanSquaredError'])

# Fit the model
r2 = model2.fit(
    X_train2, y_train2,
    validation_data=(X_val, y_val),
    batch_size=16,
    epochs=500,
    callbacks=[early_stopping],
    verbose=0,
)

# Evaluate the model
print("Train MSE:", model2.evaluate(X_train2, y_train2))
print("Test MSE:", model2.evaluate(X_val, y_val))  # Use validation data for testing

# Model summary
model2.summary()

# Plot the train and validation loss curves
plt.plot(r2.history['loss'], label='train loss')
plt.plot(r2.history['val_loss'], label='validation loss')
plt.xlabel('epochs', fontsize=12)
plt.ylabel('loss', fontsize=12)
plt.legend()
plt.title('Train and Validation Losses of Model #2', fontsize=20)
plt.tight_layout()

# Third Model
# Define the optimizer
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Define the learning rate reduction callback
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',
                                            patience=5,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.0000001)

# Create a TF model
model3 = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1)
])

# Compile the model
model3.compile(optimizer=optimizer,
               loss='mse',
               metrics=['mse', 'RootMeanSquaredError'])

# Fit the model
r3 = model3.fit(
    X_train2, y_train2,
    validation_data=(X_val, y_val),
    batch_size=16,
    epochs=50,
    callbacks=[learning_rate_reduction],
    verbose=1,
)

# Evaluate the model
print("Train MSE:", model3.evaluate(X_train2, y_train2))
print("Test MSE:", model3.evaluate(X_val, y_val))  # Use validation data for testing

# Model summary
model3.summary()

# Plot the train and validation loss curves
plt.plot(r3.history['loss'], label='train loss')
plt.plot(r3.history['val_loss'], label='validation loss')
plt.xlabel('epochs', fontsize=12)
plt.ylabel('loss', fontsize=12)
plt.legend()
plt.title('Train and Validation Losses of Model #3', fontsize=20)
plt.tight_layout()

