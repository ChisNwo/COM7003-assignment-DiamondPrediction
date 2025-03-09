import tensorflow as tf
print("TensorFlow Version:", tf.__version__)

from tensorflow import keras
print("Keras is working!")

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
print("Callbacks imported successfully!")
