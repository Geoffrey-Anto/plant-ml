import tensorflow as tf
import os

train_data_path = "data"
test_data_path = "data"
val_data_path = "data"


met  = tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")

lr = 0.01
batch_size = 64
epochs = 1

save_model_path = "models/model.h5"

file_types_for_image = [("Image File",'.jpg')]

classes = os.listdir(train_data_path)