import tensorflow as tf
import os
import config
import dataset
import model
import utils

print(tf.__version__)

# VISUALIZATION
utils.show_image([
    "./data/data/Aloevera/351.jpg",
    "./data/data/Ekka/2551.jpg",
    "./data/data/Doddapatre/5353.jpg"
])

train_dataset = dataset.train

model = model.create_model()

# model.fit(train_dataset, batch_size=config.batch_size, epochs=config.epochs)

model.save(config.save_model_path)

