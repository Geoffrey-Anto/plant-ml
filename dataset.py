from tensorflow.keras.preprocessing import image
import config

data_gen=image.ImageDataGenerator(horizontal_flip=True,
                                         rescale=1./255,
                                         rotation_range=20,
                                         zoom_range=0.2,
                                         shear_range=0.2)

train=data_gen.flow_from_directory(config.train_data_path, batch_size=64, target_size=(256, 256))