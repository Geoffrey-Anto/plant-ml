import tensorflow as tf
import config

def create_model():
    model = tf.keras.applications.resnet50.ResNet50(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=(256,256,3),
        pooling=None,
        classes=40,
        classifier_activation="softmax"
    )
    model.summary()
    model.compile(tf.keras.optimizers.Adam(config.lr), tf.losses.CategoricalCrossentropy(), metrics=config.met)
    return model