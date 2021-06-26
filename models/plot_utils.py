import tensorflow as tf


def plot_model(model_obj: tf.keras.Model, filename: str):
    return tf.keras.utils.plot_model(
        model_obj,
        show_shapes=True,
        show_dtype=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        dpi=70,
    )
