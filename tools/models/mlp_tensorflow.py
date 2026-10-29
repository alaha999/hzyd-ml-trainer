"""Simple TensorFlow MLP. Edit layers here to change the architecture."""
import numpy as np
import tensorflow as tf


def build_model(n_features, learning_rate=0.001):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(n_features,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


def predict(model, x, batch_size=16384):
    if len(x) == 0:
        return np.empty(0, dtype=np.float32)
    return model.predict(np.asarray(x, dtype=np.float32), batch_size=batch_size, verbose=0).ravel()
