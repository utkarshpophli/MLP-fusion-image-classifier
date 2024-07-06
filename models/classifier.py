import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from layers.patches import Patches
from layers.position_embedding import PositionEmbedding
from config import input_shape, patch_size, embedding_dim, num_patches, dropout_rate, num_classes, image_size

def build_classifier(blocks, positional_encoding=False):
    inputs = layers.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    patches = Patches(patch_size)(augmented)
    x = layers.Dense(units=embedding_dim)(patches)
    if positional_encoding:
        x = x + PositionEmbedding(sequence_length=num_patches)(x)
    x = blocks(x)
    representation = layers.GlobalAveragePooling1D()(x)
    representation = layers.Dropout(rate=dropout_rate)(representation)
    logits = layers.Dense(num_classes)(representation)
    return keras.Model(inputs=inputs, outputs=logits)

data_augmentation = tf.keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)