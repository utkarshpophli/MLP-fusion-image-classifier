from tensorflow import keras
from tensorflow.keras import layers
from config import num_patches, embedding_dim, dropout_rate, num_blocks

class MLPMixerLayer(layers.Layer):
    def __init__(self, num_patches, hidden_units, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mlp1 = keras.Sequential(
            [
                layers.Dense(units=num_patches, activation="gelu"),
                layers.Dense(units=num_patches),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.mlp2 = keras.Sequential(
            [
                layers.Dense(units=num_patches, activation="gelu"),
                layers.Dense(units=hidden_units),
                layers.Dropout(rate=dropout_rate),
            ]
        )
        self.normalize = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x = self.normalize(inputs)
        x_channels = keras.ops.transpose(x, axes=(0, 2, 1))
        mlp1_outputs = self.mlp1(x_channels)
        mlp1_outputs = keras.ops.transpose(mlp1_outputs, axes=(0, 2, 1))
        x = mlp1_outputs + inputs
        x_patches = self.normalize(x)
        mlp2_outputs = self.mlp2(x_patches)
        x = x + mlp2_outputs
        return x

mlpmixer_blocks = keras.Sequential(
    [MLPMixerLayer(num_patches, embedding_dim, dropout_rate) for _ in range(num_blocks)]
)