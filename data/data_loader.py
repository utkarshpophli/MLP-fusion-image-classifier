import tensorflow as tf
from config import image_size, batch_size

def load_data(data_dir, image_size, batch_size):
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        labels='inferred',
        image_size=(image_size, image_size),
        validation_split=0.2,
        subset="both",
        seed=1337,
        batch_size=batch_size,
    )
    
    return train_ds, val_ds