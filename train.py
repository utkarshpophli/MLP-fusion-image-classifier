import tensorflow as tf
from tensorflow import keras
from config import num_epochs, weight_decay, batch_size, mlpmixer_lr, fnet_lr, gmlp_lr
from utils.visualization import plot_history

def run_experiment(model, train_ds, val_ds, learning_rate):
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=learning_rate,
            decay_steps=num_epochs * len(train_ds),
        ),
        weight_decay=weight_decay,
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = f'./checkpoints/{model.name}'
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=num_epochs,
        callbacks=[checkpoint_callback, early_stopping],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(val_ds)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history

def train_all_models(models, train_ds, val_ds):
    histories = []
    for model, lr in zip(models, [mlpmixer_lr, fnet_lr, gmlp_lr]):
        print(f"\nTraining {model.name}...")
        history = run_experiment(model, train_ds, val_ds, lr)
        histories.append(history)
        
        # Plot training history
        plot_history("accuracy", history)
        plot_history("top-5-accuracy", history)
    
    return histories

if __name__ == "__main__":
    from data.data_loader import load_data
    from models.classifier import build_classifier
    from models.mlp_mixer import mlpmixer_blocks
    from models.fnet import fnet_blocks
    from models.gmlp import gmlp_blocks

    # Load data
    train_ds, val_ds = load_data()

    # Build models
    mlpmixer_classifier = build_classifier(mlpmixer_blocks, name="MLP-Mixer")
    fnet_classifier = build_classifier(fnet_blocks, positional_encoding=True, name="FNet")
    gmlp_classifier = build_classifier(gmlp_blocks, name="gMLP")

    models = [mlpmixer_classifier, fnet_classifier, gmlp_classifier]

    # Train all models
    histories = train_all_models(models, train_ds, val_ds)

    print("Training completed for all models.")