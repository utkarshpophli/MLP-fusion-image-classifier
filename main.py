from config import *
from data.data_loader import load_data
from models.classifier import build_classifier
from models.mlp_mixer import mlpmixer_blocks
from models.fnet import fnet_blocks
from models.gmlp import gmlp_blocks
from train import run_experiment
from utils.visualization import plot_history
from utils.evaluation import evaluate_models

def main():
    # Load data
    train_ds, val_ds = load_data(data_dir, image_size, batch_size)

    # Train and evaluate MLP-Mixer
    mlpmixer_classifier = build_classifier(mlpmixer_blocks)
    history_mlp = run_experiment(mlpmixer_classifier, train_ds, val_ds)
    plot_history("acc", history_mlp)
    plot_history("top5-acc", history_mlp)

    # Train and evaluate FNet
    fnet_classifier = build_classifier(fnet_blocks, positional_encoding=True)
    history_fnet = run_experiment(fnet_classifier, train_ds, val_ds)
    plot_history("acc", history_fnet)
    plot_history("top5-acc", history_fnet)

    # Train and evaluate gMLP
    gmlp_classifier = build_classifier(gmlp_blocks)
    history_gmlp = run_experiment(gmlp_classifier, train_ds, val_ds)
    plot_history("acc", history_gmlp)
    plot_history("top5-acc", history_gmlp)

    # Evaluate all models
    models = [mlpmixer_classifier, fnet_classifier, gmlp_classifier]
    evaluate_models(models, val_ds, val_ds.class_names)

if __name__ == "__main__":
    main()