import matplotlib.pyplot as plt

def plot_history(item, model_history):
    plt.figure()
    plt.plot(model_history.history[item], label=item)
    plt.plot(model_history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()