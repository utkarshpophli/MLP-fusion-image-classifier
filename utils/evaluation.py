import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def evaluate_models(models, test_ds, class_names, num_images=16, image_size=(224, 224)):
    for i, model in enumerate(models):
        print(f"\nModel {i+1} Evaluation:")
        
        # Collect true labels and predictions
        y_test = np.concatenate([y for x, y in test_ds], axis=0)
        y_test = np.argmax(y_test, axis=1)
        predictions = model.predict(test_ds)
        y_pred = np.argmax(predictions, axis=1)
        
        y_test = np.ravel(y_test)
        y_pred = np.ravel(y_pred)
        
        print(f"Length of y_test: {len(y_test)}")
        print(f"Length of y_pred: {len(y_pred)}")
        
        # Create DataFrame for actual vs predicted
        df2 = pd.DataFrame({'Actual': y_test, 'Prediction': y_pred})
        print("\nSample of Actual vs Predicted:")
        print(df2.head())
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - Model {i+1}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        # Visualize sample predictions
        batch = next(iter(test_ds))
        images = batch[0]
        
        plt.figure(figsize=(20, 20))
        for n in range(num_images):
            plt.subplot(4, 4, n+1)
            plt.imshow(images[n].numpy().astype("uint8"))
            plt.axis('off')
            true_label = class_names[y_test[n]]
            pred_label = class_names[y_pred[n]]
            color = 'green' if true_label == pred_label else 'red'
            plt.title(f'True: {true_label}\nPred: {pred_label}', color=color)
        
        plt.suptitle(f'Sample Predictions - Model {i+1}', fontsize=24)
        plt.tight_layout()
        plt.show()