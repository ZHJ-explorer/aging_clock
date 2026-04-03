import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

PLOTS_DIR = 'plots'


def plot_dl_prediction_vs_actual(predictions: np.ndarray, targets: np.ndarray,
                                   model_name: str = 'Deep Learning',
                                   metrics: dict = None, save_path: str = None):
    plt.figure(figsize=(8, 6))
    plt.scatter(targets, predictions, alpha=0.5, edgecolors='none')
    min_val = min(min(targets), min(predictions))
    max_val = max(max(targets), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    plt.xlabel('Actual Age', fontsize=12)
    plt.ylabel('Predicted Age', fontsize=12)
    plt.title(f'{model_name}: Prediction vs Actual', fontsize=14)

    if metrics:
        metrics_text = f"MAE: {metrics.get('mae', 0):.4f}\nRMSE: {metrics.get('rmse', 0):.4f}\nR²: {metrics.get('r2', 0):.4f}"
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(PLOTS_DIR, f'{model_name.lower()}_prediction_vs_actual.png'), dpi=150)
        plt.close()


def plot_training_history(history: dict, model_name: str = 'Deep Learning', save_path: str = None):
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title(f'{model_name}: Loss Curves', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_mae'], 'b-', label='Train MAE', linewidth=2)
    axes[1].plot(epochs, history['val_mae'], 'r-', label='Val MAE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title(f'{model_name}: MAE Curves', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(PLOTS_DIR, f'{model_name.lower()}_training_history.png'), dpi=150)
        plt.close()


def plot_residuals(predictions: np.ndarray, targets: np.ndarray,
                    model_name: str = 'Deep Learning', save_path: str = None):
    residuals = targets - predictions

    plt.figure(figsize=(8, 6))
    plt.scatter(predictions, residuals, alpha=0.5, edgecolors='none')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Age', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.title(f'{model_name}: Residual Plot', fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(PLOTS_DIR, f'{model_name.lower()}_residuals.png'), dpi=150)
        plt.close()


def plot_error_distribution(predictions: np.ndarray, targets: np.ndarray,
                              model_name: str = 'Deep Learning', save_path: str = None):
    errors = np.abs(predictions - targets)

    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(x=np.mean(errors), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
    plt.axvline(x=np.median(errors), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.2f}')
    plt.xlabel('Absolute Error', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'{model_name}: Error Distribution', fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(PLOTS_DIR, f'{model_name.lower()}_error_distribution.png'), dpi=150)
        plt.close()


def generate_all_plots(model_name: str, predictions: np.ndarray, targets: np.ndarray,
                        history: dict = None, metrics: dict = None, save_dir: str = None):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plot_dl_prediction_vs_actual(predictions, targets, model_name, metrics,
                                  os.path.join(save_dir, f'{model_name.lower()}_prediction_vs_actual.png') if save_dir else None)

    if history:
        plot_training_history(history, model_name,
                               os.path.join(save_dir, f'{model_name.lower()}_training_history.png') if save_dir else None)

    plot_residuals(predictions, targets, model_name,
                    os.path.join(save_dir, f'{model_name.lower()}_residuals.png') if save_dir else None)

    plot_error_distribution(predictions, targets, model_name,
                            os.path.join(save_dir, f'{model_name.lower()}_error_distribution.png') if save_dir else None)


if __name__ == "__main__":
    import pandas as pd
    from scripts.deep_learning.evaluation.metrics import compute_regression_metrics

    test_dir = 'models/deep_learning'
    results_file = os.path.join(test_dir, 'test_predictions.csv')

    if not os.path.exists(results_file):
        print(f"Test results file not found: {results_file}")
        print("Please run train_dnn.py first to generate predictions.")
        exit(1)

    df = pd.read_csv(results_file, index_col=0)
    predictions = df['prediction'].values
    targets = df['actual'].values

    model_name = 'DNN'
    metrics = compute_regression_metrics(targets, predictions)

    print(f"Model: {model_name}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")

    generate_all_plots(model_name, predictions, targets, metrics=metrics)
    print(f"Plots saved to {PLOTS_DIR}/")