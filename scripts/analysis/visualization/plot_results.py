import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scripts.config import PLOTS_DIR, Config

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


def convert_test_result_to_image(test_file='test_result.txt'):
    """Convert test_result.txt data to images"""
    print(f"Converting {test_file} to images...")

    Config.ensure_directories_exist()
    
    try:
        # Read test result file
        with open(test_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Parse file content
        models = []
        metrics = []
        predictions = {}
        
        current_model = None
        current_predictions = []
        
        for line in lines:
            line = line.strip()
            if '模型测试结果:' in line or 'Model test results:' in line:
                # Save previous model's prediction data
                if current_model and current_predictions:
                    predictions[current_model] = current_predictions
                # Start new model
                current_model = line.split(' ')[0]
                models.append(current_model)
                current_predictions = []
            elif line.startswith('MAE:'):
                mae = float(line.split(':')[1].strip())
            elif line.startswith('RMSE:'):
                rmse = float(line.split(':')[1].strip())
            elif line.startswith('R²:') or line.startswith('R^2:') or line.startswith('R2:'):
                r2 = float(line.split(':')[1].strip())
                metrics.append({'model': current_model, 'mae': mae, 'rmse': rmse, 'r2': r2})
            elif line == '预测值,实际值' or line == 'Predicted,Actual':
                # Start reading prediction data
                continue
            elif ',' in line:
                # Read prediction-actual pairs
                pred, actual = map(float, line.split(','))
                current_predictions.append((pred, actual))
        
        # Save last model's prediction data
        if current_model and current_predictions:
            predictions[current_model] = current_predictions
        
        # Skip metrics comparison plot
        # plt.figure(figsize=(12, 8))
        # 
        # # Plot MAE comparison
        # plt.subplot(3, 1, 1)
        # model_names = [m['model'] for m in metrics]
        # mae_values = [m['mae'] for m in metrics]
        # plt.bar(model_names, mae_values)
        # plt.title('MAE Comparison')
        # plt.ylabel('MAE')
        # # Add value labels
        # for i, v in enumerate(mae_values):
        #     plt.text(i, v + 0.1, f'{v:.4f}', ha='center')
        # 
        # # Plot RMSE comparison
        # plt.subplot(3, 1, 2)
        # rmse_values = [m['rmse'] for m in metrics]
        # plt.bar(model_names, rmse_values)
        # plt.title('RMSE Comparison')
        # plt.ylabel('RMSE')
        # # Add value labels
        # for i, v in enumerate(rmse_values):
        #     plt.text(i, v + 0.1, f'{v:.4f}', ha='center')
        # 
        # # Plot R² comparison
        # plt.subplot(3, 1, 3)
        # r2_values = [m['r2'] for m in metrics]
        # plt.bar(model_names, r2_values)
        # plt.title('R² Comparison')
        # plt.ylabel('R²')
        # # Add value labels
        # for i, v in enumerate(r2_values):
        #     plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
        # 
        # plt.tight_layout()
        # plt.savefig(os.path.join(PLOTS_DIR, 'metrics_comparison.png'))
        
        # Plot prediction vs actual scatter for each model
        for model, preds in predictions.items():
            y_pred = [p[0] for p in preds]
            y_test = [p[1] for p in preds]
            
            # Get model metrics
            model_metric = next((m for m in metrics if m['model'] == model), None)
            
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred)
            min_val = min(min(y_test), min(y_pred))
            max_val = max(max(y_test), max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            plt.xlabel('Actual Age')
            plt.ylabel('Predicted Age')
            plt.title(f'{model} Model Prediction vs Actual')
            
            # Add model metrics
            if model_metric:
                metrics_text = f"MAE: {model_metric['mae']:.4f}\nRMSE: {model_metric['rmse']:.4f}\nR²: {model_metric['r2']:.4f}"
                plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
            
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f'{model.lower()}_prediction_vs_actual.png'))
        
        print("Test result images generated successfully")
        
    except Exception as e:
        print(f"Failed to convert test results to images: {e}")


if __name__ == "__main__":
    # Run test result conversion to images
    convert_test_result_to_image()
