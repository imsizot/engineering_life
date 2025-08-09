# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 19:56:10 2025

@author: YAHIAOUI

Script for optimizing machine learning models using Particle Swarm Optimization (PSO) and generating visualizations.
Supports multiple models via selected_model as a string, list (e.g., ["ANN", "KNN"]), or None.
Saves model-specific results in subfolders (e.g., Results_PSO/ANN).
Saves global results (comparative figures, Excel, PDF) in Results_PSO.
Includes progress bars (tqdm) for model processing and PSO per model.
Values on histograms are removed, and table values are formatted to 6 decimal places.
Cross-validation is used only during hyperparameter optimization (configurable cv folds on 80% training data).
Final evaluation uses 80/20 train/test split.
"""

import os
import platform
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import xgboost as xgb
import lightgbm as lgb
try:
    from pyswarm import pso
except ImportError as e:
    print(f"Error: pyswarm is not installed. Install it with 'pip install pyswarm'. Details: {str(e)}")
    exit(1)
try:
    from tqdm import tqdm
except ImportError as e:
    print(f"Error: tqdm is not installed. Install it with 'pip install tqdm'. Details: {str(e)}")
    exit(1)
import json
import warnings
import sys
import logging
import argparse
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# Clear console at the start
if platform.system() == "Windows":
    os.system('cls')
else:
    os.system('clear')

# Configuration
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("shap").setLevel(logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.ERROR)
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 14  # Set default font size
plt.rcParams['axes.spines.top'] = True    # Ensure top spine is visible
plt.rcParams['axes.spines.right'] = True  # Ensure right spine is visible
plt.rcParams['axes.spines.bottom'] = True # Ensure bottom spine (x-axis) is visible
plt.rcParams['axes.spines.left'] = True   # Ensure left spine (y-axis) is visible
plt.rcParams['axes.linewidth'] = 1.0      # Set spine linewidth
plt.rcParams['axes.edgecolor'] = 'black'  # Set spine color to black

# Utility Functions
def create_results_folder(base_folder, method, model_name=None):
    """Creates a folder to store results."""
    logger.info("Creating results folder")
    results_folder = os.path.join(base_folder, f'Results_{method}')
    if model_name:
        results_folder = os.path.join(results_folder, model_name.replace(" ", "_"))
    try:
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
            logger.info(f"Folder created at: {results_folder}")
        else:
            logger.info(f"Folder already exists at: {results_folder}")
        return results_folder
    except Exception as e:
        logger.error(f"Error creating folder: {str(e)}")
        raise

def load_and_prepare_data(data_path):
    """Loads and prepares data from an Excel file."""
    logger.info("Loading and preparing data")
    try:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File {data_path} does not exist")
        df = pd.read_excel(data_path)
        logger.info(f"Data loaded with {len(df)} rows")
        df = df.dropna(subset=[df.columns[-1]])
        df = df[df[df.columns[-1]] != 0]
        feature_names = df.columns[:-1].tolist()
        target_name = df.columns[-1]
        X = df[feature_names]
        y = df[target_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logger.info("Data prepared successfully")
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, target_name
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def get_initial_param_bounds(model_name, config_path='model_config.json'):
    """Returns initial hyperparameter bounds from a config file."""
    logger.info(f"Retrieving bounds for {model_name} from {config_path}")
    try:
        with open(config_path, 'r') as f:
            bounds_dict = json.load(f)
        result = bounds_dict.get(model_name, {'numeric': {}, 'categorical': {}})
        # Convert lists to tuples for numeric bounds
        if 'numeric' in result:
            for param, bounds in result['numeric'].items():
                result['numeric'][param] = tuple(bounds)
        logger.info(f"Bounds for {model_name}: {result}")
        return result
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {config_path}")
        return {'numeric': {}, 'categorical': {}}
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {config_path}")
        return {'numeric': {}, 'categorical': {}}

def optimize_hyperparameters(model_class, param_bounds, X_train, y_train, pso_maxiter, pso_swarmsize, cv):
    """Optimizes hyperparameters using PSO with a progress bar."""
    logger.info(f"Optimizing for {model_class.__name__} with {cv}-fold CV")
    numeric_params = param_bounds['numeric']
    categorical_params = param_bounds['categorical']
    param_names = list(numeric_params.keys())
    bounds = list(numeric_params.values())

    if not param_names:
        logger.info(f"No numeric parameters to optimize for {model_class.__name__}.")
        return categorical_params.copy()

    lb = np.array([float(b[0]) for b in bounds])
    ub = np.array([float(b[1]) for b in bounds])
    logger.info(f"Parameters: {param_names}, Lower bounds: {lb}, Upper bounds: {ub}")

    def objective(x):
        param_dict = categorical_params.copy()
        try:
            for j, name in enumerate(param_names):
                value = max(lb[j], min(ub[j], x[j]))
                if name in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf',
                            'n_neighbors', 'max_iter', 'min_child_weight']:
                    param_dict[name] = int(round(value))
                elif name in ['C', 'learning_rate', 'learning_rate_init', 'subsample', 'colsample_bytree', 'alpha']:
                    param_dict[name] = float(value)
                elif name == 'hidden_layer_sizes':
                    param_dict[name] = (int(round(value)),)
            if model_class == GaussianProcessRegressor and 'kernel' not in param_dict:
                param_dict['kernel'] = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
            if model_class == lgb.LGBMRegressor:
                param_dict['verbosity'] = -1
                param_dict.setdefault('min_child_samples', 1)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model = model_class(**param_dict)
                score = -np.mean(cross_val_score(model, X_train, y_train, cv=cv,
                                                scoring='neg_mean_squared_error', n_jobs=-1))
            return score
        except Exception as e:
            logger.error(f"Error in objective for {model_class.__name__}: {str(e)}")
            return float('inf')

    logger.info("Starting PSO optimization. This may take some time...")
    try:
        best_x, best_score = pso(objective, lb, ub, swarmsize=pso_swarmsize, maxiter=pso_maxiter, debug=True)
        logger.info(f"Best score for {model_class.__name__}: {best_score}")
    except Exception as e:
        logger.error(f"Error in PSO for {model_class.__name__}: {str(e)}")
        return categorical_params.copy()

    best_params = categorical_params.copy()
    for i, name in enumerate(param_names):
        value = max(lb[i], min(ub[i], best_x[i]))
        if name in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf',
                    'n_neighbors', 'max_iter', 'min_child_weight']:
            best_params[name] = int(round(value))
        elif name in ['C', 'learning_rate', 'learning_rate_init', 'subsample', 'colsample_bytree', 'alpha']:
            best_params[name] = float(value)
        elif name == 'hidden_layer_sizes':
            best_params[name] = (int(round(value)),)

    if model_class == GaussianProcessRegressor and 'kernel' not in best_params:
        best_params['kernel'] = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    if model_class == lgb.LGBMRegressor:
        best_params['verbosity'] = -1
        best_params.setdefault('min_child_samples', 1)

    logger.info(f"Optimized parameters for {model_class.__name__}: {best_params}")
    return best_params

def evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test):
    """Evaluates a model."""
    logger.info(f"Evaluating {model.__class__.__name__}")
    try:
        if isinstance(model, lgb.LGBMRegressor):
            model.set_params(verbosity=-1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model.fit(X_train_scaled, y_train)
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        results = {
            'mse_train': mean_squared_error(y_train, y_train_pred),
            'r2_train': r2_score(y_train, y_train_pred),
            'mae_train': mean_absolute_error(y_train, y_train_pred),
            'mse_test': mean_squared_error(y_test, y_test_pred),
            'r2_test': r2_score(y_test, y_test_pred),
            'mae_test': mean_absolute_error(y_test, y_test_pred)
        }
        logger.info(f"Results: {results}")
        return results, y_train_pred, y_test_pred
    except Exception as e:
        logger.error(f"Error evaluating {model.__class__.__name__}: {str(e)}")
        return None, None, None

# Visualization Functions
def load_results(model_folder, method, model_name):
    """Loads results for a model from its subfolder."""
    logger.info(f"Loading for {method}, model {model_name} from {model_folder}")
    try:
        opt_file = os.path.join(model_folder, 'optimization_results.json')
        pred_file = os.path.join(model_folder, 'predictions.json')
        data_file = os.path.join(model_folder, 'data.npz')
        for f in [opt_file, pred_file, data_file]:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Missing file: {f}")

        with open(opt_file, 'r') as f:
            data = json.load(f)
            if method.lower() not in data:
                raise KeyError(f"Key '{method.lower()}' not found in {opt_file}")
            optimized_params = data[method.lower()]

        with open(pred_file, 'r') as f:
            predictions = json.load(f)

        data = np.load(data_file, allow_pickle=True)
        X_train_scaled = data['X_train_scaled']
        X_test_scaled = data['X_test_scaled']
        y_train = data['y_train']
        y_test = data['y_test']
        feature_names = data['feature_names'].tolist()
        target_name = data['target_name'].item()

        logger.info(f"Results loaded for {model_name}")
        return (X_train_scaled, X_test_scaled, y_train, y_test, feature_names, target_name,
                optimized_params, predictions['train'], predictions['test'])
    except Exception as e:
        logger.error(f"Error loading for {model_name}: {str(e)}")
        raise

def load_global_results(main_results_folder, method):
    """Loads metrics from all subfolders for global results."""
    logger.info(f"Loading global results from {main_results_folder}")
    results_dict = {}
    optimized_params = {}
    subdirs = [d for d in os.listdir(main_results_folder)
               if os.path.isdir(os.path.join(main_results_folder, d)) and d != 'Global_Results']

    for model_name_clean in subdirs:
        model_folder = os.path.join(main_results_folder, model_name_clean)
        model_name = model_name_clean.replace("_", " ").strip()
        try:
            X_train_scaled, X_test_scaled, y_train, y_test, feature_names, target_name, \
            opt_params, y_train_preds, y_test_preds = load_results(model_folder, method, model_name)

            y_train_pred = np.array(y_train_preds[model_name])
            y_test_pred = np.array(y_test_preds[model_name])

            results_dict[model_name] = {
                'mse_train': mean_squared_error(y_train, y_train_pred),
                'r2_train': r2_score(y_train, y_train_pred),
                'mae_train': mean_absolute_error(y_train, y_train_pred),
                'mse_test': mean_squared_error(y_test, y_test_pred),
                'r2_test': r2_score(y_test, y_test_pred),
                'mae_test': mean_absolute_error(y_test, y_test_pred)
            }
            optimized_params[model_name] = opt_params.get(model_name, {})
            logger.info(f"Loaded results for {model_name}")
        except Exception as e:
            logger.error(f"Error loading {model_name}: {str(e)}")
            continue

    return results_dict, optimized_params

def plot_actual_vs_predicted_separate(y_true, y_pred, results_folder, model_name, dataset_type):
    """Generates an Actual vs Predicted plot with visible x and y axes."""
    logger.info(f"Creating Actual vs Predicted ({dataset_type}) for {model_name}")
    try:
        fig = plt.figure(figsize=(8, 8))
        fig.patch.set_facecolor('white')
        ax = plt.gca()
        color = 'blue' if dataset_type == 'train' else 'green'
        plt.scatter(y_true, y_pred, c=color, edgecolor='black', alpha=0.5, s=60, linewidths=1.5,
                    label=f'{dataset_type.capitalize()}')
        min_val, max_val = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'MSE: {mse:.3f}\nR²: {r2:.3f}', transform=ax.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=14)
        plt.title(f'Actual vs Predicted - {model_name} ({dataset_type.capitalize()})', fontsize=14)
        plt.xlabel('Actual Values', fontsize=14)
        plt.ylabel('Predicted Values', fontsize=14)
        plt.legend(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', labelsize=14)
        model_name_clean = model_name.replace(" ", "_")
        output_path = os.path.join(results_folder, f'actual_vs_predicted_{model_name_clean}_{dataset_type}.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Plot saved: {output_path}")
    except Exception as e:
        logger.error(f"Error in plot_actual_vs_predicted_separate ({dataset_type}): {str(e)}")
        plt.close()

def plot_actual_vs_predicted_combined(y_train, y_train_pred, y_test, y_test_pred, results_folder, model_name):
    """Generates a combined Actual vs Predicted plot with visible x and y axes."""
    logger.info(f"Creating combined Actual vs Predicted for {model_name}")
    try:
        fig = plt.figure(figsize=(8, 8))
        fig.patch.set_facecolor('white')
        ax = plt.gca()
        plt.scatter(y_train, y_train_pred, c='blue', edgecolor='black', alpha=0.5, s=60, linewidths=1.5,
                    label='Train')
        plt.scatter(y_test, y_test_pred, c='green', edgecolor='black', alpha=0.5, s=60, linewidths=1.5,
                    label='Test')
        min_val = min(min(y_train), min(y_train_pred), min(y_test), min(y_test_pred))
        max_val = max(max(y_train), max(y_train_pred), max(y_test), max(y_test_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        mse_train = mean_squared_error(y_train, y_train_pred)
        r2_train = r2_score(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        r2_test = r2_score(y_test, y_test_pred)
        metrics_text = f'Train - MSE: {mse_train:.3f}, R²: {r2_train:.3f}\nTest - MSE: {mse_test:.3f}, R²: {r2_test:.3f}'
        plt.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=14)
        plt.title(f'Actual vs Predicted - {model_name} (Train + Test)', fontsize=14)
        plt.xlabel('Actual Values', fontsize=14)
        plt.ylabel('Predicted Values', fontsize=14)
        plt.legend(fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', labelsize=14)
        model_name_clean = model_name.replace(" ", "_")
        output_path = os.path.join(results_folder, f'actual_vs_predicted_{model_name_clean}_combined.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Plot saved: {output_path}")
    except Exception as e:
        logger.error(f"Error in plot_actual_vs_predicted_combined: {str(e)}")
        plt.close()

def plot_shap_analysis(model, X_train_scaled, feature_names, results_folder, model_name):
    """Generates a SHAP Beeswarm plot."""
    logger.info(f"Creating SHAP for {model_name}")
    try:
        sample_size = min(int(0.25 * X_train_scaled.shape[0]), X_train_scaled.shape[0])
        sample_indices = np.random.choice(X_train_scaled.shape[0], sample_size, replace=False)
        X_sample = X_train_scaled[sample_indices]
        logger.info(f"SHAP sample: {X_sample.shape}")

        if model_name in ['RF', 'Extra Trees', 'XGBoost', 'LightGBM', 'Gradient Boosting']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
        else:
            explainer = shap.KernelExplainer(model.predict, shap.sample(X_sample, 50))
            shap_values = explainer.shap_values(X_sample, nsamples=100)

        fig = plt.figure(figsize=(8, 6))
        fig.patch.set_facecolor('white')
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="dot", show=False)
        plt.title(f'SHAP Beeswarm - {model_name}', fontsize=14)
        plt.xlabel('SHAP Value', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.tick_params(axis='both', labelsize=14)
        model_name_clean = model_name.replace(" ", "_")
        output_path = os.path.join(results_folder, f'shap_beeswarm_{model_name_clean}.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP plot saved: {output_path}")
    except Exception as e:
        logger.error(f"Error in plot_shap_analysis: {str(e)}")
        plt.close()

def plot_feature_importance(model, feature_names, results_folder, model_name):
    """Generates a feature importance plot."""
    logger.info(f"Creating feature importance for {model_name}")
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            if np.all(importances == 0):
                logger.warning(f"Zero importances for {model_name}.")
                return
            indices = np.argsort(importances)[::-1]
            sorted_importances = importances[indices]
            sorted_features = [feature_names[i] for i in indices]
            fig = plt.figure(figsize=(8, 6))
            fig.patch.set_facecolor('white')
            plt.barh(range(len(sorted_importances)), sorted_importances, align='center', color='steelblue')
            plt.yticks(range(len(sorted_importances)), sorted_features, fontsize=14)
            plt.xlabel('Importance', fontsize=14)
            plt.ylabel('Features', fontsize=14)
            plt.title(f'Feature Importance - {model_name}', fontsize=14)
            plt.gca().invert_yaxis()
            plt.tick_params(axis='x', labelsize=14)
            plt.tight_layout()
            model_name_clean = model_name.replace(" ", "_")
            output_path = os.path.join(results_folder, f'feature_importance_{model_name_clean}.png')
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
            logger.info(f"Feature importance plot saved: {output_path}")
        else:
            logger.info(f"Feature importance not available for {model_name}")
    except Exception as e:
        logger.error(f"Error in plot_feature_importance: {str(e)}")
        plt.close()

def plot_model_performance_metrics(results_dict, results_folder):
    """Generates a metrics comparison plot without values on bars."""
    logger.info("Creating comparison plot")
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.patch.set_facecolor('white')
        models = list(results_dict.keys())
        metrics_train = {
            'MSE': [results_dict[m]['mse_train'] for m in models],
            'RMSE': [np.sqrt(results_dict[m]['mse_train']) for m in models],
            'R2': [results_dict[m]['r2_train'] for m in models],
            'MAE': [results_dict[m]['mae_train'] for m in models]
        }
        metrics_test = {
            'MSE': [results_dict[m]['mse_test'] for m in models],
            'RMSE': [np.sqrt(results_dict[m]['mse_test']) for m in models],
            'R2': [results_dict[m]['r2_test'] for m in models],
            'MAE': [results_dict[m]['mae_test'] for m in models]
        }
        axes = {'MSE': ax1, 'RMSE': ax2, 'R2': ax3, 'MAE': ax4}
        x = np.arange(len(models))
        width = 0.35
        for metric, ax in axes.items():
            ax.bar(x - width/2, metrics_train[metric], width, color='steelblue', label='Train')
            ax.bar(x + width/2, metrics_test[metric], width, color='orange', label='Test')
            ax.set_title(f'{metric} by Model', fontsize=14)
            ax.set_xlabel('Model', fontsize=14)
            ax.set_ylabel(metric, fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right', fontsize=14)
            ax.legend(fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='y', labelsize=14)
        plt.tight_layout()
        output_path = os.path.join(results_folder, 'model_performance_comparison.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Plot saved: {output_path}")
    except Exception as e:
        logger.error(f"Error in plot_model_performance_metrics: {str(e)}")
        plt.close()

def plot_model_performance_metrics_side_by_side(results_dict, results_folder):
    """Generates an alternative metrics comparison plot without values on bars."""
    logger.info("Creating side-by-side comparison plot")
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.patch.set_facecolor('white')
        models = list(results_dict.keys())
        metrics_train = {
            'MSE': [results_dict[m]['mse_train'] for m in models],
            'RMSE': [np.sqrt(results_dict[m]['mse_train']) for m in models],
            'R2': [results_dict[m]['r2_train'] for m in models],
            'MAE': [results_dict[m]['mae_train'] for m in models]
        }
        metrics_test = {
            'MSE': [results_dict[m]['mse_test'] for m in models],
            'RMSE': [np.sqrt(results_dict[m]['mse_test']) for m in models],
            'R2': [results_dict[m]['r2_test'] for m in models],
            'MAE': [results_dict[m]['mae_test'] for m in models]
        }
        axes = {'MSE': ax1, 'RMSE': ax2, 'R2': ax3, 'MAE': ax4}
        x = np.arange(len(models))
        width = 0.35
        for metric, ax in axes.items():
            ax.bar(x - width/2, metrics_train[metric], width, color='steelblue', label='Train')
            ax.bar(x + width/2, metrics_test[metric], width, color='orange', label='Test')
            ax.set_title(f'{metric} by Model (Side by Side)', fontsize=14)
            ax.set_xlabel('Model', fontsize=14)
            ax.set_ylabel(metric, fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right', fontsize=14)
            ax.legend(fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='y', labelsize=14)
        plt.tight_layout()
        output_path = os.path.join(results_folder, 'model_performance_comparison_side_by_side.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Plot saved: {output_path}")
    except Exception as e:
        logger.error(f"Error in plot_model_performance_metrics_side_by_side: {str(e)}")
        plt.close()

def save_metrics_to_excel(results_dict, results_folder):
    """Saves metrics to an Excel file with 6 decimal places."""
    logger.info("Saving metrics to Excel")
    try:
        data = []
        for model_name in results_dict.keys():
            data.append({
                'Model': model_name,
                'MSE_Train': f"{results_dict[model_name]['mse_train']:.6f}",
                'RMSE_Train': f"{np.sqrt(results_dict[model_name]['mse_train']):.6f}",
                'R2_Train': f"{results_dict[model_name]['r2_train']:.6f}",
                'MAE_Train': f"{results_dict[model_name]['mae_train']:.6f}",
                'MSE_Test': f"{results_dict[model_name]['mse_test']:.6f}",
                'RMSE_Test': f"{np.sqrt(results_dict[model_name]['mse_test']):.6f}",
                'R2_Test': f"{results_dict[model_name]['r2_test']:.6f}",
                'MAE_Test': f"{results_dict[model_name]['mae_test']:.6f}"
            })
        df_metrics = pd.DataFrame(data)
        output_path = os.path.join(results_folder, 'model_metrics.xlsx')
        df_metrics.to_excel(output_path, index=False)
        logger.info(f"Metrics saved: {output_path}")
        return df_metrics
    except Exception as e:
        logger.error(f"Error in save_metrics_to_excel: {str(e)}")
        return None

def generate_pdf_report(results_folder, optimized_params, df_metrics, method, image_size=(4*inch, 4*inch)):
    """Generates a PDF report aggregating results."""
    logger.info("Generating PDF report")
    try:
        pdf_path = os.path.join(results_folder, 'model_analysis_report.pdf')
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        image_list = []

        title = f"Machine Learning Models Analysis Report - {method}"
        elements.append(Paragraph(title, styles['Title']))
        elements.append(Spacer(1, 12))

        for model_name in optimized_params.keys():
            model_name_clean = model_name.replace(" ", "_")
            model_folder = os.path.join(results_folder, model_name_clean)
            elements.append(Paragraph(f"1. {model_name} - Actual vs Predicted", styles['Heading1']))
            elements.append(Spacer(1, 12))
            for dataset_type in ['train', 'test', 'combined']:
                img_path = os.path.join(model_folder, f'actual_vs_predicted_{model_name_clean}_{dataset_type}.png')
                if os.path.exists(img_path):
                    image_list.append((f"{model_name} - {dataset_type.capitalize()}", img_path))

            elements.append(Paragraph(f"2. {model_name} - SHAP Analysis", styles['Heading1']))
            elements.append(Spacer(1, 12))
            img_path = os.path.join(model_folder, f'shap_beeswarm_{model_name_clean}.png')
            if os.path.exists(img_path):
                image_list.append((f"{model_name} - SHAP Beeswarm", img_path))

            elements.append(Paragraph(f"3. {model_name} - Feature Importance", styles['Heading1']))
            elements.append(Spacer(1, 12))
            img_path = os.path.join(model_folder, f'feature_importance_{model_name_clean}.png')
            if os.path.exists(img_path):
                image_list.append((f"{model_name} - Feature Importance", img_path))

        elements.append(Paragraph("4. Metrics Comparison", styles['Heading1']))
        elements.append(Spacer(1, 12))
        for plot_type in ['comparison', 'comparison_side_by_side']:
            img_path = os.path.join(results_folder, f'model_performance_{plot_type}.png')
            if os.path.exists(img_path):
                image_list.append((f"Metrics - {plot_type.replace('_', ' ').capitalize()}", img_path))

        width, height = image_size
        for i in range(0, len(image_list), 2):
            group = image_list[i:i + 2]
            flowables = []
            for j, (title, img_path) in enumerate(group):
                if j == 0:
                    flowables.append(Paragraph(title, styles['Heading2']))
                else:
                    flowables.append(Paragraph("", styles['Normal']))
                img = Image(img_path, width=width, height=height)
                flowables.append(img)
            elements.append(KeepTogether(flowables))
            if i + 2 < len(image_list):
                elements.append(Spacer(1, 12))

        elements.append(Paragraph("5. Metrics Table", styles['Heading1']))
        elements.append(Spacer(1, 12))
        if df_metrics is not None:
            table_data = [df_metrics.columns.tolist()] + df_metrics.values.tolist()
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 6),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 4),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ]))
            elements.append(table)

        doc.build(elements)
        logger.info(f"PDF report generated: {pdf_path}")
    except Exception as e:
        logger.error(f"Error in generate_pdf_report: {str(e)}")

def visualize_model_results(model_folder, method, model_name, optimized_models, all_models):
    """Generates visualizations for a model."""
    logger.info(f"Visualizing for {method}, model {model_name}")
    try:
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names, target_name, \
        optimized_params, y_train_preds, y_test_preds = load_results(model_folder, method, model_name)
        logger.info(f"Data - X_train_scaled: {X_train_scaled.shape}, y_train: {y_train.shape}")

        model = optimized_models.get(model_name)
        if not model:
            logger.warning(f"Model {model_name} not found. Recreating.")
            model_config = all_models.get(model_name)
            if not model_config:
                logger.error(f"Model {model_name} not defined in all_models.")
                return
            model_class = model_config['class']
            logger.info(f"Parameters: {optimized_params[model_name]}")
            model = model_class(**optimized_params[model_name])
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    model.fit(X_train_scaled, y_train)
                logger.info(f"Model {model_name} trained.")
            except Exception as e:
                logger.error(f"Error during training: {str(e)}")
                return

        y_train_pred = np.array(y_train_preds[model_name])
        y_test_pred = np.array(y_test_preds[model_name])
        logger.info(f"Predictions - y_train_pred: {y_train_pred.shape}, y_test_pred: {y_test_pred.shape}")

        plot_actual_vs_predicted_separate(y_train, y_train_pred, model_folder, model_name, 'train')
        plot_actual_vs_predicted_separate(y_test, y_test_pred, model_folder, model_name, 'test')
        plot_actual_vs_predicted_combined(y_train, y_train_pred, y_test, y_test_pred, model_folder, model_name)
        plot_shap_analysis(model, X_train_scaled, feature_names, model_folder, model_name)
        plot_feature_importance(model, feature_names, model_folder, model_name)

    except Exception as e:
        logger.error(f"Failed to visualize for {model_name}: {str(e)}")

def main(args):
    """Main function for optimization and visualization."""
    logger.info("Starting main")

    # Configurable parameters
    pso_maxiter = args.pso_maxiter
    pso_swarmsize = args.pso_swarmsize
    cv = args.cv
    method = args.method
    selected_model = args.selected_model

    script_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Script directory: {script_dir}")
    base_folder = args.base_folder or os.path.join(script_dir, 'Results_Hybrid_Optimization')
    main_results_folder = create_results_folder(base_folder, method)
    data_path = args.data_path or os.path.join(script_dir, 'DataHole.xlsx')

    try:
        X_train_scaled, X_test_scaled, y_train, y_test, feature_names, target_name = load_and_prepare_data(data_path)
    except Exception as e:
        logger.error(f"Error in load_and_prepare_data: {str(e)}")
        sys.exit(1)

    logger.info(f"Dataset size: {len(y_train) + len(y_test)} points")
    logger.info(f"Training set: {len(y_train)} points (~80%)")
    logger.info(f"Test set: {len(y_test)} points (~20%)")
    logger.info(f"Features: {', '.join(feature_names)}")
    logger.info(f"Target: {target_name}")

    all_models = {
        'RF': {
            'class': RandomForestRegressor,
            'param_bounds': get_initial_param_bounds('RF')
        },
        'KNN': {
            'class': KNeighborsRegressor,
            'param_bounds': get_initial_param_bounds('KNN')
        },
        'SVM': {
            'class': SVR,
            'param_bounds': get_initial_param_bounds('SVM')
        },
        'ANN': {
            'class': MLPRegressor,
            'param_bounds': get_initial_param_bounds('ANN')
        },
        'XGBoost': {
            'class': xgb.XGBRegressor,
            'param_bounds': get_initial_param_bounds('XGBoost')
        },
        'LightGBM': {
            'class': lgb.LGBMRegressor,
            'param_bounds': get_initial_param_bounds('LightGBM')
        },
        'Gaussian Process': {
            'class': GaussianProcessRegressor,
            'param_bounds': get_initial_param_bounds('Gaussian Process')
        },
        'Extra Trees': {
            'class': ExtraTreesRegressor,
            'param_bounds': get_initial_param_bounds('Extra Trees')
        },
        'AdaBoost': {
            'class': AdaBoostRegressor,
            'param_bounds': get_initial_param_bounds('AdaBoost')
        },
        'Gradient Boosting': {
            'class': GradientBoostingRegressor,
            'param_bounds': get_initial_param_bounds('Gradient Boosting')
        }
    }

    # Handle selected_model as string, list, or None
    if isinstance(selected_model, str):
        if selected_model not in all_models:
            logger.error(f"Invalid model: {selected_model}")
            sys.exit(1)
        selected_models = [selected_model]
    elif isinstance(selected_model, list):
        selected_models = selected_model
        for model in selected_models:
            if model not in all_models:
                logger.error(f"Invalid model in list: {model}")
                sys.exit(1)
    else:
        selected_models = ['RF', 'KNN', 'SVM', 'ANN', 'XGBoost']

    models = {name: all_models[name] for name in selected_models if name in all_models}

    if not models:
        logger.error("No valid models selected.")
        sys.exit(1)
    logger.info(f"Selected models: {list(models.keys())}")

    optimized_models = {}
    results = {}
    models_results = {}
    results_df = None

    for name, config in tqdm(models.items(), desc="Processing models", unit="model"):
        logger.info(f"\nOptimizing {name}...")
        results_folder = create_results_folder(base_folder, method, model_name=name)

        logger.info(f"Running {method}...")
        best_params = optimize_hyperparameters(config['class'], config['param_bounds'],
                                             X_train_scaled, y_train, pso_maxiter, pso_swarmsize, cv)
        logger.info(f"Best parameters: {best_params}")

        logger.info("Training and evaluating...")
        final_model = config['class'](**best_params)
        metrics, y_train_pred, y_test_pred = evaluate_model(final_model, X_train_scaled, X_test_scaled, y_train, y_test)
        if metrics is None:
            logger.warning(f"Skipping {name}: evaluation error.")
            continue
        results[name] = metrics
        optimized_models[name] = final_model

        models_results[name] = {
            'test_metrics': metrics
        }

        try:
            results_df = pd.DataFrame({
                name: {
                    'mse_train': f"{metrics['mse_train']:.6f}",
                    'r2_train': f"{metrics['r2_train']:.6f}",
                    'mae_train': f"{metrics['mae_train']:.6f}",
                    'mse_test': f"{metrics['mse_test']:.6f}",
                    'r2_test': f"{metrics['r2_test']:.6f}",
                    'mae_test': f"{metrics['mae_test']:.6f}"
                }
            }).T
            results_df.to_excel(os.path.join(results_folder, 'model_results.xlsx'))
            with open(os.path.join(results_folder, 'optimization_results.json'), 'w') as f:
                json.dump({method.lower(): {name: best_params}}, f, indent=4)
            with open(os.path.join(results_folder, 'predictions.json'), 'w') as f:
                json.dump({'train': {name: y_train_pred.tolist()}, 'test': {name: y_test_pred.tolist()}}, f, indent=4)
            np.savez(os.path.join(results_folder, 'data.npz'),
                     X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled,
                     y_train=y_train.values, y_test=y_test.values,
                     feature_names=feature_names, target_name=target_name)
            logger.info(f"Results saved in {results_folder}")
        except Exception as e:
            logger.error(f"Error saving results for {name}: {str(e)}")
            continue

        logger.info(f"\nGenerating visualizations for {name}...")
        visualize_model_results(results_folder, method, name, optimized_models, all_models)

    logger.info("\nGenerating global results...")
    results_dict, optimized_params = load_global_results(main_results_folder, method)
    if results_dict:
        try:
            plot_model_performance_metrics(results_dict, main_results_folder)
            plot_model_performance_metrics_side_by_side(results_dict, main_results_folder)
            df_metrics = save_metrics_to_excel(results_dict, main_results_folder)
            if df_metrics is not None:
                generate_pdf_report(main_results_folder, optimized_params, df_metrics, method)
            else:
                logger.warning("Metrics DataFrame is None. Skipping PDF report.")
        except Exception as e:
            logger.error(f"Error generating global results: {str(e)}")

    logger.info("Main completed")
    return results_df, optimized_models, models_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML model optimization.")
    parser.add_argument('--data_path', type=str, help='Path to the input data file.')
    parser.add_argument('--method', type=str, default='PSO', help='Optimization method.')
    parser.add_argument('--selected_model', nargs='+', default=['KNN', 'ANN', 'SVM', 'RF', 'XGBoost'],
                        help='List of models to run.')
    parser.add_argument('--pso_maxiter', type=int, default=100, help='Max iterations for PSO.')
    parser.add_argument('--pso_swarmsize', type=int, default=20, help='Swarm size for PSO.')
    parser.add_argument('--cv', type=int, default=5, help='Number of cross-validation folds.')
    parser.add_argument('--base_folder', type=str, help='Base folder to save results.')

    args = parser.parse_args()

    try:
        results_df, optimized_models, models_results = main(args)
        logger.info("Script executed successfully")
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1)
