import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    mean_absolute_error,
)
from sklearn.model_selection import KFold
from colorama import Fore, Style, init

init(autoreset=True)

logging.basicConfig(
    level=logging.INFO,
    format=f"{Fore.CYAN}%(asctime)s{Style.RESET_ALL} - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ensemble_model")


def create_ensemble_models(base_rf_params: Dict[str, Any]) -> Dict[str, Any]:
    """Create multiple base models for ensemble"""
    
    # Optimized Random Forest (primary model)
    rf_model = RandomForestRegressor(
        n_estimators=base_rf_params.get('n_estimators', 150),
        max_depth=base_rf_params.get('max_depth', 20),
        min_samples_split=base_rf_params.get('min_samples_split', 5),
        min_samples_leaf=base_rf_params.get('min_samples_leaf', 2),
        max_features=base_rf_params.get('max_features', 'sqrt'),
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    
    # Gradient Boosting (complementary algorithm)
    gb_model = GradientBoostingRegressor(
        n_estimators=120,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        max_features='sqrt',
        random_state=42
    )
    
    # Decision Tree with different parameters
    dt_model = DecisionTreeRegressor(
        max_depth=25,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42
    )
    
    # Linear models for diversity
    ridge_model = Ridge(alpha=1.0, random_state=42)
    elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
    
    models = {
        'random_forest': rf_model,
        'gradient_boosting': gb_model,
        'decision_tree': dt_model,
        'ridge': ridge_model,
        'elastic_net': elastic_model
    }
    
    return models


def train_voting_ensemble(
    X: pd.DataFrame, 
    y: pd.Series, 
    base_rf_params: Dict[str, Any]
) -> Tuple[VotingRegressor, float]:
    """Train a voting ensemble with multiple algorithms"""
    
    logger.info(f"{Fore.GREEN}Training voting ensemble with multiple algorithms...{Style.RESET_ALL}")
    
    # Get base models
    models = create_ensemble_models(base_rf_params)
    
    # Create voting ensemble with optimized weights
    voting_estimators = [
        ('rf', models['random_forest']),
        ('gb', models['gradient_boosting']),
        ('dt', models['decision_tree']),
        ('ridge', models['ridge']),
    ]
    
    voting_model = VotingRegressor(
        estimators=voting_estimators,
        weights=[0.4, 0.3, 0.2, 0.1],  # RF gets highest weight
        n_jobs=-1
    )
    
    # Train the ensemble
    voting_model.fit(X, y)
    
    # Evaluate with cross-validation
    cv_mape, _, _ = evaluate_ensemble_model(voting_model, X, y, cross_validate=True, log_transformed=True)
    
    logger.info(f"{Fore.GREEN}Voting ensemble trained - CV MAPE: {cv_mape*100:.2f}%{Style.RESET_ALL}")
    
    return voting_model, cv_mape


def train_stacking_ensemble(
    X: pd.DataFrame, 
    y: pd.Series, 
    base_rf_params: Dict[str, Any]
) -> Tuple[StackingRegressor, float]:
    """Train a stacking ensemble with meta-learner"""
    
    logger.info(f"{Fore.GREEN}Training stacking ensemble with meta-learner...{Style.RESET_ALL}")
    
    # Get base models
    models = create_ensemble_models(base_rf_params)
    
    # Create stacking ensemble
    base_estimators = [
        ('rf', models['random_forest']),
        ('gb', models['gradient_boosting']),
        ('dt', models['decision_tree']),
        ('ridge', models['ridge']),
        ('elastic', models['elastic_net'])
    ]
    
    # Use Ridge as meta-learner
    meta_learner = Ridge(alpha=0.1, random_state=42)
    
    stacking_model = StackingRegressor(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=5,  # Cross-validation for meta-features
        n_jobs=-1,
        passthrough=False  # Don't pass original features to meta-learner
    )
    
    # Train the ensemble
    stacking_model.fit(X, y)
    
    # Evaluate with cross-validation
    cv_mape, _, _ = evaluate_ensemble_model(stacking_model, X, y, cross_validate=True, log_transformed=True)
    
    logger.info(f"{Fore.GREEN}Stacking ensemble trained - CV MAPE: {cv_mape*100:.2f}%{Style.RESET_ALL}")
    
    return stacking_model, cv_mape


def train_blended_ensemble(
    X: pd.DataFrame, 
    y: pd.Series, 
    base_rf_params: Dict[str, Any]
) -> Tuple[Any, float]:
    """Train a custom blended ensemble with optimized weights"""
    
    logger.info(f"{Fore.GREEN}Training blended ensemble with optimized weights...{Style.RESET_ALL}")
    
    # Get base models
    models = create_ensemble_models(base_rf_params)
    
    # Train individual models
    trained_models = {}
    individual_scores = {}
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Clone and train model
            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X_train, y_train)
            
            # Predict and evaluate
            predictions = model_clone.predict(X_val)
            predictions_original = np.expm1(predictions)
            y_val_original = np.expm1(y_val)
            
            mape = mean_absolute_percentage_error(y_val_original, predictions_original)
            cv_scores.append(mape)
        
        avg_mape = np.mean(cv_scores)
        individual_scores[name] = avg_mape
        
        # Train on full dataset
        model.fit(X, y)
        trained_models[name] = model
        
        logger.info(f"  {name}: CV MAPE = {avg_mape*100:.2f}%")
    
    # Calculate optimal weights (inverse of MAPE)
    inverse_scores = {name: 1.0 / max(score, 0.001) for name, score in individual_scores.items()}
    total_inverse = sum(inverse_scores.values())
    optimal_weights = {name: weight / total_inverse for name, weight in inverse_scores.items()}
    
    logger.info(f"Optimal weights: {optimal_weights}")
    
    # Create blended ensemble class
    class BlendedEnsemble:
        def __init__(self, models, weights):
            self.models = models
            self.weights = weights
            self._is_log_transformed = True
        
        def predict(self, X):
            predictions = np.zeros(len(X))
            for name, model in self.models.items():
                weight = self.weights[name]
                pred = model.predict(X)
                predictions += weight * pred
            return predictions
        
        def get_params(self, deep=True):
            return {'models': self.models, 'weights': self.weights}
    
    blended_model = BlendedEnsemble(trained_models, optimal_weights)
    
    # Evaluate blended model
    cv_mape, _, _ = evaluate_ensemble_model(blended_model, X, y, cross_validate=True, log_transformed=True)
    
    logger.info(f"{Fore.GREEN}Blended ensemble trained - CV MAPE: {cv_mape*100:.2f}%{Style.RESET_ALL}")
    
    return blended_model, cv_mape


def evaluate_ensemble_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cross_validate: bool = False,
    cv: int = 5,
    log_transformed: bool = False,
) -> Tuple[float, List[float], Dict[str, float]]:
    """Evaluate ensemble model performance"""
    
    try:
        if cross_validate:
            kf = KFold(n_splits=cv, shuffle=True, random_state=42)
            
            mape_scores = []
            r2_scores = []
            mae_scores = []
            rmse_scores = []
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # For ensemble models, we need to handle training differently
                if hasattr(model, 'estimators_'):  # VotingRegressor or StackingRegressor
                    fold_model = model.__class__(**model.get_params())
                    fold_model.fit(X_train, y_train)
                elif hasattr(model, 'models'):  # BlendedEnsemble
                    # For blended ensemble, use the pre-trained models
                    fold_model = model
                else:
                    # Single model
                    fold_model = model.__class__(**model.get_params())
                    fold_model.fit(X_train, y_train)
                
                predictions = fold_model.predict(X_val)
                
                # Transform predictions back for evaluation if needed
                if log_transformed:
                    predictions_original = np.expm1(predictions)
                    y_val_original = np.expm1(y_val)
                    
                    mape = mean_absolute_percentage_error(y_val_original, predictions_original)
                    r2 = r2_score(y_val_original, predictions_original)
                    mae = mean_absolute_error(y_val_original, predictions_original)
                    rmse = np.sqrt(mean_squared_error(y_val_original, predictions_original))
                else:
                    mape = mean_absolute_percentage_error(y_val, predictions)
                    r2 = r2_score(y_val, predictions)
                    mae = mean_absolute_error(y_val, predictions)
                    rmse = np.sqrt(mean_squared_error(y_val, predictions))
                
                mape_scores.append(mape)
                r2_scores.append(r2)
                mae_scores.append(mae)
                rmse_scores.append(rmse)
            
            avg_mape = np.mean(mape_scores)
            avg_r2 = np.mean(r2_scores)
            avg_mae = np.mean(mae_scores)
            avg_rmse = np.mean(rmse_scores)
            
            predictions = model.predict(X)
            
            metrics = {
                "mape": avg_mape,
                "r2": avg_r2,
                "mae": avg_mae,
                "rmse": avg_rmse,
                "mape_cv": mape_scores,
                "r2_cv": r2_scores,
                "mae_cv": mae_scores,
                "rmse_cv": rmse_scores,
            }
            
            logger.info(
                f"{Fore.YELLOW}CV results - MAPE: {avg_mape*100:.2f}%, R²: {avg_r2:.4f}, RMSE: {avg_rmse:.2f}{Style.RESET_ALL}"
            )
            return avg_mape, predictions.tolist(), metrics
        
        else:
            predictions = model.predict(X)
            
            if log_transformed:
                predictions_original = np.expm1(predictions)
                y_original = np.expm1(y)
                
                mape = mean_absolute_percentage_error(y_original, predictions_original)
                r2 = r2_score(y_original, predictions_original)
                mae = mean_absolute_error(y_original, predictions_original)
                rmse = np.sqrt(mean_squared_error(y_original, predictions_original))
            else:
                mape = mean_absolute_percentage_error(y, predictions)
                r2 = r2_score(y, predictions)
                mae = mean_absolute_error(y, predictions)
                rmse = np.sqrt(mean_squared_error(y, predictions))
            
            metrics = {"mape": mape, "r2": r2, "mae": mae, "rmse": rmse}
            
            logger.info(
                f"{Fore.YELLOW}Evaluation results - MAPE: {mape*100:.2f}%, R²: {r2:.4f}, RMSE: {rmse:.2f}{Style.RESET_ALL}"
            )
            return mape, predictions.tolist(), metrics
    
    except Exception as e:
        logger.error(f"{Fore.RED}Error evaluating ensemble model: {str(e)}{Style.RESET_ALL}")
        return 1.0, [], {"mape": 1.0, "r2": 0.0, "mae": 1e10, "rmse": 1e10}

