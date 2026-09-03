import logging
import numpy as np
import pandas as pd
import traceback
from typing import Tuple, List, Dict, Optional, Union, Any
from sklearn.ensemble import RandomForestRegressor
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
logger = logging.getLogger("rf_model")


def train_base_rf_model(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[RandomForestRegressor, float]:
    try:
        logger.info(
            f"{Fore.GREEN}Training base Random Forest model... [n_estimators=100]{Style.RESET_ALL}"
        )
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X, y)

        feature_importances = pd.Series(
            model.feature_importances_, index=X.columns
        ).sort_values(ascending=False)

        top_features = ", ".join(feature_importances.index[:5].tolist())
        logger.info(
            f"{Fore.GREEN}Base RF model trained successfully - Trees: {model.n_estimators}, Top features: {top_features}{Style.RESET_ALL}"
        )

        cv_mape, _, _ = evaluate_rf_model(model, X, y, cross_validate=True, log_transformed=True)

        return model, cv_mape

    except Exception as e:
        logger.error(
            f"{Fore.RED}Error training base Random Forest model: {str(e)}{Style.RESET_ALL}"
        )
        raise RuntimeError(f"Failed to train base RF model: {str(e)}")


def evaluate_rf_model(
    model: RandomForestRegressor,
    X: pd.DataFrame,
    y: pd.Series,
    cross_validate: bool = False,
    cv: int = 5,
    log_transformed: bool = False,
) -> Tuple[float, List[float], Dict[str, float]]:
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

                fold_model = RandomForestRegressor(**model.get_params())
                fold_model.fit(X_train, y_train)

                predictions = fold_model.predict(X_val)
                
                # If using log-transformed data, transform predictions back for proper evaluation
                if log_transformed:
                    predictions_original = np.expm1(predictions)
                    y_val_original = np.expm1(y_val)
                    # Calculate ALL metrics on original scale for consistency
                    mape = mean_absolute_percentage_error(y_val_original, predictions_original)
                    r2 = r2_score(y_val_original, predictions_original)
                    mae = mean_absolute_error(y_val_original, predictions_original)
                    rmse = np.sqrt(mean_squared_error(y_val_original, predictions_original))
                else:
                    predictions_original = predictions
                    y_val_original = y_val
                    mape = mean_absolute_percentage_error(y_val_original, predictions_original)
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
                f"{Fore.YELLOW}Cross-validation results - MAPE: {avg_mape*100:.2f}%, R²: {avg_r2:.4f}, RMSE: {avg_rmse:.2f}{Style.RESET_ALL}"
            )
            return avg_mape, predictions.tolist(), metrics

        else:
            predictions = model.predict(X)
            
            # If using log-transformed data, transform predictions back for proper evaluation
            if log_transformed:
                predictions_original = np.expm1(predictions)
                y_original = np.expm1(y)
                # Calculate ALL metrics on original scale for consistency
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
        logger.error(
            f"{Fore.RED}Error evaluating Random Forest model: {str(e)}{Style.RESET_ALL}"
        )

        return 1.0, [], {"mape": 1.0, "r2": 0.0, "mae": 1e10, "rmse": 1e10}


def train_optimized_rf_model(
    X: pd.DataFrame, y: pd.Series, params: Union[Tuple, List, Dict[str, Any]]
) -> Tuple[RandomForestRegressor, float]:
    try:

        if isinstance(params, dict):
            logger.info(f"Converting dictionary params to tuple: {params}")
            param_tuple = (
                params.get("n_estimators", 100),
                params.get("max_depth", None),
                params.get("min_samples_split", 2),
                params.get("min_samples_leaf", 1),
                params.get("max_features", "sqrt"),
                params.get("bootstrap", True),
            )
            params = param_tuple
            logger.info(f"Converted to tuple: {params}")

        if not isinstance(params, (list, tuple)):
            raise TypeError(
                f"Expected params to be a list, tuple, or dict, got {type(params).__name__}: {params}"
            )

        if len(params) < 4:
            raise ValueError(
                f"Expected at least 4 parameters, got {len(params)}: {params}"
            )

        try:
            n_estimators = int(params[0])
            if n_estimators <= 0:
                raise ValueError(f"n_estimators must be positive, got {n_estimators}")

            max_depth = int(params[1]) if params[1] and params[1] > 0 else None

            min_samples_split = int(params[2])
            if min_samples_split < 2:
                raise ValueError(
                    f"min_samples_split must be at least 2, got {min_samples_split}"
                )

            min_samples_leaf = int(params[3])
            if min_samples_leaf < 1:
                raise ValueError(
                    f"min_samples_leaf must be at least 1, got {min_samples_leaf}"
                )
        except (ValueError, TypeError) as e:
            raise ValueError(f"Error converting basic parameters: {e}") from e

        max_features = "sqrt"
        if len(params) > 4:
            mf_val = params[4]
            logger.info(
                f"Processing max_features parameter with value: {mf_val} (type: {type(mf_val).__name__})"
            )

            if isinstance(mf_val, str):
                if mf_val in ["auto", "sqrt", "log2", None]:
                    max_features = mf_val
                else:
                    raise ValueError(f"Invalid string value for max_features: {mf_val}")

            elif isinstance(mf_val, (int, float)):

                if 0.0 < mf_val <= 1.0:

                    max_features = float(mf_val)
                    logger.info(f"Using max_features as a float value: {max_features}")

                elif mf_val < 0.33:
                    max_features = "sqrt"
                elif mf_val < 0.66:
                    max_features = "log2"
                else:
                    max_features = None
            else:
                logger.warning(
                    f"Unexpected type for max_features: {type(mf_val).__name__}, using default 'sqrt'"
                )

        bootstrap = True
        if len(params) > 5:
            bootstrap_val = params[5]
            if isinstance(bootstrap_val, bool):
                bootstrap = bootstrap_val
            elif isinstance(bootstrap_val, (int, float)):
                bootstrap = bootstrap_val > 0.5
            else:
                logger.warning(
                    f"Unexpected type for bootstrap: {type(bootstrap_val).__name__}, using default True"
                )

        param_str = (
            f"n_estimators={n_estimators}, max_depth={max_depth}, "
            f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, "
            f"max_features={max_features} (type: {type(max_features).__name__}), bootstrap={bootstrap}"
        )
        logger.info(
            f"{Fore.GREEN}Training optimized RF model with parameters: {param_str}{Style.RESET_ALL}"
        )

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=42,
            n_jobs=-1,
        )

        logger.info(f"Fitting model with X shape: {X.shape}, y shape: {y.shape}")
        try:
            model.fit(X, y)
            logger.info(
                f"{Fore.GREEN}Optimized Random Forest model trained successfully{Style.RESET_ALL}"
            )
        except Exception as fit_error:
            logger.error(
                f"{Fore.RED}Error during model.fit(): {str(fit_error)}{Style.RESET_ALL}"
            )
            raise

        cv_mape, _, _ = evaluate_rf_model(model, X, y, cross_validate=True, log_transformed=True)

        return model, cv_mape

    except Exception as e:

        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(
            f"{Fore.RED}Error training optimized Random Forest model: {error_msg}{Style.RESET_ALL}"
        )
        logger.error(
            f"{Fore.RED}Input params type: {type(params).__name__}, value: {params}{Style.RESET_ALL}"
        )

        logger.warning(
            f"{Fore.YELLOW}Falling back to default Random Forest model{Style.RESET_ALL}"
        )
        return train_base_rf_model(X, y)


def predict_price_rf(
    model: RandomForestRegressor,
    bedroom: Union[int, float],
    bathroom: Union[int, float],
    lt: Union[int, float],
    lb: Union[int, float],
    carport: Optional[Union[int, float]] = None,
    kecamatan_encoded: Optional[Union[int, float]] = None,
    listing_age_days: Optional[Union[int, float]] = None,
    fallback_value: Optional[float] = None,
) -> float:
    try:
        # Calculate basic features
        features = {
            "bedroom": [bedroom],
            "bathroom": [bathroom],
            "LT": [lt],
            "LB": [lb],
        }

        # Add optional features if provided
        if carport is not None:
            features["carport"] = [carport]
        else:
            features["carport"] = [0]  # Default to 0 if not provided

        if kecamatan_encoded is not None:
            features["kecamatan_encoded"] = [kecamatan_encoded]
        else:
            features["kecamatan_encoded"] = [0]  # Default encoding
            
        if listing_age_days is not None:
            features["listing_age_days"] = [listing_age_days]
        else:
            features["listing_age_days"] = [30]  # Default to 30 days

        # Calculate legitimate derived features (not price-based)
        if "building_efficiency" in model.feature_names_in_:
            features["building_efficiency"] = [lb / lt if lt > 0 else 0]
            
        if "total_rooms" in model.feature_names_in_:
            features["total_rooms"] = [bedroom + bathroom]

        # Calculate fuzzy features if they exist in the model
        if "fuzzy_area_quality" in model.feature_names_in_ or "fuzzy_quality_score" in model.feature_names_in_:
            try:
                # Import here to avoid circular imports
                import sys
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(current_dir)
                sys.path.insert(0, os.path.join(parent_dir, "utils"))
                from utils.preprocessing import create_fuzzy_systems
                
                fuzzy_systems = create_fuzzy_systems()
                
                # Calculate fuzzy area quality
                fuzzy_area_quality = 0.5  # Default value
                if "area" in fuzzy_systems:
                    try:
                        area_system = fuzzy_systems["area"]
                        area_system.input["lt"] = min(lt, 2000)
                        area_system.input["lb"] = min(lb, 1000)
                        area_system.compute()
                        fuzzy_area_quality = area_system.output["area_quality"] / 100.0
                    except Exception as e:
                        logger.warning(f"Could not calculate fuzzy area quality: {e}")
                        
                # Calculate fuzzy room quality
                fuzzy_room_quality = 0.5  # Default value
                if "room" in fuzzy_systems:
                    try:
                        room_system = fuzzy_systems["room"]
                        room_system.input["bedroom"] = min(bedroom, 10)
                        room_system.input["bathroom"] = min(bathroom, 10)
                        room_system.compute()
                        fuzzy_room_quality = room_system.output["room_quality"] / 100.0
                    except Exception as e:
                        logger.warning(f"Could not calculate fuzzy room quality: {e}")
                        
                if "fuzzy_area_quality" in model.feature_names_in_:
                    features["fuzzy_area_quality"] = [fuzzy_area_quality]
                    
                if "fuzzy_quality_score" in model.feature_names_in_:
                    features["fuzzy_quality_score"] = [(fuzzy_area_quality + fuzzy_room_quality) / 2.0]
                    
            except Exception as e:
                logger.warning(f"Could not calculate fuzzy features: {e}")
                if "fuzzy_area_quality" in model.feature_names_in_:
                    features["fuzzy_area_quality"] = [0.5]
                if "fuzzy_quality_score" in model.feature_names_in_:
                    features["fuzzy_quality_score"] = [0.5]

        X_pred = pd.DataFrame(features)

        # Check for required columns and add defaults for missing ones
        required_columns = model.feature_names_in_
        missing_columns = set(required_columns) - set(X_pred.columns)

        if missing_columns:
            logger.warning(
                f"{Fore.YELLOW}Missing columns for prediction: {missing_columns}{Style.RESET_ALL}"
            )
            # Only set defaults for legitimate missing features
            for col in missing_columns:
                if "price" in col.lower():
                    logger.error(f"Model incorrectly expects price-derived feature: {col}")
                    return fallback_value if fallback_value is not None else 2000000000
                else:
                    X_pred[col] = 0  # Default value for other missing features

        # Ensure column order matches model expectation
        X_pred = X_pred[required_columns]

        prediction = model.predict(X_pred)[0]
        
        # If model was trained on log-transformed data, transform back
        # This will be determined by the model metadata or attributes
        try:
            # Check if this is a log-transformed model by examining prediction range
            if hasattr(model, '_is_log_transformed') and model._is_log_transformed:
                prediction = np.expm1(prediction)  # Inverse of log1p
                logger.info(f"Applied inverse log transformation to prediction")
            elif prediction < 30:  # Log-transformed predictions are typically < 30
                prediction = np.expm1(prediction)  # Likely log-transformed
                logger.info(f"Detected log-transformed model, applied inverse transformation")
        except Exception as e:
            logger.warning(f"Error applying inverse transformation: {e}")
        
        # Sanity check the prediction
        if prediction <= 0 or prediction > 50000000000:  # 50 billion max
            logger.warning(f"Prediction seems unreasonable: {prediction}, using fallback")
            return fallback_value if fallback_value is not None else 2000000000

        return prediction

    except Exception as e:
        logger.error(
            f"{Fore.RED}Error predicting price with RF model: {str(e)}{Style.RESET_ALL}"
        )
        return fallback_value if fallback_value is not None else 2000000000
