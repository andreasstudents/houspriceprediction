import re
import pandas as pd
import numpy as np
import logging
import json
import os
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from scipy import stats


class PreprocessingError(Exception):

    pass


class DataValidationError(PreprocessingError):

    pass


class FeatureEngineeringError(PreprocessingError):

    pass


class DataScalingError(PreprocessingError):

    pass


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("house_price_prediction")


QUALITY_THRESHOLDS = {
    "price": {
        "min": 100e6,
        "max": 20e9,
    },
    "LT": {
        "min": 15,
        "max": 5000,
    },
    "LB": {
        "min": 10,
        "max": 2000,
    },
    "bedroom": {"min": 1, "max": 20},
    "bathroom": {"min": 1, "max": 15},
    "carport": {"min": 0, "max": 10},
}


IQR_THRESHOLDS = {
    "price": 2.0,
    "LT": 1.75,
    "LB": 1.75,
    "bedroom": 1.5,
    "bathroom": 1.5,
    "carport": 2.0,
}


Z_SCORE_THRESHOLDS = {
    "price": 3.0,
    "LT": 3.0,
    "LB": 3.0,
    "bedroom": 2.5,
    "bathroom": 2.5,
    "carport": 3.0,
}


def select_features(
    X: pd.DataFrame, y: pd.Series, method: str = "f_regression", k: int = None
) -> Tuple[pd.DataFrame, List[str]]:
    if k is None:
        k = max(1, X.shape[1] // 2)

    if method == "f_regression":
        score_func = f_regression
    elif method == "mutual_info":
        score_func = mutual_info_regression
    else:
        raise ValueError(f"Unknown feature selection method: {method}")

    try:

        selector = SelectKBest(score_func=score_func, k=k)
        X_new = selector.fit_transform(X, y)

        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()

        X_selected = pd.DataFrame(X_new, columns=selected_features, index=X.index)

        feature_scores = pd.DataFrame(
            {"Feature": X.columns, "Score": selector.scores_}
        ).sort_values(by="Score", ascending=False)

        logger.info(
            f"Feature selection completed using {method}. Selected {len(selected_features)} features: {', '.join(selected_features)}"
        )
        logger.info(
            f"Top 5 features: {', '.join(feature_scores['Feature'].head(5).tolist())}"
        )

        return X_selected, selected_features

    except Exception as e:
        logger.error(f"Feature selection failed: {str(e)}")
        raise ValueError(f"Feature selection failed: {str(e)}")


def scale_features(
    df: pd.DataFrame,
    price_cols: List[str] = None,
    numeric_cols: List[str] = None,
    fit_scalers: bool = True,
    scalers: Dict[str, Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df_scaled = df.copy()

    if price_cols is None:
        price_cols = [col for col in df.columns if "price" in col.lower()]

    if numeric_cols is None:

        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        numeric_cols = [col for col in numeric_cols if col not in price_cols]

    if scalers is None:
        scalers = {}

    try:

        if price_cols:
            price_data = df_scaled[price_cols]

            if fit_scalers:
                price_scaler = RobustScaler()
                scalers["price_scaler"] = price_scaler
            else:
                price_scaler = scalers.get("price_scaler")
                if price_scaler is None:
                    raise DataScalingError(
                        "Price scaler not provided but fit_scalers=False"
                    )

            if fit_scalers:
                df_scaled[price_cols] = price_scaler.fit_transform(price_data)
            else:
                df_scaled[price_cols] = price_scaler.transform(price_data)

        if numeric_cols:
            numeric_data = df_scaled[numeric_cols]

            if fit_scalers:
                numeric_scaler = StandardScaler()
                scalers["numeric_scaler"] = numeric_scaler
            else:
                numeric_scaler = scalers.get("numeric_scaler")
                if numeric_scaler is None:
                    raise DataScalingError(
                        "Numeric scaler not provided but fit_scalers=False"
                    )

            if fit_scalers:
                df_scaled[numeric_cols] = numeric_scaler.fit_transform(numeric_data)
            else:
                df_scaled[numeric_cols] = numeric_scaler.transform(numeric_data)

        logger.info(
            f"Data scaling completed. Scaled {len(price_cols)} price features and {len(numeric_cols)} numeric features"
        )
        return df_scaled, scalers

    except Exception as e:
        logger.error(f"Error during data scaling: {str(e)}")
        raise DataScalingError(f"Data scaling failed: {str(e)}")


def remove_outliers(
    df: pd.DataFrame, columns: List[str], method: str = "both"
) -> pd.DataFrame:
    df_clean = df.copy()
    original_count = len(df_clean)

    if method in ["iqr", "both"]:
        for column in columns:
            if column in df_clean.columns:

                threshold = IQR_THRESHOLDS.get(column, 1.5)

                Q1 = df_clean[column].quantile(0.25)
                Q3 = df_clean[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outliers_count = df_clean[
                    (df_clean[column] < lower_bound) | (df_clean[column] > upper_bound)
                ].shape[0]

                df_clean = df_clean[
                    (df_clean[column] >= lower_bound)
                    & (df_clean[column] <= upper_bound)
                ]

                logger.info(
                    f"IQR method: Removed {outliers_count} outliers from {column} "
                    f"(threshold: {threshold})"
                )

    if method in ["zscore", "both"]:
        for column in columns:
            if column in df_clean.columns:

                threshold = Z_SCORE_THRESHOLDS.get(column, 3.0)

                z_scores = np.abs(stats.zscore(df_clean[column], nan_policy="omit"))

                outliers_count = df_clean[z_scores > threshold].shape[0]

                df_clean = df_clean[z_scores <= threshold]

                logger.info(
                    f"Z-score method: Removed {outliers_count} outliers from {column} "
                    f"(threshold: {threshold})"
                )

    removed_count = original_count - len(df_clean)
    removed_percentage = (
        (removed_count / original_count) * 100 if original_count > 0 else 0
    )
    logger.info(
        f"Removed {removed_count} outliers in total ({removed_percentage:.2f}% of data)"
    )

    return df_clean


def validate_dataframe(
    df: pd.DataFrame, required_columns: List[str], raise_exception: bool = False
) -> Tuple[bool, str]:
    if df.empty:
        error_msg = "DataFrame is empty"
        if raise_exception:
            raise DataValidationError(error_msg)
        return False, error_msg

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        error_msg = f"Missing required columns: {', '.join(missing_columns)}"
        if raise_exception:
            raise DataValidationError(error_msg)
        return False, error_msg

    null_columns = [
        col for col in required_columns if col in df.columns and df[col].isna().all()
    ]
    if null_columns:
        error_msg = f"Columns with all null values: {', '.join(null_columns)}"
        if raise_exception:
            raise DataValidationError(error_msg)
        return False, error_msg

    type_issues = []
    for col in df.columns:
        if col in df.select_dtypes(include=["object"]).columns:

            if any(
                col.lower().endswith(suffix)
                for suffix in ["price", "area", "lt", "lb", "size"]
            ):
                try:
                    pd.to_numeric(df[col], errors="raise")
                except:
                    type_issues.append(col)

    if type_issues:
        error_msg = f"Columns with potential type issues: {', '.join(type_issues)}"
        if raise_exception:
            raise DataValidationError(error_msg)
        return False, error_msg

    return True, "Validation passed"


def convert_to_numeric(value: Union[str, int, float, None]) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)

    if value is None or pd.isna(value):
        return None

    try:

        value_str = str(value)
        value_numeric = re.sub(r"Rp\s?", "", value_str)

        if "Miliar" in value_numeric:
            cleaned_value = re.sub(r"\s?Miliar", "", value_numeric).replace(",", ".")
            return float(cleaned_value) * 1e9

        elif "Juta" in value_numeric:
            cleaned_value = re.sub(r"\s?Juta", "", value_numeric).replace(",", ".")
            return float(cleaned_value) * 1e6

        elif re.match(r"^\d+(\.\d+)?$", value_numeric.strip()):
            return float(value_numeric)

        else:
            logger.warning(f"Couldn't parse price value: {value}")
            return None

    except (ValueError, TypeError) as e:
        logger.warning(f"Error converting price value {value}: {str(e)}")
        return None


def preprocess_updated(updated: Any) -> Optional[datetime]:
    if updated is None or pd.isna(updated):
        return None

    if isinstance(updated, datetime):
        return updated

    try:
        updated_str = str(updated).lower()
        today = datetime.today()

        number_match = re.search(r"(\d+)", updated_str)
        if not number_match:
            logger.warning(f"Couldn't extract number from updated value: {updated}")
            return None

        time_value = int(number_match.group(1))

        if "minggu" in updated_str:
            return today - timedelta(weeks=time_value)
        elif "bulan" in updated_str:

            return today - timedelta(days=time_value * 30.44)
        elif "hari" in updated_str:
            return today - timedelta(days=time_value)
        elif "jam" in updated_str:
            return today - timedelta(hours=time_value)
        elif "menit" in updated_str:
            return today - timedelta(minutes=time_value)
        else:

            try:

                for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d %b %Y"]:
                    try:
                        return datetime.strptime(updated_str, fmt)
                    except ValueError:
                        continue

                logger.warning(f"Unknown time format in updated value: {updated}")
                return None
            except:
                logger.warning(f"Failed to parse date: {updated}")
                return None

    except Exception as e:
        logger.warning(f"Error processing updated value {updated}: {str(e)}")
        return None


def preprocess_area(area_str: Any) -> Optional[float]:
    """
    Extract and convert area values from strings.

    Args:
        area_str: String containing area information (e.g., "150 m²")

    Returns:
        Numeric area value in square meters or None if conversion fails
    """

    if isinstance(area_str, (int, float)):
        return float(area_str) if not pd.isna(area_str) else None

    if area_str is None or pd.isna(area_str):
        return None

    try:

        area_text = str(area_str).strip()

        match = re.search(r"(\d+(\.\d+)?)\s*m²", area_text)
        if match:
            return float(match.group(1))

        match = re.search(r"(\d+(\.\d+)?)m²", area_text)
        if match:
            return float(match.group(1))

        match = re.search(r"^(\d+(\.\d+)?)$", area_text)
        if match:
            return float(match.group(1))

        match = re.search(r"^(\d+,\d+)$", area_text)
        if match:
            return float(match.group(1).replace(",", "."))

        logger.warning(f"Couldn't parse area value: {area_str}")
        return None

    except (ValueError, TypeError) as e:
        logger.warning(f"Error processing area value {area_str}: {str(e)}")
        return None


def preprocess_carport(carport_str: Any) -> int:
    """
    Preprocess carport data by converting it to numeric values.

    Args:
        carport_str: String or integer representing the number of carports

    Returns:
        Integer representing the number of carports or 0 if missing
    """
    try:
        if pd.isna(carport_str):
            return 0
        elif isinstance(carport_str, (int, float)) and not pd.isna(carport_str):
            return int(carport_str)
        elif isinstance(carport_str, str) and carport_str.strip().isdigit():
            return int(carport_str)
        else:

            match = re.search(r"(\d+)", str(carport_str))
            if match:
                return int(match.group(1))

            logger.warning(
                f"Could not parse carport value: {carport_str}, defaulting to 0"
            )
            return 0
    except Exception as e:
        logger.warning(f"Error processing carport value {carport_str}: {str(e)}")
        return 0


def preprocess_location(location):
    if pd.isna(location):
        return pd.Series([None, None])
    split_location = location.split(",")
    kecamatan = split_location[0].strip() if len(split_location) > 0 else None
    kabupaten_kota = split_location[1].strip() if len(split_location) > 1 else None
    return pd.Series([kecamatan, kabupaten_kota])


def preprocess_data(
    df: pd.DataFrame,
    validate: bool = True,
    scale_data: bool = False,
    feature_selection: bool = False,
    required_columns: List[str] = None,
    output_dir: str = "reports",
) -> pd.DataFrame:
    if required_columns is None:
        required_columns = ["price", "LT", "LB", "location", "bedroom", "bathroom"]

    if validate:
        is_valid, error_message = validate_dataframe(
            df, required_columns, raise_exception=False
        )
        if not is_valid:
            logger.error(f"Data validation failed: {error_message}")
            if validate:
                raise DataValidationError(error_message)

    logger.info(f"Starting preprocessing of {len(df)} rows of data")
    df_processed = df.copy()

    df_processed["price"] = df_processed["price"].apply(convert_to_numeric)
    df_processed["LT"] = df_processed["LT"].apply(preprocess_area)
    df_processed["LB"] = df_processed["LB"].apply(preprocess_area)
    df_processed["updated"] = df_processed["updated"].apply(preprocess_updated)
    df_processed[["kecamatan", "kabupaten_kota"]] = df_processed["location"].apply(
        preprocess_location
    )
    df_processed["carport"] = df_processed["carport"].apply(preprocess_carport)

    df_processed["updated"] = pd.to_datetime(df_processed["updated"])

    df_processed = df_processed[
        ~df_processed["title"].str.contains("hotel|kost|Kos", case=False, na=False)
    ]

    df_processed = df_processed.drop_duplicates(subset="title", keep="first")

    sixty_days_ago = datetime.today() - timedelta(days=60)
    df_processed = df_processed[df_processed["updated"] >= sixty_days_ago]

    df_processed["price"] = df_processed["price"].fillna(df_processed["price"].median())
    df_processed["LT"] = df_processed["LT"].fillna(df_processed["LT"].median())
    df_processed["LB"] = df_processed["LB"].fillna(df_processed["LB"].median())
    df_processed["bedroom"] = df_processed["bedroom"].fillna(
        df_processed["bedroom"].median()
    )
    df_processed["bathroom"] = df_processed["bathroom"].fillna(
        df_processed["bathroom"].median()
    )

    df_processed = df_processed.dropna(
        subset=["updated", "price", "LT", "LB", "bedroom", "bathroom"]
    )

    try:
        logger.info("Performing data quality checks...")

        price_min = QUALITY_THRESHOLDS["price"]["min"]
        price_max = QUALITY_THRESHOLDS["price"]["max"]
        invalid_prices = df_processed[
            (df_processed["price"] < price_min) | (df_processed["price"] > price_max)
        ]

        if not invalid_prices.empty:
            logger.warning(
                f"Found {len(invalid_prices)} rows with price values outside reasonable range "
                f"({price_min:,.0f} - {price_max:,.0f})"
            )
            df_processed = df_processed[
                (df_processed["price"] >= price_min)
                & (df_processed["price"] <= price_max)
            ]

        invalid_areas = df_processed[df_processed["LB"] > df_processed["LT"] * 1.1]
        if not invalid_areas.empty:
            logger.warning(
                f"Found {len(invalid_areas)} rows where building area (LB) > land area (LT)"
            )

            df_processed.loc[
                df_processed["LB"] > df_processed["LT"] * 1.1, "area_ratio_flag"
            ] = True
            df_processed["area_ratio_flag"] = df_processed["area_ratio_flag"].fillna(
                False
            )

        avg_room_size = 9
        df_processed["total_rooms"] = df_processed["bedroom"] + df_processed["bathroom"]
        df_processed["min_area_needed"] = df_processed["total_rooms"] * avg_room_size

        invalid_rooms = df_processed[
            df_processed["min_area_needed"] > df_processed["LB"]
        ]
        if not invalid_rooms.empty:
            logger.warning(
                f"Found {len(invalid_rooms)} rows with too many rooms for the building size"
            )

            df_processed.loc[
                df_processed["min_area_needed"] > df_processed["LB"], "room_count_flag"
            ] = True
            df_processed["room_count_flag"] = df_processed["room_count_flag"].fillna(
                False
            )

        logger.info(
            f"Data quality checks completed. Remaining rows: {len(df_processed)}"
        )

    except Exception as e:
        logger.error(f"Error during data quality checks: {str(e)}")

    if "kecamatan" in df_processed.columns and df_processed["kecamatan"].notna().any():

        df_processed["kecamatan"] = df_processed["kecamatan"].fillna("Unknown")

        le = LabelEncoder()
        df_processed["kecamatan_encoded"] = le.fit_transform(df_processed["kecamatan"])

        kecamatan_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

        os.makedirs(output_dir, exist_ok=True)
        try:
            with open(f"{output_dir}/kecamatan_encoding.json", "w") as f:
                json.dump(kecamatan_mapping, f, indent=4)
        except Exception as e:
            logger.warning(f"Failed to save kecamatan encoding mapping: {str(e)}")

    try:
        logger.info("Performing feature engineering...")

        df_processed["price_per_m2_land"] = df_processed.apply(
            lambda row: row["price"] / row["LT"] if row["LT"] > 0 else np.nan, axis=1
        )

        df_processed["price_per_m2_building"] = df_processed.apply(
            lambda row: row["price"] / row["LB"] if row["LB"] > 0 else np.nan, axis=1
        )

        df_processed["building_efficiency"] = df_processed.apply(
            lambda row: row["LB"] / row["LT"] if row["LT"] > 0 else np.nan, axis=1
        )

        df_processed["listing_age_days"] = (
            datetime.today() - df_processed["updated"]
        ).dt.days

        if "kecamatan" in df_processed.columns:
            freq_encoding = (
                df_processed["kecamatan"].value_counts(normalize=True).to_dict()
            )
            df_processed["kecamatan_freq"] = df_processed["kecamatan"].map(
                freq_encoding
            )

        numeric_cols = df_processed.select_dtypes(include=["float64", "int64"]).columns
        for col in numeric_cols:
            if df_processed[col].isna().any():
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())

        logger.info(
            f"Feature engineering completed, added {4 + ('kecamatan' in df_processed.columns)} new features"
        )

    except Exception as e:
        logger.error(f"Error during feature engineering: {str(e)}")
        raise FeatureEngineeringError(f"Feature engineering failed: {str(e)}")

    try:
        quality_metrics = {
            "total_rows": len(df_processed),
            "duplicate_count": len(df) - len(df.drop_duplicates()),
            "missing_values": {
                col: int(df_processed[col].isna().sum()) for col in df_processed.columns
            },
            "numeric_features": {
                col: {
                    "mean": float(df_processed[col].mean()),
                    "median": float(df_processed[col].median()),
                    "min": float(df_processed[col].min()),
                    "max": float(df_processed[col].max()),
                    "std": float(df_processed[col].std()),
                }
                for col in df_processed.select_dtypes(
                    include=["float64", "int64"]
                ).columns
            },
            "categorical_features": {
                col: {
                    "unique_values": int(df_processed[col].nunique()),
                    "most_common": (
                        df_processed[col].value_counts().index[0]
                        if not df_processed[col].value_counts().empty
                        else None
                    ),
                }
                for col in df_processed.select_dtypes(
                    include=["object", "category"]
                ).columns
            },
        }

        os.makedirs("reports", exist_ok=True)
        with open("reports/data_quality_metrics.json", "w") as f:
            json.dump(quality_metrics, f, indent=4, default=str)

        logger.info(
            f"Data quality report generated with {len(quality_metrics['numeric_features'])} numeric and {len(quality_metrics['categorical_features'])} categorical features"
        )

    except Exception as e:
        logger.warning(f"Failed to generate data quality report: {str(e)}")

    scalers = {}
    if scale_data:
        try:
            logger.info("Scaling numeric features...")

            price_cols = [col for col in df_processed.columns if "price" in col.lower()]
            numeric_cols = df_processed.select_dtypes(
                include=["float64", "int64"]
            ).columns
            numeric_cols = [col for col in numeric_cols if col not in price_cols]

            df_processed, scalers = scale_features(
                df_processed, price_cols, numeric_cols
            )

            try:
                import pickle

                os.makedirs(output_dir, exist_ok=True)
                with open(f"{output_dir}/scalers.pkl", "wb") as f:
                    pickle.dump(scalers, f)
                logger.info(f"Saved scalers to {output_dir}/scalers.pkl")
            except Exception as e:
                logger.warning(f"Failed to save scalers: {str(e)}")

        except Exception as e:
            logger.error(f"Error during data scaling: {str(e)}")

    logger.info(f"Preprocessing completed. Final dataset shape: {df_processed.shape}")
    logger.info(f"Columns in processed data: {', '.join(df_processed.columns)}")

    return df_processed
