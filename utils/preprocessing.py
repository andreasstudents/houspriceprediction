import re
import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from sklearn.preprocessing import (
    LabelEncoder,
    RobustScaler,
    StandardScaler,
    OneHotEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from scipy import stats
import skfuzzy as fuzz
from skfuzzy import control as ctrl


DATA_QUALITY_CONFIG = {
    "price": {
        "min": 100e6,
        "max": 10e9,  # 10 billion max - more realistic for Indonesian market
    },
    "LT": {
        "min": 15,
        "max": 5000,
    },
    "LB": {
        "min": 10,
        "max": 2000,
    },
    "bedroom": {
        "min": 1,
        "max": 20,
    },
    "bathroom": {
        "min": 1,
        "max": 15,
    },
    "carport": {
        "min": 0,
        "max": 10,
    },
}


IQR_THRESHOLDS = {
    "price": 3.0,  # More lenient for price outliers
    "LT": 3.0,     # More lenient for land area
    "LB": 3.0,     # More lenient for building area
    "bedroom": 2.5, # More lenient for bedrooms
    "bathroom": 2.5, # More lenient for bathrooms
    "carport": 3.0,  # Much more lenient for carport
    "price_per_m2_land": 3.0,
    "price_per_m2_building": 3.0,
    "building_efficiency": 3.0,
}


Z_SCORE_THRESHOLDS = {
    "price": 3.0,
    "LT": 3.0,
    "LB": 3.0,
    "bedroom": 2.5,
    "bathroom": 2.5,
    "carport": 3.0,
    "price_per_m2_land": 3.5,
    "price_per_m2_building": 3.5,
    "building_efficiency": 3.0,
}


REQUIRED_COLUMNS = [
    "price",
    "LT",
    "LB",
    "bedroom",
    "bathroom",
    "location",
    "title",
    "updated",
]


DEFAULT_FEATURES = [
    "bedroom",
    "bathroom",
    "LT",
    "LB",
    "carport",
    "kecamatan_encoded",
    "price_per_m2_land",
    "price_per_m2_building",
    "building_efficiency",
    "listing_age_days",
]


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("preprocessing")


class PreprocessingError(Exception):

    pass


class DataValidationError(PreprocessingError):

    pass


class FeatureEngineeringError(PreprocessingError):

    pass


class DataScalingError(PreprocessingError):

    pass


def convert_to_numeric(value: Union[str, int, float, None]) -> Optional[float]:
    """
    Convert price string to numeric value.

    Args:
        value: Price value to convert, which can be a formatted string or numeric

    Returns:
        Numeric price value or None if conversion fails
    """

    if isinstance(value, (int, float)):
        return float(value)

    if value is None or pd.isna(value):
        return None

    try:

        value_str = str(value)

        if "HEMAT" in value_str:

            value_str = value_str.split("HEMAT")[0]

        value_numeric = re.sub(r"Rp\s?", "", value_str)

        if "Miliar" in value_numeric:

            cleaned_value = re.sub(r"\s?Miliar.*$", "", value_numeric).replace(",", ".")
            return float(cleaned_value) * 1e9

        elif "Juta" in value_numeric:

            cleaned_value = re.sub(r"\s?Juta.*$", "", value_numeric).replace(",", ".")
            return float(cleaned_value) * 1e6

        elif "M" in value_numeric:

            cleaned_value = value_numeric.split("M")[0].replace(",", ".")
            return float(cleaned_value) * 1e6

        elif re.match(r"^\d+(\.\d+)?$", value_numeric.strip()):
            return float(value_numeric)

        else:
            number_match = re.search(r"(\d+[.,]?\d*)", value_numeric)
            if number_match:

                extracted_number = number_match.group(1).replace(",", ".")
                if "Miliar" in value_numeric or value_numeric.endswith("B"):
                    return float(extracted_number) * 1e9
                elif "Juta" in value_numeric or value_numeric.endswith("M"):
                    return float(extracted_number) * 1e6
                return float(extracted_number)

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

        match = re.search(
            r"[LT|LB]?\s*:?\s*(\d+[\.,]?\d*)\s*m²", area_text, re.IGNORECASE
        )
        if match:
            return float(match.group(1).replace(",", "."))

        match = re.search(r"(\d+[\.,]?\d*)\s*m²", area_text)
        if match:
            return float(match.group(1).replace(",", "."))

        match = re.search(r"(\d+[\.,]?\d*)m²", area_text)
        if match:
            return float(match.group(1).replace(",", "."))

        match = re.search(r"^(\d+[\.,]?\d*)$", area_text)
        if match:
            return float(match.group(1).replace(",", "."))

        match = re.search(r"(\d+[\.,]?\d*)", area_text)
        if match:
            return float(match.group(1).replace(",", "."))

        logger.warning(f"Couldn't parse area value: {area_str}")
        return None

    except (ValueError, TypeError) as e:
        logger.warning(f"Error processing area value {area_str}: {str(e)}")
        return None


def preprocess_carport(carport_str: Any) -> int:
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


def preprocess_location(location) -> pd.Series:
    if pd.isna(location):
        return pd.Series([None, None])
    split_location = location.split(",")
    kecamatan = split_location[0].strip() if len(split_location) > 0 else None
    kabupaten_kota = split_location[1].strip() if len(split_location) > 1 else None
    return pd.Series([kecamatan, kabupaten_kota])


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str] = REQUIRED_COLUMNS,
    raise_exception: bool = False,
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


def check_data_quality(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform quality checks on the data and return flags for suspicious entries.

    Args:
        df: DataFrame to check

    Returns:
        Tuple of (processed_df, quality_report)
    """
    df_processed = df.copy()
    quality_flags = {}
    quality_stats = {}

    try:

        price_min = DATA_QUALITY_CONFIG["price"]["min"]
        price_max = DATA_QUALITY_CONFIG["price"]["max"]
        
        # Additional filter for extremely unrealistic prices (data entry errors)
        extreme_price_max = 50e9  # 50 billion - beyond any reasonable Indonesian property price
        
        invalid_prices = df_processed[
            (df_processed["price"] < price_min) | 
            (df_processed["price"] > price_max) |
            (df_processed["price"] > extreme_price_max)
        ]

        if not invalid_prices.empty:
            logger.warning(
                f"Found {len(invalid_prices)} rows with price values outside reasonable range "
                f"({price_min:,.0f} - {price_max:,.0f})"
            )

            quality_flags["unreasonable_price"] = invalid_prices.index.tolist()
            quality_stats["unreasonable_price_count"] = len(invalid_prices)

        invalid_areas = df_processed[df_processed["LB"] > df_processed["LT"] * 1.1]
        if not invalid_areas.empty:
            logger.warning(
                f"Found {len(invalid_areas)} rows where building area (LB) > land area (LT)"
            )

            quality_flags["invalid_area_ratio"] = invalid_areas.index.tolist()
            quality_stats["invalid_area_ratio_count"] = len(invalid_areas)

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

            quality_flags["too_many_rooms"] = invalid_rooms.index.tolist()
            quality_stats["too_many_rooms_count"] = len(invalid_rooms)

        missing_values = df_processed[REQUIRED_COLUMNS].isna().any(axis=1)
        missing_entries = df_processed[missing_values]
        if not missing_entries.empty:
            logger.warning(f"Found {len(missing_entries)} rows with missing values")
            quality_flags["missing_values"] = missing_entries.index.tolist()
            quality_stats["missing_values_count"] = len(missing_entries)

        total_checks = 4
        passed_checks = total_checks - len(quality_flags)
        quality_score = (passed_checks / total_checks) * 100
        quality_stats["quality_score"] = quality_score

        logger.info(
            f"Data quality check completed. Quality score: {quality_score:.2f}%"
        )

    except Exception as e:
        logger.error(f"Error during data quality check: {str(e)}")
        quality_stats["error"] = str(e)

    quality_report = {"flags": quality_flags, "stats": quality_stats}

    return df_processed, quality_report


def remove_outliers(
    df: pd.DataFrame, columns: List[str], method: str = "both"
) -> pd.DataFrame:
    df_clean = df.copy()
    original_count = len(df_clean)
    
    # Safety check: don't remove more than 80% of data
    max_removable = int(original_count * 0.8)

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

                before_removal = len(df_clean)
                temp_df = df_clean[
                    (df_clean[column] >= lower_bound)
                    & (df_clean[column] <= upper_bound)
                ]
                after_removal = len(temp_df)
                removed = before_removal - after_removal
                
                # Safety check: if this would remove too much data, skip this column
                total_removed = original_count - after_removal
                if total_removed > max_removable:
                    logger.warning(f"Outlier removal for {column} would remove too much data ({total_removed}/{original_count}), skipping this column")
                    outliers_count = 0  # Don't count as removed since we're skipping
                else:
                    df_clean = temp_df  # Apply the outlier removal
                    outliers_count = removed

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


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_processed = df.copy()

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

        df_processed["room_density"] = df_processed.apply(
            lambda row: (
                (row["bedroom"] + row["bathroom"]) / row["LB"]
                if row["LB"] > 0
                else np.nan
            ),
            axis=1,
        )

        df_processed["total_rooms"] = df_processed["bedroom"] + df_processed["bathroom"]

        df_processed["bathroom_bedroom_ratio"] = df_processed.apply(
            lambda row: (
                row["bathroom"] / row["bedroom"] if row["bedroom"] > 0 else np.nan
            ),
            axis=1,
        )

        numeric_cols = df_processed.select_dtypes(include=["float64", "int64"]).columns
        for col in numeric_cols:
            if df_processed[col].isna().any():
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())

        logger.info(f"Feature engineering completed, added 7 new features")

        return df_processed

    except Exception as e:
        logger.error(f"Error during feature engineering: {str(e)}")
        raise FeatureEngineeringError(f"Feature engineering failed: {str(e)}")


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
    encoding_method: str = "label",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df_processed = df.copy()
    encoders = {}

    if categorical_cols is None:
        categorical_cols = [
            col
            for col in df.select_dtypes(include=["object", "category"]).columns
            if col != "title" and col != "url" and col != "description"
        ]

    if not categorical_cols:
        logger.info("No categorical columns to encode")
        return df_processed, encoders

    logger.info(
        f"Encoding {len(categorical_cols)} categorical columns using {encoding_method}"
    )

    for col in categorical_cols:
        if col not in df_processed.columns:
            logger.warning(f"Column {col} not found in DataFrame, skipping")
            continue

        df_processed[col] = df_processed[col].fillna("Unknown")

        if encoding_method == "label":

            le = LabelEncoder()
            df_processed[f"{col}_encoded"] = le.fit_transform(df_processed[col])
            encoders[col] = le

            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            logger.info(f"Label encoded {col} with {len(mapping)} unique values")

        elif encoding_method == "onehot":

            ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
            encoded = ohe.fit_transform(df_processed[[col]])

            encoded_df = pd.DataFrame(
                encoded,
                columns=[f"{col}_{cat}" for cat in ohe.categories_[0]],
                index=df_processed.index,
            )

            df_processed = pd.concat([df_processed, encoded_df], axis=1)
            encoders[col] = ohe

            logger.info(f"One-hot encoded {col} with {encoded_df.shape[1]} categories")

        elif encoding_method == "frequency":

            freq_encoding = df_processed[col].value_counts(normalize=True).to_dict()
            df_processed[f"{col}_freq"] = df_processed[col].map(freq_encoding)
            encoders[col] = freq_encoding

            logger.info(
                f"Frequency encoded {col} with {len(freq_encoding)} unique values"
            )

        else:
            logger.warning(f"Unknown encoding method: {encoding_method}")

    return df_processed, encoders


def select_features(
    X: pd.DataFrame, y: pd.Series, method: str = "f_regression", k: Optional[int] = None
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


class FuzzyLogicError(PreprocessingError):

    pass


def create_fuzzy_systems() -> Dict[str, Any]:
    fuzzy_systems = {}

    try:

        logger.info("Creating price fuzzy system...")

        price = ctrl.Antecedent(np.linspace(0, 15000000000, 100), "price")
        price["very_low"] = fuzz.trapmf(price.universe, [0, 0, 500000000, 1000000000])
        price["low"] = fuzz.trimf(price.universe, [500000000, 1200000000, 2000000000])
        price["medium"] = fuzz.trimf(
            price.universe, [1500000000, 3000000000, 5000000000]
        )
        price["high"] = fuzz.trimf(
            price.universe, [4000000000, 7000000000, 10000000000]
        )
        price["very_high"] = fuzz.trapmf(
            price.universe, [8000000000, 12000000000, 15000000000, 15000000000]
        )

        fuzzy_price_value = ctrl.Consequent(np.linspace(0, 100, 100), "price_value")
        fuzzy_price_value["very_low"] = fuzz.trapmf(
            fuzzy_price_value.universe, [0, 0, 20, 30]
        )
        fuzzy_price_value["low"] = fuzz.trimf(fuzzy_price_value.universe, [20, 35, 50])
        fuzzy_price_value["medium"] = fuzz.trimf(
            fuzzy_price_value.universe, [40, 50, 70]
        )
        fuzzy_price_value["high"] = fuzz.trimf(fuzzy_price_value.universe, [60, 75, 90])
        fuzzy_price_value["very_high"] = fuzz.trapmf(
            fuzzy_price_value.universe, [80, 90, 100, 100]
        )

        price_rules = [
            ctrl.Rule(price["very_low"], fuzzy_price_value["very_low"]),
            ctrl.Rule(price["low"], fuzzy_price_value["low"]),
            ctrl.Rule(price["medium"], fuzzy_price_value["medium"]),
            ctrl.Rule(price["high"], fuzzy_price_value["high"]),
            ctrl.Rule(price["very_high"], fuzzy_price_value["very_high"]),
        ]

        logger.info("Creating price control system...")
        try:
            price_ctrl = ctrl.ControlSystem(price_rules)
            fuzzy_systems["price"] = ctrl.ControlSystemSimulation(price_ctrl)
        except Exception as e:
            logger.error(f"Error creating price fuzzy system: {str(e)}")

        logger.info("Creating room quality fuzzy system...")
        bedroom = ctrl.Antecedent(np.arange(0, 12, 0.5), "bedroom")
        bedroom["few"] = fuzz.trapmf(bedroom.universe, [0, 0, 1, 2.5])
        bedroom["standard"] = fuzz.trimf(bedroom.universe, [1.5, 3, 5])
        bedroom["many"] = fuzz.trapmf(bedroom.universe, [4, 6, 11, 11])

        bathroom = ctrl.Antecedent(np.arange(0, 12, 0.5), "bathroom")
        bathroom["few"] = fuzz.trapmf(bathroom.universe, [0, 0, 1, 2])
        bathroom["standard"] = fuzz.trimf(bathroom.universe, [1, 2.5, 4])
        bathroom["many"] = fuzz.trapmf(bathroom.universe, [3, 5, 11, 11])

        room_quality = ctrl.Consequent(np.linspace(0, 100, 100), "room_quality")
        room_quality["poor"] = fuzz.trapmf(room_quality.universe, [0, 0, 25, 40])
        room_quality["standard"] = fuzz.trimf(room_quality.universe, [30, 50, 70])
        room_quality["excellent"] = fuzz.trapmf(
            room_quality.universe, [60, 80, 100, 100]
        )

        room_rules = [
            ctrl.Rule(bedroom["few"] & bathroom["few"], room_quality["poor"]),
            ctrl.Rule(bedroom["few"] & bathroom["standard"], room_quality["standard"]),
            ctrl.Rule(bedroom["few"] & bathroom["many"], room_quality["standard"]),
            ctrl.Rule(bedroom["standard"] & bathroom["few"], room_quality["standard"]),
            ctrl.Rule(
                bedroom["standard"] & bathroom["standard"], room_quality["standard"]
            ),
            ctrl.Rule(
                bedroom["standard"] & bathroom["many"], room_quality["excellent"]
            ),
            ctrl.Rule(bedroom["many"] & bathroom["few"], room_quality["standard"]),
            ctrl.Rule(
                bedroom["many"] & bathroom["standard"], room_quality["excellent"]
            ),
            ctrl.Rule(bedroom["many"] & bathroom["many"], room_quality["excellent"]),
        ]

        try:
            room_ctrl = ctrl.ControlSystem(room_rules)
            fuzzy_systems["room"] = ctrl.ControlSystemSimulation(room_ctrl)
            logger.info("Room quality fuzzy system created successfully")
        except Exception as e:
            logger.error(f"Error creating room quality fuzzy system: {str(e)}")

        logger.info("Creating area quality fuzzy system...")
        lt = ctrl.Antecedent(np.linspace(0, 2000, 100), "lt")
        lt["very_small"] = fuzz.trapmf(lt.universe, [0, 0, 60, 110])
        lt["small"] = fuzz.trimf(lt.universe, [80, 150, 250])
        lt["medium"] = fuzz.trimf(lt.universe, [200, 350, 600])
        lt["large"] = fuzz.trimf(lt.universe, [450, 750, 1200])
        lt["very_large"] = fuzz.trapmf(lt.universe, [900, 1500, 2000, 2000])

        lb = ctrl.Antecedent(np.linspace(0, 1000, 100), "lb")
        lb["very_small"] = fuzz.trapmf(lb.universe, [0, 0, 40, 80])
        lb["small"] = fuzz.trimf(lb.universe, [60, 100, 170])
        lb["medium"] = fuzz.trimf(lb.universe, [140, 220, 380])
        lb["large"] = fuzz.trimf(lb.universe, [320, 480, 650])
        lb["very_large"] = fuzz.trapmf(lb.universe, [550, 750, 1000, 1000])

        area_quality = ctrl.Consequent(np.linspace(0, 100, 100), "area_quality")
        area_quality["poor"] = fuzz.trapmf(area_quality.universe, [0, 0, 20, 40])
        area_quality["standard"] = fuzz.trimf(area_quality.universe, [30, 50, 70])
        area_quality["excellent"] = fuzz.trapmf(
            area_quality.universe, [60, 80, 100, 100]
        )

        area_rules = [
            ctrl.Rule(lt["very_small"] & lb["very_small"], area_quality["poor"]),
            ctrl.Rule(lt["very_small"] & lb["small"], area_quality["poor"]),
            ctrl.Rule(lt["small"] & lb["very_small"], area_quality["poor"]),
            ctrl.Rule(lt["small"] & lb["small"], area_quality["standard"]),
            ctrl.Rule(lt["small"] & lb["medium"], area_quality["standard"]),
            ctrl.Rule(lt["medium"] & lb["small"], area_quality["standard"]),
            ctrl.Rule(lt["medium"] & lb["medium"], area_quality["standard"]),
            ctrl.Rule(lt["medium"] & lb["large"], area_quality["excellent"]),
            ctrl.Rule(lt["large"] & lb["medium"], area_quality["excellent"]),
            ctrl.Rule(lt["large"] & lb["large"], area_quality["excellent"]),
            ctrl.Rule(lt["very_large"] & lb["large"], area_quality["excellent"]),
            ctrl.Rule(lt["very_large"] & lb["very_large"], area_quality["excellent"]),
        ]

        try:
            area_ctrl = ctrl.ControlSystem(area_rules)
            fuzzy_systems["area"] = ctrl.ControlSystemSimulation(area_ctrl)
            logger.info("Area quality fuzzy system created successfully")
        except Exception as e:
            logger.error(f"Error creating area quality fuzzy system: {str(e)}")

        logger.info("Creating carport evaluation fuzzy system...")
        try:
            carport = ctrl.Antecedent(np.arange(0, 7, 1), "carport")
            carport["none"] = fuzz.trimf(carport.universe, [0, 0, 1])
            carport["few"] = fuzz.trimf(carport.universe, [0, 1, 3])
            carport["many"] = fuzz.trapmf(carport.universe, [2, 3, 6, 6])

            carport_quality = ctrl.Consequent(
                np.linspace(0, 100, 100), "carport_quality"
            )
            carport_quality["poor"] = fuzz.trapmf(
                carport_quality.universe, [0, 0, 20, 40]
            )
            carport_quality["average"] = fuzz.trimf(
                carport_quality.universe, [30, 50, 70]
            )
            carport_quality["excellent"] = fuzz.trapmf(
                carport_quality.universe, [60, 80, 100, 100]
            )

            carport_rules = [
                ctrl.Rule(carport["none"], carport_quality["poor"]),
                ctrl.Rule(carport["few"], carport_quality["average"]),
                ctrl.Rule(carport["many"], carport_quality["excellent"]),
            ]

            carport_ctrl = ctrl.ControlSystem(carport_rules)
            fuzzy_systems["carport"] = ctrl.ControlSystemSimulation(carport_ctrl)
            logger.info("Carport evaluation fuzzy system created successfully")
        except Exception as e:
            logger.error(f"Error creating carport fuzzy system: {str(e)}")

        return fuzzy_systems

    except Exception as e:
        logger.error(f"Error creating fuzzy systems: {str(e)}")
        raise FuzzyLogicError(f"Failed to create fuzzy systems: {str(e)}")


def validate_fuzzy_features(df: pd.DataFrame) -> Tuple[bool, str]:
    required_features = ["price", "bedroom", "bathroom", "LT", "LB"]
    missing_features = [feat for feat in required_features if feat not in df.columns]

    if missing_features:
        return (
            False,
            f"Missing required features for fuzzy logic: {', '.join(missing_features)}",
        )

    numeric_errors = []

    if "price" in df.columns and df["price"].notna().any():
        if df["price"].max() > 50000000000:
            numeric_errors.append("price has unreasonably high values")

    for feat in ["bedroom", "bathroom"]:
        if feat in df.columns and df[feat].notna().any():
            if df[feat].max() > 20:
                numeric_errors.append(f"{feat} has unreasonably high values")

    if numeric_errors:
        return False, f"Invalid numeric ranges: {', '.join(numeric_errors)}"

    return True, "Fuzzy feature validation passed"


def apply_fuzzy_logic(df: pd.DataFrame) -> pd.DataFrame:
    df_fuzzy = df.copy()

    try:

        is_valid, message = validate_fuzzy_features(df_fuzzy)
        if not is_valid:
            logger.warning(f"Fuzzy logic validation failed: {message}")

        logger.info("Applying fuzzy logic to preprocess data features...")

        fuzzy_systems = create_fuzzy_systems()

        fuzzy_price_values = []
        fuzzy_room_values = []
        fuzzy_area_values = []

        for _, row in df_fuzzy.iterrows():

            if "price" in row and pd.notna(row["price"]):
                try:
                    price_system = fuzzy_systems["price"]
                    price_system.input["price"] = min(row["price"], 20000000000 - 1)
                    price_system.compute()
                    fuzzy_price_values.append(price_system.output["price_value"])
                except Exception as e:
                    logger.warning(f"Failed to apply price fuzzy logic: {str(e)}")
                    fuzzy_price_values.append(np.nan)
            else:
                fuzzy_price_values.append(np.nan)

            if all(
                feature in row and pd.notna(row[feature])
                for feature in ["bedroom", "bathroom"]
            ):
                try:
                    room_system = fuzzy_systems["room"]
                    room_system.input["bedroom"] = min(row["bedroom"], 10)
                    room_system.input["bathroom"] = min(row["bathroom"], 10)
                    room_system.compute()
                    fuzzy_room_values.append(room_system.output["room_quality"])
                except Exception as e:
                    logger.warning(f"Failed to apply room fuzzy logic: {str(e)}")
                    fuzzy_room_values.append(np.nan)
            else:
                fuzzy_room_values.append(np.nan)

            if all(
                feature in row and pd.notna(row[feature]) for feature in ["LT", "LB"]
            ):
                try:
                    area_system = fuzzy_systems["area"]
                    area_system.input["lt"] = min(row["LT"], 2000)
                    area_system.input["lb"] = min(row["LB"], 1000)
                    area_system.compute()
                    fuzzy_area_values.append(area_system.output["area_quality"])
                except Exception as e:
                    logger.warning(f"Failed to apply area fuzzy logic: {str(e)}")
                    fuzzy_area_values.append(np.nan)
            else:
                fuzzy_area_values.append(np.nan)

            fuzzy_carport_values = []
            if (
                "carport" in fuzzy_systems
                and "carport" in row
                and pd.notna(row["carport"])
            ):
                try:
                    carport_system = fuzzy_systems["carport"]
                    carport_system.input["carport"] = min(row["carport"], 6)
                    carport_system.compute()
                    fuzzy_carport_values.append(
                        carport_system.output["carport_quality"]
                    )
                except Exception as e:
                    logger.warning(f"Failed to apply carport fuzzy logic: {str(e)}")
                    fuzzy_carport_values.append(np.nan)
            else:
                fuzzy_carport_values.append(np.nan)

        df_fuzzy["fuzzy_area_quality"] = fuzzy_area_values

        for col in ["fuzzy_price_value", "fuzzy_room_quality", "fuzzy_area_quality"]:
            if col in df_fuzzy.columns:
                df_fuzzy[col] = df_fuzzy[col] / 100.0

        quality_cols = [
            col
            for col in ["fuzzy_price_value", "fuzzy_room_quality", "fuzzy_area_quality"]
            if col in df_fuzzy.columns
        ]

        if quality_cols:
            df_fuzzy["fuzzy_quality_score"] = df_fuzzy[quality_cols].mean(axis=1)
            logger.info(
                f"Added fuzzy quality score based on {len(quality_cols)} fuzzy metrics"
            )

        if "building_efficiency" in df_fuzzy.columns:

            df_fuzzy["efficiency_category"] = pd.cut(
                df_fuzzy["building_efficiency"],
                bins=[0, 0.3, 0.5, 0.7, 1.0, float("inf")],
                labels=["very_low", "low", "medium", "high", "very_high"],
            )

        if all(
            col in df_fuzzy.columns
            for col in ["fuzzy_price_value", "fuzzy_room_quality", "fuzzy_area_quality"]
        ):

            price_weight = 0.3
            room_weight = 0.35
            area_weight = 0.35

            df_fuzzy["fuzzy_value_assessment"] = (
                (1 - df_fuzzy["fuzzy_price_value"]) * price_weight
                + df_fuzzy["fuzzy_room_quality"] * room_weight
                + df_fuzzy["fuzzy_area_quality"] * area_weight
            )

        logger.info("Fuzzy logic preprocessing completed successfully")
        return df_fuzzy

    except Exception as e:
        logger.error(f"Error applying fuzzy logic: {str(e)}")

        return df_fuzzy


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


def preprocess_data(
    df: pd.DataFrame,
    validate: bool = True,
    remove_outliers_method: str = "both",
    apply_feature_engineering: bool = True,
    categorical_encoding: str = "label",
    scale_data: bool = False,
    feature_selection_method: Optional[str] = None,
    num_features: Optional[int] = None,
    required_columns: List[str] = REQUIRED_COLUMNS,
    filter_city: Optional[str] = None,
    output_dir: str = "reports",
    save_artifacts: bool = True,
    apply_fuzzy: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    preprocessing_metadata = {
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original_shape": df.shape,
        "steps_performed": [],
        "quality_report": {},
        "artifacts": {},
    }

    if save_artifacts:
        os.makedirs(output_dir, exist_ok=True)

    if validate:
        logger.info("Validating input data...")
        preprocessing_metadata["steps_performed"].append("validation")

        is_valid, error_message = validate_dataframe(
            df, required_columns, raise_exception=False
        )
        if not is_valid:
            logger.error(f"Data validation failed: {error_message}")
            preprocessing_metadata["validation_error"] = error_message
            if validate:
                raise DataValidationError(error_message)

    logger.info(f"Starting preprocessing of {len(df)} rows of data")
    df_processed = df.copy()

    try:
        logger.info("Converting data types...")
        preprocessing_metadata["steps_performed"].append("type_conversion")

        df_processed["price"] = df_processed["price"].apply(convert_to_numeric)
        df_processed["LT"] = df_processed["LT"].apply(preprocess_area)
        df_processed["LB"] = df_processed["LB"].apply(preprocess_area)
        df_processed["updated"] = df_processed["updated"].apply(preprocess_updated)

        df_processed[["kecamatan", "kabupaten_kota"]] = df_processed["location"].apply(
            preprocess_location
        )

        if "carport" in df_processed.columns:
            df_processed["carport"] = df_processed["carport"].apply(preprocess_carport)

        df_processed["updated"] = pd.to_datetime(df_processed["updated"])

    except Exception as e:
        logger.error(f"Error during type conversion: {str(e)}")
        preprocessing_metadata["type_conversion_error"] = str(e)
        raise PreprocessingError(f"Type conversion failed: {str(e)}")

    try:
        logger.info("Cleaning data...")
        preprocessing_metadata["steps_performed"].append("data_cleaning")

        initial_count = len(df_processed)
        if "title" in df_processed.columns:
            df_processed = df_processed[
                ~df_processed["title"].str.contains(
                    "hotel|kost|Kos", case=False, na=False
                )
            ]
            non_residential_removed = initial_count - len(df_processed)
            logger.info(f"Removed {non_residential_removed} non-residential properties")
            preprocessing_metadata["non_residential_removed"] = non_residential_removed

        initial_count = len(df_processed)
        df_processed = df_processed.drop_duplicates(subset="title", keep="first")
        duplicates_removed = initial_count - len(df_processed)
        logger.info(f"Removed {duplicates_removed} duplicate listings")
        preprocessing_metadata["duplicates_removed"] = duplicates_removed

        initial_count = len(df_processed)
        sixty_days_ago = datetime.today() - timedelta(days=60)
        df_processed = df_processed[df_processed["updated"] >= sixty_days_ago]
        old_listings_removed = initial_count - len(df_processed)
        logger.info(f"Removed {old_listings_removed} listings older than 60 days")
        preprocessing_metadata["old_listings_removed"] = old_listings_removed

        if filter_city is not None and "kabupaten_kota" in df_processed.columns:
            initial_count = len(df_processed)
            df_processed = df_processed[df_processed["kabupaten_kota"] == filter_city]
            city_filtered = initial_count - len(df_processed)
            logger.info(f"Filtered to {len(df_processed)} listings in {filter_city}")
            preprocessing_metadata["city_filtered"] = {
                "city": filter_city,
                "listings_retained": len(df_processed),
            }

    except Exception as e:
        logger.error(f"Error during data cleaning: {str(e)}")
        preprocessing_metadata["data_cleaning_error"] = str(e)
        raise PreprocessingError(f"Data cleaning failed: {str(e)}")

    try:
        logger.info("Checking data quality...")
        preprocessing_metadata["steps_performed"].append("quality_check")

        df_processed, quality_report = check_data_quality(df_processed)
        preprocessing_metadata["quality_report"] = quality_report

        for col in ["price", "LT", "LB", "bedroom", "bathroom"]:
            if col in df_processed.columns and df_processed[col].isna().any():
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                logger.info(
                    f"Filled {df_processed[col].isna().sum()} missing values in {col} with median"
                )

        initial_count = len(df_processed)
        critical_columns = ["price", "LT", "LB", "bedroom", "bathroom"]
        df_processed = df_processed.dropna(
            subset=[col for col in critical_columns if col in df_processed.columns]
        )
        missing_removed = initial_count - len(df_processed)
        logger.info(
            f"Removed {missing_removed} rows with missing values in critical columns"
        )
        preprocessing_metadata["missing_values_removed"] = missing_removed

    except Exception as e:
        logger.error(f"Error during data quality check: {str(e)}")
        preprocessing_metadata["quality_check_error"] = str(e)
        raise PreprocessingError(f"Data quality check failed: {str(e)}")

    if remove_outliers_method:
        try:
            logger.info(f"Removing outliers using {remove_outliers_method} method...")
            preprocessing_metadata["steps_performed"].append("outlier_removal")

            columns_to_check = [
                col
                for col in ["price", "LT", "LB", "bedroom", "bathroom", "carport"]
                if col in df_processed.columns
            ]

            initial_count = len(df_processed)
            df_processed = remove_outliers(
                df_processed, columns_to_check, method=remove_outliers_method
            )
            outliers_removed = initial_count - len(df_processed)

            preprocessing_metadata["outliers_removed"] = {
                "method": remove_outliers_method,
                "count": outliers_removed,
                "percentage": (
                    (outliers_removed / initial_count * 100) if initial_count > 0 else 0
                ),
            }

        except Exception as e:
            logger.error(f"Error during outlier removal: {str(e)}")
            preprocessing_metadata["outlier_removal_error"] = str(e)

    if apply_feature_engineering:
        try:
            logger.info("Performing feature engineering...")
            preprocessing_metadata["steps_performed"].append("feature_engineering")

            df_processed = engineer_features(df_processed)
            preprocessing_metadata["engineered_features"] = [
                "price_per_m2_land",
                "price_per_m2_building",
                "building_efficiency",
                "listing_age_days",
                "room_density",
                "total_rooms",
                "bathroom_bedroom_ratio",
            ]

        except Exception as e:
            logger.error(f"Error during feature engineering: {str(e)}")
            preprocessing_metadata["feature_engineering_error"] = str(e)
            raise FeatureEngineeringError(f"Feature engineering failed: {str(e)}")

    if apply_fuzzy:
        try:
            logger.info("Applying fuzzy logic preprocessing...")
            preprocessing_metadata["steps_performed"].append("fuzzy_logic")

            df_processed = apply_fuzzy_logic(df_processed)

            fuzzy_features = [
                col for col in df_processed.columns if col.startswith("fuzzy_")
            ]
            preprocessing_metadata["fuzzy_features"] = fuzzy_features

            logger.info(f"Added {len(fuzzy_features)} fuzzy features")
        except Exception as e:
            logger.error(f"Error during fuzzy logic application: {str(e)}")
            preprocessing_metadata["fuzzy_logic_error"] = str(e)

    try:
        logger.info(
            f"Encoding categorical features using {categorical_encoding} method..."
        )
        preprocessing_metadata["steps_performed"].append("categorical_encoding")

        categorical_cols = [
            col
            for col in df_processed.select_dtypes(include=["object"]).columns
            if col in ["kecamatan", "kabupaten_kota"] or "type" in col.lower()
        ]

        df_processed, encoders = encode_categorical_features(
            df_processed, categorical_cols, encoding_method=categorical_encoding
        )

        if save_artifacts and encoders:
            try:

                encoder_path = os.path.join(output_dir, "encoder_mappings.json")

                serializable_encoders = {}
                for col, encoder in encoders.items():
                    if isinstance(encoder, LabelEncoder):
                        serializable_encoders[col] = {
                            "type": "label",
                            "mapping": dict(
                                zip(
                                    encoder.classes_.tolist(),
                                    encoder.transform(encoder.classes_).tolist(),
                                )
                            ),
                        }
                    elif isinstance(encoder, dict):
                        serializable_encoders[col] = {
                            "type": "frequency",
                            "mapping": encoder,
                        }

                with open(encoder_path, "w") as f:
                    json.dump(serializable_encoders, f, indent=4, default=str)
                logger.info(f"Saved encoder mappings to {encoder_path}")
                preprocessing_metadata["artifacts"]["encoders"] = encoder_path
            except Exception as e:
                logger.warning(f"Failed to save encoder mappings: {str(e)}")

    except Exception as e:
        logger.error(f"Error during categorical encoding: {str(e)}")
        preprocessing_metadata["categorical_encoding_error"] = str(e)

    if scale_data:
        try:
            logger.info("Scaling numeric features...")
            preprocessing_metadata["steps_performed"].append("feature_scaling")

            price_cols = [col for col in df_processed.columns if "price" in col.lower()]
            numeric_cols = df_processed.select_dtypes(
                include=["float64", "int64"]
            ).columns
            numeric_cols = [col for col in numeric_cols if col not in price_cols]

            df_processed, scalers = scale_features(
                df_processed, price_cols, numeric_cols
            )

            if save_artifacts:
                try:
                    import pickle

                    scaler_path = os.path.join(output_dir, "scalers.pkl")
                    with open(scaler_path, "wb") as f:
                        pickle.dump(scalers, f)
                    logger.info(f"Saved scalers to {scaler_path}")
                    preprocessing_metadata["artifacts"]["scalers"] = scaler_path
                except Exception as e:
                    logger.warning(f"Failed to save scalers: {str(e)}")

        except Exception as e:
            logger.error(f"Error during feature scaling: {str(e)}")
            preprocessing_metadata["scaling_error"] = str(e)

    selected_features = None
    if feature_selection_method and df_processed.shape[1] > 5:
        try:
            logger.info(
                f"Selecting features using {feature_selection_method} method..."
            )
            preprocessing_metadata["steps_performed"].append("feature_selection")

            if "price" in df_processed.columns:
                y = df_processed["price"]
                X = df_processed.drop(columns=["price"])

                feature_cols = list(
                    X.select_dtypes(include=["int64", "float64"]).columns
                )

                excluded_cols = ["title", "url", "description", "location", "updated"]
                feature_cols = [
                    col
                    for col in feature_cols
                    if not any(excl in col for excl in excluded_cols)
                ]

                if feature_cols:
                    X_features = X[feature_cols]

                    X_selected, selected_features = select_features(
                        X_features, y, method=feature_selection_method, k=num_features
                    )

                    preprocessing_metadata["selected_features"] = selected_features

                    if save_artifacts:
                        try:
                            feature_info_path = os.path.join(
                                output_dir, "feature_selection.json"
                            )
                            feature_info = {
                                "method": feature_selection_method,
                                "num_features": len(selected_features),
                                "selected_features": selected_features,
                            }
                            with open(feature_info_path, "w") as f:
                                json.dump(feature_info, f, indent=4)
                            logger.info(
                                f"Saved feature selection info to {feature_info_path}"
                            )
                            preprocessing_metadata["artifacts"][
                                "feature_selection"
                            ] = feature_info_path
                        except Exception as e:
                            logger.warning(
                                f"Failed to save feature selection info: {str(e)}"
                            )
                else:
                    logger.warning("No suitable features found for feature selection")
                    preprocessing_metadata["feature_selection_warning"] = (
                        "No suitable features found"
                    )
            else:
                logger.warning(
                    "Target variable 'price' not found, skipping feature selection"
                )
                preprocessing_metadata["feature_selection_warning"] = (
                    "Target variable not found"
                )

        except Exception as e:
            logger.error(f"Error during feature selection: {str(e)}")
            preprocessing_metadata["feature_selection_error"] = str(e)

    try:

        preprocessing_metadata["final_shape"] = df_processed.shape
        preprocessing_metadata["columns"] = list(df_processed.columns)

        numeric_columns = df_processed.select_dtypes(
            include=["int64", "float64"]
        ).columns
        stats = {}
        for col in numeric_columns:
            col_stats = {
                "mean": float(df_processed[col].mean()),
                "median": float(df_processed[col].median()),
                "min": float(df_processed[col].min()),
                "max": float(df_processed[col].max()),
                "std": float(df_processed[col].std()),
            }
            stats[col] = col_stats

        preprocessing_metadata["numeric_stats"] = stats

        categorical_columns = df_processed.select_dtypes(
            include=["object", "category"]
        ).columns
        cat_stats = {}
        for col in categorical_columns:
            if col not in ["title", "url", "description"]:
                value_counts = df_processed[col].value_counts()
                if not value_counts.empty:
                    most_common = value_counts.index[0]
                    cat_stats[col] = {
                        "unique_values": len(value_counts),
                        "most_common": most_common,
                        "most_common_count": int(value_counts.iloc[0]),
                    }

        preprocessing_metadata["categorical_stats"] = cat_stats

        if save_artifacts:
            try:
                report_path = os.path.join(output_dir, "preprocessing_report.json")
                with open(report_path, "w") as f:
                    json.dump(preprocessing_metadata, f, indent=4, default=str)
                logger.info(f"Saved preprocessing report to {report_path}")
            except Exception as e:
                logger.warning(f"Failed to save preprocessing report: {str(e)}")

    except Exception as e:
        logger.error(f"Error generating final data quality report: {str(e)}")
        preprocessing_metadata["report_generation_error"] = str(e)

    try:

        if len(df_processed) < 10:
            warning_msg = f"Very small dataset: only {len(df_processed)} samples after preprocessing"
            logger.warning(warning_msg)
            preprocessing_metadata["warnings"] = preprocessing_metadata.get(
                "warnings", []
            ) + [warning_msg]

        missing_counts = df_processed.isna().sum()
        if missing_counts.sum() > 0:
            columns_with_missing = [
                col
                for col in df_processed.columns
                if df_processed[col].isna().sum() > 0
            ]
            warning_msg = f"Final dataset still has missing values in columns: {columns_with_missing}"
            logger.warning(warning_msg)
            preprocessing_metadata["warnings"] = preprocessing_metadata.get(
                "warnings", []
            ) + [warning_msg]

            for col in df_processed.columns:
                if df_processed[col].isna().any():
                    if df_processed[col].dtype in ["int64", "float64"]:
                        df_processed[col] = df_processed[col].fillna(
                            df_processed[col].median()
                        )
                    else:
                        df_processed[col] = df_processed[col].fillna("Unknown")

            logger.info("Filled remaining missing values in final dataset")

        required_feature_cols = ["bedroom", "bathroom", "LT", "LB"]
        missing_features = [
            col for col in required_feature_cols if col not in df_processed.columns
        ]
        if missing_features:
            warning_msg = f"Missing essential feature columns: {missing_features}"
            logger.warning(warning_msg)
            preprocessing_metadata["warnings"] = preprocessing_metadata.get(
                "warnings", []
            ) + [warning_msg]

    except Exception as e:
        logger.error(f"Error during final validation: {str(e)}")
        preprocessing_metadata["validation_error"] = str(e)

    preprocessing_metadata["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    preprocessing_metadata["rows_removed"] = len(df) - len(df_processed)
    preprocessing_metadata["rows_removed_percentage"] = (
        (len(df) - len(df_processed)) / len(df) * 100 if len(df) > 0 else 0
    )

    logger.info(f"Preprocessing completed. Final dataset shape: {df_processed.shape}")

    if selected_features:

        logger.info(
            f"Using {len(selected_features)} selected features for the final dataset"
        )
        feature_columns = selected_features

        if "price" in df_processed.columns and "price" not in feature_columns:
            feature_columns = ["price"] + feature_columns
    else:

        fuzzy_cols = [col for col in df_processed.columns if col.startswith("fuzzy_")]
        extended_defaults = DEFAULT_FEATURES + fuzzy_cols

        available_features = [
            col for col in extended_defaults if col in df_processed.columns
        ]
        if len(available_features) >= 4:
            feature_columns = available_features
            logger.info(
                f"Using {len(feature_columns)} default features for the final dataset"
            )
        else:

            feature_columns = list(
                df_processed.select_dtypes(include=["int64", "float64"]).columns
            )

            excluded_cols = ["title", "url", "description"]
            feature_columns = [
                col
                for col in feature_columns
                if not any(excl in col for excl in excluded_cols)
            ]
            logger.info(
                f"Using {len(feature_columns)} available features for the final dataset"
            )

    if "price" in df_processed.columns and "price" not in feature_columns:
        feature_columns = ["price"] + feature_columns

    logger.info(f"Final features: {', '.join(feature_columns)}")
    preprocessing_metadata["final_features"] = feature_columns

    return df_processed, preprocessing_metadata


def preprocess_data_legacy(df):

    logger.warning(
        "Using legacy preprocessing function. Consider updating to the new API."
    )
    preprocessed_df, _ = preprocess_data(
        df,
        validate=True,
        remove_outliers_method="both",
        apply_feature_engineering=True,
        categorical_encoding="label",
        scale_data=False,
        feature_selection_method=None,
    )
    return preprocessed_df
