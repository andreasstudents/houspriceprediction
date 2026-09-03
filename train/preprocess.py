import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
import logging
from colorama import Fore, Back, Style, init
from datetime import datetime


init()


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


from utils.preprocessing import (
    preprocess_data,
    DataValidationError,
    FeatureEngineeringError,
    PreprocessingError,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(parent_dir, "logs/preprocessing.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("preprocessing")


output_dirs = [
    os.path.join(parent_dir, "model"),
    os.path.join(parent_dir, "logs"),
    os.path.join(parent_dir, "processed_data"),
]

for dir_path in output_dirs:
    os.makedirs(dir_path, exist_ok=True)


def clean_numeric(val):
    """Clean numeric values from strings with better error handling."""
    try:
        if pd.isna(val):
            return None
        if not isinstance(val, str):
            return float(val)

        val = str(val)
        val = (
            val.replace("Rp", "")
            .replace("Miliar", "000000000")
            .replace("Juta", "000000")
        )
        val = val.replace("²", "").replace(" ", "")
        val = val.split("HEMAT")[0]
        val = val.split("M")[0]

        val = "".join(c for c in val if c.isdigit() or c == ".")
        if val:
            try:
                return float(val)
            except ValueError:
                return None
        return None
    except Exception as e:
        print(f"Error cleaning value '{val}': {str(e)}")
        return None


def preprocess_and_save(input_file, output_dir, city_filter=None):
    """
    Load raw data, preprocess it, and save the preprocessed data

    Args:
        input_file: Path to input CSV file
        output_dir: Directory to save preprocessed data
        city_filter: Filter data by city (e.g., "Yogyakarta")

    Returns:
        Path to the saved preprocessed data file
    """
    print(
        f"{Back.BLUE}{Fore.WHITE} House Price Data Preprocessing System {Style.RESET_ALL}"
    )
    print(f"{Fore.CYAN}Loading and preprocessing data...{Style.RESET_ALL}")

    try:

        df = pd.read_csv(input_file)
        print(f"Initial data shape: {df.shape}")

        print("Cleaning numeric data...")
        original_counts = {
            "price": df["price"].notna().sum(),
            "LT": df["LT"].notna().sum(),
            "LB": df["LB"].notna().sum(),
        }

        df["price"] = df["price"].apply(clean_numeric)
        df["LT"] = df["LT"].apply(clean_numeric)
        df["LB"] = df["LB"].apply(clean_numeric)

        cleaned_counts = {
            "price": df["price"].notna().sum(),
            "LT": df["LT"].notna().sum(),
            "LB": df["LB"].notna().sum(),
        }

        print("\nData cleaning results:")
        for col in ["price", "LT", "LB"]:
            print(
                f"{col}: {original_counts[col]} -> {cleaned_counts[col]} valid values"
            )

        print("\nUnique values after cleaning:")
        print("\nPrice values:")
        print(df["price"].value_counts().head())
        print("\nLT values:")
        print(df["LT"].value_counts().head())
        print("\nLB values:")
        print(df["LB"].value_counts().head())

        df_cleaned = df.dropna(subset=["price", "LT", "LB"])
        print(
            f"\nRows after cleaning: {len(df_cleaned)} (removed {len(df) - len(df_cleaned)} rows)"
        )

        if len(df_cleaned) == 0:
            raise PreprocessingError(
                "All data was removed during cleaning. Check the data format and cleaning logic."
            )

        df = df_cleaned

        print(f"{Fore.CYAN}Applying advanced preprocessing...{Style.RESET_ALL}")
        df_processed, preprocessing_info = preprocess_data(
            df,
            validate=True,
            remove_outliers_method="iqr",
            apply_feature_engineering=True,
            categorical_encoding="label",
            scale_data=False,
            filter_city=city_filter,
            output_dir=os.path.join(parent_dir, "reports"),
        )

        print(f"{Fore.YELLOW}Preprocessing Statistics:{Style.RESET_ALL}")
        print(f"  - Original rows: {preprocessing_info['original_shape'][0]}")
        print(f"  - Final rows: {preprocessing_info['final_shape'][0]}")
        print(
            f"  - Rows removed: {preprocessing_info['rows_removed']} ({preprocessing_info['rows_removed_percentage']:.2f}%)"
        )

        if (
            "quality_report" in preprocessing_info
            and "stats" in preprocessing_info["quality_report"]
        ):
            quality_stats = preprocessing_info["quality_report"]["stats"]
            if "quality_score" in quality_stats:
                print(f"  - Data quality score: {quality_stats['quality_score']:.2f}%")

        if "final_features" in preprocessing_info:
            feature_columns = preprocessing_info["final_features"]
            print(
                f"{Fore.CYAN}Features selected: {', '.join(feature_columns)}{Style.RESET_ALL}"
            )
        else:

            feature_columns = ["bedroom", "bathroom", "LT", "LB", "carport"]

            if "kecamatan_encoded" in df_processed.columns:
                feature_columns.append("kecamatan_encoded")

            for feature in [
                "price_per_m2_land",
                "price_per_m2_building",
                "building_efficiency",
            ]:
                if feature in df_processed.columns:
                    feature_columns.append(feature)

            print(
                f"{Fore.CYAN}Using default features: {', '.join(feature_columns)}{Style.RESET_ALL}"
            )

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"preprocessed_data_{timestamp}.pkl")

        output_data = {
            "data": df_processed,
            "feature_columns": feature_columns,
            "preprocessing_info": preprocessing_info,
            "timestamp": timestamp,
        }

        with open(output_file, "wb") as f:
            pickle.dump(output_data, f)

        csv_file = os.path.join(output_dir, f"preprocessed_data_{timestamp}.csv")
        df_processed.to_csv(csv_file, index=False)

        import json

        metadata_file = os.path.join(
            output_dir, f"preprocessing_metadata_{timestamp}.json"
        )

        metadata = preprocessing_info.copy()

        for key in metadata:
            if isinstance(metadata[key], np.ndarray):
                metadata[key] = metadata[key].tolist()
            elif isinstance(metadata[key], np.integer):
                metadata[key] = int(metadata[key])
            elif isinstance(metadata[key], np.floating):
                metadata[key] = float(metadata[key])

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4, default=str)

        print(f"{Fore.GREEN}Preprocessed data saved to: {output_file}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Preprocessed CSV saved to: {csv_file}{Style.RESET_ALL}")
        print(
            f"{Fore.GREEN}Preprocessing metadata saved to: {metadata_file}{Style.RESET_ALL}"
        )

        return output_file

    except (DataValidationError, PreprocessingError, FeatureEngineeringError) as e:
        print(f"{Fore.RED}Preprocessing error: {str(e)}{Style.RESET_ALL}")
        logger.error(f"Preprocessing failed: {str(e)}")

        print(f"{Fore.YELLOW}Falling back to basic preprocessing...{Style.RESET_ALL}")
        try:

            df_preprocessed = preprocess_data(
                df,
                validate=False,
                remove_outliers_method="iqr",
                apply_feature_engineering=False,
                categorical_encoding="label",
                scale_data=False,
                feature_selection_method=None,
                apply_fuzzy=False,
            )

            if isinstance(df_preprocessed, tuple) and len(df_preprocessed) > 0:
                df_processed = df_preprocessed[0]
                preprocessing_info = df_preprocessed[1]
            else:
                df_processed = df_preprocessed
                preprocessing_info = {"fallback": True}

            if df_processed is None or df_processed.empty:
                raise PreprocessingError("Preprocessing resulted in empty DataFrame")

            if city_filter is not None and "kabupaten_kota" in df_processed.columns:
                df_yogyakarta = df_processed[
                    df_processed["kabupaten_kota"] == city_filter
                ]
                print(
                    f"{Fore.YELLOW}Number of houses in {city_filter}: {len(df_yogyakarta)}{Style.RESET_ALL}"
                )
                df_processed = df_yogyakarta

            feature_columns = ["bedroom", "bathroom", "LT", "LB"]
            if "carport" in df_processed.columns:
                feature_columns.append("carport")
            if "kecamatan_encoded" in df_processed.columns:
                feature_columns.append("kecamatan_encoded")
            if "price" not in feature_columns:
                feature_columns = ["price"] + feature_columns

            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(
                output_dir, f"preprocessed_data_fallback_{timestamp}.pkl"
            )

            output_data = {
                "data": df_processed,
                "feature_columns": feature_columns,
                "preprocessing_info": preprocessing_info,
                "timestamp": timestamp,
                "fallback": True,
            }

            with open(output_file, "wb") as f:
                pickle.dump(output_data, f)

            csv_file = os.path.join(
                output_dir, f"preprocessed_data_fallback_{timestamp}.csv"
            )
            df_processed.to_csv(csv_file, index=False)

            print(
                f"{Fore.YELLOW}Fallback preprocessed data saved to: {output_file}{Style.RESET_ALL}"
            )
            print(f"{Fore.YELLOW}Fallback CSV saved to: {csv_file}{Style.RESET_ALL}")

            return output_file

        except Exception as e:
            logger.error(f"Error during fallback preprocessing: {str(e)}")
            raise PreprocessingError(
                f"Both standard and fallback preprocessing failed: {str(e)}"
            )


def main():
    parser = argparse.ArgumentParser(description="Preprocess house price data")
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(parent_dir, "dataset/houses.csv"),
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(parent_dir, "processed_data"),
        help="Directory to save processed data",
    )
    parser.add_argument(
        "--city",
        type=str,
        default="Yogyakarta",
        help="Filter data by city (e.g., 'Yogyakarta')",
    )

    args = parser.parse_args()

    try:
        output_file = preprocess_and_save(args.input, args.output_dir, args.city)
        print(f"{Fore.GREEN}Preprocessing completed successfully!{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Preprocessed data saved to: {output_file}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Preprocessing failed: {str(e)}{Style.RESET_ALL}")
        logger.error(f"Preprocessing failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
