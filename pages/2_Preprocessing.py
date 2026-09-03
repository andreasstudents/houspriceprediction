import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import datetime
from io import BytesIO
import pickle
import glob
from PIL import Image


st.set_page_config(page_title="Data Preprocessing", page_icon="🧹", layout="wide")


def load_latest_preprocessed_data():
    """Load the latest preprocessed data from processed_data directory."""
    csv_files = glob.glob("processed_data/preprocessed_data_*.csv")
    if not csv_files:
        st.error("No preprocessed data found. Please run preprocessing first.")
        return None, None, None

    latest_file = max(csv_files, key=os.path.getctime)

    timestamp = latest_file.split("_")[-1].split(".")[0]

    metadata_file = f"processed_data/preprocessing_metadata_20250506_140027.json"
    try:
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        metadata = {}
        st.warning(f"Metadata file not found: {metadata_file}")

    data = pd.read_csv(latest_file)

    pickle_file = f"processed_data/preprocessed_data_20250506_140027.pkl"
    pkl_data = None
    try:
        with open(pickle_file, "rb") as f:
            pkl_data = pickle.load(f)
    except FileNotFoundError:
        st.warning(f"Pickle file not found: {pickle_file}")

    return data, metadata, pkl_data


def load_raw_data():
    """
    Function to load raw data if it exists, otherwise show a message.
    """
    try:
        raw_data_path = "data/raw_data.csv"
        if os.path.exists(raw_data_path):
            return pd.read_csv(raw_data_path)
        else:
            st.info(
                "Raw data file not found. Only preprocessed data will be displayed."
            )
            return None
    except Exception as e:
        st.error(f"Error loading raw data: {e}")
        return None


st.title("🧹 Data Preprocessing")
st.markdown(
    """
This page demonstrates the preprocessing steps applied to the real estate data from Yogyakarta area.
The preprocessing transforms raw data into a format suitable for exploratory data analysis and modeling.
"""
)


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Data Overview",
        "🧼 Data Cleaning",
        "🔍 Feature Engineering",
        "📈 Data Exploration",
        "⬇️ Download Data",
    ]
)


data, metadata, pkl_data = load_latest_preprocessed_data()
raw_data = load_raw_data()

with tab1:
    st.header("Data Overview")

    if data is not None:
        st.subheader("Preprocessed Data Sample")
        st.dataframe(data.head())

        st.subheader("Data Information")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Number of rows:** {data.shape[0]}")
            st.markdown(f"**Number of columns:** {data.shape[1]}")
        with col2:
            if metadata and "preprocessing_date" in metadata:
                st.markdown(f"**Preprocessing date:** {metadata['preprocessing_date']}")

            if raw_data is not None:
                st.markdown(f"**Original data rows:** {raw_data.shape[0]}")
                st.markdown(
                    f"**Rows retained:** {data.shape[0]/raw_data.shape[0]*100:.1f}%"
                )

        st.subheader("Data Types")
        dtype_df = pd.DataFrame(
            {
                "Column": data.columns,
                "Data Type": data.dtypes.astype(str),
                "Non-Null Count": data.count().values,
                "Null Count": data.isnull().sum().values,
                "Null Percentage": (data.isnull().sum() / len(data) * 100).values.round(
                    2
                ),
            }
        )
        st.dataframe(dtype_df)

        st.subheader("Statistical Summary")
        st.write("Basic statistics for numerical columns:")
        st.dataframe(data.describe())

        st.subheader("Categorical Data Overview")
        categorical_cols = data.select_dtypes(include=["object"]).columns
        if len(categorical_cols) > 0:
            selected_cat_column = st.selectbox(
                "Select a categorical column to see its distribution:", categorical_cols
            )

            if selected_cat_column:
                cat_counts = data[selected_cat_column].value_counts()

                fig, ax = plt.subplots(figsize=(10, 6))
                cat_counts.plot(kind="bar", ax=ax)
                plt.title(f"Distribution of {selected_cat_column}")
                plt.tight_layout()
                st.pyplot(fig)

                st.write("Value counts:")
                st.dataframe(
                    cat_counts.reset_index().rename(
                        columns={
                            "index": selected_cat_column,
                            selected_cat_column: "Count",
                        }
                    )
                )
    else:
        st.warning("No data available to display.")

with tab2:
    st.header("Data Cleaning Process")

    st.subheader("Cleaning Steps Applied")

    if metadata and "cleaning_steps" in metadata:
        for idx, step in enumerate(metadata["cleaning_steps"], 1):
            st.markdown(f"**Step {idx}:** {step['description']}")
            if "details" in step:
                st.markdown(f"*Details:* {step['details']}")
            if "metrics" in step:
                metrics_df = pd.DataFrame([step["metrics"]])
                st.dataframe(metrics_df)
    else:

        cleaning_steps = [
            {
                "step": "Handle Missing Values",
                "description": "Identified and handled missing values in the dataset by either imputation or removal.",
            },
            {
                "step": "Remove Duplicates",
                "description": "Removed duplicate entries from the dataset to ensure data integrity.",
            },
            {
                "step": "Format Standardization",
                "description": "Standardized the format of columns like price, area, and location.",
            },
            {
                "step": "Outlier Detection",
                "description": "Identified and handled outliers in numerical columns.",
            },
            {
                "step": "Text Normalization",
                "description": "Normalized text in categorical columns by removing special characters and standardizing case.",
            },
        ]

        for step in cleaning_steps:
            st.markdown(f"**{step['step']}**")
            st.markdown(step["description"])

    if raw_data is not None and data is not None:
        st.subheader("Before & After Cleaning Comparison")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Raw Data**")
            st.markdown(f"Rows: {raw_data.shape[0]}")
            st.markdown(f"Columns: {raw_data.shape[1]}")
            if not raw_data.empty:
                st.markdown(f"Missing values: {raw_data.isnull().sum().sum()}")
                has_duplicates = raw_data.duplicated().any()
                st.markdown(f"Has duplicates: {'Yes' if has_duplicates else 'No'}")

        with col2:
            st.markdown("**Cleaned Data**")
            st.markdown(f"Rows: {data.shape[0]}")
            st.markdown(f"Columns: {data.shape[1]}")
            if not data.empty:
                st.markdown(f"Missing values: {data.isnull().sum().sum()}")
                has_duplicates = data.duplicated().any()
                st.markdown(f"Has duplicates: {'Yes' if has_duplicates else 'No'}")

    st.subheader("Data Quality Metrics")

    if data is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Completeness**")
            completeness = (1 - data.isnull().sum() / len(data)) * 100
            completeness_df = pd.DataFrame(
                {
                    "Column": completeness.index,
                    "Completeness (%)": completeness.values.round(2),
                }
            )
            st.dataframe(completeness_df)

        with col2:
            st.markdown("**Uniqueness**")
            uniqueness = (data.nunique() / len(data)) * 100
            uniqueness_df = pd.DataFrame(
                {
                    "Column": uniqueness.index,
                    "Uniqueness (%)": uniqueness.values.round(2),
                }
            )
            st.dataframe(uniqueness_df)
    else:
        st.warning("No data available to calculate quality metrics.")

with tab3:
    st.header("Feature Engineering")

    st.subheader("Feature Engineering Process")

    engineered_features = []
    if data is not None and raw_data is not None:
        engineered_features = list(set(data.columns) - set(raw_data.columns))

    if engineered_features:
        st.markdown("The following features were created during preprocessing:")
        for feature in engineered_features:
            st.markdown(f"- **{feature}**")

    feature_engineering_explanations = {
        "price_per_m2_land": "Price per square meter of land, calculated as price/LT.",
        "price_per_m2_building": "Price per square meter of building, calculated as price/LB.",
        "building_efficiency": "Ratio of building area to land area (LB/LT).",
        "total_rooms": "Sum of bedrooms and bathrooms.",
        "min_area_needed": "Minimum area needed based on number of people (estimated from rooms).",
        "listing_age_days": "Age of the listing in days from the latest update date.",
        "room_density": "Number of rooms per building area (total_rooms/LB).",
        "bathroom_bedroom_ratio": "Ratio of bathrooms to bedrooms.",
        "fuzzy_area_quality": "Quality score based on land and building areas.",
        "fuzzy_quality_score": "Comprehensive quality score based on multiple features.",
        "efficiency_category": "Categorization based on building efficiency ratio.",
        "kecamatan_encoded": "Encoded version of the kecamatan (district) categorical variable.",
        "kabupaten_kota_encoded": "Encoded version of the kabupaten/kota (regency/city) categorical variable.",
    }

    st.subheader("Feature Engineering Explanations")

    if data is not None:
        actual_features = [
            f for f in feature_engineering_explanations.keys() if f in data.columns
        ]

        if actual_features:
            for feature in actual_features:
                with st.expander(f"{feature}"):
                    st.markdown(feature_engineering_explanations[feature])

                    if data[feature].dtype in ["int64", "float64"]:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        sns.histplot(data[feature].dropna(), kde=True, ax=ax)
                        plt.title(f"Distribution of {feature}")
                        st.pyplot(fig)
                    elif (
                        data[feature].dtype == "object" or data[feature].nunique() < 20
                    ):
                        fig, ax = plt.subplots(figsize=(10, 4))
                        data[feature].value_counts().plot(kind="bar", ax=ax)
                        plt.title(f"Distribution of {feature}")
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
        else:
            st.info("No engineered features found in the dataset.")
    else:
        st.warning("No data available to display feature engineering examples.")

with tab4:
    st.header("Data Exploration")

    if data is not None:

        st.subheader("Interactive Data Exploration")

        st.markdown("### Filter Data")

        col1, col2, col3 = st.columns(3)

        with col1:
            min_price = int(data["price"].min())
            max_price = int(data["price"].max())
            price_range = st.slider(
                "Price Range (Rp)", min_price, max_price, (min_price, max_price)
            )

        with col2:
            min_land = int(data["LT"].min())
            max_land = int(data["LT"].max())
            land_range = st.slider(
                "Land Area Range (m²)", min_land, max_land, (min_land, max_land)
            )

        with col3:
            min_building = int(data["LB"].min())
            max_building = int(data["LB"].max())
            building_range = st.slider(
                "Building Area Range (m²)",
                min_building,
                max_building,
                (min_building, max_building),
            )

        col1, col2 = st.columns(2)
        with col1:
            bedroom_options = sorted(data["bedroom"].unique())
            selected_bedrooms = st.multiselect(
                "Number of Bedrooms", bedroom_options, default=bedroom_options
            )

        with col2:
            bathroom_options = sorted(data["bathroom"].unique())
            selected_bathrooms = st.multiselect(
                "Number of Bathrooms", bathroom_options, default=bathroom_options
            )

        if "kecamatan" in data.columns:
            location_options = sorted(data["kecamatan"].unique())
            selected_locations = st.multiselect(
                "Location (Kecamatan)",
                location_options,
                default=(
                    location_options[:5]
                    if len(location_options) > 5
                    else location_options
                ),
            )
        else:
            selected_locations = None

        filtered_data = data[
            (data["price"] >= price_range[0])
            & (data["price"] <= price_range[1])
            & (data["LT"] >= land_range[0])
            & (data["LT"] <= land_range[1])
            & (data["LB"] >= building_range[0])
            & (data["LB"] <= building_range[1])
            & (data["bedroom"].isin(selected_bedrooms))
            & (data["bathroom"].isin(selected_bathrooms))
        ]

        if selected_locations is not None:
            filtered_data = filtered_data[
                filtered_data["kecamatan"].isin(selected_locations)
            ]

        st.write(f"Showing {len(filtered_data)} properties matching your criteria")

        with st.expander("View Filtered Data", expanded=False):
            st.dataframe(filtered_data)

        st.subheader("Data Visualizations")

        viz_tab1, viz_tab2, viz_tab3 = st.tabs(
            ["Scatter Plots", "Distributions", "Correlation Analysis"]
        )

        with viz_tab1:
            st.markdown("### Scatter Plots")

            x_options = [
                col
                for col in filtered_data.columns
                if filtered_data[col].dtype in ["int64", "float64"]
            ]
            x_axis = st.selectbox(
                "Select X-axis",
                options=x_options,
                index=x_options.index("LT") if "LT" in x_options else 0,
            )

            y_options = [
                col
                for col in filtered_data.columns
                if filtered_data[col].dtype in ["int64", "float64"]
            ]
            y_axis = st.selectbox(
                "Select Y-axis",
                options=y_options,
                index=y_options.index("price") if "price" in y_options else 0,
            )

            color_options = ["None"] + [
                col
                for col in filtered_data.columns
                if filtered_data[col].nunique() < 20
            ]
            color_var = st.selectbox("Color by", options=color_options)

            fig = plt.figure(figsize=(10, 6))
            if color_var != "None":
                scatter = sns.scatterplot(
                    data=filtered_data, x=x_axis, y=y_axis, hue=color_var, alpha=0.7
                )
                plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            else:
                scatter = sns.scatterplot(
                    data=filtered_data, x=x_axis, y=y_axis, alpha=0.7
                )

            plt.title(f"{y_axis} vs {x_axis}")
            plt.tight_layout()
            st.pyplot(fig)

        with viz_tab2:
            st.markdown("### Distribution Plots")

            dist_feature = st.selectbox(
                "Select feature to visualize distribution",
                options=[
                    col
                    for col in filtered_data.columns
                    if filtered_data[col].dtype in ["int64", "float64"]
                ],
            )

            fig = plt.figure(figsize=(10, 6))
            sns.histplot(filtered_data[dist_feature].dropna(), kde=True)
            plt.title(f"Distribution of {dist_feature}")
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("### Statistical Summary")
            st.dataframe(filtered_data[dist_feature].describe().to_frame())

        with viz_tab3:
            st.markdown("### Correlation Analysis")

            numerical_cols = filtered_data.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()

            if len(numerical_cols) > 1:

                correlation_matrix = filtered_data[numerical_cols].corr()

                fig, ax = plt.subplots(figsize=(12, 10))
                mask = np.triu(correlation_matrix)
                heatmap = sns.heatmap(
                    correlation_matrix,
                    annot=True,
                    mask=mask,
                    cmap="coolwarm",
                    linewidths=0.5,
                    fmt=".2f",
                    ax=ax,
                )
                plt.title("Correlation Matrix of Numerical Features")
                plt.tight_layout()
                st.pyplot(fig)

                if "price" in numerical_cols:
                    st.markdown("### Price Correlation Analysis")
                    price_correlations = (
                        correlation_matrix["price"]
                        .sort_values(ascending=False)
                        .drop("price")
                    )

                    fig, ax = plt.subplots(figsize=(10, 6))
                    price_correlations.plot(kind="bar", ax=ax)
                    plt.title("Features Correlation with Price")
                    plt.ylabel("Correlation Coefficient")
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.info("Not enough numerical columns for correlation analysis.")
    else:
        st.warning("No data available for exploration.")

with tab5:
    st.header("Download Preprocessed Data")

    if data is not None:
        st.markdown(
            """
        You can download the preprocessed data for further analysis or model building.
        The data is provided in CSV format and includes all the cleaned and engineered features.
        """
        )

        csv = data.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Full Preprocessed Dataset (CSV)",
            data=csv,
            file_name="yogyakarta_real_estate_preprocessed.csv",
            mime="text/csv",
        )

        st.markdown("### Custom Download Options")
        st.markdown("Select specific columns to include in your download:")

        selected_columns = st.multiselect(
            "Select columns to include",
            options=data.columns.tolist(),
            default=["price", "LT", "LB", "bedroom", "bathroom"],
        )

        if selected_columns:
            custom_csv = data[selected_columns].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Custom Dataset (CSV)",
                data=custom_csv,
                file_name="yogyakarta_real_estate_custom.csv",
                mime="text/csv",
            )

        if metadata:
            metadata_json = json.dumps(metadata, indent=4).encode("utf-8")
            st.download_button(
                label="Download Preprocessing Metadata (JSON)",
                data=metadata_json,
                file_name="preprocessing_metadata.json",
                mime="application/json",
            )
    else:
        st.warning("No data available to download.")


st.markdown("---")
st.markdown(
    """
**About this preprocessing page:**  
This page was created to demonstrate the preprocessing workflow for Yogyakarta real estate data.  
It includes data cleaning, feature engineering, and exploratory analysis tools to prepare data for machine learning models.
"""
)
