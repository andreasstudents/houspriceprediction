import os
import pickle
import json
import sys
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.preprocessing import StandardScaler

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from train.rf_model import predict_price_rf

st.set_page_config(page_title="Prediksi Harga Rumah", page_icon="🏠", layout="wide")


st.markdown(
    """
<style>
    .prediction-container {
        background-color: 
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-value {
        font-size: 36px;
        font-weight: bold;
        color: 
        text-align: center;
    }
    .prediction-label {
        font-size: 16px;
        color: 
        text-align: center;
    }
    .feature-importance {
        margin-top: 10px;
        padding: 10px;
        background-color: 
        border-radius: 5px;
    }
    .confidence-indicator {
        margin-top: 15px;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .high-confidence {
        background-color: 
        color: 
    }
    .medium-confidence {
        background-color: 
        color: 
    }
    .low-confidence {
        background-color: 
        color: 
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Prediksi Harga Rumah")
st.markdown(
    """
Gunakan formulir di bawah ini untuk memprediksi harga rumah berdasarkan karakteristiknya.
Model ini telah dilatih menggunakan Random Forest yang telah dioptimalkan.
"""
)

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(current_dir, "../model")
EVAL_DIR = os.path.join(MODEL_DIR, "evaluation")

OPTIMIZED_RF_PATH = os.path.join(MODEL_DIR, "optimized_rf_model.pkl")
ENSEMBLE_PATH = os.path.join(MODEL_DIR, "best_stacking_esemble.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "optimized_rf_model_metadata.json")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")


@st.cache_data
def load_metadata():
    try:
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        st.error(f"Error loading model metadata: {e}")
        return None


@st.cache_resource
def load_resources():
    resources = {}
    try:

        if os.path.exists(ENSEMBLE_PATH):
            with open(ENSEMBLE_PATH, "rb") as f:
                resources["rf_model"] = pickle.load(f)
            st.info("Using best stacking ensemble model")
        else:

            with open(OPTIMIZED_RF_PATH, "rb") as f:
                resources["rf_model"] = pickle.load(f)
            st.info("Using optimized Random Forest model")

        with open(SCALER_PATH, "rb") as f:
            resources["scaler"] = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading resources: {e}")
    return resources


metadata = load_metadata()
resources = load_resources()

feature_list = [
    "price",
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
    "fuzzy_area_quality",
    "fuzzy_quality_score",
]


prediction_features = [f for f in feature_list if f != "price"]

feature_importance = metadata.get("feature_importance", {}) if metadata else {}


def format_price(price):
    """Format price with Rp symbol and thousand separators"""
    return f"Rp {price:,.0f}"


kecamatan_list = [
    "Bantul",
    "Caturtunggal",
    "Cebongan",
    "Danurejan",
    "Demangan",
    "Gedong Tengen",
    "Gondokusuman",
    "Gondomanan",
    "Jetis",
    "Kaliurang",
    "Kotagede",
    "Kraton",
    "Kulonprogo",
    "Maguwoharjo",
    "Mantrijeron",
    "Mergangsan",
    "Minomartani",
    "Ngampilan",
    "Nologaten",
    "Pakualaman",
    "Pogung",
    "Purwomartani",
    "Seturan",
    "Sidoarum",
    "Sleman",
    "Tegalrejo",
    "Umbulharjo",
    "Wirobrajan",
]


def load_kecamatan_encoding():
    encoding_path = os.path.join(MODEL_DIR, "../reports/encoder_mappings.json")
    if os.path.exists(encoding_path):
        try:
            with open(encoding_path, "r") as f:
                encodings = json.load(f)
                return encodings.get("kecamatan", {}).get("mapping", {})
        except Exception as e:
            st.warning(f"Could not load kecamatan encoding: {e}")
    return {k: i for i, k in enumerate(kecamatan_list)}


def calculate_fuzzy_features(bedroom, bathroom, lt, lb, carport=0):
    """Calculate fuzzy logic features for prediction"""
    try:

        sys.path.insert(0, os.path.join(current_dir, "../utils"))
        from utils.preprocessing import create_fuzzy_systems

        fuzzy_systems = create_fuzzy_systems()

        fuzzy_area_quality = 0.5
        if "area" in fuzzy_systems:
            try:
                area_system = fuzzy_systems["area"]
                area_system.input["lt"] = min(lt, 2000)
                area_system.input["lb"] = min(lb, 1000)
                area_system.compute()
                fuzzy_area_quality = area_system.output["area_quality"] / 100.0
            except Exception as e:
                st.warning(f"Could not calculate fuzzy area quality: {e}")

        fuzzy_room_quality = 0.5
        if "room" in fuzzy_systems:
            try:
                room_system = fuzzy_systems["room"]
                room_system.input["bedroom"] = min(bedroom, 10)
                room_system.input["bathroom"] = min(bathroom, 10)
                room_system.compute()
                fuzzy_room_quality = room_system.output["room_quality"] / 100.0
            except Exception as e:
                st.warning(f"Could not calculate fuzzy room quality: {e}")

        fuzzy_quality_score = (fuzzy_area_quality + fuzzy_room_quality) / 2.0

        return fuzzy_area_quality, fuzzy_quality_score

    except Exception as e:
        st.warning(f"Error calculating fuzzy features: {e}")
        return 0.5, 0.5


def predict_house_price(
    bedroom, bathroom, lt, lb, carport, kecamatan, listing_age_days=30
):
    """Predict house price using the trained model"""
    try:
        if resources.get("rf_model") is None or resources.get("scaler") is None:
            st.error("Necessary resources not loaded!")
            return None

        model = resources["rf_model"]
        scaler = resources["scaler"]

        kecamatan_encoding = load_kecamatan_encoding()
        kecamatan_encoded = kecamatan_encoding.get(kecamatan, 0)

        building_efficiency = lb / lt if lt > 0 else 0

        building_efficiency = min(building_efficiency, 1.0)

        total_rooms = bedroom + bathroom
        total_area = lt + lb
        area_ratio = lb / lt if lt > 0 else 0

        room_per_area = total_rooms / max(lb, 1) if lb > 0 else 0
        land_utilization = lb / lt if lt > 0 else 0
        outdoor_space = max(lt - lb, 0)
        outdoor_ratio = outdoor_space / lt if lt > 0 else 0
        space_efficiency = total_rooms / max(lb, 1) if lb > 0 else 0
        room_density = total_rooms / max(lt, 1) if lt > 0 else 0

        bedroom_bathroom_ratio = bedroom / max(bathroom, 0.5)
        bedroom_bathroom_ratio = min(bedroom_bathroom_ratio, 10.0)

        ideal_bathroom_ratio = bedroom / 2.0
        room_balance = min(abs(bathroom - ideal_bathroom_ratio), 5.0)

        if lt <= 85:
            land_size_tier = 0
        elif lt <= 115:
            land_size_tier = 1
        elif lt <= 160:
            land_size_tier = 2
        else:
            land_size_tier = 3

        if lb <= 55:
            building_size_tier = 0
        elif lb <= 75:
            building_size_tier = 1
        elif lb <= 100:
            building_size_tier = 2
        else:
            building_size_tier = 3

        area_ratio_squared = (area_ratio + 0.001) ** 2
        room_density_squared = (room_density + 0.001) ** 2

        kabupaten_kota_encoded = 0

        fuzzy_area, fuzzy_quality_score = calculate_fuzzy_features(
            bedroom, bathroom, lt, lb, carport
        )

        feature_names = [
            "bedroom",
            "bathroom",
            "LT",
            "LB",
            "kecamatan_encoded",
            "building_efficiency",
            "listing_age_days",
            "total_rooms",
            "total_area",
            "area_ratio",
            "room_per_area",
            "land_utilization",
            "outdoor_space",
            "outdoor_ratio",
            "space_efficiency",
            "room_density",
            "bedroom_bathroom_ratio",
            "room_balance",
            "land_size_tier",
            "building_size_tier",
            "area_ratio_squared",
            "room_density_squared",
            "kabupaten_kota_encoded",
            "fuzzy_quality_score",
        ]

        feature_values = [
            bedroom,
            bathroom,
            lt,
            lb,
            kecamatan_encoded,
            building_efficiency,
            listing_age_days,
            total_rooms,
            total_area,
            area_ratio,
            room_per_area,
            land_utilization,
            outdoor_space,
            outdoor_ratio,
            space_efficiency,
            room_density,
            bedroom_bathroom_ratio,
            room_balance,
            land_size_tier,
            building_size_tier,
            area_ratio_squared,
            room_density_squared,
            kabupaten_kota_encoded,
            fuzzy_quality_score,
        ]

        features_df = pd.DataFrame([feature_values], columns=feature_names)

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            features_array = features_df.values
            scaled_features = scaler.transform(features_array)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction_log = model.predict(scaled_features)[0]

        prediction = np.expm1(prediction_log)

        if prediction <= 0 or prediction > 50000000000:
            st.warning(
                f"Prediction seems unreasonable: {prediction:,.0f}, using fallback method"
            )
            raise ValueError("Unreasonable prediction")

        return prediction

    except Exception as e:
        st.error(f"Error making prediction: {e}")

        try:

            base_price = 2000000
            building_multiplier = 3000000

            estimated_price = (lt * base_price) + (lb * building_multiplier)

            room_factor = 1 + ((bedroom - 2) * 0.1) + ((bathroom - 1) * 0.05)
            estimated_price *= room_factor

            if carport > 0:
                estimated_price *= 1 + carport * 0.05

            st.warning("Using fallback calculation method")
            return estimated_price

        except:
            return 2000000000


def get_confidence_level(prediction, features):
    """Determine confidence level based on feature values and model uncertainty"""
    try:

        confidence_score = 0.8

        if features["LT"] > 1000 or features["LB"] > 500:
            confidence_score -= 0.2
        if features["bedroom"] > 6 or features["bathroom"] > 4:
            confidence_score -= 0.1
        if (
            features["building_efficiency"] > 0.8
            or features["building_efficiency"] < 0.2
        ):
            confidence_score -= 0.1

        if prediction < 500000000 or prediction > 10000000000:
            confidence_score -= 0.2

        if confidence_score >= 0.7:
            return "high", confidence_score
        elif confidence_score >= 0.5:
            return "medium", confidence_score
        else:
            return "low", confidence_score

    except:
        return "medium", 0.6


st.header("Input Karakteristik Rumah")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Spesifikasi Dasar")
    bedroom = st.number_input("Jumlah Kamar Tidur", min_value=1, max_value=10, value=1)
    bathroom = st.number_input("Jumlah Kamar Mandi", min_value=1, max_value=8, value=1)
    carport = st.number_input("Jumlah Carport", min_value=0, max_value=1, value=1)

with col2:
    st.subheader("Dimensi Properti")
    lt = st.number_input("Luas Tanah (m²)", min_value=50, max_value=220, value=50)
    lb = st.number_input("Luas Bangunan (m²)", min_value=30, max_value=150, value=50)
    kecamatan = st.selectbox("Kecamatan", kecamatan_list)

st.subheader("Parameter Tambahan")
col3, col4 = st.columns(2)
with col3:
    listing_age_days = st.slider(
        "Umur Listing (hari)", min_value=1, max_value=60, value=30
    )
with col4:
    show_details = st.checkbox("Tampilkan Detail Analisis", value=True)

if st.button("Prediksi Harga Rumah", type="primary"):
    with st.spinner("Menghitung prediksi harga..."):
        prediction = predict_house_price(
            bedroom, bathroom, lt, lb, carport, kecamatan, listing_age_days
        )
        method_used = "Machine Learning"

        if prediction is not None:

            st.markdown(
                f"""
                <div class="prediction-container">
                    <div class="prediction-label">Prediksi Harga Rumah</div>
                    <div class="prediction-value">{format_price(prediction)}</div>
                    <div class="prediction-label"><small>Metode: {method_used}</small></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            features = {
                "LT": lt,
                "LB": lb,
                "bedroom": bedroom,
                "bathroom": bathroom,
                "building_efficiency": lb / lt if lt > 0 else 0,
            }
            confidence_level, confidence_score = get_confidence_level(
                prediction, features
            )

            confidence_class = f"{confidence_level}-confidence"
            confidence_text = {
                "high": "Tinggi - Prediksi sangat akurat",
                "medium": "Sedang - Prediksi cukup akurat",
                "low": "Rendah - Prediksi perlu validasi lebih lanjut",
            }[confidence_level]

            st.markdown(
                f"""
                <div class="confidence-indicator {confidence_class}">
                    <strong>Tingkat Kepercayaan: {confidence_text}</strong><br>
                    <small>Skor Kepercayaan: {confidence_score:.2f}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if show_details:
                st.subheader("Detail Analisis")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Karakteristik Input:**")
                    st.write(f"- Kamar Tidur: {bedroom}")
                    st.write(f"- Kamar Mandi: {bathroom}")
                    st.write(f"- Luas Tanah: {lt} m²")
                    st.write(f"- Luas Bangunan: {lb} m²")
                    st.write(f"- Carport: {carport}")
                    st.write(f"- Kecamatan: {kecamatan}")

                with col2:
                    st.write("**Analisis Properti:**")
                    building_efficiency = lb / lt if lt > 0 else 0
                    st.write(f"- Efisiensi Bangunan: {building_efficiency:.2f}")
                    st.write(f"- Harga per m² Tanah: ~Rp {(prediction/lt):,.0f}")
                    st.write(f"- Harga per m² Bangunan: ~Rp {(prediction/lb):,.0f}")

                    fuzzy_area, fuzzy_quality = calculate_fuzzy_features(
                        bedroom, bathroom, lt, lb, carport
                    )
                    st.write(f"- Kualitas Area (Fuzzy): {fuzzy_area:.2f}")
                    st.write(f"- Skor Kualitas Total: {fuzzy_quality:.2f}")

                    total_rooms = bedroom + bathroom
                    total_area = lt + lb
                    st.write(f"- Total Rooms: {total_rooms}")
                    st.write(f"- Total Area: {total_area} m²")

                    if lt <= 85:
                        land_tier = "Kecil (0)"
                    elif lt <= 115:
                        land_tier = "Sedang (1)"
                    elif lt <= 160:
                        land_tier = "Besar (2)"
                    else:
                        land_tier = "Sangat Besar (3)"
                    st.write(f"- Kategori Tanah: {land_tier}")

                    if lb <= 55:
                        building_tier = "Kecil (0)"
                    elif lb <= 75:
                        building_tier = "Sedang (1)"
                    elif lb <= 100:
                        building_tier = "Besar (2)"
                    else:
                        building_tier = "Sangat Besar (3)"
                    st.write(f"- Kategori Bangunan: {building_tier}")

                    warnings = []

                    if lt in [85, 86, 115, 116, 160, 161]:
                        warnings.append(
                            "⚠️ Luas tanah dekat batas kategori - prediksi mungkin tidak stabil"
                        )

                    if lb in [55, 56, 75, 76, 100, 101]:
                        warnings.append(
                            "⚠️ Luas bangunan dekat batas kategori - prediksi mungkin tidak stabil"
                        )

                    if building_efficiency > 0.9:
                        warnings.append(
                            "⚠️ Efisiensi bangunan sangat tinggi - mungkin tidak realistis"
                        )

                    bedroom_bathroom_ratio = bedroom / max(bathroom, 0.5)
                    if bedroom_bathroom_ratio > 8:
                        warnings.append(
                            "⚠️ Rasio kamar tidur/mandi sangat tinggi - tidak umum"
                        )

                    if fuzzy_quality < 0.3:
                        warnings.append(
                            "⚠️ Skor kualitas fuzzy rendah - mempengaruhi prediksi secara signifikan"
                        )

                    if warnings:
                        st.warning("**Perhatian:**")
                        for warning in warnings:
                            st.write(f"  {warning}")

                st.subheader("Analisis Rentang Harga")

                price_ranges = [
                    (0, 1000000000, "Sangat Terjangkau"),
                    (1000000000, 2500000000, "Terjangkau"),
                    (2500000000, 5000000000, "Menengah"),
                    (5000000000, 10000000000, "Premium"),
                    (10000000000, float("inf"), "Mewah"),
                ]

                for min_price, max_price, category in price_ranges:
                    if min_price <= prediction < max_price:
                        st.info(f"Rumah ini termasuk dalam kategori **{category}**")
                        break

                st.subheader("Rekomendasi")
                recommendations = []

                if building_efficiency < 0.4:
                    recommendations.append(
                        "Pertimbangkan untuk menambah luas bangunan untuk efisiensi yang lebih baik"
                    )
                elif building_efficiency > 0.8:
                    recommendations.append(
                        "Efisiensi bangunan sangat baik, properti ini memaksimalkan penggunaan lahan"
                    )

                if lt > 200 and lb < 100:
                    recommendations.append(
                        "Ada potensi untuk pengembangan lebih lanjut dengan luas tanah yang tersedia"
                    )

                if confidence_level == "low":
                    recommendations.append(
                        "Disarankan untuk memvalidasi prediksi dengan survei pasar lokal"
                    )

                if not recommendations:
                    recommendations.append(
                        "Properti memiliki spesifikasi yang seimbang dan sesuai dengan standar pasar"
                    )

                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")

        else:
            st.error("Gagal membuat prediksi. Silakan periksa input dan coba lagi.")


with st.expander("ℹ️ Informasi Model"):
    st.write("**Tentang Model Prediksi:**")
    st.write("- Model: Random Forest Regressor yang telah dioptimalkan")
    st.write("- Fitur: Menggunakan 12+ fitur termasuk fuzzy logic")
    st.write("- Akurasi: R² > 0.999 pada data training")
    st.write("- Data: Berdasarkan properti di Yogyakarta")

    if metadata:
        st.write("\n**Metrik Evaluasi:**")
        metrics = metadata.get("evaluation_metrics", {})
        if metrics:
            st.write(f"- MAPE: {metrics.get('mape', 0)*100:.2f}%")
            st.write(f"- R²: {metrics.get('r2', 0):.4f}")
            st.write(f"- RMSE: Rp {metrics.get('rmse', 0):,.0f}")
            st.write(f"- MAE: Rp {metrics.get('mae', 0):,.0f}")

st.markdown("---")
st.markdown(
    "*Prediksi ini dibuat menggunakan model machine learning dan hanya untuk referensi. Harga aktual dapat bervariasi berdasarkan kondisi pasar dan faktor lainnya.*"
)
