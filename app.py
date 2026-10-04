# ============================================================
# SuperKart Sales Prediction - Streamlit Application
# ============================================================

from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. Application Configuration
# ============================================================

st.set_page_config(
    page_title="SuperKart Sales Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


MODEL_FILE = "superkart_model.joblib"
METADATA_FILE = "superkart_model_metadata.json"
PREDICTION_COLUMN = "Predicted_Product_Store_Sales_Total"

EXPECTED_FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]

NUMERICAL_FEATURES = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Age_Years",
]

CATEGORICAL_FEATURES = [
    "Product_Sugar_Content",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Product_Type_Category",
]


# ============================================================
# 2. Styling
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.3rem;
            font-weight: 700;
            color: #1565C0;
            margin-bottom: 0.2rem;
        }

        .sub-title {
            font-size: 1.05rem;
            color: #555555;
            margin-bottom: 1.5rem;
        }

        .prediction-box {
            padding: 1.4rem;
            border-radius: 12px;
            background-color: #E8F5E9;
            border: 1px solid #66BB6A;
            text-align: center;
            margin-top: 1rem;
        }

        .prediction-value {
            font-size: 2rem;
            font-weight: 700;
            color: #1B5E20;
        }

        .info-box {
            padding: 1rem;
            border-radius: 10px;
            background-color: #E3F2FD;
            border-left: 5px solid #1976D2;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. Model-Loading Function
# ============================================================

@st.cache_resource
def load_model():
    """
    Load the serialized preprocessing and machine-learning pipeline.

    The model is loaded only once and cached by Streamlit.
    """

    base_directory = Path(__file__).resolve().parent
    model_path = base_directory / MODEL_FILE

    if not model_path.exists():
        raise FileNotFoundError(
            f"'{MODEL_FILE}' was not found. "
            "Place the model file in the same folder as app.py."
        )

    loaded_model = joblib.load(model_path)

    if not hasattr(loaded_model, "predict"):
        raise TypeError(
            "The loaded file does not contain a valid prediction model."
        )

    return loaded_model


# ============================================================
# 4. Metadata-Loading Function
# ============================================================

@st.cache_data
def load_metadata():
    """
    Load optional model metadata.

    The app will continue working if the metadata file is unavailable.
    """

    import json

    base_directory = Path(__file__).resolve().parent
    metadata_path = base_directory / METADATA_FILE

    if not metadata_path.exists():
        return {}

    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        return metadata if isinstance(metadata, dict) else {}

    except (OSError, json.JSONDecodeError):
        return {}


# ============================================================
# 5. Input Validation Functions
# ============================================================

def validate_schema(data: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Check whether all required model input columns are present.
    """

    missing_columns = [
        column
        for column in EXPECTED_FEATURES
        if column not in data.columns
    ]

    return len(missing_columns) == 0, missing_columns


def prepare_input_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate, clean and arrange incoming data for prediction.
    """

    is_valid, missing_columns = validate_schema(data)

    if not is_valid:
        raise ValueError(
            "The following required columns are missing: "
            + ", ".join(missing_columns)
        )

    prepared_data = data[EXPECTED_FEATURES].copy()

    # Convert numerical columns to numeric values.
    for column in NUMERICAL_FEATURES:
        prepared_data[column] = pd.to_numeric(
            prepared_data[column],
            errors="coerce",
        )

    # Check for missing or invalid numerical values.
    invalid_numeric_columns = [
        column
        for column in NUMERICAL_FEATURES
        if prepared_data[column].isna().any()
    ]

    if invalid_numeric_columns:
        raise ValueError(
            "Invalid or missing numerical values were found in: "
            + ", ".join(invalid_numeric_columns)
        )

    # Check for infinite numerical values.
    finite_values = np.isfinite(
        prepared_data[NUMERICAL_FEATURES].to_numpy(dtype=float)
    )

    if not finite_values.all():
        raise ValueError(
            "Numerical columns cannot contain infinite values."
        )

    # Clean categorical columns.
    for column in CATEGORICAL_FEATURES:
        prepared_data[column] = (
            prepared_data[column]
            .astype("string")
            .str.strip()
        )

    # Standardize known sugar-content variations.
    prepared_data["Product_Sugar_Content"] = (
        prepared_data["Product_Sugar_Content"].replace(
            {
                "reg": "Regular",
                "Reg": "Regular",
                "REG": "Regular",
                "regular": "Regular",
                "low sugar": "Low Sugar",
                "no sugar": "No Sugar",
            }
        )
    )

    # Check for blank categorical values.
    for column in CATEGORICAL_FEATURES:
        if (
            prepared_data[column].isna().any()
            or prepared_data[column].eq("").any()
        ):
            raise ValueError(
                f"'{column}' contains missing or blank values."
            )

    # Business-rule validation.
    if (prepared_data["Product_Weight"] <= 0).any():
        raise ValueError(
            "Product_Weight must be greater than zero."
        )

    if (
        (prepared_data["Product_Allocated_Area"] < 0)
        | (prepared_data["Product_Allocated_Area"] > 1)
    ).any():
        raise ValueError(
            "Product_Allocated_Area must be between 0 and 1."
        )

    if (prepared_data["Product_MRP"] <= 0).any():
        raise ValueError(
            "Product_MRP must be greater than zero."
        )

    if (prepared_data["Store_Age_Years"] < 0).any():
        raise ValueError(
            "Store_Age_Years cannot be negative."
        )

    return prepared_data


def make_predictions(model, input_data: pd.DataFrame) -> np.ndarray:
    """
    Generate model predictions and verify that the results are valid.
    """

    prepared_data = prepare_input_data(input_data)

    predictions = np.asarray(
        model.predict(prepared_data),
        dtype=float,
    ).reshape(-1)

    if not np.isfinite(predictions).all():
        raise ValueError(
            "The model generated an invalid prediction."
        )

    return predictions


# ============================================================
# 6. Load Model
# ============================================================

try:
    model = load_model()
    metadata = load_metadata()

except Exception as error:
    st.error("The SuperKart model could not be loaded.")
    st.code(str(error))
    st.info(
        "Ensure that app.py and superkart_model.joblib "
        "are saved in the same folder."
    )
    st.stop()


# ============================================================
# 7. Sidebar
# ============================================================

with st.sidebar:
    st.header("🛒 SuperKart")

    st.success("Model loaded successfully")

    st.markdown("---")

    page = st.radio(
        "Select prediction mode",
        options=[
            "Single Prediction",
            "Batch Prediction",
            "About the Model",
        ],
    )

    st.markdown("---")

    st.caption(
        "This application estimates product-store sales using "
        "a trained machine-learning pipeline."
    )


# ============================================================
# 8. Header
# ============================================================

st.markdown(
    '<div class="main-title">🛒 SuperKart Sales Predictor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sub-title">
        Predict product-store sales using product and store information.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 9. Single Prediction Page
# ============================================================

if page == "Single Prediction":

    st.subheader("Single Product Sales Prediction")

    st.write(
        "Enter the product and store information below. "
        "Then click **Predict Sales**."
    )

    with st.expander(
        "ℹ️ How should I fill out this form?",
        expanded=False,
    ):
        st.markdown(
            """
            - **Product Weight:** Weight of the product.
            - **Sugar Content:** Product sugar-content category.
            - **Allocated Area:** Store display-area ratio between 0 and 1.
            - **Product MRP:** Maximum retail price of the product.
            - **Store Size:** Small, Medium or High.
            - **City Tier:** Tier 1, Tier 2 or Tier 3.
            - **Store Type:** Type of retail store.
            - **Product ID Prefix:** First two letters of the product ID.
            - **Store Age:** Number of years since store establishment.
            - **Product Category:** Perishable or non-perishable.
            """
        )

    with st.form("single_prediction_form"):

        st.markdown("### Product Information")

        product_column_1, product_column_2 = st.columns(2)

        with product_column_1:
            product_weight = st.number_input(
                "Product Weight",
                min_value=0.01,
                max_value=100.00,
                value=12.66,
                step=0.01,
                help="Enter the weight of the product.",
            )

            product_sugar_content = st.selectbox(
                "Product Sugar Content",
                options=[
                    "Low Sugar",
                    "Regular",
                    "No Sugar",
                ],
            )

            product_allocated_area = st.number_input(
                "Product Allocated Area",
                min_value=0.000,
                max_value=1.000,
                value=0.027,
                step=0.001,
                format="%.3f",
                help="Enter a value between 0 and 1.",
            )

        with product_column_2:
            product_mrp = st.number_input(
                "Product MRP",
                min_value=0.01,
                max_value=10000.00,
                value=117.08,
                step=0.01,
            )

            product_id_prefix = st.selectbox(
                "Product ID Prefix",
                options=[
                    "FD",
                    "DR",
                    "NC",
                ],
                help=(
                    "FD = Food, DR = Drinks, "
                    "NC = Non-consumable"
                ),
            )

            product_type_category = st.selectbox(
                "Product Type Category",
                options=[
                    "Perishables",
                    "Non Perishables",
                ],
            )

        st.markdown("### Store Information")

        store_column_1, store_column_2 = st.columns(2)

        with store_column_1:
            store_size = st.selectbox(
                "Store Size",
                options=[
                    "Small",
                    "Medium",
                    "High",
                ],
                index=1,
            )

            city_type = st.selectbox(
                "Store Location City Type",
                options=[
                    "Tier 1",
                    "Tier 2",
                    "Tier 3",
                ],
                index=1,
            )

        with store_column_2:
            store_type = st.selectbox(
                "Store Type",
                options=[
                    "Departmental Store",
                    "Food Mart",
                    "Supermarket Type1",
                    "Supermarket Type2",
                ],
                index=3,
            )

            store_age = st.number_input(
                "Store Age in Years",
                min_value=0,
                max_value=100,
                value=16,
                step=1,
                help=(
                    "The model was developed using a fixed "
                    "reference year of 2025."
                ),
            )

        submitted = st.form_submit_button(
            "🔮 Predict Sales",
            use_container_width=True,
            type="primary",
        )

    if submitted:

        single_input = pd.DataFrame(
            [
                {
                    "Product_Weight": product_weight,
                    "Product_Sugar_Content": product_sugar_content,
                    "Product_Allocated_Area": product_allocated_area,
                    "Product_MRP": product_mrp,
                    "Store_Size": store_size,
                    "Store_Location_City_Type": city_type,
                    "Store_Type": store_type,
                    "Product_Id_char": product_id_prefix,
                    "Store_Age_Years": store_age,
                    "Product_Type_Category": product_type_category,
                }
            ],
            columns=EXPECTED_FEATURES,
        )

        try:
            with st.spinner("Generating sales prediction..."):
                prediction = make_predictions(
                    model,
                    single_input,
                )[0]

            st.markdown(
                f"""
                <div class="prediction-box">
                    <div>Predicted Product-Store Sales</div>
                    <div class="prediction-value">
                        ₹{prediction:,.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("View submitted input"):
                st.dataframe(
                    single_input,
                    use_container_width=True,
                    hide_index=True,
                )

            st.caption(
                "This is a model estimate based on historical data. "
                "Actual sales can be affected by demand, seasonality, "
                "competition, promotions and market conditions."
            )

        except Exception as error:
            st.error("The prediction could not be completed.")
            st.code(str(error))


# ============================================================
# 10. Batch Prediction Page
# ============================================================

elif page == "Batch Prediction":

    st.subheader("Batch Sales Prediction")

    st.write(
        "Upload a CSV file containing multiple product-store records."
    )

    st.markdown(
        """
        <div class="info-box">
            The uploaded CSV must contain the exact ten model-input
            columns shown below. Extra columns are allowed but will not
            be used for prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Required CSV Columns")

    required_columns_table = pd.DataFrame(
        {
            "Position": range(1, len(EXPECTED_FEATURES) + 1),
            "Required Column": EXPECTED_FEATURES,
            "Type": [
                (
                    "Numerical"
                    if feature in NUMERICAL_FEATURES
                    else "Categorical"
                )
                for feature in EXPECTED_FEATURES
            ],
        }
    )

    st.dataframe(
        required_columns_table,
        use_container_width=True,
        hide_index=True,
    )

    template_data = pd.DataFrame(
        [
            {
                "Product_Weight": 12.66,
                "Product_Sugar_Content": "Low Sugar",
                "Product_Allocated_Area": 0.027,
                "Product_MRP": 117.08,
                "Store_Size": "Medium",
                "Store_Location_City_Type": "Tier 2",
                "Store_Type": "Supermarket Type2",
                "Product_Id_char": "FD",
                "Store_Age_Years": 16,
                "Product_Type_Category": "Non Perishables",
            }
        ],
        columns=EXPECTED_FEATURES,
    )

    template_csv = template_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download CSV Template",
        data=template_csv,
        file_name="superkart_batch_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "Upload your batch CSV file",
        type=["csv"],
        help="Maximum recommended size: 10,000 rows.",
    )

    if uploaded_file is not None:

        try:
            batch_data = pd.read_csv(uploaded_file)

            if batch_data.empty:
                st.warning("The uploaded CSV file contains no rows.")
                st.stop()

            st.success(
                f"CSV uploaded successfully: "
                f"{len(batch_data):,} row(s)"
            )

            st.markdown("### Uploaded Data Preview")

            st.dataframe(
                batch_data.head(10),
                use_container_width=True,
                hide_index=True,
            )

            is_valid, missing_columns = validate_schema(batch_data)

            if not is_valid:
                st.error(
                    "The CSV cannot be processed because required "
                    "columns are missing."
                )

                st.write("Missing columns:")

                for column in missing_columns:
                    st.write(f"- `{column}`")

            elif len(batch_data) > 10000:
                st.error(
                    "The uploaded CSV contains more than 10,000 rows. "
                    "Please upload a smaller file."
                )

            else:
                if st.button(
                    "🔮 Generate Batch Predictions",
                    use_container_width=True,
                    type="primary",
                ):
                    with st.spinner(
                        "Generating batch predictions..."
                    ):
                        predictions = make_predictions(
                            model,
                            batch_data,
                        )

                    result_data = batch_data.copy()
                    result_data[PREDICTION_COLUMN] = predictions.round(2)

                    st.success(
                        f"Predictions generated successfully for "
                        f"{len(result_data):,} record(s)."
                    )

                    metric_column_1, metric_column_2 = st.columns(2)

                    with metric_column_1:
                        st.metric(
                            "Records Processed",
                            f"{len(result_data):,}",
                        )

                    with metric_column_2:
                        st.metric(
                            "Average Predicted Sales",
                            f"₹{predictions.mean():,.2f}",
                        )

                    st.markdown("### Prediction Results")

                    st.dataframe(
                        result_data.head(50),
                        use_container_width=True,
                        hide_index=True,
                    )

                    result_csv = result_data.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="⬇️ Download Prediction Results",
                        data=result_csv,
                        file_name="superkart_batch_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    st.caption(
                        "The preview displays the first 50 rows. "
                        "The downloaded CSV contains all records."
                    )

        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty.")

        except pd.errors.ParserError:
            st.error(
                "The uploaded file could not be read as a valid CSV."
            )

        except Exception as error:
            st.error("Batch prediction could not be completed.")
            st.code(str(error))


# ============================================================
# 11. About Page
# ============================================================

elif page == "About the Model":

    st.subheader("About the SuperKart Model")

    information_column_1, information_column_2 = st.columns(2)

    with information_column_1:
        st.markdown(
            """
            ### Model Purpose

            The application estimates product-store sales using
            product characteristics and store characteristics.

            ### Final Model

            The selected model is a **tuned Random Forest regressor**
            bundled together with its preprocessing pipeline.
            """
        )

    with information_column_2:
        st.markdown(
            """
            ### Model Inputs

            The model accepts:

            - Four numerical features
            - Six categorical features
            - Ten total input features

            Preprocessing is performed automatically by the saved
            machine-learning pipeline.
            """
        )

    st.markdown("---")

    st.markdown("### Model Performance")

    metric_column_1, metric_column_2, metric_column_3 = st.columns(3)

    with metric_column_1:
        st.metric(
            "Test RMSE",
            "277.00",
        )

    with metric_column_2:
        st.metric(
            "Test MAE",
            "107.75",
        )

    with metric_column_3:
        st.metric(
            "Test R²",
            "0.9328",
        )

    st.markdown("---")

    st.markdown("### Technical Information")

    technical_information = {
        "Model artifact": MODEL_FILE,
        "Pipeline structure": "Preprocessor + Random Forest",
        "Number of input features": len(EXPECTED_FEATURES),
        "Target": "Product_Store_Sales_Total",
        "Deployment": "Streamlit Community Cloud",
        "Backend API": "Not required",
        "Hugging Face": "Not used",
    }

    st.json(technical_information)

    if metadata:
        with st.expander("View saved model metadata"):
            st.json(metadata)

    st.warning(
        "This application provides statistical estimates. "
        "Predictions do not guarantee future sales because real sales "
        "may be affected by conditions not included in the dataset."
    )


# ============================================================
# 12. Footer
# ============================================================

st.markdown("---")

st.caption(
    "SuperKart Sales Prediction | Machine Learning Regression Project"
)