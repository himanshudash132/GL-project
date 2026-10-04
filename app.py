# ============================================================
# SuperKart Sales Forecasting Application
# Standalone Streamlit deployment
# ============================================================

from pathlib import Path
import io
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# 1. Application configuration
# ============================================================

st.set_page_config(
    page_title="SuperKart Sales Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_TITLE = "SuperKart Sales Forecasting System"
TARGET_NAME = "Product_Store_Sales_Total"

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "superkart_model.joblib"
METADATA_PATH = BASE_DIR / "superkart_model_metadata.json"

MAX_BATCH_ROWS = 10_000


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


CATEGORY_OPTIONS = {
    "Product_Sugar_Content": [
        "Low Sugar",
        "Regular",
        "No Sugar",
    ],
    "Store_Size": [
        "Small",
        "Medium",
        "High",
    ],
    "Store_Location_City_Type": [
        "Tier 1",
        "Tier 2",
        "Tier 3",
    ],
    "Store_Type": [
        "Departmental Store",
        "Supermarket Type1",
        "Supermarket Type2",
        "Food Mart",
    ],
    "Product_Id_char": [
        "FD",
        "DR",
        "NC",
    ],
    "Product_Type_Category": [
        "Perishables",
        "Non Perishables",
    ],
}


# ============================================================
# 2. Model and metadata loading
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model(model_path: Path):
    """
    Load the complete serialized preprocessing and regression pipeline.

    The model is cached so that it is not loaded again after every
    Streamlit interaction.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file '{model_path.name}' was not found. "
            "Place superkart_model.joblib in the same directory as app.py."
        )

    if model_path.stat().st_size == 0:
        raise ValueError(
            f"Model file '{model_path.name}' exists but is empty."
        )

    loaded_model = joblib.load(model_path)

    if not hasattr(loaded_model, "predict"):
        raise TypeError(
            "The loaded model does not provide a predict() method."
        )

    return loaded_model


@st.cache_data(show_spinner=False)
def load_metadata(metadata_path: Path):
    """Load optional model metadata without stopping the application."""
    if not metadata_path.exists():
        return {}

    try:
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)

        if isinstance(metadata, dict):
            return metadata

        return {}

    except (OSError, json.JSONDecodeError):
        return {}


try:
    model = load_model(MODEL_PATH)
    model_metadata = load_metadata(METADATA_PATH)
    MODEL_AVAILABLE = True
    MODEL_ERROR = None

except Exception as error:
    model = None
    model_metadata = {}
    MODEL_AVAILABLE = False
    MODEL_ERROR = str(error)


# ============================================================
# 3. Validation functions
# ============================================================

def validate_feature_schema(dataframe: pd.DataFrame):
    """
    Validate the required input columns.

    Returns:
        missing_columns, unexpected_columns
    """
    missing_columns = [
        column
        for column in EXPECTED_FEATURES
        if column not in dataframe.columns
    ]

    unexpected_columns = [
        column
        for column in dataframe.columns
        if column not in EXPECTED_FEATURES
    ]

    return missing_columns, unexpected_columns


def prepare_input_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and prepare raw input data for the serialized pipeline.

    No imputation, encoding, or scaling is performed here because
    those transformations are already included in the saved pipeline.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be supplied as a pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("The input dataset is empty.")

    if dataframe.columns.duplicated().any():
        duplicate_columns = dataframe.columns[
            dataframe.columns.duplicated()
        ].tolist()

        raise ValueError(
            f"Duplicate columns detected: {duplicate_columns}"
        )

    missing_columns, unexpected_columns = validate_feature_schema(dataframe)

    if missing_columns:
        raise ValueError(
            "The following required columns are missing: "
            + ", ".join(missing_columns)
        )

    prepared_data = dataframe[EXPECTED_FEATURES].copy()

    # Convert numerical variables safely.
    for column in NUMERICAL_FEATURES:
        prepared_data[column] = pd.to_numeric(
            prepared_data[column],
            errors="coerce",
        )

    # Clean categorical values without encoding them.
    for column in CATEGORICAL_FEATURES:
        prepared_data[column] = prepared_data[column].apply(
            lambda value: (
                value.strip()
                if isinstance(value, str)
                else value
            )
        )

    # Standardize known equivalent sugar-content labels.
    sugar_mapping = {
        "reg": "Regular",
        "Reg": "Regular",
        "REG": "Regular",
        "regular": "Regular",
        "Regular": "Regular",
        "low sugar": "Low Sugar",
        "Low sugar": "Low Sugar",
        "LOW SUGAR": "Low Sugar",
        "no sugar": "No Sugar",
        "No sugar": "No Sugar",
        "NO SUGAR": "No Sugar",
    }

    prepared_data["Product_Sugar_Content"] = (
        prepared_data["Product_Sugar_Content"]
        .replace(sugar_mapping)
    )

    # Ensure that numerical values are finite.
    numeric_array = prepared_data[NUMERICAL_FEATURES].to_numpy(
        dtype=float
    )

    if np.isinf(numeric_array).any():
        raise ValueError(
            "Infinite numerical values are not allowed."
        )

    # Business-rule validations.
    invalid_weight = (
        prepared_data["Product_Weight"].notna()
        & (prepared_data["Product_Weight"] <= 0)
    )

    if invalid_weight.any():
        invalid_rows = prepared_data.index[invalid_weight].tolist()

        raise ValueError(
            "Product_Weight must be greater than zero. "
            f"Invalid row indices: {invalid_rows[:10]}"
        )

    invalid_area = (
        prepared_data["Product_Allocated_Area"].notna()
        & (
            (prepared_data["Product_Allocated_Area"] < 0)
            | (prepared_data["Product_Allocated_Area"] > 1)
        )
    )

    if invalid_area.any():
        invalid_rows = prepared_data.index[invalid_area].tolist()

        raise ValueError(
            "Product_Allocated_Area must be between 0 and 1. "
            f"Invalid row indices: {invalid_rows[:10]}"
        )

    invalid_mrp = (
        prepared_data["Product_MRP"].notna()
        & (prepared_data["Product_MRP"] <= 0)
    )

    if invalid_mrp.any():
        invalid_rows = prepared_data.index[invalid_mrp].tolist()

        raise ValueError(
            "Product_MRP must be greater than zero. "
            f"Invalid row indices: {invalid_rows[:10]}"
        )

    invalid_age = (
        prepared_data["Store_Age_Years"].notna()
        & (prepared_data["Store_Age_Years"] < 0)
    )

    if invalid_age.any():
        invalid_rows = prepared_data.index[invalid_age].tolist()

        raise ValueError(
            "Store_Age_Years must be zero or greater. "
            f"Invalid row indices: {invalid_rows[:10]}"
        )

    # Reject empty categorical values.
    empty_category_details = {}

    for column in CATEGORICAL_FEATURES:
        empty_mask = (
            prepared_data[column].isna()
            | prepared_data[column].astype(str).str.strip().eq("")
        )

        if empty_mask.any():
            empty_category_details[column] = (
                prepared_data.index[empty_mask].tolist()[:10]
            )

    if empty_category_details:
        raise ValueError(
            "Missing or empty categorical values were detected: "
            f"{empty_category_details}"
        )

    # Unexpected columns are intentionally excluded after validation.
    if unexpected_columns:
        st.info(
            "The following extra columns will not be used for prediction: "
            + ", ".join(unexpected_columns)
        )

    return prepared_data


def generate_predictions(prepared_data: pd.DataFrame) -> np.ndarray:
    """Generate finite numeric predictions from the loaded pipeline."""
    if not MODEL_AVAILABLE or model is None:
        raise RuntimeError(
            "The prediction model is not currently available."
        )

    predictions = np.asarray(
        model.predict(prepared_data),
        dtype=float,
    ).reshape(-1)

    if len(predictions) != len(prepared_data):
        raise ValueError(
            "The number of predictions does not match "
            "the number of input records."
        )

    if not np.isfinite(predictions).all():
        raise ValueError(
            "The model returned one or more invalid predictions."
        )

    return predictions


def create_blank_template() -> bytes:
    """Create a blank CSV template containing the expected columns."""
    template = pd.DataFrame(columns=EXPECTED_FEATURES)
    return template.to_csv(index=False).encode("utf-8")


def create_example_template() -> bytes:
    """Create a one-row example CSV for batch prediction."""
    example = pd.DataFrame(
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
        ]
    )

    return example.to_csv(index=False).encode("utf-8")


# ============================================================
# 4. Header
# ============================================================

st.title(APP_TITLE)

st.markdown(
    """
    Predict quarterly product-store sales revenue using product
    characteristics and store attributes.

    The application uses a serialized machine learning pipeline that
    performs preprocessing and prediction consistently.
    """
)

st.info(
    "The prediction is a decision-support estimate and does not "
    "guarantee future sales revenue."
)


# ============================================================
# 5. Sidebar
# ============================================================

with st.sidebar:
    st.header("Application Information")

    if MODEL_AVAILABLE:
        st.success("Prediction model loaded")
    else:
        st.error("Prediction model unavailable")

    st.write(f"**Target:** `{TARGET_NAME}`")
    st.write(f"**Expected features:** {len(EXPECTED_FEATURES)}")
    st.write("**Deployment:** Streamlit Community Cloud")

    if model_metadata:
        st.divider()
        st.subheader("Model Metadata")

        final_model_name = (
            model_metadata.get("final_model_name")
            or model_metadata.get("model_name")
            or model_metadata.get("Final model name")
        )

        primary_metric = (
            model_metadata.get("primary_metric")
            or model_metadata.get("Primary metric")
        )

        if final_model_name:
            st.write(f"**Model:** {final_model_name}")

        if primary_metric:
            st.write(f"**Primary metric:** {primary_metric}")

    st.divider()

    with st.expander("Expected input fields"):
        for feature_number, feature in enumerate(
            EXPECTED_FEATURES,
            start=1,
        ):
            st.write(f"{feature_number}. `{feature}`")

    st.caption(
        "No manual encoding is required. The serialized pipeline "
        "handles preprocessing internally."
    )


# Stop only after showing a useful model-loading error.

if not MODEL_AVAILABLE:
    st.error(
        "The application could not load the prediction model."
    )

    st.code(
        MODEL_ERROR or "Unknown model-loading error",
        language="text",
    )

    st.warning(
        "Verify that `superkart_model.joblib` is present in the same "
        "GitHub directory as `app.py`, and that requirements.txt "
        "contains the required package versions."
    )

    st.stop()


# ============================================================
# 6. Prediction tabs
# ============================================================

single_tab, batch_tab, information_tab = st.tabs(
    [
        "Single Prediction",
        "Batch Prediction",
        "Model Information",
    ]
)


# ============================================================
# 7. Single prediction
# ============================================================

with single_tab:
    st.subheader("Single Product-Store Prediction")

    st.write(
        "Enter the product and store information below. "
        "All required preprocessing is performed by the saved model pipeline."
    )

    with st.form("single_prediction_form"):
        column_1, column_2 = st.columns(2)

        with column_1:
            product_weight = st.number_input(
                "Product Weight",
                min_value=0.01,
                value=12.66,
                step=0.01,
                format="%.2f",
                help="Weight of the product. The value must be positive.",
            )

            product_sugar_content = st.selectbox(
                "Product Sugar Content",
                options=CATEGORY_OPTIONS["Product_Sugar_Content"],
                index=0,
            )

            product_allocated_area = st.number_input(
                "Product Allocated Area",
                min_value=0.0,
                max_value=1.0,
                value=0.027,
                step=0.001,
                format="%.3f",
                help="Proportion of store area allocated to the product.",
            )

            product_mrp = st.number_input(
                "Product MRP",
                min_value=0.01,
                value=117.08,
                step=0.01,
                format="%.2f",
                help="Maximum retail price of the product.",
            )

            product_id_char = st.selectbox(
                "Product ID Category",
                options=CATEGORY_OPTIONS["Product_Id_char"],
                index=0,
                help="Two-character product-family prefix.",
            )

        with column_2:
            store_size = st.selectbox(
                "Store Size",
                options=CATEGORY_OPTIONS["Store_Size"],
                index=1,
            )

            store_city_type = st.selectbox(
                "Store Location City Type",
                options=CATEGORY_OPTIONS[
                    "Store_Location_City_Type"
                ],
                index=1,
            )

            store_type = st.selectbox(
                "Store Type",
                options=CATEGORY_OPTIONS["Store_Type"],
                index=2,
            )

            store_age_years = st.number_input(
                "Store Age in Years",
                min_value=0,
                value=16,
                step=1,
                help="Store age calculated using the project reference year.",
            )

            product_type_category = st.selectbox(
                "Product Type Category",
                options=CATEGORY_OPTIONS["Product_Type_Category"],
                index=1,
            )

        submitted = st.form_submit_button(
            "Predict Sales Revenue",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        single_payload = {
            "Product_Weight": float(product_weight),
            "Product_Sugar_Content": product_sugar_content,
            "Product_Allocated_Area": float(
                product_allocated_area
            ),
            "Product_MRP": float(product_mrp),
            "Store_Size": store_size,
            "Store_Location_City_Type": store_city_type,
            "Store_Type": store_type,
            "Product_Id_char": product_id_char,
            "Store_Age_Years": int(store_age_years),
            "Product_Type_Category": product_type_category,
        }

        try:
            single_input = pd.DataFrame([single_payload])
            prepared_single_input = prepare_input_dataframe(
                single_input
            )

            with st.spinner("Generating the sales prediction..."):
                single_prediction = generate_predictions(
                    prepared_single_input
                )[0]

            st.success("Prediction generated successfully.")

            metric_column, records_column = st.columns(2)

            with metric_column:
                st.metric(
                    label="Predicted Product-Store Sales Revenue",
                    value=f"{single_prediction:,.2f}",
                )

            with records_column:
                st.metric(
                    label="Records Processed",
                    value="1",
                )

            with st.expander("Submitted input"):
                display_input = prepared_single_input.copy()
                st.dataframe(
                    display_input,
                    use_container_width=True,
                    hide_index=True,
                )

            st.caption(
                "The displayed prediction has been rounded to two decimal "
                "places. The model calculation uses the full numeric value."
            )

        except Exception as error:
            st.error(
                "The prediction could not be generated. "
                f"Please review the input values. Details: {error}"
            )


# ============================================================
# 8. Batch prediction
# ============================================================

with batch_tab:
    st.subheader("Batch Sales Prediction")

    st.write(
        "Upload a CSV file containing the ten required model features. "
        "The uploaded file must not contain the target value."
    )

    template_column, example_column = st.columns(2)

    with template_column:
        st.download_button(
            label="Download Blank CSV Template",
            data=create_blank_template(),
            file_name="superkart_batch_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with example_column:
        st.download_button(
            label="Download Example CSV",
            data=create_example_template(),
            file_name="superkart_batch_example.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("Required CSV columns"):
        st.code(
            "\n".join(EXPECTED_FEATURES),
            language="text",
        )

    uploaded_file = st.file_uploader(
        "Upload the batch CSV file",
        type=["csv"],
        help=f"Maximum supported batch size: {MAX_BATCH_ROWS:,} rows.",
    )

    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)

            file_column, row_column, feature_column = st.columns(3)

            with file_column:
                st.metric(
                    "Filename",
                    uploaded_file.name,
                )

            with row_column:
                st.metric(
                    "Rows",
                    f"{len(batch_data):,}",
                )

            with feature_column:
                st.metric(
                    "Columns",
                    len(batch_data.columns),
                )

            st.write("#### Uploaded Data Preview")

            st.dataframe(
                batch_data.head(),
                use_container_width=True,
                hide_index=True,
            )

            if len(batch_data) > MAX_BATCH_ROWS:
                raise ValueError(
                    f"The uploaded file contains {len(batch_data):,} rows. "
                    f"The maximum supported batch size is "
                    f"{MAX_BATCH_ROWS:,} rows."
                )

            missing_columns, unexpected_columns = (
                validate_feature_schema(batch_data)
            )

            if missing_columns:
                st.error(
                    "Missing required columns: "
                    + ", ".join(missing_columns)
                )

            if unexpected_columns:
                st.warning(
                    "The following additional columns will be excluded "
                    "from prediction: "
                    + ", ".join(unexpected_columns)
                )

            if not missing_columns:
                prepared_batch_data = prepare_input_dataframe(
                    batch_data
                )

                st.success(
                    "The uploaded CSV passed schema validation."
                )

                predict_batch = st.button(
                    "Generate Batch Predictions",
                    type="primary",
                    use_container_width=True,
                )

                if predict_batch:
                    with st.spinner(
                        f"Generating predictions for "
                        f"{len(prepared_batch_data):,} records..."
                    ):
                        batch_predictions = generate_predictions(
                            prepared_batch_data
                        )

                    batch_results = batch_data.copy()

                    batch_results[
                        "Predicted_Product_Store_Sales_Total"
                    ] = batch_predictions

                    st.success(
                        f"Predictions generated successfully for "
                        f"{len(batch_results):,} records."
                    )

                    result_column, prediction_column = st.columns(2)

                    with result_column:
                        st.metric(
                            "Records Processed",
                            f"{len(batch_results):,}",
                        )

                    with prediction_column:
                        st.metric(
                            "Predictions Generated",
                            f"{len(batch_predictions):,}",
                        )

                    st.write("#### Prediction Results Preview")

                    st.dataframe(
                        batch_results.head(10),
                        use_container_width=True,
                        hide_index=True,
                    )

                    result_csv = batch_results.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Batch Prediction Results",
                        data=result_csv,
                        file_name="superkart_batch_predictions.csv",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True,
                    )

                    st.info(
                        "The batch data does not contain actual target "
                        "values. Therefore, evaluation metrics such as "
                        "RMSE, MAE, and R-squared are not calculated."
                    )

        except pd.errors.EmptyDataError:
            st.error(
                "The uploaded CSV is empty or does not contain readable data."
            )

        except pd.errors.ParserError:
            st.error(
                "The uploaded file could not be parsed as a valid CSV."
            )

        except UnicodeDecodeError:
            st.error(
                "The CSV encoding could not be read. "
                "Please upload a UTF-8 encoded CSV file."
            )

        except Exception as error:
            st.error(
                "Batch validation or prediction failed. "
                f"Details: {error}"
            )


# ============================================================
# 9. Model information
# ============================================================

with information_tab:
    st.subheader("Model and Deployment Information")

    information_column_1, information_column_2 = st.columns(2)

    with information_column_1:
        st.write("#### Application")

        st.write(f"**Application:** {APP_TITLE}")
        st.write(f"**Prediction target:** `{TARGET_NAME}`")
        st.write("**Interface:** Streamlit")
        st.write("**Inference type:** Single and batch")
        st.write("**Model file:** `superkart_model.joblib`")

    with information_column_2:
        st.write("#### Model Status")

        st.write(
            f"**Model loaded:** {'Yes' if MODEL_AVAILABLE else 'No'}"
        )

        if model is not None:
            st.write(
                f"**Loaded object:** `{type(model).__name__}`"
            )

            if hasattr(model, "named_steps"):
                pipeline_steps = list(model.named_steps.keys())

                st.write(
                    "**Pipeline steps:** "
                    + ", ".join(pipeline_steps)
                )

    st.divider()

    st.write("#### Expected Feature Schema")

    feature_schema = pd.DataFrame(
        {
            "Feature": EXPECTED_FEATURES,
            "Type": [
                (
                    "Numerical"
                    if feature in NUMERICAL_FEATURES
                    else "Categorical"
                )
                for feature in EXPECTED_FEATURES
            ],
            "Required": ["Yes"] * len(EXPECTED_FEATURES),
        }
    )

    st.dataframe(
        feature_schema,
        use_container_width=True,
        hide_index=True,
    )

    if model_metadata:
        st.divider()
        st.write("#### Saved Model Metadata")

        safe_metadata = {
            key: value
            for key, value in model_metadata.items()
            if not any(
                sensitive_word in key.lower()
                for sensitive_word in [
                    "token",
                    "password",
                    "secret",
                    "credential",
                ]
            )
        }

        st.json(safe_metadata)

    st.divider()

    st.warning(
        "The model predicts product-store sales revenue rather than "
        "physical unit demand. Inventory decisions should also consider "
        "current stock, unit prices, supplier lead times, shelf life, "
        "promotions, and operational constraints."
    )


# ============================================================
# 10. Footer
# ============================================================

st.divider()

st.caption(
    "SuperKart Sales Forecasting System | "
    "Machine Learning Model Deployment Project"
)
