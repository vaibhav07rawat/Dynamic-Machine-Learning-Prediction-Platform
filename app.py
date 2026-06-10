import hashlib
import io
import os
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import ParserError
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.ml_pipeline import DataCleaner, EnsembleFeatureSelector, ProblemAnalyzer


st.set_page_config(
    page_title="Dynamic ML + TinyML Dashboard",
    page_icon="ML",
    layout="wide",
)


DEFAULT_DATA_PATH = Path("data/iris_data.csv")
RANDOM_STATE = 42
tf.keras.utils.set_random_seed(RANDOM_STATE)


def apply_page_style() -> None:
    """Add clean visual styling while keeping the app simple for students to understand."""
    st.markdown(
        """
        <style>
        .prediction-card {
            border-left: 5px solid #1f77b4;
            border-radius: 8px;
            padding: 1.1rem 1.25rem;
            background: #f7fbff;
            color: #102a43;
        }
        .small-note {
            color: #5f6b7a;
            font-size: 0.93rem;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def make_dataset_key(df: pd.DataFrame, target_column: str) -> str:
    """Create a stable cache key so Streamlit retrains only when the data or target changes."""
    sample_hash = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.md5(sample_hash + target_column.encode("utf-8")).hexdigest()


@st.cache_data
def load_default_dataset(path: str) -> pd.DataFrame:
    """Load the built-in CSV dataset used when the user has not uploaded another file."""
    return pd.read_csv(path)


def read_dataset_file(uploaded_file) -> pd.DataFrame:
    """Read uploaded CSV or Excel files with practical encoding fallbacks."""
    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    # Some files are accidentally renamed as .csv even though they are Excel files.
    # XLSX files begin with PK because they are ZIP containers.
    is_xlsx_content = file_bytes.startswith(b"PK\x03\x04")
    is_xls_content = file_bytes.startswith(b"\xD0\xCF\x11\xE0")
    if file_name.endswith((".xlsx", ".xls")) or is_xlsx_content or is_xls_content:
        return pd.read_excel(io.BytesIO(file_bytes))

    encodings = ["utf-8", "utf-8-sig", "utf-16", "utf-16le", "utf-16be", "latin1", "cp1252", "iso-8859-1"]
    read_attempts = [
        {"sep": None, "engine": "python"},
        {"sep": ","},
        {"sep": ";"},
        {"sep": "\t"},
        {"sep": "|"},
    ]
    last_error = None

    for encoding in encodings:
        for options in read_attempts:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding, **options)
                if df.shape[1] > 1:
                    return df
                last_error = ValueError("Only one column detected. Trying another delimiter.")
            except (UnicodeDecodeError, ParserError, ValueError) as error:
                last_error = error

    raise ValueError(
        "Unable to read the uploaded file. Please upload a valid CSV/Excel file. "
        "If it is CSV, save it with a clear delimiter such as comma, semicolon, tab, or pipe."
    ) from last_error


def load_uploaded_or_default_dataset() -> tuple[pd.DataFrame, str]:
    """Load a user CSV/Excel upload, falling back to the bundled Iris CSV."""
    uploaded_file = st.sidebar.file_uploader("Upload dataset", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        df = read_dataset_file(uploaded_file)
        return df, uploaded_file.name

    if not DEFAULT_DATA_PATH.exists():
        st.error("No uploaded file found and the default dataset is missing.")
        st.stop()

    return load_default_dataset(str(DEFAULT_DATA_PATH)), "Default Iris Dataset"


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove fully empty rows/columns and normalize column names to strings."""
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return df


def get_dataset_info(df: pd.DataFrame) -> str:
    """Return the same useful text produced by pandas df.info()."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


def detect_columns(df: pd.DataFrame, target_column: str) -> tuple[list[str], list[str], list[str]]:
    """Automatically detect feature, numeric, and categorical columns."""
    feature_columns = [col for col in df.columns if col != target_column]
    numeric_columns = df[feature_columns].select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [col for col in feature_columns if col not in numeric_columns]
    return feature_columns, numeric_columns, categorical_columns


def make_one_hot_encoder() -> OneHotEncoder:
    """Create a OneHotEncoder compatible with modern scikit-learn versions."""
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def create_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    """Build preprocessing dynamically for the columns present in the uploaded dataset."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_one_hot_encoder()),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def build_tensorflow_model(input_dim: int, output_dim: int) -> tf.keras.Model:
    """Build a neural network whose input size adapts to the dataset."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(output_dim, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_sklearn_models(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> dict[str, Pipeline]:
    """Create all classical ML models with dynamic preprocessing pipelines."""
    scaled_preprocessor = create_preprocessor(numeric_columns, categorical_columns, scale_numeric=True)
    unscaled_preprocessor = create_preprocessor(numeric_columns, categorical_columns, scale_numeric=False)

    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", scaled_preprocessor),
                ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("preprocessor", unscaled_preprocessor),
                ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", unscaled_preprocessor),
                ("model", RandomForestClassifier(n_estimators=120, random_state=RANDOM_STATE)),
            ]
        ),
        "SVM": Pipeline(
            [
                ("preprocessor", scaled_preprocessor),
                ("model", CalibratedClassifierCV(SVC(kernel="rbf", random_state=RANDOM_STATE), cv=3)),
            ]
        ),
        "Naive Bayes": Pipeline(
            [
                ("preprocessor", scaled_preprocessor),
                ("model", GaussianNB()),
            ]
        ),
    }


def model_size_bytes(model) -> int:
    """Estimate saved model size by serializing sklearn objects or saving a Keras model."""
    if isinstance(model, tf.keras.Model):
        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
            temp_path = tmp.name
        try:
            model.save(temp_path)
            return os.path.getsize(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    return len(pickle.dumps(model))


def bytes_to_kb(size_bytes: int) -> float:
    return round(size_bytes / 1024, 2)


def prepare_target(df: pd.DataFrame, target_column: str) -> tuple[pd.Series, np.ndarray, LabelEncoder]:
    """Encode the selected target column for classification."""
    target = df[target_column].copy()
    valid_target_rows = target.notna()
    df.drop(index=df.index[~valid_target_rows], inplace=True)
    target = df[target_column].astype(str)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(target)
    return target, y, label_encoder


def get_train_test_data(df: pd.DataFrame, target_column: str):
    """Split the selected dataset into features and target arrays."""
    working_df = df.copy()
    _, y, label_encoder = prepare_target(working_df, target_column)
    feature_columns, numeric_columns, categorical_columns = detect_columns(working_df, target_column)
    x = working_df[feature_columns].copy()

    class_counts = pd.Series(y).value_counts()
    can_stratify = len(class_counts) > 1 and class_counts.min() >= 2
    stratify = y if can_stratify else None

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    return x, y, x_train, x_test, y_train, y_test, label_encoder, feature_columns, numeric_columns, categorical_columns


def evaluate_tflite_model(tflite_model: bytes, x_test_processed: np.ndarray, y_test: np.ndarray):
    """Run TensorFlow Lite inference sample by sample."""
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    predictions = []
    for row in x_test_processed.astype(np.float32):
        interpreter.set_tensor(input_details[0]["index"], row.reshape(1, -1))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        predictions.append(int(np.argmax(output)))

    predictions = np.array(predictions)
    return accuracy_score(y_test, predictions), predictions


def get_preprocessed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return readable feature names after numeric preprocessing and categorical encoding."""
    try:
        names = preprocessor.get_feature_names_out()
        return [name.replace("numeric__", "").replace("categorical__", "") for name in names]
    except Exception:
        return [f"Feature {index + 1}" for index in range(len(preprocessor.transformers_))]


def aggregate_feature_importance(
    model_pipeline: Pipeline,
    original_feature_columns: list[str],
) -> pd.DataFrame:
    """Aggregate encoded feature importance back to original dataset feature names."""
    importances = model_pipeline.named_steps["model"].feature_importances_
    transformed_names = get_preprocessed_feature_names(model_pipeline.named_steps["preprocessor"])

    rows = []
    for transformed_name, importance in zip(transformed_names, importances):
        matched_feature = next(
            (feature for feature in original_feature_columns if transformed_name == feature or transformed_name.startswith(f"{feature}_")),
            transformed_name,
        )
        rows.append({"Feature": matched_feature, "Importance": importance})

    return (
        pd.DataFrame(rows)
        .groupby("Feature", as_index=False)["Importance"]
        .sum()
        .sort_values("Importance", ascending=False)
    )


@st.cache_resource(show_spinner="Training dynamic ML and TinyML models...")
def train_and_evaluate_models(df: pd.DataFrame, target_column: str, dataset_key: str) -> dict:
    """Train models, compare metrics, run cross-validation, and convert TensorFlow to TFLite."""
    del dataset_key  # The value is used by Streamlit cache to invalidate old training results.
    (
        x,
        y,
        x_train,
        x_test,
        y_train,
        y_test,
        label_encoder,
        feature_columns,
        numeric_columns,
        categorical_columns,
    ) = get_train_test_data(df, target_column)

    sklearn_models = create_sklearn_models(numeric_columns, categorical_columns)
    trained_models = {}
    metrics_rows = []
    predictions = {}

    for name, model in sklearn_models.items():
        start_time = time.perf_counter()
        model.fit(x_train, y_train)
        training_time = time.perf_counter() - start_time

        y_pred = model.predict(x_test)
        predictions[name] = y_pred
        trained_models[name] = model

        metrics_rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
                "F1 Score": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "Training Time (sec)": training_time,
                "Saved Model Size (KB)": bytes_to_kb(model_size_bytes(model)),
            }
        )

    tf_preprocessor = create_preprocessor(numeric_columns, categorical_columns, scale_numeric=True)
    x_train_processed = tf_preprocessor.fit_transform(x_train).astype(np.float32)
    x_test_processed = tf_preprocessor.transform(x_test).astype(np.float32)

    tf_model = build_tensorflow_model(x_train_processed.shape[1], len(label_encoder.classes_))
    tf_start = time.perf_counter()
    history = tf_model.fit(
        x_train_processed,
        y_train,
        validation_data=(x_test_processed, y_test),
        epochs=30,
        verbose=0,
    )
    tf_training_time = time.perf_counter() - tf_start

    tf_probabilities = tf_model.predict(x_test_processed, verbose=0)
    tf_predictions = np.argmax(tf_probabilities, axis=1)
    predictions["TensorFlow Neural Network"] = tf_predictions
    trained_models["TensorFlow Neural Network"] = tf_model

    tf_model_size = model_size_bytes(tf_model)
    metrics_rows.append(
        {
            "Model": "TensorFlow Neural Network",
            "Accuracy": accuracy_score(y_test, tf_predictions),
            "Precision": precision_score(y_test, tf_predictions, average="macro", zero_division=0),
            "Recall": recall_score(y_test, tf_predictions, average="macro", zero_division=0),
            "F1 Score": f1_score(y_test, tf_predictions, average="macro", zero_division=0),
            "Training Time (sec)": tf_training_time,
            "Saved Model Size (KB)": bytes_to_kb(tf_model_size),
        }
    )

    tflite_start = time.perf_counter()
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    tflite_model = converter.convert()
    tflite_conversion_time = time.perf_counter() - tflite_start
    tflite_accuracy, tflite_predictions = evaluate_tflite_model(tflite_model, x_test_processed, y_test)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df["Accuracy (%)"] = (metrics_df["Accuracy"] * 100).round(2)
    metrics_df["Precision (%)"] = (metrics_df["Precision"] * 100).round(2)
    metrics_df["Recall (%)"] = (metrics_df["Recall"] * 100).round(2)
    metrics_df["F1 Score (%)"] = (metrics_df["F1 Score"] * 100).round(2)
    metrics_df["Training Time (sec)"] = metrics_df["Training Time (sec)"].round(4)

    cv_results = run_cross_validation(x, y, label_encoder, numeric_columns, categorical_columns)

    return {
        "x": x,
        "y": y,
        "x_test": x_test,
        "x_test_processed": x_test_processed,
        "y_test": y_test,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "label_encoder": label_encoder,
        "tf_preprocessor": tf_preprocessor,
        "trained_models": trained_models,
        "metrics_df": metrics_df,
        "predictions": predictions,
        "history": history.history,
        "tf_model_size": tf_model_size,
        "tflite_model": tflite_model,
        "tflite_accuracy": tflite_accuracy,
        "tflite_predictions": tflite_predictions,
        "tflite_size": len(tflite_model),
        "tflite_conversion_time": tflite_conversion_time,
        "tf_training_time": tf_training_time,
        "cv_results": cv_results,
    }


def run_cross_validation(
    x: pd.DataFrame,
    y: np.ndarray,
    label_encoder: LabelEncoder,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    """Run 5-fold cross-validation for every model with the current feature set."""
    class_counts = pd.Series(y).value_counts()
    n_splits = int(min(5, class_counts.min()))

    if n_splits < 2:
        return pd.DataFrame(
            {
                "Model": ["Not available"],
                "Fold 1": [np.nan],
                "Fold 2": [np.nan],
                "Fold 3": [np.nan],
                "Fold 4": [np.nan],
                "Fold 5": [np.nan],
                "Average Accuracy": [np.nan],
            }
        )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for name, model in create_sklearn_models(numeric_columns, categorical_columns).items():
        scores = cross_val_score(model, x, y, cv=cv, scoring="accuracy")
        rows.append(make_cv_row(name, scores))

    tf_scores = []
    for train_index, test_index in cv.split(x, y):
        x_train_fold, x_test_fold = x.iloc[train_index], x.iloc[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        preprocessor = create_preprocessor(numeric_columns, categorical_columns, scale_numeric=True)
        x_train_fold = preprocessor.fit_transform(x_train_fold).astype(np.float32)
        x_test_fold = preprocessor.transform(x_test_fold).astype(np.float32)

        fold_model = build_tensorflow_model(x_train_fold.shape[1], len(label_encoder.classes_))
        fold_model.fit(x_train_fold, y_train_fold, epochs=30, verbose=0)
        fold_probabilities = fold_model.predict(x_test_fold, verbose=0)
        fold_predictions = np.argmax(fold_probabilities, axis=1)
        tf_scores.append(accuracy_score(y_test_fold, fold_predictions))

    rows.append(make_cv_row("TensorFlow Neural Network", np.array(tf_scores)))

    cv_df = pd.DataFrame(rows)
    for col in ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Average Accuracy"]:
        cv_df[col] = (cv_df[col] * 100).round(2)
    return cv_df


def make_cv_row(model_name: str, scores: np.ndarray) -> dict:
    """Create a fixed five-fold display row, even when a small dataset permits fewer folds."""
    row = {"Model": model_name}
    for index in range(5):
        row[f"Fold {index + 1}"] = scores[index] if index < len(scores) else np.nan
    row["Average Accuracy"] = scores.mean()
    return row


def make_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, color: str = "#1f77b4"):
    fig = px.bar(df, x=x_col, y=y_col, text=y_col, title=title)
    fig.update_traces(marker_color=color, texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(yaxis_title=y_col, xaxis_title="", uniformtext_minsize=8)
    return fig


def render_dataset_section(df: pd.DataFrame, dataset_name: str, target_column: str, results: dict) -> None:
    st.header("Dataset Overview")
    st.caption(f"Current dataset: {dataset_name}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Feature Columns", len(results["feature_columns"]))
    col4.metric("Target Classes", df[target_column].nunique(dropna=True))

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Dataset Shape")
        st.write(f"The dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
        st.subheader("Column List")
        st.write(list(df.columns))
        st.subheader("Missing Values Check")
        st.dataframe(df.isnull().sum().rename("Missing Values"), use_container_width=True)
    with right:
        st.subheader("Dataset Statistics")
        st.dataframe(df.describe(include="all"), use_container_width=True)

    st.subheader("Dataset Information")
    st.code(get_dataset_info(df), language="text")


def render_eda_section(df: pd.DataFrame, target_column: str, numeric_columns: list[str]) -> None:
    st.header("Exploratory Data Analysis")

    if not numeric_columns:
        st.warning("No numeric feature columns were detected, so numeric EDA charts cannot be generated.")
        return

    st.subheader("Histograms for All Numeric Features")
    hist_df = df.melt(id_vars=target_column, value_vars=numeric_columns, var_name="Feature", value_name="Value")
    fig = px.histogram(hist_df, x="Value", color=target_column, facet_col="Feature", facet_col_wrap=2, nbins=18)
    fig.update_layout(height=max(520, 260 * int(np.ceil(len(numeric_columns) / 2))))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<p class="small-note">Histograms show the distribution of every detected numeric feature. When new numeric columns are added, they appear here automatically.</p>', unsafe_allow_html=True)

    st.subheader("Boxplots for All Numeric Features")
    fig = px.box(hist_df, x="Feature", y="Value", color=target_column, points="outliers")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<p class="small-note">Boxplots summarize median, spread, and outliers for each numeric feature grouped by the selected target.</p>', unsafe_allow_html=True)

    if len(numeric_columns) >= 2:
        st.subheader("Correlation Heatmap")
        correlation = df[numeric_columns].corr()
        fig = px.imshow(correlation, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<p class="small-note">Correlation measures how strongly numeric features move together. Strong correlations may indicate redundant or related measurements.</p>', unsafe_allow_html=True)

        st.subheader("Scatter Plot Matrix / Pair Plot")
        pair_columns = numeric_columns[:8]
        if len(numeric_columns) > 8:
            st.info("Showing the first 8 numeric features to keep the scatter matrix readable.")
        fig = px.scatter_matrix(df, dimensions=pair_columns, color=target_column)
        fig.update_layout(height=780)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<p class="small-note">The scatter matrix compares feature pairs and helps reveal class separation patterns automatically.</p>', unsafe_allow_html=True)

    st.subheader("Class Distribution Chart")
    class_counts = df[target_column].astype(str).value_counts().reset_index()
    class_counts.columns = [target_column, "Count"]
    fig = px.bar(class_counts, x=target_column, y="Count", color=target_column, text="Count")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<p class="small-note">Class distribution shows whether the selected target is balanced. Imbalanced classes can affect model performance.</p>', unsafe_allow_html=True)

    st.subheader("Feature Distribution Analysis")
    fig = px.violin(hist_df, x="Feature", y="Value", color=target_column, box=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<p class="small-note">Violin plots show both spread and density, making them useful for comparing feature behavior across classes.</p>', unsafe_allow_html=True)


def render_model_section(results: dict) -> None:
    st.header("Machine Learning Model Evaluation")
    metrics_df = results["metrics_df"]
    display_columns = [
        "Model",
        "Accuracy (%)",
        "Precision (%)",
        "Recall (%)",
        "F1 Score (%)",
        "Training Time (sec)",
        "Saved Model Size (KB)",
    ]
    st.dataframe(metrics_df[display_columns], use_container_width=True)

    st.subheader("Model Comparison Visualizations")
    charts = [
        ("Accuracy (%)", "Accuracy Comparison Graph", "#1f77b4"),
        ("Precision (%)", "Precision Comparison Graph", "#2ca02c"),
        ("Recall (%)", "Recall Comparison Graph", "#ff7f0e"),
        ("F1 Score (%)", "F1 Score Comparison Graph", "#9467bd"),
        ("Training Time (sec)", "Training Time Comparison Graph", "#d62728"),
        ("Saved Model Size (KB)", "Model Size Comparison Graph", "#17becf"),
    ]
    for metric, title, color in charts:
        st.plotly_chart(make_bar_chart(metrics_df, "Model", metric, title, color), use_container_width=True)


def render_cross_validation_section(results: dict) -> None:
    st.header("Cross Validation")
    cv_df = results["cv_results"]
    st.dataframe(cv_df, use_container_width=True)
    if cv_df["Average Accuracy"].notna().any():
        fig = make_bar_chart(cv_df.dropna(subset=["Average Accuracy"]), "Model", "Average Accuracy", "Cross Validation Comparison Chart", "#1f9d55")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('<p class="small-note">Cross-validation tests model consistency across multiple train-test splits. The app automatically reduces folds when a dataset has very small classes.</p>', unsafe_allow_html=True)


def render_confusion_matrices(results: dict) -> None:
    st.header("Confusion Matrix")
    label_encoder = results["label_encoder"]
    y_test = results["y_test"]

    for model_name, y_pred in results["predictions"].items():
        st.subheader(model_name)
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
            x=label_encoder.classes_,
            y=label_encoder.classes_,
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_feature_importance(results: dict) -> None:
    st.header("Dynamic Feature Importance")
    rf_pipeline = results["trained_models"]["Random Forest"]
    importance_df = aggregate_feature_importance(rf_pipeline, results["feature_columns"])

    fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h", text="Importance")
    fig.update_traces(texttemplate="%{text:.3f}", marker_color="#1f77b4")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    best_feature = importance_df.iloc[0]["Feature"]
    st.info(
        f"Random Forest finds **{best_feature}** to be the most important current feature. "
        "If new dataset columns are added, this chart automatically includes them after retraining."
    )


def render_tensorflow_history(results: dict) -> None:
    st.header("TensorFlow Training History")
    history_df = pd.DataFrame(results["history"])
    history_df["Epoch"] = np.arange(1, len(history_df) + 1)

    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(x=history_df["Epoch"], y=history_df["loss"], mode="lines+markers", name="Training Loss"))
    loss_fig.add_trace(go.Scatter(x=history_df["Epoch"], y=history_df["val_loss"], mode="lines+markers", name="Validation Loss"))
    loss_fig.update_layout(title="Training Loss vs Epoch and Validation Loss vs Epoch", xaxis_title="Epoch", yaxis_title="Loss")
    st.plotly_chart(loss_fig, use_container_width=True)

    accuracy_fig = go.Figure()
    accuracy_fig.add_trace(go.Scatter(x=history_df["Epoch"], y=history_df["accuracy"], mode="lines+markers", name="Training Accuracy"))
    accuracy_fig.add_trace(go.Scatter(x=history_df["Epoch"], y=history_df["val_accuracy"], mode="lines+markers", name="Validation Accuracy"))
    accuracy_fig.update_layout(title="Training Accuracy vs Epoch and Validation Accuracy vs Epoch", xaxis_title="Epoch", yaxis_title="Accuracy")
    st.plotly_chart(accuracy_fig, use_container_width=True)

    with st.expander("Key TensorFlow Training Terms"):
        st.markdown(
            """
            **Epoch:** One complete pass through the training dataset.

            **Loss:** A number that shows prediction error. Lower loss usually means the model is learning better.

            **Accuracy:** The percentage of correct predictions.

            **Overfitting:** A condition where training accuracy is high but validation accuracy is low. It means the model memorized training data instead of learning general patterns.
            """
        )


def render_tinyml_section(results: dict) -> None:
    st.header("TinyML and TensorFlow Lite")
    tf_accuracy = results["metrics_df"].loc[
        results["metrics_df"]["Model"] == "TensorFlow Neural Network", "Accuracy"
    ].iloc[0]
    tflite_accuracy = results["tflite_accuracy"]
    tf_model_size = results["tf_model_size"]
    tflite_size = results["tflite_size"]
    size_reduction = (1 - (tflite_size / tf_model_size)) * 100

    summary_df = pd.DataFrame(
        [
            {"Metric": "TensorFlow Accuracy (%)", "Value": round(tf_accuracy * 100, 2)},
            {"Metric": "TensorFlow Lite Accuracy (%)", "Value": round(tflite_accuracy * 100, 2)},
            {"Metric": "TensorFlow Model Size (KB)", "Value": bytes_to_kb(tf_model_size)},
            {"Metric": "TensorFlow Lite Model Size (KB)", "Value": bytes_to_kb(tflite_size)},
            {"Metric": "Size Reduction (%)", "Value": round(size_reduction, 2)},
            {"Metric": "TensorFlow Training Time (sec)", "Value": round(results["tf_training_time"], 4)},
            {"Metric": "TensorFlow Lite Conversion Time (sec)", "Value": round(results["tflite_conversion_time"], 4)},
        ]
    )
    st.dataframe(summary_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        size_chart = pd.DataFrame(
            {"Model": ["TensorFlow", "TensorFlow Lite"], "Size (KB)": [bytes_to_kb(tf_model_size), bytes_to_kb(tflite_size)]}
        )
        st.plotly_chart(make_bar_chart(size_chart, "Model", "Size (KB)", "TensorFlow vs TensorFlow Lite Size", "#17becf"), use_container_width=True)
    with col2:
        acc_chart = pd.DataFrame(
            {"Model": ["TensorFlow", "TensorFlow Lite"], "Accuracy (%)": [round(tf_accuracy * 100, 2), round(tflite_accuracy * 100, 2)]}
        )
        st.plotly_chart(make_bar_chart(acc_chart, "Model", "Accuracy (%)", "TensorFlow vs TensorFlow Lite Accuracy", "#2ca02c"), use_container_width=True)

    st.subheader("TinyML Explanation")
    explanations = {
        "What is TinyML?": "TinyML is the practice of running machine learning models on small, low-power devices such as microcontrollers, sensors, and embedded boards.",
        "What is TensorFlow?": "TensorFlow is an open-source machine learning framework used to build, train, and deploy models, including neural networks.",
        "What is TensorFlow Lite?": "TensorFlow Lite is a lightweight version of TensorFlow designed for mobile, embedded, and edge devices.",
        "Why TensorFlow Lite is used?": "TensorFlow Lite reduces model size and improves inference speed, making models easier to deploy on devices with limited memory and power.",
        "Benefits of TinyML": "TinyML enables low latency, offline prediction, lower cloud cost, better privacy, and energy-efficient intelligent devices.",
        "Edge Computing Concepts": "Edge computing processes data near the source of generation instead of sending everything to the cloud, which improves speed and privacy.",
    }
    for title, explanation in explanations.items():
        with st.expander(title):
            st.write(explanation)


def render_dynamic_input(feature: str, df: pd.DataFrame) -> object:
    """Create the correct input widget for each feature automatically."""
    series = df[feature]
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        numeric_series = pd.to_numeric(series, errors="coerce")
        min_value = float(numeric_series.min()) if numeric_series.notna().any() else 0.0
        max_value = float(numeric_series.max()) if numeric_series.notna().any() else 1.0
        mean_value = float(numeric_series.mean()) if numeric_series.notna().any() else min_value
        if min_value == max_value:
            return st.number_input(feature, value=mean_value)
        return st.slider(feature, min_value=min_value, max_value=max_value, value=mean_value)

    options = sorted(series.dropna().astype(str).unique().tolist())
    if not options:
        options = [""]
    return st.selectbox(feature, options)


def predict_with_model(model_name: str, input_df: pd.DataFrame, results: dict) -> np.ndarray:
    """Return class probabilities for the selected model and dynamic input row."""
    model = results["trained_models"][model_name]

    if model_name == "TensorFlow Neural Network":
        model_input = results["tf_preprocessor"].transform(input_df).astype(np.float32)
        return model.predict(model_input, verbose=0)[0]

    if hasattr(model, "predict_proba"):
        return model.predict_proba(input_df)[0]

    predicted_class = model.predict(input_df)[0]
    probabilities = np.zeros(len(results["label_encoder"].classes_))
    probabilities[predicted_class] = 1.0
    return probabilities


def render_prediction_section(df: pd.DataFrame, results: dict) -> None:
    st.header("Interactive Dynamic Prediction")
    metrics_df = results["metrics_df"]
    best_model_name = metrics_df.sort_values(["Accuracy", "F1 Score"], ascending=False).iloc[0]["Model"]
    model_names = metrics_df["Model"].tolist()

    selected_model = st.selectbox(
        "Choose a model for prediction",
        model_names,
        index=model_names.index(best_model_name),
    )

    st.subheader("Enter Feature Values")
    input_values = {}
    feature_columns = results["feature_columns"]
    columns_per_row = 3
    for row_start in range(0, len(feature_columns), columns_per_row):
        cols = st.columns(min(columns_per_row, len(feature_columns) - row_start))
        for col, feature in zip(cols, feature_columns[row_start : row_start + columns_per_row]):
            with col:
                input_values[feature] = render_dynamic_input(feature, df)

    input_df = pd.DataFrame([input_values], columns=feature_columns)

    if st.button("Predict Class", type="primary"):
        probabilities = predict_with_model(selected_model, input_df, results)
        label_encoder = results["label_encoder"]
        predicted_index = int(np.argmax(probabilities))
        predicted_class = label_encoder.inverse_transform([predicted_index])[0]
        confidence = float(probabilities[predicted_index] * 100)

        st.markdown(
            f"""
            <div class="prediction-card">
                <h3>Predicted Class: {predicted_class}</h3>
                <p><strong>Model:</strong> {selected_model}</p>
                <p><strong>Confidence:</strong> {confidence:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        probability_df = pd.DataFrame(
            {
                "Class": label_encoder.classes_,
                "Probability (%)": np.round(probabilities * 100, 2),
            }
        )
        st.dataframe(probability_df, use_container_width=True)
        fig = px.bar(probability_df, x="Class", y="Probability (%)", text="Probability (%)", color="Class")
        st.plotly_chart(fig, use_container_width=True)


def render_project_summary(results: dict) -> None:
    st.header("Project Summary and Automatic Conclusions")
    metrics_df = results["metrics_df"]
    best_accuracy = metrics_df.sort_values(["Accuracy", "F1 Score"], ascending=False).iloc[0]
    fastest = metrics_df.sort_values("Training Time (sec)").iloc[0]
    smallest = metrics_df.sort_values("Saved Model Size (KB)").iloc[0]

    tf_size = results["tf_model_size"]
    tflite_size = results["tflite_size"]
    tflite_accuracy = results["tflite_accuracy"] * 100
    size_reduction = (1 - (tflite_size / tf_size)) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Accuracy Model", best_accuracy["Model"], f"{best_accuracy['Accuracy (%)']:.2f}%")
    col2.metric("Fastest Model", fastest["Model"], f"{fastest['Training Time (sec)']:.4f} sec")
    col3.metric("Smallest Model", smallest["Model"], f"{smallest['Saved Model Size (KB)']:.2f} KB")
    col4.metric("Best TinyML Model", "TensorFlow Lite", f"{tflite_accuracy:.2f}%")

    st.success(
        f"The dashboard used **{len(results['feature_columns'])} detected feature columns** and target **{results['target_column']}**. "
        f"The best accuracy model is **{best_accuracy['Model']}** with **{best_accuracy['Accuracy (%)']:.2f}%** accuracy. "
        f"TensorFlow Lite reached **{tflite_accuracy:.2f}%** accuracy and reduced model size by about **{size_reduction:.2f}%**."
    )


def validate_classification_setup(df: pd.DataFrame, target_column: str) -> None:
    """Stop early with clear student-friendly messages when a dataset cannot be classified."""
    if df.shape[1] < 2:
        st.error("The dataset must contain at least one feature column and one target column.")
        st.stop()

    target_unique = df[target_column].dropna().nunique()
    if target_unique < 2:
        st.error("The selected target column must contain at least two classes.")
        st.stop()

    if df[target_column].notna().sum() < 10:
        st.error("The selected target column has too few usable rows for train-test splitting.")
        st.stop()


def main() -> None:
    apply_page_style()
    st.title("Dynamic Machine Learning + TinyML Dashboard")
    st.caption("Upload any CSV classification dataset, select the target column, and let the app detect features automatically.")

    raw_df, dataset_name = load_uploaded_or_default_dataset()
    df = clean_dataset(raw_df)

    if df.empty:
        st.error("The dataset is empty after removing blank rows and columns.")
        st.stop()

    default_target_index = list(df.columns).index("Species") if "Species" in df.columns else len(df.columns) - 1
    target_column = st.sidebar.selectbox("Select target column", df.columns.tolist(), index=default_target_index)
    validate_classification_setup(df, target_column)

    dataset_key = make_dataset_key(df, target_column)
    results = train_and_evaluate_models(df, target_column, dataset_key)

    st.sidebar.subheader("Detected Columns")
    st.sidebar.write(f"Features: {len(results['feature_columns'])}")
    st.sidebar.write(f"Numeric: {len(results['numeric_columns'])}")
    st.sidebar.write(f"Categorical: {len(results['categorical_columns'])}")

    section = st.sidebar.radio(
        "Navigate",
        [
            "Dataset",
            "EDA",
            "Model Evaluation",
            "Cross Validation",
            "Confusion Matrix",
            "Feature Importance",
            "TensorFlow History",
            "TinyML",
            "Prediction",
            "Summary",
        ],
    )

    if section == "Dataset":
        render_dataset_section(df, dataset_name, target_column, results)
    elif section == "EDA":
        render_eda_section(df, target_column, results["numeric_columns"])
    elif section == "Model Evaluation":
        render_model_section(results)
    elif section == "Cross Validation":
        render_cross_validation_section(results)
    elif section == "Confusion Matrix":
        render_confusion_matrices(results)
    elif section == "Feature Importance":
        render_feature_importance(results)
    elif section == "TensorFlow History":
        render_tensorflow_history(results)
    elif section == "TinyML":
        render_tinyml_section(results)
    elif section == "Prediction":
        render_prediction_section(df, results)
    else:
        render_project_summary(results)


if __name__ == "__main__":
    main()
