# Dynamic ML + TinyML Streamlit Dashboard

This project is a complete B.Tech final-year style dashboard for tabular classification datasets.
It starts with the Iris dataset by default, but it also supports uploading any CSV file and selecting
the target column dynamically.

## Project Features

- CSV upload with `st.file_uploader`
- Dynamic target selection with `st.selectbox`
- Automatic feature detection by excluding only the selected target column
- Automatic numeric and categorical column detection
- Missing value handling, label encoding, one-hot encoding, and scaling where required
- Dataset preview, shape, column list, statistics, information, and missing value checks
- EDA with Plotly histograms, boxplots, heatmap, scatter matrix, class distribution, and distribution analysis
- Model training and comparison:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine
  - Naive Bayes
  - TensorFlow Neural Network
- Accuracy, precision, recall, F1 score, training time, and saved model size
- 5-fold cross-validation for every model
- Confusion matrices for every model
- Dynamic Random Forest feature importance
- TensorFlow training history charts
- TensorFlow Lite conversion and TinyML comparison
- Dynamic interactive prediction inputs for all detected features
- Automatic project summary and conclusions

## Folder Structure

```text
.
+-- app.py
+-- requirements.txt
+-- README.md
+-- data/
|   +-- iris_data.csv
+-- DATASETS/
    +-- (Add your CSV and Excel files here)
```

## Datasets Folder

The `DATASETS/` folder is dedicated to storing all CSV and Excel files that you want to use with the platform. Simply place your dataset files here, and you can:

1. Upload them through the Streamlit dashboard interface, or
2. Reference them directly in the application

### Supported File Formats
- `.csv` - Comma-separated values
- `.xlsx` - Excel spreadsheets
- `.xls` - Legacy Excel files

## How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit application:

```bash
streamlit run app.py
```

The app will open in your browser. If it does not open automatically, use the local URL shown in the terminal.

## Dynamic Dataset Support

The app no longer requires hardcoded feature columns. It automatically uses every column except the
selected target column as input features.

The original Iris Excel file was converted into `data/iris_data.csv`, which is used as the default
demo dataset when no CSV is uploaded.
