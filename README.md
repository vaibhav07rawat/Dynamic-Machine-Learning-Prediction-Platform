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
+-- DATASETS/
|   +-- README.md
|   +-- bank-full.csv
|   +-- bank_full_clean.csv
|   +-- cleaned_data.csv
|   +-- iris_data_final.csv.xlsx
|   +-- wdbc.csv
```

## Available Datasets

All CSV and Excel files are stored in the **DATASETS** folder. Here's what's available:

- **bank-full.csv** - Bank marketing dataset (4.6 MB)
  - Features: Banking customer data for marketing campaigns
  - Use case: Classification and prediction models

- **bank_full_clean.csv** - Cleaned bank marketing dataset (3.7 MB)
  - Features: Pre-processed bank data with cleaned values
  - Use case: Ready-to-use dataset for model training

- **cleaned_data.csv** - Pre-processed dataset (2.3 MB)
  - Features: General cleaned dataset
  - Use case: Intermediate analysis and modeling

- **iris_data_final.csv.xlsx** - Iris dataset in Excel format (14.6 KB)
  - Features: Classic iris flower classification data
  - Use case: Quick demo and benchmarking

- **wdbc.csv** - Breast Cancer Diagnostic Dataset (124 KB)
  - Features: Medical diagnostic features
  - Use case: Classification models for medical analysis

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

The platform now supports multiple datasets stored in the **DATASETS** folder. Simply select any CSV or Excel file from the DATASETS folder using the file uploader, and the application will:
- Automatically detect features and target variables
- Preprocess the data
- Train multiple models
- Generate predictions

For more details about available datasets, see [DATASETS/README.md](DATASETS/README.md).

## Adding New Datasets

To add new datasets to the platform:
1. Place your CSV or Excel files in the **DATASETS** folder
2. No code modifications are required
3. The files will be automatically available in the application through the file uploader
