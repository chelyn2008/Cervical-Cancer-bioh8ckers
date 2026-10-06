# Cervical Cancer Classification — bioh8ckers

A Python machine learning project exploring binary classification of the `Dx` diagnosis label using patient demographic and clinical data.

The project follows a workflow from data preparation and exploratory analysis to model training, evaluation, and comparison. It is implemented in Jupyter notebooks.

## Project Objectives

- Prepare patient data for analysis and modelling.
- Explore feature distributions, correlations, and class imbalance.
- Compare multiple classification algorithms.
- Evaluate performance using accuracy, precision, recall, F1-score, and ROC-AUC.

## Repository Contents

| File | Purpose |
| --- | --- |
| [data_prep_visualisation.ipynb](./data_prep_visualisation.ipynb) | Data cleaning, exploratory analysis, visualisation, and export of the cleaned dataset. |
| [prediction_model.ipynb](./prediction_model.ipynb) | Feature scaling, model training, evaluation, comparison, and model export. |

## Workflow

### 1. Data Preparation and Exploration

The preparation notebook:
- Fills missing binary and duration values with zero and selected numerical values with their medians.
- Removes duplicate rows and columns containing only zeros.
- Visualises the distribution of the target label and feature correlations.
- Examines feature distributions for records with `Dx = 1`.
- Exports the processed data to `cleaned_cervical_cancer.csv`.

### 2. Model Training

The modelling notebook uses `Dx` as the target and the remaining columns as input features. It:
- Splits the data into 80% training and 20% testing sets with `random_state=42`.
- Fits a `StandardScaler` on the training features and transforms both sets.
- Applies an experimental multiplier of 1.5 to the scaled `Dx:HPV` and `Hormonal Contraceptives` features.
- Trains Logistic Regression, Random Forest, Support Vector Machine (SVM), and XGBoost classifiers.

Balanced class weights are used for Logistic Regression, Random Forest, and SVM.

### 3. Evaluation

Models are compared using classification reports, confusion matrices, ROC curves, and a summary of evaluation metrics. The notebook also exports the fitted Random Forest classifier as `random_forest_model.pkl`.

## Recorded Results

The saved modelling notebook contains the following results from one train/test split:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9817 | 0.7143 | 0.8333 | 0.7692 | 0.9979 |
| Random Forest | 0.9939 | 0.8571 | 1.0000 | 0.9231 | 0.9989 |
| SVM | 0.9817 | 0.7143 | 0.8333 | 0.7692 | 0.9968 |
| XGBoost | 0.9878 | 0.8333 | 0.8333 | 0.8333 | 0.9968 |

Precision, recall, and F1-score refer to the positive class. Random Forest achieved the highest scores in this recorded run.

These results describe the notebook experiment. The test set contained 164 records, with only 6 positive examples, so the positive-class metrics are sensitive to individual predictions. Results may vary when rerunning the notebooks.

## Technologies

- **Python**
- **Pandas and NumPy** — data preparation and analysis
- **Matplotlib and Seaborn** — visualisation
- **scikit-learn** — preprocessing, classifiers, and evaluation
- **XGBoost** — gradient boosting classifier
- **Joblib** — model export
- **Jupyter Notebook** — interactive development

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/chelyn2008/Cervical-Cancer-bioh8ckers.git
cd Cervical-Cancer-bioh8ckers
```

### 2. Install dependencies

Using a Python 3 environment:

```bash
python -m pip install notebook pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 4. Run the notebooks in order

1. Open `data_prep_visualisation.ipynb` and run its cells from top to bottom to generate `cleaned_cervical_cancer.csv`.
2. Open `prediction_model.ipynb` and run its cells from top to bottom to train and evaluate the classifiers.

Run both notebooks with the repository root as the working directory so their relative CSV paths resolve correctly. The cleaned CSV and exported model are generated locally and are not currently included in the repository.

## Scope and Limitations

This is an exploratory learning project and is not a clinically validated diagnostic tool.

The current workflow performs imputation before the train/test split and retains diagnosis-related and screening-result columns as predictors. Future evaluation should fit preprocessing only on the training data and review which features would be available at prediction time to address possible data leakage.

The exported model contains only the classifier. Reusing it requires the same feature order, fitted scaler, and feature multipliers used during training; those preprocessing steps are not bundled into the exported file.
