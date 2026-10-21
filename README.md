# Diabetes Classification: Model Comparison and Threshold Analysis

## Project Overview

This project explores how machine learning models classify diabetes outcomes using clinical measurements from the Pima Indians Diabetes dataset.

I focused on three questions: how well different models perform, which features are most useful for prediction, and how changing the classification threshold affects the number of positive cases identified.

The project covers data cleaning, exploratory analysis, model training, cross-validation, feature importance, and threshold analysis.

## Dataset

The Pima Indians Diabetes dataset contains 768 observations from women aged 21 or older with Pima Indian heritage.

It includes eight input features:

`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, and `Age`.

The target variable is `Outcome`, where 0 represents a negative outcome and 1 represents a positive outcome.

Some measurements contain zero values that were treated as missing during preprocessing.

## Tools and Libraries

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook
- Git and GitHub

## Analysis Workflow

1. **Data Cleaning and EDA:** Explored the dataset, identified invalid zero values, examined feature distributions, and visualized relationships with the outcome.
2. **Preprocessing:** Used median imputation and scaling where appropriate, with preprocessing included in scikit-learn pipelines.
3. **Model Training:** Trained Logistic Regression, Random Forest, and Gradient Boosting.
4. **Model Evaluation:** Compared accuracy, precision, recall, F1-score, confusion matrices, and ROC-AUC.
5. **Cross-Validation:** Used stratified five-fold cross-validation on the training data to compare model performance.
6. **Feature Importance:** Examined which features contributed most to Random Forest predictions.
7. **Threshold Analysis:** Explored how different classification thresholds changed precision, recall, and prediction errors.

## Key Results

### Model Comparison

The following results are mean scores from five-fold cross-validation on the training data.

| Model | Accuracy | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.780 | 0.573 | 0.641 | 0.836 |
| Random Forest | 0.769 | 0.583 | 0.634 | 0.821 |
| Gradient Boosting | 0.756 | 0.578 | 0.620 | 0.807 |

Logistic Regression achieved the highest mean accuracy, F1-score, and ROC-AUC. Random Forest had slightly higher recall, but the difference was small.

The results did not show a clear advantage from using a more complex model.

### Feature Importance

Glucose had the highest impurity-based feature importance in Random Forest, followed by BMI and Age. These scores describe how the model uses the features; they do not establish causal relationships.

### Threshold Analysis

I also examined the effect of changing the Logistic Regression classification threshold using out-of-fold training predictions.

| Metric | Threshold 0.50 | Threshold 0.30 |
|---|---:|---:|
| Precision | 0.738 | 0.586 |
| Recall | 0.572 | 0.765 |

Lowering the threshold increased recall but also produced more false positive predictions.

The threshold of 0.30 was an exploratory candidate based on an illustrative 75% recall target, not a clinically validated threshold.

## Project Structure

```text
diabetes-risk-prediction-ml/
├── data/
│   └── diabetes.csv
├── notebooks/
│   └── Pima_Indians_Diabetes_Classification.ipynb
├── README.md
└── requirements.txt
```

## How to Run

Clone the repository and install the required packages:

```bash
git clone https://github.com/sati2799/diabetes-risk-prediction-ml.git
cd diabetes-risk-prediction-ml
python -m pip install -r requirements.txt
```

Open the notebook from its directory so that its relative data path works:

```bash
cd notebooks
jupyter notebook Pima_Indians_Diabetes_Classification.ipynb
```

The notebook reads the dataset from `../data/diabetes.csv`.

## Limitations

- The dataset is relatively small and represents a specific population.
- Some clinical measurements contain missing or invalid values.
- Recall for the positive class remains limited at the default threshold.
- The test set was examined during model development and is not a fully untouched final evaluation set.
- The exploratory threshold selection was not independently validated.
- The models have not been evaluated on external clinical data.

This project is intended for learning and analytical comparison, not medical diagnosis or clinical decision-making.

## Next Steps

Possible improvements include independent validation, probability calibration, and a more rigorous assessment of threshold selection.

## References

- Pima Indians Diabetes dataset — UCI Machine Learning Repository
- Scikit-learn documentation: https://scikit-learn.org/stable/
- Pandas documentation: https://pandas.pydata.org/docs/
