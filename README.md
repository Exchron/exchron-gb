# Exoplanet Prediction Using Gradient Boosting

This project builds a machine learning model to predict whether a Kepler Object of Interest (KOI) is likely to be an exoplanet candidate using gradient boosting classifier. The model outputs probabilities for both candidate and non-candidate classifications and includes comprehensive hyperparameter tuning for optimal performance.

## 📁 Project Structure

```
exoplanet-prediction/
├── data/
│   ├── KOI-Playground-Train-Data.csv    # Training dataset
│   └── KOI-Playground-Test-Data.csv     # Test dataset
├── exoplanet_prediction.ipynb           # Main training notebook
├── model_evaluation.ipynb               # Model evaluation notebook
├── exoplanet_gradient_boosting_model.joblib  # Saved trained model
├── label_encoder.joblib                 # Label encoder for target variable
├── feature_names.csv                    # List of model features
├── test_predictions.csv                 # Test set predictions (generated)
├── test_performance_metrics.csv         # Test performance metrics (generated)
└── README.md                           # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Required packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`

### Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Usage

1. **Training the Model** (already completed):
   ```bash
   jupyter notebook exoplanet_prediction.ipynb
   ```

2. **Evaluating on Test Data**:
   ```bash
   jupyter notebook model_evaluation.ipynb
   ```

3. **Using the Trained Model** (Python script):
   ```python
   import joblib
   import pandas as pd
   
   # Load the model
   model = joblib.load('exoplanet_gradient_boosting_model.joblib')
   label_encoder = joblib.load('label_encoder.joblib')
   feature_names = pd.read_csv('feature_names.csv')['feature_names'].tolist()
   
   # Make predictions
   predictions = model.predict_proba(X_new)
   ```

## 📊 Dataset Description

The project uses the Kepler Object of Interest (KOI) dataset containing astronomical measurements of potential exoplanets:

### Features Used by the Model
The model uses the following astronomical features (exact list in `feature_names.csv`):
- **koi_period**: Orbital period of the KOI
- **koi_prad**: Planet radius in Earth radii
- **koi_teq**: Equilibrium temperature of the planet
- **koi_insol**: Insolation flux received by the planet
- **koi_dor**: Planet-star distance over star radius
- And other KOI-specific measurements

### Target Variable
- **candidate**: Object is likely an exoplanet
- **non-candidate**: Object is likely not an exoplanet

## 🤖 Model Details

### Algorithm
- **Model Type**: Gradient Boosting Classifier
- **Library**: scikit-learn
- **Hyperparameter Tuning**: GridSearchCV with 4-fold cross-validation

### Key Features
- **Probability Output**: Returns confidence scores for each prediction
- **Feature Importance**: Identifies most predictive astronomical measurements
- **Cross-validation**: 5-fold CV for robust performance estimation
- **Missing Value Handling**: Median imputation for numerical features

### Performance Metrics
The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC (primary metric for hyperparameter tuning)

## 📓 Notebook Descriptions

### 1. `exoplanet_prediction.ipynb` - Main Training Notebook

**Purpose**: Complete model development pipeline from data exploration to model export.

**Sections**:
1. **Import Libraries**: Load required packages for ML and visualization
2. **Data Exploration**: Analyze dataset structure, missing values, and distributions  
3. **Data Visualization**: Create plots showing feature distributions and correlations
4. **Data Preprocessing**: Handle missing values, encode target variable
5. **Feature Engineering**: Prepare feature matrix and target vector
6. **Baseline Training**: Train initial model for baseline performance
7. **Hyperparameter Tuning**: Grid search optimization using ROC-AUC
8. **Final Training**: Train optimized model on full dataset
9. **Model Evaluation**: Comprehensive performance analysis with visualizations
10. **Model Export**: Save model, encoder, and feature names

**Key Outputs**:
- Trained gradient boosting model
- Performance visualizations (confusion matrix, ROC curve, feature importance)
- Cross-validation scores
- Model artifacts for deployment

### 2. `model_evaluation.ipynb` - Test Set Evaluation

**Purpose**: Evaluate the trained model on unseen test data.

**Sections**:
1. **Import Libraries**: Load evaluation packages
2. **Load Model Components**: Import saved model, encoder, and features
3. **Load Test Data**: Import and explore test dataset
4. **Data Preprocessing**: Apply same preprocessing as training
5. **Generate Predictions**: Create predictions and probabilities
6. **Performance Evaluation**: Calculate metrics (if true labels available)
7. **Visualizations**: Generate evaluation plots and charts
8. **Export Results**: Save predictions and performance metrics

**Key Outputs**:
- Test set predictions with probabilities
- Performance metrics comparison
- Prediction confidence analysis
- Exportable results for reporting

## 🔧 Model Configuration

### Hyperparameter Grid (Optimized)
The model was tuned across these parameter ranges:
- **n_estimators**: [100, 200, 300]
- **learning_rate**: [0.05, 0.1, 0.15] 
- **max_depth**: [3, 5, 7]
- **subsample**: [0.8, 0.9, 1.0]
- **min_samples_split**: [2, 5, 10]

### Best Parameters (Found via GridSearch)
The optimal parameters are stored in the trained model and can be accessed via:
```python
model = joblib.load('exoplanet_gradient_boosting_model.joblib')
print(model.get_params())
```

## 📈 Performance Summary

### Training Results
- **Cross-validation ROC-AUC**: [Displayed in training notebook]
- **Feature Count**: [Number shown in notebook output]
- **Training Samples**: 7,651 samples

### Key Insights
- **Most Important Feature**: [Shown in feature importance analysis]
- **Model Type**: Gradient Boosting provides excellent performance for this tabular data
- **Confidence Scores**: Model provides reliable probability estimates
- **Generalization**: Cross-validation ensures robust performance on unseen data

## 🔍 Feature Importance

The model automatically identifies the most predictive astronomical measurements for exoplanet detection. Feature importance rankings are:

1. Available in training notebook visualizations
2. Exportable from the trained model
3. Used to understand which measurements are most critical

## 📝 Usage Examples

### Loading and Using the Model
```python
import joblib
import pandas as pd
import numpy as np

# Load model components
model = joblib.load('exoplanet_gradient_boosting_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')
features = pd.read_csv('feature_names.csv')['feature_names'].tolist()

# Prepare your data (same preprocessing as training)
# X_new should have the same features in the same order

# Make predictions
probabilities = model.predict_proba(X_new)
predictions = model.predict(X_new)

# Interpret results
for i in range(len(predictions)):
    pred_class = label_encoder.inverse_transform([predictions[i]])[0]
    confidence = np.max(probabilities[i])
    print(f"Sample {i+1}: {pred_class} (confidence: {confidence:.3f})")
```

### Batch Prediction
```python
# Load test data
test_data = pd.read_csv('your_test_data.csv')

# Preprocess (same steps as training)
# ... preprocessing code ...

# Generate predictions
results = pd.DataFrame({
    'id': test_data.index,
    'prediction': label_encoder.inverse_transform(model.predict(X_test)),
    'candidate_prob': model.predict_proba(X_test)[:, 0],
    'non_candidate_prob': model.predict_proba(X_test)[:, 1]
})

# Save results
results.to_csv('predictions.csv', index=False)
```

## 🛠️ Troubleshooting

### Common Issues

1. **Feature Mismatch Error**
   - Ensure your data has the exact same features as in `feature_names.csv`
   - Check feature order and names

2. **Missing Values**
   - Apply the same median imputation used during training
   - Check the preprocessing steps in the training notebook

3. **Prediction Format**
   - Use `predict_proba()` for probability scores
   - Use `predict()` for class predictions
   - Use `label_encoder.inverse_transform()` to get original class names

### Model Retraining
To retrain the model with new data:
1. Add new data to the training set
2. Run the complete `exoplanet_prediction.ipynb` notebook
3. The model will be retrained and saved automatically

## 📚 Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
joblib>=1.1.0
```

## 🎯 Next Steps

### Model Improvements
- **Ensemble Methods**: Combine with Random Forest or XGBoost
- **Feature Engineering**: Create polynomial or interaction features  
- **Advanced Tuning**: Try Bayesian optimization for hyperparameters
- **Deep Learning**: Experiment with neural networks for comparison

### Deployment Options
- **API Deployment**: Create REST API using Flask/FastAPI
- **Batch Processing**: Set up automated prediction pipelines
- **Real-time Inference**: Deploy for live astronomical data streams
- **Model Monitoring**: Track performance drift over time

### Data Enhancements
- **Feature Selection**: Use statistical tests or recursive elimination
- **Data Augmentation**: Synthetic data generation techniques
- **Cross-validation Strategy**: Time-series or stratified splits
- **External Data**: Incorporate additional astronomical catalogs

## 📄 License

This project is provided for educational and research purposes. Please cite appropriately if used in academic work.

## 📧 Contact

For questions about the model implementation, performance, or usage, please refer to the detailed documentation in the Jupyter notebooks or create an issue in this repository.

---

**Last Updated**: October 2025  
**Model Version**: 1.0  
**Python Version**: 3.7+