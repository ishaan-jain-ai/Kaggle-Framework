# Kaggle Framework

A comprehensive, object-oriented machine learning framework designed to streamline Kaggle competition workflows. This framework provides modular components for data handling, feature engineering, model training, and evaluation.

## 🚀 Features

### ✅ **Completed Modules**

#### 📊 **Data Handler Module** (`ml-data-handler/`)
- **Multi-format Data Reading**: Support for CSV, Excel, text, and image files
- **Intelligent Analysis**: Automatic data type detection and analysis
- **Memory Optimization**: Efficient memory usage tracking and optimization
- **Text Analytics**: Advanced text column analysis and preprocessing

#### 🔧 **Feature Engineering Module** (`ml-feature-engg/`)
- **Tabular Processing**: Missing value handling, outlier detection, feature scaling
- **Text Processing**: NLP preprocessing with NLTK integration
- **Flexible Architecture**: Extensible processor classes for different data types
- **Processing History**: Track and replay feature engineering steps

#### 🤖 **Model Training & Evaluation Module** (`ml-model-trainer/`) - **NEW!**
- **Multiple Algorithms**: Support for 6+ classification and regression algorithms
- **Comprehensive Evaluation**: Detailed metrics, visualizations, and reports
- **Model Comparison**: Side-by-side comparison of multiple models
- **Kaggle Submission**: Direct submission file generation
- **High-level API**: Easy-to-use ModelManager for complete workflows

## 📁 Project Structure

```
kaggle-framework/
├── ml-data-handler/          # Data reading and analysis
│   ├── src/
│   │   ├── core/            # Core reader classes
│   │   └── utils/           # Data analysis utilities
├── ml-feature-engg/         # Feature engineering
│   ├── src/
│   │   ├── core/            # Processing classes
│   │   └── utils/           # Preprocessing utilities
├── ml-model-trainer/        # Model training & evaluation
│   ├── src/
│   │   ├── core/            # Trainer and evaluator classes
│   │   └── utils/           # Model management utilities
│   └── examples/            # Usage examples and demos
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd kaggle-framework
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download NLTK data** (for text processing):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 🚀 Quick Start

### Complete Workflow Example

```python
import pandas as pd
from ml_model_trainer.src.utils.model_manager import ModelManager

# Load your data
df = pd.read_csv('your_data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Initialize model manager
manager = ModelManager('classification', random_state=42)

# Quick experiment with multiple algorithms
results = manager.quick_experiment(X, y, test_size=0.2)
print(results)

# Get best model and create submission
test_data = pd.read_csv('test_data.csv')
test_ids = test_data['id']
test_features = test_data.drop('id', axis=1)

submission = manager.create_submission(
    test_features, test_ids, 
    filename='my_submission.csv'
)
```

### Individual Module Usage

#### Data Reading
```python
from ml_data_handler.src.core.tabular_reader import TabularReader

reader = TabularReader('data.csv')
reader.read_data()
details = reader.get_details()
print(f"Dataset has {details['num_columns']} columns")
```

#### Feature Engineering
```python
from ml_feature_engg.src.core.tabular_processor import TabularProcessor

processor = TabularProcessor(data)
processed_data = processor.clean().engineer_features().data
```

#### Model Training
```python
from ml_model_trainer.src.core.sklearn_trainer import SklearnTrainer
from ml_model_trainer.src.core.model_evaluator import ModelEvaluator

# Train model
trainer = SklearnTrainer('random_forest', 'classification')
trainer.train(X_train, y_train, cv_folds=5)

# Evaluate model
evaluator = ModelEvaluator('classification')
results = evaluator.evaluate_model(trainer, X_test, y_test)
print(f"Accuracy: {results['accuracy']:.4f}")
```

## 📋 Supported Algorithms

### Classification
- Random Forest
- Gradient Boosting
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree

### Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Linear Regression
- Ridge Regression
- Lasso Regression
- Support Vector Regression (SVR)
- K-Nearest Neighbors Regressor
- Decision Tree Regressor

## 📊 Evaluation Metrics

### Classification
- Accuracy, Precision, Recall, F1-Score
- ROC AUC (binary classification)
- Confusion Matrix
- Classification Report
- Cross-validation scores

### Regression
- R² Score, RMSE, MAE, MSE
- Residual Analysis
- Cross-validation scores

## 🎯 Use Cases

This framework is perfect for:

- **Kaggle Competitions**: Complete pipeline from data to submission
- **Data Science Projects**: Structured approach to ML workflows
- **Model Comparison**: Easy comparison of multiple algorithms
- **Prototyping**: Quick experimentation with different approaches
- **Learning**: Well-documented, educational codebase

## 🔧 Advanced Features

### Model Manager Capabilities
- **Batch Training**: Train multiple models simultaneously
- **Auto-Selection**: Automatically select best performing model
- **Model Persistence**: Save and load trained models
- **Comprehensive Reporting**: Detailed evaluation reports
- **Visualization**: Built-in plotting for evaluation results

### Extensibility
- **Custom Algorithms**: Easy to add new algorithms
- **Custom Processors**: Extend feature engineering capabilities
- **Custom Metrics**: Add domain-specific evaluation metrics

## 📈 Example Outputs

### Model Comparison
```
Model Comparison:
        model_name  accuracy  precision    recall  f1_score  cv_mean   cv_std
0  random_forest     0.9200     0.9180    0.9200    0.9189   0.9050   0.0127
1  gradient_boosting 0.9100     0.9090    0.9100    0.9095   0.8980   0.0145
2  logistic_regression 0.8800   0.8750    0.8800    0.8775   0.8720   0.0098
```

### Kaggle Submission
```
Submission saved to submission.csv
Submission shape: (10000, 2)
```

## 🛣️ Roadmap

### Planned Features
- **XGBoost/LightGBM Integration**: Advanced gradient boosting
- **Neural Network Support**: Deep learning capabilities
- **AutoML Features**: Automated hyperparameter tuning
- **Time Series Support**: Specialized time series algorithms
- **Ensemble Methods**: Advanced model combination techniques

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your enhancements
4. Follow the existing code structure and patterns
5. Add tests and documentation
6. Submit a pull request

## 📜 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

Built with love for the Kaggle community. Special thanks to:
- scikit-learn for the robust ML algorithms
- pandas for data manipulation
- The open-source ML community

---

**Ready to dominate Kaggle competitions? Get started with the framework today!** 🏆