"""
Comprehensive demo of the Kaggle Framework Model Training Module

This script demonstrates:
1. Using individual components (trainer, evaluator)
2. Using the high-level ModelManager
3. Complete workflow from data to submission
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

# Add the src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.sklearn_trainer import SklearnTrainer
from core.model_evaluator import ModelEvaluator
from utils.model_manager import ModelManager

def demo_classification():
    """Demonstrate classification workflow"""
    print("="*60)
    print("CLASSIFICATION DEMO")
    print("="*60)
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=8,
        n_redundant=2, n_clusters_per_class=1, random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    print(f"Dataset: {X_df.shape[0]} samples, {X_df.shape[1]} features")
    print(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Method 1: Using individual components
    print("\n--- Method 1: Individual Components ---")
    
    # Create and train model
    trainer = SklearnTrainer('random_forest', 'classification', {'n_estimators': 100})
    
    # Split data manually
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
    )
    
    # Train model
    trainer.train(X_train, y_train, cv_folds=5)
    
    # Evaluate
    evaluator = ModelEvaluator('classification')
    results = evaluator.evaluate_model(trainer, X_test, y_test, X_train, y_train)
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    
    # Generate report
    report = evaluator.generate_report()
    print("\nModel Report:")
    print(report)
    
    # Method 2: Using ModelManager (Recommended)
    print("\n--- Method 2: ModelManager (High-level) ---")
    
    manager = ModelManager('classification', random_state=42)
    
    # Quick experiment with multiple algorithms
    comparison = manager.quick_experiment(X_df, y_series, test_size=0.2)
    
    # Get best model
    best_id, best_model = manager.get_best_model()
    print(f"\nBest model: {best_id}")
    
    # Create mock test data for submission
    X_test_submission = pd.DataFrame(
        np.random.randn(100, X_df.shape[1]), 
        columns=X_df.columns
    )
    test_ids = pd.Series(range(1000, 1100))
    
    # Create submission
    submission = manager.create_submission(
        X_test_submission, test_ids, 
        filename='classification_submission.csv'
    )
    print(f"Submission created: {submission.shape}")

def demo_regression():
    """Demonstrate regression workflow"""
    print("\n" + "="*60)
    print("REGRESSION DEMO")
    print("="*60)
    
    # Generate sample data
    X, y = make_regression(
        n_samples=1000, n_features=10, noise=0.1, random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    print(f"Dataset: {X_df.shape[0]} samples, {X_df.shape[1]} features")
    print(f"Target stats: mean={y_series.mean():.2f}, std={y_series.std():.2f}")
    
    # Using ModelManager for regression
    manager = ModelManager('regression', random_state=42)
    
    # Add specific models with custom parameters
    manager.add_model('rf_100', 'random_forest', {'n_estimators': 100})
    manager.add_model('rf_200', 'random_forest', {'n_estimators': 200})
    manager.add_model('ridge', 'ridge', {'alpha': 1.0})
    manager.add_model('lasso', 'lasso', {'alpha': 0.1})
    
    # Prepare data
    manager.prepare_data(X_df, y_series, test_size=0.2)
    
    # Train all models
    manager.train_all_models(cv_folds=5)
    
    # Evaluate all models
    manager.evaluate_all_models()
    
    # Compare models
    comparison = manager.compare_models()
    
    # Generate full report
    full_report = manager.generate_full_report('regression_report.txt')
    
    print("Full report generated!")

def demo_advanced_usage():
    """Demonstrate advanced features"""
    print("\n" + "="*60)
    print("ADVANCED FEATURES DEMO")
    print("="*60)
    
    # Generate data
    X, y = make_classification(n_samples=500, n_features=15, n_informative=10, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])
    y_series = pd.Series(y, name='target')
    
    manager = ModelManager('classification')
    manager.prepare_data(X_df, y_series)
    
    # Add models with different parameters
    manager.add_model('rf_small', 'random_forest', {'n_estimators': 50, 'max_depth': 5})
    manager.add_model('rf_large', 'random_forest', {'n_estimators': 200, 'max_depth': 15})
    manager.add_model('gb_model', 'gradient_boosting', {'n_estimators': 100, 'learning_rate': 0.1})
    
    # Train models
    manager.train_all_models()
    
    # Evaluate models
    manager.evaluate_all_models()
    
    # Model comparison
    comparison = manager.compare_models()
    
    # Feature importance for best model
    best_id, best_model = manager.get_best_model()
    importance = best_model.get_feature_importance()
    
    if importance is not None:
        print(f"\nTop 5 features for {best_id}:")
        print(importance.head().to_string(index=False))
    
    # Save best model
    manager.save_model(best_id, f'models/{best_id}_model.pkl')
    
    print(f"\nAdvanced demo completed!")

def main():
    """Run all demonstrations"""
    # Create directories for outputs
    os.makedirs('models', exist_ok=True)
    
    try:
        demo_classification()
        demo_regression()
        demo_advanced_usage()
        
        print("\n" + "="*60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Generated files:")
        print("- classification_submission.csv")
        print("- regression_report.txt")
        print("- models/[best_model]_model.pkl")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()