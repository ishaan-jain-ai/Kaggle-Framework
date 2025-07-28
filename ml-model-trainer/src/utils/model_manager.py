import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from core.sklearn_trainer import SklearnTrainer
from core.model_evaluator import ModelEvaluator

class ModelManager:
    """
    High-level manager for orchestrating the complete ML pipeline.
    Handles multiple models, comparisons, and provides easy-to-use interface.
    """
    
    def __init__(self, task_type: str = 'classification', random_state: int = 42):
        """
        Initialize the model manager
        
        Args:
            task_type: 'classification' or 'regression'
            random_state: Random state for reproducibility
        """
        self.task_type = task_type
        self.random_state = random_state
        self.models = {}
        self.evaluation_results = {}
        self.evaluator = ModelEvaluator(task_type)
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, 
                    test_size: float = 0.2, stratify: bool = True) -> None:
        """
        Split data into training and testing sets
        
        Args:
            X: Features dataframe
            y: Target series
            test_size: Fraction of data to use for testing
            stratify: Whether to stratify the split (for classification)
        """
        stratify_param = y if (stratify and self.task_type == 'classification') else None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=stratify_param
        )
        
        print(f"Data split completed:")
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Features: {self.X_train.shape[1]}")
        
    def add_model(self, model_id: str, algorithm: str, 
                 model_params: Dict[str, Any] = None) -> 'ModelManager':
        """
        Add a model to the manager
        
        Args:
            model_id: Unique identifier for the model
            algorithm: Algorithm name (from SklearnTrainer)
            model_params: Parameters for the model
            
        Returns:
            Self for method chaining
        """
        model = SklearnTrainer(
            algorithm=algorithm,
            task_type=self.task_type,
            model_params=model_params,
            random_state=self.random_state
        )
        model.model_name = f"{self.task_type}_{algorithm}_{model_id}"
        self.models[model_id] = model
        
        print(f"Added model '{model_id}' with algorithm '{algorithm}'")
        return self
        
    def train_model(self, model_id: str, cv_folds: int = 5, **kwargs) -> 'ModelManager':
        """
        Train a specific model
        
        Args:
            model_id: ID of the model to train
            cv_folds: Number of cross-validation folds
            **kwargs: Additional training parameters
            
        Returns:
            Self for method chaining
        """
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not found")
        
        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first")
        
        model = self.models[model_id]
        model.train(self.X_train, self.y_train, cv_folds=cv_folds, **kwargs)
        
        print(f"Model '{model_id}' trained successfully")
        return self
    
    def train_all_models(self, cv_folds: int = 5, **kwargs) -> 'ModelManager':
        """
        Train all models in the manager
        
        Args:
            cv_folds: Number of cross-validation folds
            **kwargs: Additional training parameters
            
        Returns:
            Self for method chaining
        """
        for model_id in self.models:
            print(f"Training model: {model_id}")
            self.train_model(model_id, cv_folds, **kwargs)
        
        print(f"All {len(self.models)} models trained")
        return self
    
    def evaluate_model(self, model_id: str, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Evaluate a specific model
        
        Args:
            model_id: ID of the model to evaluate
            cv_folds: Number of cross-validation folds
            
        Returns:
            Evaluation results dictionary
        """
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not found")
        
        if self.X_test is None:
            raise ValueError("Data not prepared. Call prepare_data() first")
        
        model = self.models[model_id]
        if not model.is_trained:
            raise ValueError(f"Model '{model_id}' not trained")
        
        results = self.evaluator.evaluate_model(
            model, self.X_test, self.y_test,
            self.X_train, self.y_train, cv_folds
        )
        
        self.evaluation_results[model_id] = results
        
        print(f"Model '{model_id}' evaluated")
        return results
    
    def evaluate_all_models(self, cv_folds: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all trained models
        
        Args:
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary of all evaluation results
        """
        for model_id in self.models:
            if self.models[model_id].is_trained:
                print(f"Evaluating model: {model_id}")
                self.evaluate_model(model_id, cv_folds)
        
        print(f"Evaluated {len(self.evaluation_results)} models")
        return self.evaluation_results
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all evaluated models
        
        Returns:
            Comparison dataframe sorted by performance
        """
        if not self.evaluation_results:
            raise ValueError("No models evaluated yet")
        
        comparison = self.evaluator.compare_models(list(self.evaluation_results.values()))
        
        print("Model Comparison:")
        print(comparison.to_string(index=False))
        
        return comparison
    
    def get_best_model(self) -> Tuple[str, Dict[str, Any]]:
        """
        Get the best performing model
        
        Returns:
            Tuple of (model_id, model_object)
        """
        if not self.evaluation_results:
            raise ValueError("No models evaluated yet")
        
        # Determine best metric based on task type
        if self.task_type == 'classification':
            best_metric = 'accuracy'
        else:
            best_metric = 'r2_score'
        
        best_model_id = None
        best_score = -np.inf
        
        for model_id, results in self.evaluation_results.items():
            score = results.get(best_metric, -np.inf)
            if score > best_score:
                best_score = score
                best_model_id = model_id
        
        return best_model_id, self.models[best_model_id]
    
    def create_submission(self, test_data: pd.DataFrame, test_ids: pd.Series,
                         model_id: str = None, filename: str = 'submission.csv',
                         target_column: str = 'target') -> pd.DataFrame:
        """
        Create Kaggle submission using specified or best model
        
        Args:
            test_data: Test features
            test_ids: Test IDs
            model_id: Specific model to use (if None, uses best model)
            filename: Output filename
            target_column: Target column name
            
        Returns:
            Submission dataframe
        """
        if model_id is None:
            model_id, _ = self.get_best_model()
            print(f"Using best model: {model_id}")
        
        model = self.models[model_id]
        if not model.is_trained:
            raise ValueError(f"Model '{model_id}' not trained")
        
        predictions = model.predict(test_data)
        
        submission = self.evaluator.create_kaggle_submission(
            predictions, test_ids, filename, target_column
        )
        
        return submission
    
    def save_model(self, model_id: str, filepath: str) -> None:
        """Save a trained model"""
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not found")
        
        self.models[model_id].save_model(filepath)
        print(f"Model '{model_id}' saved to {filepath}")
    
    def load_model(self, model_id: str, filepath: str) -> 'ModelManager':
        """Load a trained model"""
        model = SklearnTrainer('random_forest', self.task_type)  # Placeholder
        model.load_model(filepath)
        self.models[model_id] = model
        
        print(f"Model loaded as '{model_id}' from {filepath}")
        return self
    
    def generate_full_report(self, output_file: str = None) -> str:
        """
        Generate comprehensive report for all models
        
        Args:
            output_file: Optional file to save the report
            
        Returns:
            Report string
        """
        if not self.evaluation_results:
            raise ValueError("No models evaluated yet")
        
        report = []
        report.append("="*80)
        report.append("KAGGLE FRAMEWORK - COMPLETE MODEL EVALUATION REPORT")
        report.append("="*80)
        report.append(f"Task Type: {self.task_type}")
        report.append(f"Total Models: {len(self.models)}")
        report.append(f"Evaluated Models: {len(self.evaluation_results)}")
        
        if self.X_train is not None:
            report.append(f"Training Samples: {len(self.X_train)}")
            report.append(f"Test Samples: {len(self.X_test)}")
            report.append(f"Features: {self.X_train.shape[1]}")
        
        report.append("\n" + "="*80)
        report.append("MODEL COMPARISON")
        report.append("="*80)
        
        comparison = self.compare_models()
        report.append(comparison.to_string(index=False))
        
        report.append("\n" + "="*80)
        report.append("DETAILED MODEL RESULTS")
        report.append("="*80)
        
        for model_id, results in self.evaluation_results.items():
            report.append(f"\nModel: {model_id}")
            report.append("-" * 40)
            
            if self.task_type == 'classification':
                report.append(f"Accuracy:  {results.get('accuracy', 0):.4f}")
                report.append(f"Precision: {results.get('precision', 0):.4f}")
                report.append(f"Recall:    {results.get('recall', 0):.4f}")
                report.append(f"F1-Score:  {results.get('f1_score', 0):.4f}")
                if 'roc_auc' in results:
                    report.append(f"ROC AUC:   {results['roc_auc']:.4f}")
            else:
                report.append(f"R² Score: {results.get('r2_score', 0):.4f}")
                report.append(f"RMSE:     {results.get('rmse', 0):.4f}")
                report.append(f"MAE:      {results.get('mae', 0):.4f}")
            
            if 'cv_scores' in results:
                cv = results['cv_scores']
                report.append(f"CV Mean:  {cv['mean']:.4f} (±{cv['std']:.4f})")
        
        # Best model
        best_model_id, _ = self.get_best_model()
        report.append(f"\n" + "="*80)
        report.append(f"BEST MODEL: {best_model_id}")
        report.append("="*80)
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Full report saved to {output_file}")
        
        return report_text
    
    def quick_experiment(self, X: pd.DataFrame, y: pd.Series,
                        algorithms: List[str] = None,
                        test_size: float = 0.2) -> pd.DataFrame:
        """
        Run a quick experiment with multiple algorithms
        
        Args:
            X: Features
            y: Target
            algorithms: List of algorithms to try
            test_size: Test split size
            
        Returns:
            Comparison results
        """
        if algorithms is None:
            if self.task_type == 'classification':
                algorithms = ['random_forest', 'gradient_boosting', 'logistic_regression']
            else:
                algorithms = ['random_forest', 'gradient_boosting', 'linear_regression']
        
        # Prepare data
        self.prepare_data(X, y, test_size)
        
        # Add and train models
        for algo in algorithms:
            self.add_model(algo, algo)
        
        self.train_all_models()
        self.evaluate_all_models()
        
        print("\nQuick Experiment Complete!")
        return self.compare_models()