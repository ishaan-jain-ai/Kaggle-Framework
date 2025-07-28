import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score, learning_curve
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Comprehensive model evaluation class for both classification and regression tasks.
    Provides detailed metrics, visualizations, and Kaggle submission capabilities.
    """
    
    def __init__(self, task_type: str = 'classification'):
        """
        Initialize the evaluator
        
        Args:
            task_type: 'classification' or 'regression'
        """
        self.task_type = task_type
        self.evaluation_results = {}
        
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                      X_train: pd.DataFrame = None, y_train: pd.Series = None,
                      cv_folds: int = 5) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model with predict method
            X_test: Test features
            y_test: Test target
            X_train: Training features (optional, for learning curves)
            y_train: Training target (optional, for learning curves)
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing all evaluation metrics and results
        """
        results = {
            'model_name': getattr(model, 'model_name', 'Unknown'),
            'task_type': self.task_type,
            'test_samples': len(X_test),
            'feature_count': len(X_test.columns)
        }
        
        # Get predictions
        y_pred = model.predict(X_test)
        results['predictions'] = y_pred
        
        if self.task_type == 'classification':
            results.update(self._evaluate_classification(model, X_test, y_test, y_pred))
        else:
            results.update(self._evaluate_regression(X_test, y_test, y_pred))
        
        # Cross-validation scores if training data provided
        if X_train is not None and y_train is not None:
            cv_scores = cross_val_score(model.model, X_train, y_train, cv=cv_folds)
            results['cv_scores'] = {
                'mean': cv_scores.mean(),
                'std': cv_scores.std(),
                'scores': cv_scores.tolist()
            }
        
        # Feature importance if available
        if hasattr(model, 'get_feature_importance'):
            feature_importance = model.get_feature_importance()
            if feature_importance is not None:
                results['feature_importance'] = feature_importance
        
        self.evaluation_results = results
        return results
    
    def _evaluate_classification(self, model, X_test: pd.DataFrame, 
                               y_test: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
        """Evaluate classification model"""
        results = {}
        
        # Basic metrics
        results['accuracy'] = accuracy_score(y_test, y_pred)
        results['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        results['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        results['f1_score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        results['confusion_matrix'] = cm
        
        # Classification report
        results['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        
        # ROC AUC for binary classification
        if len(np.unique(y_test)) == 2:
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    results['roc_auc'] = roc_auc_score(y_test, y_proba)
                    results['probabilities'] = y_proba
                else:
                    results['roc_auc'] = roc_auc_score(y_test, y_pred)
            except Exception as e:
                results['roc_auc_error'] = str(e)
        
        return results
    
    def _evaluate_regression(self, X_test: pd.DataFrame, 
                           y_test: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
        """Evaluate regression model"""
        results = {}
        
        results['mse'] = mean_squared_error(y_test, y_pred)
        results['rmse'] = np.sqrt(results['mse'])
        results['mae'] = mean_absolute_error(y_test, y_pred)
        results['r2_score'] = r2_score(y_test, y_pred)
        
        # Additional metrics
        results['mean_residual'] = np.mean(y_test - y_pred)
        results['std_residual'] = np.std(y_test - y_pred)
        
        return results
    
    def plot_evaluation_results(self, save_path: str = None) -> None:
        """Create evaluation plots"""
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_model first.")
        
        if self.task_type == 'classification':
            self._plot_classification_results(save_path)
        else:
            self._plot_regression_results(save_path)
    
    def _plot_classification_results(self, save_path: str = None) -> None:
        """Plot classification evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Classification Evaluation: {self.evaluation_results["model_name"]}', fontsize=16)
        
        # Confusion Matrix
        cm = self.evaluation_results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 0], cmap='Blues')
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # Metrics Bar Plot
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_values = [self.evaluation_results.get(m, 0) for m in metrics]
        axes[0, 1].bar(metrics, metric_values)
        axes[0, 1].set_title('Classification Metrics')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # ROC Curve (if available)
        if 'roc_auc' in self.evaluation_results and 'probabilities' in self.evaluation_results:
            y_test = self.evaluation_results.get('y_test')  # Would need to store this
            y_proba = self.evaluation_results['probabilities']
            if y_test is not None:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                axes[1, 0].plot(fpr, tpr, label=f'ROC Curve (AUC = {self.evaluation_results["roc_auc"]:.3f})')
                axes[1, 0].plot([0, 1], [0, 1], 'k--')
                axes[1, 0].set_xlabel('False Positive Rate')
                axes[1, 0].set_ylabel('True Positive Rate')
                axes[1, 0].set_title('ROC Curve')
                axes[1, 0].legend()
        
        # Feature Importance (if available)
        if 'feature_importance' in self.evaluation_results:
            importance_df = self.evaluation_results['feature_importance']
            top_features = importance_df.head(10)
            axes[1, 1].barh(range(len(top_features)), top_features['importance'])
            axes[1, 1].set_yticks(range(len(top_features)))
            axes[1, 1].set_yticklabels(top_features['feature'])
            axes[1, 1].set_title('Top 10 Feature Importance')
            axes[1, 1].set_xlabel('Importance')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_regression_results(self, save_path: str = None) -> None:
        """Plot regression evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Regression Evaluation: {self.evaluation_results["model_name"]}', fontsize=16)
        
        # This would need actual vs predicted values stored in evaluation_results
        # For now, showing placeholder structure
        
        # Metrics Bar Plot
        metrics = ['r2_score', 'rmse', 'mae']
        metric_values = [self.evaluation_results.get(m, 0) for m in metrics]
        axes[0, 0].bar(metrics, metric_values)
        axes[0, 0].set_title('Regression Metrics')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, output_file: str = None) -> str:
        """Generate a comprehensive evaluation report"""
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_model first.")
        
        report = []
        report.append("="*60)
        report.append(f"MODEL EVALUATION REPORT")
        report.append("="*60)
        report.append(f"Model: {self.evaluation_results['model_name']}")
        report.append(f"Task Type: {self.evaluation_results['task_type']}")
        report.append(f"Test Samples: {self.evaluation_results['test_samples']}")
        report.append(f"Features: {self.evaluation_results['feature_count']}")
        report.append("-"*60)
        
        if self.task_type == 'classification':
            report.append("CLASSIFICATION METRICS:")
            report.append(f"Accuracy:  {self.evaluation_results.get('accuracy', 0):.4f}")
            report.append(f"Precision: {self.evaluation_results.get('precision', 0):.4f}")
            report.append(f"Recall:    {self.evaluation_results.get('recall', 0):.4f}")
            report.append(f"F1-Score:  {self.evaluation_results.get('f1_score', 0):.4f}")
            if 'roc_auc' in self.evaluation_results:
                report.append(f"ROC AUC:   {self.evaluation_results['roc_auc']:.4f}")
        else:
            report.append("REGRESSION METRICS:")
            report.append(f"R² Score: {self.evaluation_results.get('r2_score', 0):.4f}")
            report.append(f"RMSE:     {self.evaluation_results.get('rmse', 0):.4f}")
            report.append(f"MAE:      {self.evaluation_results.get('mae', 0):.4f}")
            report.append(f"MSE:      {self.evaluation_results.get('mse', 0):.4f}")
        
        if 'cv_scores' in self.evaluation_results:
            cv = self.evaluation_results['cv_scores']
            report.append("-"*60)
            report.append("CROSS-VALIDATION:")
            report.append(f"CV Mean:  {cv['mean']:.4f}")
            report.append(f"CV Std:   {cv['std']:.4f}")
        
        report.append("="*60)
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def create_kaggle_submission(self, test_predictions: np.ndarray, 
                               test_ids: Union[pd.Series, np.ndarray],
                               filename: str = 'submission.csv',
                               target_column: str = 'target') -> pd.DataFrame:
        """
        Create a Kaggle submission file
        
        Args:
            test_predictions: Model predictions for test set
            test_ids: Test set IDs (usually 'id' column)
            filename: Output filename
            target_column: Name of the target column
            
        Returns:
            Submission dataframe
        """
        submission = pd.DataFrame({
            'id': test_ids,
            target_column: test_predictions
        })
        
        submission.to_csv(filename, index=False)
        print(f"Submission saved to {filename}")
        print(f"Submission shape: {submission.shape}")
        
        return submission
    
    def compare_models(self, evaluation_results_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple model evaluation results
        
        Args:
            evaluation_results_list: List of evaluation result dictionaries
            
        Returns:
            Comparison dataframe
        """
        comparison_data = []
        
        for results in evaluation_results_list:
            model_data = {'model_name': results.get('model_name', 'Unknown')}
            
            if self.task_type == 'classification':
                model_data.update({
                    'accuracy': results.get('accuracy', 0),
                    'precision': results.get('precision', 0),
                    'recall': results.get('recall', 0),
                    'f1_score': results.get('f1_score', 0),
                    'roc_auc': results.get('roc_auc', 0)
                })
            else:
                model_data.update({
                    'r2_score': results.get('r2_score', 0),
                    'rmse': results.get('rmse', 0),
                    'mae': results.get('mae', 0),
                    'mse': results.get('mse', 0)
                })
            
            if 'cv_scores' in results:
                model_data['cv_mean'] = results['cv_scores']['mean']
                model_data['cv_std'] = results['cv_scores']['std']
            
            comparison_data.append(model_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df.sort_values(
            'accuracy' if self.task_type == 'classification' else 'r2_score', 
            ascending=False
        )