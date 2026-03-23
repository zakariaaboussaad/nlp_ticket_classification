import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import logging
from typing import Dict, List, Tuple, Any
import os
from src.data_preprocessing import TicketDataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TicketEvaluator:
    """Comprehensive evaluator for ticket classification model."""
    
    def __init__(self, model_path: str = 'models/best_model.pt', 
                 data_path: str = 'data/processed_tickets.pt'):
        self.model_path = model_path
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and data
        self.load_model()
        self.load_data()
        
    def load_model(self):
        """Load trained model from checkpoint."""
        logger.info(f"Loading model from {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        config = checkpoint['config']
        
        self.model = BertForSequenceClassification.from_pretrained(
            config['model_name'],
            num_labels=config['num_labels']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = BertTokenizer.from_pretrained(config['model_name'])
        self.max_length = config['max_length']
        self.num_labels = config['num_labels']
        
        logger.info(f"Model loaded successfully with {self.num_labels} classes")
    
    def load_data(self):
        """Load processed data."""
        logger.info(f"Loading data from {self.data_path}")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data not found at {self.data_path}")
        
        self.processed_data = torch.load(self.data_path, map_location=self.device)
        self.categories = self.processed_data['metadata']['categories']
        
        logger.info("Data loaded successfully")
    
    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on a list of texts."""
        self.model.eval()
        
        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
        
        return predictions.cpu().numpy(), probabilities.cpu().numpy()
    
    def evaluate_dataset(self, split: str = 'test') -> Dict[str, Any]:
        """Evaluate model on a specific dataset split."""
        logger.info(f"Evaluating on {split} set")
        
        if split not in self.processed_data:
            raise ValueError(f"Split '{split}' not found in processed data")
        
        data = self.processed_data[split]
        texts = data['texts']
        true_labels = data['labels'].cpu().numpy()
        
        # Make predictions
        predictions, probabilities = self.predict(texts)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        f1_weighted = f1_score(true_labels, predictions, average='weighted')
        f1_macro = f1_score(true_labels, predictions, average='macro')
        f1_per_class = f1_score(true_labels, predictions, average=None)
        
        # Classification report
        class_report = classification_report(
            true_labels, predictions,
            target_names=list(self.categories.values()),
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        results = {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'f1_per_class': f1_per_class,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'true_labels': true_labels,
            'probabilities': probabilities,
            'texts': texts
        }
        
        logger.info(f"Accuracy: {accuracy:.4f}, F1-weighted: {f1_weighted:.4f}, F1-macro: {f1_macro:.4f}")
        
        return results
    
    def analyze_errors(self, results: Dict[str, Any], top_k: int = 5) -> Dict[str, Any]:
        """Analyze prediction errors and identify failure cases."""
        logger.info("Analyzing prediction errors...")
        
        true_labels = results['true_labels']
        predictions = results['predictions']
        texts = results['texts']
        probabilities = results['probabilities']
        
        # Find misclassified examples
        misclassified_mask = true_labels != predictions
        misclassified_indices = np.where(misclassified_mask)[0]
        
        error_analysis = {
            'total_errors': len(misclassified_indices),
            'error_rate': len(misclassified_indices) / len(true_labels),
            'errors_by_class': {},
            'top_confident_errors': [],
            'top_uncertain_errors': []
        }
        
        # Analyze errors by class
        for class_idx in range(self.num_labels):
            class_mask = true_labels == class_idx
            class_errors = np.sum((true_labels == class_idx) & (predictions != class_idx))
            class_total = np.sum(class_mask)
            
            error_analysis['errors_by_class'][class_idx] = {
                'class_name': self.categories[class_idx],
                'errors': class_errors,
                'total': class_total,
                'error_rate': class_errors / class_total if class_total > 0 else 0
            }
        
        # Get top confident errors (high confidence but wrong)
        error_confidences = np.max(probabilities[misclassified_mask], axis=1)
        top_confident_indices = misclassified_indices[np.argsort(error_confidences)[-top_k:]]
        
        for idx in top_confident_indices:
            error_analysis['top_confident_errors'].append({
                'text': texts[idx],
                'true_label': self.categories[true_labels[idx]],
                'predicted_label': self.categories[predictions[idx]],
                'confidence': float(np.max(probabilities[idx])),
                'true_prob': float(probabilities[idx][true_labels[idx]]),
                'pred_prob': float(probabilities[idx][predictions[idx]])
            })
        
        # Get top uncertain errors (low confidence overall)
        uncertain_confidences = np.max(probabilities[misclassified_mask], axis=1)
        top_uncertain_indices = misclassified_indices[np.argsort(uncertain_confidences)[:top_k]]
        
        for idx in top_uncertain_indices:
            error_analysis['top_uncertain_errors'].append({
                'text': texts[idx],
                'true_label': self.categories[true_labels[idx]],
                'predicted_label': self.categories[predictions[idx]],
                'confidence': float(np.max(probabilities[idx])),
                'true_prob': float(probabilities[idx][true_labels[idx]]),
                'pred_prob': float(probabilities[idx][predictions[idx]])
            })
        
        return error_analysis
    
    def generate_report(self, results: Dict[str, Any], error_analysis: Dict[str, Any], 
                       output_dir: str = 'reports') -> str:
        """Generate comprehensive evaluation report."""
        logger.info(f"Generating evaluation report in {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'evaluation_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Ticket Classification Evaluation Report\n\n")
            
            # Overall metrics
            f.write("## Overall Performance Metrics\n\n")
            f.write(f"- **Accuracy**: {results['accuracy']:.4f}\n")
            f.write(f"- **F1-Score (Weighted)**: {results['f1_weighted']:.4f}\n")
            f.write(f"- **F1-Score (Macro)**: {results['f1_macro']:.4f}\n\n")
            
            # Per-class metrics
            f.write("## Per-Class Performance\n\n")
            f.write("| Class | F1-Score | Precision | Recall | Support |\n")
            f.write("|-------|----------|-----------|--------|----------|\n")
            
            for class_idx in range(self.num_labels):
                class_name = self.categories[class_idx]
                class_metrics = results['classification_report'][class_name]
                f.write(f"| {class_name} | {class_metrics['f1-score']:.4f} | "
                       f"{class_metrics['precision']:.4f} | {class_metrics['recall']:.4f} | "
                       f"{int(class_metrics['support'])} |\n")
            
            f.write("\n")
            
            # Error analysis
            f.write("## Error Analysis\n\n")
            f.write(f"- **Total Errors**: {error_analysis['total_errors']}\n")
            f.write(f"- **Error Rate**: {error_analysis['error_rate']:.4f}\n\n")
            
            f.write("### Errors by Class\n\n")
            f.write("| Class | Errors | Total | Error Rate |\n")
            f.write("|-------|--------|-------|------------|\n")
            
            for class_idx, error_info in error_analysis['errors_by_class'].items():
                f.write(f"| {error_info['class_name']} | {error_info['errors']} | "
                       f"{error_info['total']} | {error_info['error_rate']:.4f} |\n")
            
            f.write("\n")
            
            # Top confident errors
            f.write("### Top Confident Errors (High Confidence but Wrong)\n\n")
            for i, error in enumerate(error_analysis['top_confident_errors'], 1):
                f.write(f"**{i}.** Confidence: {error['confidence']:.4f}\n")
                f.write(f"- **Text**: {error['text']}\n")
                f.write(f"- **True Label**: {error['true_label']}\n")
                f.write(f"- **Predicted**: {error['predicted_label']}\n")
                f.write(f"- **True Prob**: {error['true_prob']:.4f}\n")
                f.write(f"- **Pred Prob**: {error['pred_prob']:.4f}\n\n")
            
            # Top uncertain errors
            f.write("### Top Uncertain Errors (Low Confidence)\n\n")
            for i, error in enumerate(error_analysis['top_uncertain_errors'], 1):
                f.write(f"**{i}.** Confidence: {error['confidence']:.4f}\n")
                f.write(f"- **Text**: {error['text']}\n")
                f.write(f"- **True Label**: {error['true_label']}\n")
                f.write(f"- **Predicted**: {error['predicted_label']}\n")
                f.write(f"- **True Prob**: {error['true_prob']:.4f}\n")
                f.write(f"- **Pred Prob**: {error['pred_prob']:.4f}\n\n")
            
            # Trade-offs analysis
            f.write("## Precision vs Recall Trade-offs\n\n")
            f.write("Based on the per-class metrics, here are the key observations:\n\n")
            
            for class_idx in range(self.num_labels):
                class_name = self.categories[class_idx]
                class_metrics = results['classification_report'][class_name]
                precision = class_metrics['precision']
                recall = class_metrics['recall']
                
                if precision > recall:
                    f.write(f"- **{class_name}**: Higher precision ({precision:.3f}) than recall ({recall:.3f}). "
                           f"Model is conservative, may miss some cases but predictions are reliable.\n")
                elif recall > precision:
                    f.write(f"- **{class_name}**: Higher recall ({recall:.3f}) than precision ({precision:.3f}). "
                           f"Model catches most cases but may have false positives.\n")
                else:
                    f.write(f"- **{class_name}**: Balanced precision and recall ({precision:.3f}). "
                           f"Good trade-off between catching cases and avoiding false positives.\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("1. **Data Quality**: Consider collecting more examples for classes with high error rates.\n")
            f.write("2. **Model Thresholds**: For classes with low precision, consider adjusting decision thresholds.\n")
            f.write("3. **Feature Engineering**: Analyze common patterns in misclassified examples.\n")
            f.write("4. **Ensemble Methods**: Consider model ensembles for improved performance.\n")
        
        logger.info(f"Evaluation report saved to {report_path}")
        return report_path
    
    def plot_confusion_matrix(self, results: Dict[str, Any], output_dir: str = 'reports'):
        """Plot and save confusion matrix."""
        os.makedirs(output_dir, exist_ok=True)
        
        cm = results['confusion_matrix']
        class_names = list(self.categories.values())
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {plot_path}")
        return plot_path
    
    def plot_class_metrics(self, results: Dict[str, Any], output_dir: str = 'reports'):
        """Plot per-class metrics."""
        os.makedirs(output_dir, exist_ok=True)
        
        class_names = list(self.categories.values())
        precision = [results['classification_report'][name]['precision'] for name in class_names]
        recall = [results['classification_report'][name]['recall'] for name in class_names]
        f1 = [results['classification_report'][name]['f1-score'] for name in class_names]
        
        x = np.arange(len(class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'class_metrics.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Class metrics plot saved to {plot_path}")
        return plot_path

def main():
    """Main evaluation function."""
    logger.info("Starting model evaluation...")
    
    # Initialize evaluator
    evaluator = TicketEvaluator()
    
    # Evaluate on test set
    results = evaluator.evaluate_dataset('test')
    
    # Analyze errors
    error_analysis = evaluator.analyze_errors(results, top_k=5)
    
    # Generate report
    report_path = evaluator.generate_report(results, error_analysis)
    
    # Generate plots
    cm_plot = evaluator.plot_confusion_matrix(results)
    metrics_plot = evaluator.plot_class_metrics(results)
    
    logger.info("Evaluation completed successfully!")
    logger.info(f"Report: {report_path}")
    logger.info(f"Confusion Matrix: {cm_plot}")
    logger.info(f"Metrics Plot: {metrics_plot}")

if __name__ == "__main__":
    main()
